"""
LIDA Kinetics endpoints — Lesion-Induced DNA Amplification.

Three endpoints, registered as a Flask Blueprint into the existing app:
    POST /api/kinetics-fit              — replication-kinetics ODE fit per group
    POST /api/kinetics-suggest          — RandomForest active learning
    POST /api/kinetics-sequence-predict — greedy sequence optimization

The kinetic model below is the same antimony-defined replication system that
is used in the wet-lab Tellurium notebook. It is implemented here directly
via scipy.integrate.solve_ivp so the backend does not require the heavy
Tellurium / libroadrunner native stack on Render — same math, lighter deploy.

Antimony reference (matches the notebook verbatim):

    model replication_kinetics
        var A, a, B, b, Aa, Bb, R
        J1: A + a -> Aa;     ku * A * a
        J2: B + b -> Bb;     ku * B * b
        J3: Aa + Bb -> R;    k1 * Aa * Bb - k2 * R
        J4: A + a + Bb -> R; kr * A * a * R
        J5: B + b + Aa -> R; kr * B * b * R
        ku = 0.01; k1 = 1.0; k2 = 1.0; kr = 1.0;
        A = 2.8; a = 2.8; B = 1.4; b = 1.4; Aa = 0; Bb = 0; R = 0;
    end
"""
from flask import Blueprint, request, jsonify
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from sklearn.ensemble import RandomForestRegressor

lida_bp = Blueprint('lida_kinetics', __name__, url_prefix='/api')

BASES = ['A', 'C', 'G', 'T']

# Default initial concentrations (uM) — match the notebook's antimony defaults.
DEFAULT_A0 = 2.8
DEFAULT_B0 = 1.4
DEFAULT_LIMIT_UM = 1.4   # Max yieldable [R] (uM); matches LIMIT_B in the notebook.


# ════════════════════════════════════════════════════════════════════════
# Replication kinetics ODE
# ════════════════════════════════════════════════════════════════════════

def _replication_ode(t, y, ku, k1, k2, kr):
    A, a, B, b, Aa, Bb, R = y
    rJ1 = ku * A * a               # A + a -> Aa
    rJ2 = ku * B * b               # B + b -> Bb
    rJ3 = k1 * Aa * Bb - k2 * R    # Aa + Bb <-> R
    rJ4 = kr * A * a * R           # A + a + Bb -> R   (autocatalytic in R)
    rJ5 = kr * B * b * R           # B + b + Aa -> R   (autocatalytic in R)

    dA  = -rJ1 - rJ4
    da  = -rJ1 - rJ4
    dB  = -rJ2 - rJ5
    db  = -rJ2 - rJ5
    dAa =  rJ1 - rJ3 - rJ5         # Aa consumed by J3 (forward) and J5
    dBb =  rJ2 - rJ3 - rJ4         # Bb consumed by J3 (forward) and J4
    dR  =  rJ3 + rJ4 + rJ5
    return [dA, da, dB, db, dAa, dBb, dR]


def _simulate_R_at(params, initial_R, t_eval, A0, B0):
    """Integrate the replication ODE and return [R] at the supplied time points.
    Uses LSODA in the fitting hot loop — fast, adequate accuracy for the optimizer
    to find the correct minimum. BDF is reserved for the display-quality dense pass."""
    ku, k1, k2, kr = params
    y0 = [A0, A0, B0, B0, 0.0, 0.0, float(initial_R)]
    t_max = float(t_eval[-1]) + 1e-6
    try:
        sol = solve_ivp(
            _replication_ode, (0.0, t_max), y0,
            args=(ku, k1, k2, kr),
            t_eval=t_eval, method='LSODA',
            rtol=1e-5, atol=1e-8,
        )
        if not sol.success:
            return None
        return sol.y[6]
    except Exception:
        return None


def _simulate_R_dense(params, initial_R, t_max, A0, B0, n=100):
    """Fine-grid simulation used once per group for the frontend overlay."""
    ku, k1, k2, kr = params
    y0 = [A0, A0, B0, B0, 0.0, 0.0, float(initial_R)]
    t_grid = np.linspace(0.0, t_max, n)
    try:
        sol = solve_ivp(
            _replication_ode, (0.0, t_max), y0,
            args=(ku, k1, k2, kr),
            t_eval=t_grid, method='BDF',
            rtol=1e-8, atol=1e-10,
        )
        if not sol.success:
            return None, None
        return sol.t, sol.y[6]
    except Exception:
        return None, None


def _residuals(params, t_data, y_data, initial_R, A0, B0):
    """Concentration-domain residuals; large penalty for negative-slope simulations."""
    if np.any(np.array(params) < 0):
        return np.full_like(y_data, 1e6)
    y_sim = _simulate_R_at(params, initial_R, t_data, A0, B0)
    if y_sim is None or np.any(np.isnan(y_sim)):
        return np.full_like(y_data, 1e6)
    diffs = np.diff(y_sim)
    penalty = 1e6 if np.any(diffs < -1e-5) else 0.0
    return (y_sim - y_data) + penalty


def _build_initial_guesses():
    """14 multi-start seeds matching the notebook exactly:
    4 hand-picked + 10 log-uniform random (numpy seed 42), k2 pinned to 1e-11."""
    rng = np.random.RandomState(42)
    fixed = [
        [1e-6, 1e-3, 1e-11, 1e-6],
        [1e-3, 1e-1, 1e-11, 1e-1],
        [1e-2, 1.0,  1e-11, 1.0 ],
        [1e-5, 1.0,  1e-11, 10.0],
    ]
    random_starts = []
    for _ in range(10):
        g = np.power(10, rng.uniform(-4, 1.5, size=4)).tolist()
        g[2] = 1e-11
        random_starts.append(g)
    return fixed + random_starts


# ════════════════════════════════════════════════════════════════════════
# Featurisation for AL/sequence-prediction endpoints
# ════════════════════════════════════════════════════════════════════════

def _encode_sequence(seq, max_len):
    out = np.zeros(max_len * 4, dtype=float)
    for i, b in enumerate(seq[:max_len]):
        if b in BASES:
            out[i * 4 + BASES.index(b)] = 1.0
    return out


def _featurize(exp, max_len, env_vocab):
    c = exp['conditions']
    env_oh = np.zeros(len(env_vocab))
    if c.get('env') in env_vocab:
        env_oh[env_vocab.index(c['env'])] = 1.0
    return np.concatenate([
        np.array([
            float(c.get('temperature', 0)),
            float(c.get('ligase', 0)),
            float(c.get('atp', 0)),
            float(c.get('mg2', 0)),
        ], dtype=float),
        env_oh,
        _encode_sequence(exp.get('sequence', ''), max_len),
    ])


def _max_yield(exp):
    tc = exp.get('timeCourse', [])
    if not tc:
        return 0.0
    return float(max(p['conversion'] for p in tc))


# ════════════════════════════════════════════════════════════════════════
# Endpoints
# ════════════════════════════════════════════════════════════════════════

@lida_bp.route('/kinetics-fit', methods=['POST'])
def kinetics_fit():
    """
    Fit the replication-kinetics ODE per experiment group. Each group's
    `timeCourse` is treated as conversion (%); we convert to [R] in uM by
    multiplying by `limit_uM` / 100 and then fit (ku, k1, k2, kr).

    Request:
      {
        experiments: [{ groupId, sequence, conditions: {temperature, ligase, atp, mg2, env}, timeCourse: [{time, conversion}], limit_uM? }],
        limit_uM?: 1.4,    # default per-group ceiling concentration in uM
        A0?: 2.8,          # initial [A] = [a] (uM), default 2.8
        B0?: 1.4           # initial [B] = [b] (uM), default 1.4
      }

    Response:
      { fits: [{ groupId, model, ku, k1, k2, kr, k_bg, limit_uM, seed_uM, cost, simT, simY }],
        summary: { topGroups, trendNotes } }
    """
    body = request.get_json(force=True) or {}
    experiments = body.get('experiments', [])
    default_limit = float(body.get('limit_uM', DEFAULT_LIMIT_UM))
    A0 = float(body.get('A0', DEFAULT_A0))
    B0 = float(body.get('B0', DEFAULT_B0))

    fits = []
    guesses = _build_initial_guesses()

    for g in experiments:
        gid = g.get('groupId')
        tc = g.get('timeCourse', [])
        if len(tc) < 3:
            fits.append({'groupId': gid, 'model': None,
                         'ku': None, 'k1': None, 'k2': None, 'kr': None, 'k_bg': None,
                         'limit_uM': None, 'seed_uM': None, 'cost': None,
                         'simT': [], 'simY': [], 'note': 'Need at least 3 timepoints.'})
            continue

        limit_uM = float(g.get('limit_uM', default_limit))
        env = ((g.get('conditions') or {}).get('env') or '')
        seed_pct = 0.05 if 'seed' in str(env).lower() else 0.0

        t_data = np.array([float(p['time']) for p in tc], dtype=float)
        y_pct = np.array([float(p['conversion']) for p in tc], dtype=float)
        order = np.argsort(t_data)
        t_data = t_data[order]
        y_pct = y_pct[order]

        # Convert conversion (%) → concentration (uM)
        y_data = (y_pct / 100.0) * limit_uM
        # Initial [R] = 5% of mean observed conversion if env contains "seed", else 0
        initial_R = float(seed_pct * np.mean(y_data))

        best_res, best_cost = None, np.inf
        for p0 in guesses:
            try:
                res = least_squares(
                    _residuals, p0,
                    args=(t_data, y_data, initial_R, A0, B0),
                    bounds=([0.0, 0.0, 0.0, 0.0], [100.0, 100.0, 1e-10, 100.0]),
                    ftol=1e-8, max_nfev=300,
                )
                if res.success and res.cost < best_cost:
                    best_cost = res.cost
                    best_res = res
            except Exception:
                continue

        if best_res is None:
            fits.append({'groupId': gid, 'model': None,
                         'ku': None, 'k1': None, 'k2': None, 'kr': None, 'k_bg': None,
                         'limit_uM': limit_uM, 'seed_uM': initial_R, 'cost': None,
                         'simT': [], 'simY': [], 'note': 'Fit did not converge.'})
            continue

        ku, k1, k2, kr = best_res.x
        # High-resolution simulated curve for frontend overlay (returned in % units).
        sim_t, sim_R = _simulate_R_dense(best_res.x, initial_R, float(t_data[-1]) + 5.0, A0, B0)
        sim_y_pct = (sim_R / limit_uM) * 100.0 if sim_R is not None else None

        fits.append({
            'groupId': gid,
            'model': 'replication_kinetics',
            'ku':    float(ku),
            'k1':    float(k1),
            'k2':    float(k2),
            'kr':    float(kr),
            'k_bg':  float(ku * k1),       # background formation rate ku·k1
            'limit_uM': limit_uM,
            'seed_uM':  initial_R,
            'cost':     float(best_cost),
            'simT':  sim_t.tolist() if sim_t is not None else [],
            'simY':  sim_y_pct.tolist() if sim_y_pct is not None else [],
        })

    ranked = sorted(experiments, key=_max_yield, reverse=True)[:5]
    top_max = _max_yield(ranked[0]) if ranked else 0.0
    fitted = sum(1 for f in fits if f.get('ku') is not None)
    return jsonify({
        'fits': fits,
        'summary': {
            'topGroups': [{'groupId': g.get('groupId'), 'maxConversion': _max_yield(g)} for g in ranked],
            'trendNotes': f"{fitted}/{len(fits)} groups fitted with replication kinetics; top yield {top_max:.1f}%.",
        },
    })


@lida_bp.route('/kinetics-suggest', methods=['POST'])
def kinetics_suggest():
    body = request.get_json(force=True) or {}
    exps = body.get('experiments', [])
    ranges = body.get('ranges', {}) or {}
    n = int(body.get('nSuggestions', 12))
    strategy = body.get('strategy', 'exploit')
    if not exps:
        return jsonify({'suggestions': []})

    max_len = max(len(e.get('sequence', '')) for e in exps) or 1
    env_vocab = sorted({(e.get('conditions') or {}).get('env', 'none') for e in exps})

    X = np.array([_featurize(e, max_len, env_vocab) for e in exps])
    y = np.array([_max_yield(e) for e in exps])
    rf = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=1).fit(X, y)

    rng = np.random.default_rng(42)
    n_trials = max(200, n * 30)

    # Generate all candidate experiments first, then score them in one batch.
    cand_exps = []
    for _ in range(n_trials):
        cand_exps.append({
            'sequence': ''.join(rng.choice(BASES, size=max_len)),
            'conditions': {
                'temperature': float(rng.uniform(ranges.get('tMin', 20),    ranges.get('tMax', 50))),
                'ligase':      float(rng.uniform(ranges.get('ligaseMin', 0), ranges.get('ligaseMax', 10))),
                'atp':         float(rng.uniform(ranges.get('atpMin', 0),   ranges.get('atpMax', 10))),
                'mg2':         float(rng.uniform(ranges.get('mg2Min', 0),   ranges.get('mg2Max', 20))),
                'env':         str(rng.choice(env_vocab)),
            },
        })

    Xc = np.asarray([_featurize(c, max_len, env_vocab) for c in cand_exps])
    # Vectorised: per-tree predictions for ALL candidates in a single sweep.
    # Shape: (n_estimators, n_trials). ~200x faster than per-row predict on Render.
    per_tree = np.asarray([tree.predict(Xc) for tree in rf.estimators_])
    means = per_tree.mean(axis=0)
    stds  = per_tree.std(axis=0)

    candidates = [
        {**cand_exps[i],
         'predicted_conversion': float(means[i]),
         'uncertainty':         float(stds[i])}
        for i in range(len(cand_exps))
    ]

    key = (lambda c: -c['uncertainty']) if strategy == 'explore' else (lambda c: -c['predicted_conversion'])
    candidates.sort(key=key)
    return jsonify({'suggestions': candidates[:n]})


@lida_bp.route('/kinetics-sequence-predict', methods=['POST'])
def kinetics_sequence_predict():
    body = request.get_json(force=True) or {}
    exps = body.get('experiments', [])
    fixed = body.get('fixedConditions')
    topK = int(body.get('topK', 5))
    if not exps:
        return jsonify({'candidates': []})

    max_len = max(len(e.get('sequence', '')) for e in exps) or 1
    env_vocab = sorted({(e.get('conditions') or {}).get('env', 'none') for e in exps})

    X = np.array([_featurize(e, max_len, env_vocab) for e in exps])
    y = np.array([_max_yield(e) for e in exps])
    rf = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=1).fit(X, y)

    best_exp = max(exps, key=_max_yield)
    cond = fixed or best_exp['conditions']
    seq = list(best_exp.get('sequence', 'A' * max_len))
    if len(seq) < max_len:
        seq += ['A'] * (max_len - len(seq))

    def score_position_variants(seq_list, pos):
        """Return rf-predicted yield for [seq with base=A,C,G,T at `pos`] in one batched predict."""
        feats = []
        for b in BASES:
            tmp = seq_list.copy()
            tmp[pos] = b
            feats.append(_featurize({'sequence': ''.join(tmp), 'conditions': cond}, max_len, env_vocab))
        return rf.predict(np.asarray(feats))  # shape (4,)

    rng = np.random.default_rng()
    candidates = []
    MAX_PASSES = 3   # bound greedy hill-climb time on Render's shared CPU
    for _ in range(topK):
        improved, passes = True, 0
        while improved and passes < MAX_PASSES:
            improved = False
            passes += 1
            for i in range(max_len):
                preds = score_position_variants(seq, i)
                best_idx = int(np.argmax(preds))
                if seq[i] != BASES[best_idx]:
                    seq[i] = BASES[best_idx]
                    improved = True

        # Per-position score grid for the optimum (one batched predict per position).
        per_pos = []
        for i in range(max_len):
            preds = score_position_variants(seq, i)
            per_pos.append({BASES[k]: float(preds[k]) for k in range(4)})

        feat = _featurize({'sequence': ''.join(seq), 'conditions': cond}, max_len, env_vocab).reshape(1, -1)
        candidates.append({
            'sequence': ''.join(seq),
            'predicted_conversion': float(rf.predict(feat)[0]),
            'perPositionScore': per_pos,
        })

        # Perturb one position to seed the next candidate.
        idx = int(rng.integers(0, max_len))
        seq[idx] = str(rng.choice([b for b in BASES if b != seq[idx]]))

    return jsonify({'candidates': candidates})
