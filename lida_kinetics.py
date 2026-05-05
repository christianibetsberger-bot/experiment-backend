"""
LIDA Kinetics endpoints — Lesion-Induced DNA Amplification.

Three endpoints, registered as a Flask Blueprint into the existing app:
    POST /api/kinetics-fit              — replication-kinetics ODE fit per group
    POST /api/kinetics-suggest          — METIS Bayesian active learning (XGBoost ensemble + UCB)
    POST /api/kinetics-sequence-predict — greedy sequence optimisation via XGBoost ensemble

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

Active-learning strategy — METIS (doi:10.1101/2021.12.28.474323):
    UCB = exploitation × mean_ensemble + kappa × std_ensemble
    kappa = 0   → pure exploit (maximise predicted yield)
    kappa = √2  → balanced (classic UCB1 constant, default)
    kappa = 3   → pure explore (maximise model uncertainty)
"""
from flask import Blueprint, request, jsonify
import time
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

lida_bp = Blueprint('lida_kinetics', __name__, url_prefix='/api')

BASES = ['A', 'C', 'G', 'T']

DEFAULT_A0       = 2.8
DEFAULT_B0       = 1.4
DEFAULT_LIMIT_UM = 1.4   # Max yieldable [R] (µM); matches LIMIT_B in the Tellurium notebook.

# METIS hyperparameter grid — identical to the published METIS notebook.
_PARAM_GRID = {
    'learning_rate':    [0.01, 0.03, 0.1, 0.3],
    'colsample_bytree': [0.6, 0.8, 0.9, 1.0],
    'subsample':        [0.6, 0.8, 0.9, 1.0],
    'max_depth':        [2, 3, 4, 6, 8],
    'n_estimators':     [20, 40, 80, 150, 300],
    'reg_lambda':       [1, 1.5, 2],
    'gamma':            [0, 0.1, 0.4],
    'min_child_weight': [1, 2, 4],
}


# ════════════════════════════════════════════════════════════════════════
# Replication-kinetics ODE  (unchanged)
# ════════════════════════════════════════════════════════════════════════

def _replication_ode(t, y, ku, k1, k2, kr):
    A, a, B, b, Aa, Bb, R = y
    rJ1 = ku * A * a
    rJ2 = ku * B * b
    rJ3 = k1 * Aa * Bb - k2 * R
    rJ4 = kr * A * a * R
    rJ5 = kr * B * b * R
    dA  = -rJ1 - rJ4
    da  = -rJ1 - rJ4
    dB  = -rJ2 - rJ5
    db  = -rJ2 - rJ5
    dAa =  rJ1 - rJ3 - rJ5
    dBb =  rJ2 - rJ3 - rJ4
    dR  =  rJ3 + rJ4 + rJ5
    return [dA, da, dB, db, dAa, dBb, dR]


def _simulate_R_at(params, initial_R, t_eval, A0, B0, rtol=1e-4, atol=1e-7):
    ku, k1, k2, kr = params
    y0 = [A0, A0, B0, B0, 0.0, 0.0, float(initial_R)]
    t_max = float(t_eval[-1]) + 1e-6
    try:
        sol = solve_ivp(
            _replication_ode, (0.0, t_max), y0,
            args=(ku, k1, k2, kr),
            t_eval=t_eval, method='LSODA',
            rtol=rtol, atol=atol,
        )
        if not sol.success:
            return None
        R = sol.y[6]
        return None if (np.any(np.isnan(R)) or np.any(np.isinf(R))) else R
    except Exception:
        return None


def _simulate_R_dense(params, initial_R, t_max, A0, B0, n=100):
    """
    Dense post-fit simulation for plotting. Tries progressively looser tolerances
    and a fallback solver before giving up. Tight tolerances (1e-6/1e-9) used to
    silently fail on borderline-stiff parameter combinations even though the fit
    itself converged at 1e-4/1e-7 — that left ku/k1/k2/kr populated but simT/simY
    empty, so the curve never rendered in the UI.
    """
    ku, k1, k2, kr = params
    y0 = [A0, A0, B0, B0, 0.0, 0.0, float(initial_R)]
    t_grid = np.linspace(0.0, t_max, n)

    # (method, rtol, atol) — start with the same tolerance as the fit, fall back
    # to looser settings, then a non-stiff solver as a last resort.
    attempts = [
        ('LSODA', 1e-4, 1e-7),
        ('LSODA', 1e-3, 1e-6),
        ('LSODA', 1e-2, 1e-5),
        ('RK45',  1e-3, 1e-6),
    ]
    for method, rtol, atol in attempts:
        try:
            sol = solve_ivp(
                _replication_ode, (0.0, t_max), y0,
                args=(ku, k1, k2, kr),
                t_eval=t_grid, method=method,
                rtol=rtol, atol=atol,
            )
            if not sol.success:
                continue
            R = sol.y[6]
            if np.any(np.isnan(R)) or np.any(np.isinf(R)):
                continue
            return sol.t, R
        except Exception:
            continue
    return None, None


def _safe_floats(arr):
    """Convert array to list, replacing NaN/Inf with None for valid JSON."""
    return [float(v) if np.isfinite(v) else None for v in arr]


def _residuals(params, t_data, y_data, initial_R, A0, B0):
    if np.any(np.array(params) < 0):
        return np.full_like(y_data, 1e6)
    y_sim = _simulate_R_at(params, initial_R, t_data, A0, B0)
    if y_sim is None or np.any(np.isnan(y_sim)):
        return np.full_like(y_data, 1e6)
    penalty = 1e6 if np.any(np.diff(y_sim) < -1e-5) else 0.0
    return (y_sim - y_data) + penalty


def _build_initial_guesses():
    # Three guesses spanning the biologically relevant range.
    # Removed the extreme-slow guess (1e-6) to reduce per-group compute time.
    return [
        [1e-3, 1e-1, 1e-11, 1e-1],
        [1e-2, 1.0,  1e-11, 1.0 ],
        [1e-5, 1.0,  1e-11, 10.0],
    ]


# ════════════════════════════════════════════════════════════════════════
# Feature encoding  (unchanged)
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
    """Return max conversion (%). Accepts pre-computed maxConversion from the frontend."""
    if 'maxConversion' in exp:
        return float(exp['maxConversion'])
    tc = exp.get('timeCourse', [])
    return float(max(p['conversion'] for p in tc)) if tc else 0.0


# ════════════════════════════════════════════════════════════════════════
# METIS core: ensemble builder + UCB scorer
# ════════════════════════════════════════════════════════════════════════

def _build_ensemble(X, y, ensemble_size=20, n_iter=50):
    """Train a METIS-style XGBoost ensemble via RandomizedSearchCV.

    n_iter is capped relative to dataset size so Render's shared CPU stays
    responsive at every campaign stage:
      < 5 data points  → skip RandomCV, use sensible defaults directly
      5–20             → n_iter = max(10, n_samples × 2)
      > 20             → min(n_iter, 100)
    """
    n = len(y)
    cv_folds = min(5, n)

    if n < 5:
        # Too few samples for cross-validated search; use fixed default params.
        m = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=40, learning_rate=0.1, max_depth=3,
            subsample=0.9, colsample_bytree=0.9, random_state=0,
        )
        m.fit(X, y)
        return [m] * min(ensemble_size, 5)

    effective_n_iter = min(n_iter, max(10, n * 2)) if n <= 20 else min(n_iter, 100)

    grid = RandomizedSearchCV(
        estimator=XGBRegressor(objective='reg:squarederror'),
        param_distributions=_PARAM_GRID,
        cv=cv_folds,
        scoring='neg_mean_absolute_error',
        n_jobs=1,          # Render shared tier: single thread avoids OOM
        n_iter=effective_n_iter,
        random_state=0,
    )
    grid.fit(X, y)

    results = (
        pd.DataFrame(grid.cv_results_)
        .sort_values('mean_test_score', ascending=False)
    )
    ensemble = []
    for params in results['params'].iloc[:ensemble_size]:
        m = XGBRegressor(objective='reg:squarederror', **params)
        m.fit(X, y)
        ensemble.append(m)
    return ensemble


def _ucb_score(ensemble, X_cand, kappa, exploitation=1.0):
    """Vectorised UCB acquisition over the ensemble.

    preds shape: (n_candidates, n_models)
    Returns (mean, std, ucb) each of shape (n_candidates,).
    """
    preds = np.column_stack([m.predict(X_cand) for m in ensemble])
    mean  = preds.mean(axis=1)
    std   = preds.std(axis=1)
    ucb   = exploitation * mean + kappa * std
    return mean, std, ucb


# ════════════════════════════════════════════════════════════════════════
# Endpoint: /api/kinetics-fit  (ODE fitting — unchanged logic)
# ════════════════════════════════════════════════════════════════════════

@lida_bp.route('/kinetics-fit', methods=['POST'])
def kinetics_fit():
    """
    Fit the replication-kinetics ODE per experiment group.

    Request:
      {
        experiments: [{ groupId, sequence,
                         conditions: {temperature, ligase, atp, mg2, env},
                         timeCourse: [{time, conversion}], limit_uM? }],
        limit_uM?: 1.4,
        A0?: 2.8,
        B0?: 1.4
      }

    Response:
      { fits: [{ groupId, model, ku, k1, k2, kr, k_bg,
                 limit_uM, seed_uM, cost, simT, simY }],
        summary: { topGroups, trendNotes } }
    """
    body        = request.get_json(force=True) or {}
    experiments = body.get('experiments', [])
    default_lim = float(body.get('limit_uM', DEFAULT_LIMIT_UM))
    A0          = float(body.get('A0', DEFAULT_A0))
    B0          = float(body.get('B0', DEFAULT_B0))

    fits      = []
    guesses   = _build_initial_guesses()
    t_start   = time.time()
    TIME_BUDGET = 100  # seconds — stay inside gunicorn's 120 s worker timeout

    for g in experiments:
        gid = g.get('groupId')

        if time.time() - t_start > TIME_BUDGET:
            fits.append({
                'groupId': gid, 'model': None,
                'ku': None, 'k1': None, 'k2': None, 'kr': None, 'k_bg': None,
                'limit_uM': None, 'seed_uM': None, 'cost': None,
                'simT': [], 'simY': [],
                'note': 'Time budget reached — re-fit with fewer groups or split the dataset.',
            })
            continue

        try:
            tc  = g.get('timeCourse', [])
            if len(tc) < 3:
                fits.append({'groupId': gid, 'model': None,
                             'ku': None, 'k1': None, 'k2': None, 'kr': None, 'k_bg': None,
                             'limit_uM': None, 'seed_uM': None, 'cost': None,
                             'simT': [], 'simY': [], 'note': 'Need at least 3 timepoints.'})
                continue

            limit_uM = float(g.get('limit_uM', default_lim))
            env      = ((g.get('conditions') or {}).get('env') or '')
            seed_pct = 0.05 if 'seed' in str(env).lower() else 0.0

            t_data = np.array([float(p['time'])       for p in tc], dtype=float)
            y_pct  = np.array([float(p['conversion'])  for p in tc], dtype=float)
            order  = np.argsort(t_data)
            t_data, y_pct = t_data[order], y_pct[order]

            y_data    = (y_pct / 100.0) * limit_uM
            initial_R = float(seed_pct * np.mean(y_data))

            best_res, best_cost = None, np.inf
            good_enough = max(1e-4, 1e-4 * (limit_uM ** 2) * len(t_data))

            for p0 in guesses:
                try:
                    res = least_squares(
                        _residuals, p0,
                        args=(t_data, y_data, initial_R, A0, B0),
                        bounds=([0.0]*4, [100.0, 100.0, 0.1, 100.0]),
                        ftol=1e-6, xtol=1e-6, max_nfev=60,
                    )
                    if res.success and res.cost < best_cost:
                        best_cost, best_res = res.cost, res
                        if best_cost < good_enough:
                            break
                except Exception:
                    continue

            if best_res is None:
                fits.append({'groupId': gid, 'model': None,
                             'ku': None, 'k1': None, 'k2': None, 'kr': None, 'k_bg': None,
                             'limit_uM': limit_uM, 'seed_uM': initial_R, 'cost': None,
                             'simT': [], 'simY': [], 'note': 'Fit did not converge.'})
                continue

            ku, k1, k2, kr = best_res.x
            sim_t, sim_R   = _simulate_R_dense(best_res.x, initial_R, float(t_data[-1]) + 5.0, A0, B0)
            sim_y_pct      = (sim_R / limit_uM) * 100.0 if sim_R is not None else None

            fits.append({
                'groupId': gid,
                'model':   'replication_kinetics',
                'ku':      float(ku),
                'k1':      float(k1),
                'k2':      float(k2),
                'kr':      float(kr),
                'k_bg':    float(ku * k1),
                'limit_uM': limit_uM,
                'seed_uM':  initial_R,
                'cost':     float(best_cost),
                'simT':  _safe_floats(sim_t)     if sim_t     is not None else [],
                'simY':  _safe_floats(sim_y_pct) if sim_y_pct is not None else [],
            })

        except Exception as exc:
            fits.append({'groupId': gid, 'model': None,
                         'ku': None, 'k1': None, 'k2': None, 'kr': None, 'k_bg': None,
                         'limit_uM': None, 'seed_uM': None, 'cost': None,
                         'simT': [], 'simY': [], 'note': f'Unexpected error: {exc}'})

    ranked  = sorted(experiments, key=_max_yield, reverse=True)[:5]
    top_max = _max_yield(ranked[0]) if ranked else 0.0
    fitted  = sum(1 for f in fits if f.get('ku') is not None)
    return jsonify({
        'fits': fits,
        'summary': {
            'topGroups':  [{'groupId': g.get('groupId'), 'maxConversion': _max_yield(g)} for g in ranked],
            'trendNotes': f"{fitted}/{len(fits)} groups fitted; top yield {top_max:.1f}%.",
        },
    })


# ════════════════════════════════════════════════════════════════════════
# Endpoint: /api/kinetics-suggest  (METIS UCB active learning)
# ════════════════════════════════════════════════════════════════════════

@lida_bp.route('/kinetics-suggest', methods=['POST'])
def kinetics_suggest():
    """
    METIS Bayesian active learning: XGBoost ensemble + UCB acquisition.

    Request:
      {
        experiments:      [...],   # groups; maxConversion or timeCourse
        ranges:           { tMin, tMax, ligaseMin, ligaseMax,
                            atpMin, atpMax, mg2Min, mg2Max },
        nSuggestions:     12,
        explorationCoeff: 1.41,    # kappa (0=exploit, sqrt2=balanced, 3=explore)
        ensembleSize:     20,
        poolSize:         5000,
      }

    Response:
      { suggestions: [{ sequence, conditions,
                         predicted_conversion, uncertainty, ucb }] }
    """
    body   = request.get_json(force=True) or {}
    exps   = body.get('experiments', [])
    if not exps:
        return jsonify({'suggestions': []})

    ranges        = body.get('ranges', {}) or {}
    n             = int(body.get('nSuggestions', 12))
    kappa         = float(body.get('explorationCoeff', 1.41))
    ensemble_size = int(body.get('ensembleSize', 20))
    pool_size     = int(body.get('poolSize', 5000))

    max_len   = max(len(e.get('sequence', '')) for e in exps) or 1
    env_vocab = sorted({(e.get('conditions') or {}).get('env', 'none') for e in exps})

    X = np.array([_featurize(e, max_len, env_vocab) for e in exps])
    y = np.array([_max_yield(e) for e in exps], dtype=float)

    if len(y) < 3:
        return jsonify({'suggestions': [],
                        'note': 'METIS needs ≥ 3 experiments. Add more data first.'})

    ensemble = _build_ensemble(X, y, ensemble_size=ensemble_size)

    # Candidates use only the sequences already studied — METIS explores the
    # condition space for those specific strands, not unvalidated new sequences.
    observed_seqs = [e.get('sequence', '') for e in exps]
    per_seq = max(1, pool_size // len(observed_seqs))
    rng = np.random.default_rng()
    cand_exps = []
    for seq in observed_seqs:
        for _ in range(per_seq):
            cand_exps.append({
                'sequence': seq,
                'conditions': {
                    'temperature': float(rng.uniform(ranges.get('tMin', 20),     ranges.get('tMax', 50))),
                    'ligase':      float(rng.uniform(ranges.get('ligaseMin', 0),  ranges.get('ligaseMax', 10))),
                    'atp':         float(rng.uniform(ranges.get('atpMin', 0),     ranges.get('atpMax', 10))),
                    'mg2':         float(rng.uniform(ranges.get('mg2Min', 0),     ranges.get('mg2Max', 20))),
                    'env':         str(rng.choice(env_vocab)),
                },
            })
    cand_exps = cand_exps[:pool_size]

    Xc            = np.asarray([_featurize(c, max_len, env_vocab) for c in cand_exps])
    mean, std, ucb = _ucb_score(ensemble, Xc, kappa)

    top_idx = np.argsort(ucb)[::-1][:n]
    suggestions = [{
        **cand_exps[i],
        'predicted_conversion': round(float(mean[i]), 2),
        'uncertainty':          round(float(std[i]),  2),
        'ucb':                  round(float(ucb[i]),  2),
    } for i in top_idx]

    return jsonify({'suggestions': suggestions})


# ════════════════════════════════════════════════════════════════════════
# Endpoint: /api/kinetics-sequence-predict  (METIS ensemble sequence opt)
# ════════════════════════════════════════════════════════════════════════

@lida_bp.route('/kinetics-sequence-predict', methods=['POST'])
def kinetics_sequence_predict():
    """
    Find high-yield DNA sequences by scoring a random pool against ALL observed
    conditions. Each candidate sequence is rated by its mean (and std) predicted
    conversion averaged over every condition set in the dataset, so the returned
    sequences are predicted to perform well across the entire condition space —
    not just at one fixed operating point.

    Request:
      {
        experiments:  [...],   # groups with sequence + conditions + maxConversion
        topK:         5,
        ensembleSize: 20,
        poolSize:     2000,    # random-sequence pool size (default smaller than suggest)
      }

    Response:
      { candidates: [{ sequence, predicted_conversion, uncertainty,
                        perPositionScore }] }
    """
    body   = request.get_json(force=True) or {}
    exps   = body.get('experiments', [])
    if not exps:
        return jsonify({'candidates': []})

    topK          = int(body.get('topK', 5))
    ensemble_size = int(body.get('ensembleSize', 20))
    pool_size     = int(body.get('poolSize', 2000))

    max_len   = max(len(e.get('sequence', '')) for e in exps) or 1
    env_vocab = sorted({(e.get('conditions') or {}).get('env', 'none') for e in exps})

    X = np.array([_featurize(e, max_len, env_vocab) for e in exps])
    y = np.array([_max_yield(e) for e in exps], dtype=float)

    ensemble = _build_ensemble(X, y, ensemble_size=ensemble_size)

    obs_conditions = [e['conditions'] for e in exps]
    n_cond = len(obs_conditions)
    rng = np.random.default_rng()

    # Build pool of random candidate sequences.
    cand_seqs = [''.join(rng.choice(BASES, size=max_len)) for _ in range(pool_size)]

    # Score every (sequence, observed_condition) pair in one batch, then average
    # over conditions per sequence. This finds sequences that are broadly good
    # rather than optimal only at one condition point.
    all_feats = np.asarray([
        _featurize({'sequence': seq, 'conditions': cond}, max_len, env_vocab)
        for seq in cand_seqs
        for cond in obs_conditions
    ])  # shape: (pool_size * n_cond, n_features)

    preds_flat = np.column_stack([m.predict(all_feats) for m in ensemble])
    # (pool_size * n_cond, n_models) → (pool_size, n_cond, n_models)
    preds_3d = preds_flat.reshape(pool_size, n_cond, len(ensemble))

    # Mean yield: average over conditions and ensemble models
    seq_means = preds_3d.mean(axis=(1, 2))          # (pool_size,)
    # Uncertainty: mean ensemble std over conditions
    seq_stds  = preds_3d.std(axis=2).mean(axis=1)   # (pool_size,)

    top_idx = np.argsort(seq_means)[::-1][:topK]

    # For each top sequence, compute per-position importance by mutating one base
    # at a time and averaging the prediction change over observed conditions.
    candidates = []
    for i in top_idx:
        seq_chars = list(cand_seqs[i])
        per_pos = []
        for pos in range(max_len):
            pos_scores = {}
            for b in BASES:
                tmp = seq_chars.copy(); tmp[pos] = b
                feats = np.asarray([
                    _featurize({'sequence': ''.join(tmp), 'conditions': c}, max_len, env_vocab)
                    for c in obs_conditions
                ])
                p = np.column_stack([m.predict(feats) for m in ensemble])
                pos_scores[b] = round(float(p.mean()), 2)
            per_pos.append(pos_scores)

        candidates.append({
            'sequence':             cand_seqs[i],
            'predicted_conversion': round(float(seq_means[i]), 2),
            'uncertainty':          round(float(seq_stds[i]),  2),
            'perPositionScore':     per_pos,
        })

    return jsonify({'candidates': candidates})
