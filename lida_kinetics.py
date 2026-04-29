"""
LIDA Kinetics endpoints — Lesion-Induced DNA Amplification.

Drop this file into the experiment-backend repo (alongside app.py) and
register the Blueprint in app.py with the two-line patch shown in the
adjacent README.md. The existing /api/phase-boundary and
/api/suggest-experiments endpoints are NOT modified.
"""
from flask import Blueprint, request, jsonify
import numpy as np
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor

lida_bp = Blueprint('lida_kinetics', __name__, url_prefix='/api')

BASES = ['A', 'C', 'G', 'T']


# ────────────── helpers ──────────────

def _first_order(t, Cmax, k):
    return Cmax * (1.0 - np.exp(-k * np.asarray(t)))


def _sigmoidal(t, Cmax, k, t50):
    return Cmax / (1.0 + np.exp(-k * (np.asarray(t) - t50)))


def _aic(y, yhat, k_params):
    n = len(y)
    rss = float(np.sum((np.asarray(y) - np.asarray(yhat)) ** 2))
    if rss <= 0:
        return -np.inf
    return n * np.log(rss / n) + 2 * k_params


def _encode_sequence(seq, max_len):
    """One-hot encode a sequence into a (max_len * 4) vector. Pad with zeros."""
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


# ────────────── endpoints ──────────────

@lida_bp.route('/kinetics-fit', methods=['POST'])
def kinetics_fit():
    body = request.get_json(force=True) or {}
    experiments = body.get('experiments', [])
    fits = []
    for g in experiments:
        tc = g.get('timeCourse', [])
        if len(tc) < 3:
            fits.append({'groupId': g['groupId'], 'Cmax': None, 'k': None, 't50': None, 'model': None})
            continue
        t = [p['time'] for p in tc]
        y = [p['conversion'] for p in tc]
        best = None
        for name, fn, p0, k_params in [
            ('first_order', _first_order, [max(y), 0.05], 2),
            ('sigmoidal',   _sigmoidal,   [max(y), 0.1, float(np.median(t))], 3),
        ]:
            try:
                popt, _ = curve_fit(fn, t, y, p0=p0, maxfev=4000)
                yhat = fn(t, *popt)
                aic = _aic(y, yhat, k_params)
                if best is None or aic < best['aic']:
                    best = {'name': name, 'popt': popt.tolist(), 'aic': aic}
            except Exception:
                continue
        if best is None:
            fits.append({'groupId': g['groupId'], 'Cmax': None, 'k': None, 't50': None, 'model': None})
        elif best['name'] == 'first_order':
            fits.append({
                'groupId': g['groupId'],
                'Cmax': best['popt'][0],
                'k': best['popt'][1],
                't50': None,
                'model': 'first_order',
            })
        else:
            fits.append({
                'groupId': g['groupId'],
                'Cmax': best['popt'][0],
                'k': best['popt'][1],
                't50': best['popt'][2],
                'model': 'sigmoidal',
            })

    ranked = sorted(experiments, key=_max_yield, reverse=True)[:5]
    top_max = _max_yield(ranked[0]) if ranked else 0.0
    return jsonify({
        'fits': fits,
        'summary': {
            'topGroups': [{'groupId': g['groupId'], 'maxConversion': _max_yield(g)} for g in ranked],
            'trendNotes': f"{len(fits)} groups fitted; top yield {top_max:.1f}%.",
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
    rf = RandomForestRegressor(n_estimators=200, random_state=0).fit(X, y)

    rng = np.random.default_rng(42)
    candidates = []
    n_trials = max(500, n * 50)
    for _ in range(n_trials):
        seq = ''.join(rng.choice(BASES, size=max_len))
        cand_exp = {
            'sequence': seq,
            'conditions': {
                'temperature': float(rng.uniform(ranges.get('tMin', 20), ranges.get('tMax', 50))),
                'ligase':      float(rng.uniform(ranges.get('ligaseMin', 0), ranges.get('ligaseMax', 10))),
                'atp':         float(rng.uniform(ranges.get('atpMin', 0), ranges.get('atpMax', 10))),
                'mg2':         float(rng.uniform(ranges.get('mg2Min', 0), ranges.get('mg2Max', 20))),
                'env':         str(rng.choice(env_vocab)),
            },
        }
        x = _featurize(cand_exp, max_len, env_vocab).reshape(1, -1)
        per_tree = np.array([t.predict(x)[0] for t in rf.estimators_])
        candidates.append({
            **cand_exp,
            'predicted_conversion': float(per_tree.mean()),
            'uncertainty': float(per_tree.std()),
        })

    if strategy == 'explore':
        candidates.sort(key=lambda c: -c['uncertainty'])
    else:
        candidates.sort(key=lambda c: -c['predicted_conversion'])
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
    rf = RandomForestRegressor(n_estimators=200, random_state=0).fit(X, y)

    best_exp = max(exps, key=_max_yield)
    cond = fixed or best_exp['conditions']
    seq = list(best_exp.get('sequence', 'A' * max_len))
    if len(seq) < max_len:
        seq += ['A'] * (max_len - len(seq))

    rng = np.random.default_rng()
    candidates = []
    for _ in range(topK):
        # Greedy hill-climb until no single-position change improves yield.
        improved = True
        while improved:
            improved = False
            for i in range(max_len):
                best_b, best_yhat = seq[i], -np.inf
                for b in BASES:
                    seq[i] = b
                    x = _featurize({'sequence': ''.join(seq), 'conditions': cond}, max_len, env_vocab).reshape(1, -1)
                    yhat = float(rf.predict(x)[0])
                    if yhat > best_yhat:
                        best_b, best_yhat = b, yhat
                if seq[i] != best_b:
                    seq[i] = best_b
                    improved = True

        # Per-position score grid (4 bases × max_len) for the optimum.
        per_pos = []
        for i in range(max_len):
            scores = {}
            for b in BASES:
                tmp = seq.copy()
                tmp[i] = b
                x = _featurize({'sequence': ''.join(tmp), 'conditions': cond}, max_len, env_vocab).reshape(1, -1)
                scores[b] = float(rf.predict(x)[0])
            per_pos.append(scores)

        x = _featurize({'sequence': ''.join(seq), 'conditions': cond}, max_len, env_vocab).reshape(1, -1)
        candidates.append({
            'sequence': ''.join(seq),
            'predicted_conversion': float(rf.predict(x)[0]),
            'perPositionScore': per_pos,
        })

        # Perturb one position to seed the next candidate.
        idx = int(rng.integers(0, max_len))
        seq[idx] = str(rng.choice([b for b in BASES if b != seq[idx]]))

    return jsonify({'candidates': candidates})
