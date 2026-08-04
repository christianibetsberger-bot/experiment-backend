from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial import distance_matrix
from scipy.ndimage import gaussian_filter
from itertools import combinations
from lida_kinetics import lida_bp
import math
import os

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.register_blueprint(lida_bp)
RANDOM_STATE = 42
EPS = 1e-9

# Total candidate rows the box can score before the RandomForest becomes the
# bottleneck. Beyond this the axes are STRIDE-subsampled (every k-th grid point),
# never re-stepped — every value stays on the user's own min + n·step lattice.
GRID_BUDGET = 250_000
MAX_POINTS_PER_AXIS = 100_000

def safe_float(val, fallback):
    try:
        if pd.isna(val) or val is None or str(val).strip() == '': return fallback
        return float(val)
    except (ValueError, TypeError):
        return fallback

def axis_spec(config, key, default_min=0.0, default_max=1.0, default_step=0.1):
    """(lo, hi, step) for one component, ordered and with a usable step."""
    lo = safe_float(config.get(key + 'Min'), default_min)
    hi = safe_float(config.get(key + 'Max'), default_max)
    if hi < lo: lo, hi = hi, lo
    step = safe_float(config.get(key + 'Step'), default_step)
    if step <= 0:
        step = (hi - lo) / 20.0 or 1.0
    return lo, hi, step

def axis_grid(lo, hi, step, stride=1):
    """Grid points lo + k·step·stride that stay inside [lo, hi].

    Unlike np.arange(lo, hi + step*0.1, step) this can never overshoot `hi` when
    the range is not a whole multiple of the step, and a stride only thins the
    lattice — it never moves a point off it.
    """
    span = hi - lo
    n = int(math.floor(span / step + 1e-9)) + 1 if step > 0 else 1
    n = max(n, 1)
    if stride > 1:
        n = (n - 1) // stride + 1
    return lo + np.arange(n) * (step * stride)

def axis_point_count(lo, hi, step, stride=1):
    n = int(math.floor((hi - lo) / step + 1e-9)) + 1 if step > 0 else 1
    n = max(n, 1)
    return (n - 1) // stride + 1 if stride > 1 else n

def fit_grid_to_budget(specs, budget=GRID_BUDGET):
    """Thin the largest axis (stride ×2) until the mesh fits the budget.

    Returns {key: stride} plus human-readable notes. The user's step is never
    rewritten — the suggestions stay pipettable on their configured lattice, they
    are just sampled more sparsely, and we say so instead of doing it silently.
    """
    strides = {k: 1 for k, _, _, _ in specs}
    notes = []
    def counts():
        return {k: axis_point_count(lo, hi, st, strides[k]) for k, lo, hi, st in specs}

    for k, lo, hi, st in specs:
        n = axis_point_count(lo, hi, st)
        if n > MAX_POINTS_PER_AXIS:
            strides[k] = int(math.ceil(n / MAX_POINTS_PER_AXIS))

    guard = 0
    while guard < 64:
        guard += 1
        c = counts()
        total = 1
        for v in c.values(): total *= v
        if total <= budget: break
        widest = max(c, key=lambda k: c[k])
        if c[widest] <= 2: break
        strides[widest] *= 2

    for k, lo, hi, st in specs:
        if strides[k] > 1:
            notes.append({
                "axis": k,
                "requested_step": st,
                "sampled_step": st * strides[k],
                "message": f"grid too large — sampled every {st * strides[k]:g} instead of {st:g} "
                           f"(still on your {st:g} lattice)"
            })
    return strides, notes

def snap_to_axis(values, lo, step):
    """Nearest lo + n·step. Used only when a link asks to stay on the target grid."""
    if step <= 0: return values
    return lo + np.round((values - lo) / step) * step

class LinkInfeasible(Exception):
    """A component link that no well in the configured ranges can satisfy."""
    pass

def comp_label(config, key):
    return config.get(key + 'Name') or {"anion": "A", "cation": "B", "salt": "C", "compD": "D"}[key]

def comp_unit(config, key):
    return config.get(key + 'Unit') or ''

def apply_dependencies(df, config, feature_cols):
    """Restrict the candidate grid to combinations the component links allow.

    fixed  — target is DERIVED: target = source·factor + offset. Candidates whose
             derived value leaves the target's own [min, max] are dropped rather
             than clipped, because clipping would silently break the ratio.
    range  — target is CONSTRAINED: keep only candidates already inside the band,
             so the engine still chooses the target itself, on its own grid.

    Doing this here (instead of overwriting the answer afterwards in the UI) is the
    whole point: the entropy/midpoint ranking is then computed on wells that can
    actually be pipetted, not on ones that get rewritten after being chosen.
    """
    deps = config.get('dependencies') or []
    notes = []
    for dep in deps:
        if not isinstance(dep, dict): continue
        src, tgt = dep.get('source'), dep.get('target')
        if not src or not tgt or src == tgt: continue
        if src not in feature_cols or tgt not in feature_cols: continue
        if df.empty: break

        factor = safe_float(dep.get('factor'), 1.0)
        offset = safe_float(dep.get('offset'), 0.0)
        t_lo, t_hi, t_step = axis_spec(config, tgt)
        before = len(df)
        s = df[src].to_numpy()

        src_name, tgt_name = comp_label(config, src), comp_label(config, tgt)
        unit = comp_unit(config, tgt)
        arrow = f"link {src_name} → {tgt_name}"

        if dep.get('mode') == 'range':
            f_max = safe_float(dep.get('factorMax'), factor)
            o_max = safe_float(dep.get('offsetMax'), offset)
            a, b = s * factor + offset, s * f_max + o_max
            band_lo, band_hi = np.minimum(a, b), np.maximum(a, b)
            t = df[tgt].to_numpy()
            df = df.loc[(t >= band_lo - EPS) & (t <= band_hi + EPS)]
            if df.empty:
                raise LinkInfeasible(
                    f"{arrow} (between {factor:g}× and {f_max:g}× {src_name}) asks for {tgt_name} "
                    f"somewhere in {band_lo.min():g}–{band_hi.max():g} {unit}, but {tgt_name} is set to "
                    f"{t_lo:g}–{t_hi:g} {unit} in steps of {t_step:g} and no step lands in the band. "
                    f"Widen {tgt_name}'s range, use a finer step, or change the link's multipliers.")
        else:
            derived = s * factor + offset
            if dep.get('snapStep'):
                derived = snap_to_axis(derived, t_lo, t_step)
            keep = (derived >= t_lo - EPS) & (derived <= t_hi + EPS)
            if not keep.any():
                raise LinkInfeasible(
                    f"{arrow} (= {factor:g} × {src_name}{f' + {offset:g}' if offset else ''}) needs "
                    f"{tgt_name} between {derived.min():g} and {derived.max():g} {unit}, but {tgt_name} "
                    f"is set to {t_lo:g}–{t_hi:g} {unit}. Widen {tgt_name}'s range, narrow {src_name}'s, "
                    f"change the factor, or reverse the link so {src_name} is derived from {tgt_name}.")
            df = df.loc[keep].copy()
            df[tgt] = np.round(derived[keep], 6)

        df = df.drop_duplicates(subset=feature_cols)
        notes.append({
            "link": f"{src}->{tgt}",
            "mode": dep.get('mode', 'fixed'),
            "candidates_before": before,
            "candidates_after": len(df),
        })
    return df.reset_index(drop=True), notes

def value_decimals(step):
    """Enough decimals to represent the user's own step (rounding to 2 dp would
    collapse neighbouring points on a fine grid into the same well)."""
    if step <= 0 or not np.isfinite(step): return 3
    d = max(0, int(math.ceil(-math.log10(step))) + 1)
    return int(min(max(d, 2), 6))

def fps_sampling(X, n):
    if len(X) == 0: return []
    n = min(n, len(X))
    selected = [np.random.randint(len(X))]
    distances = np.full(len(X), np.inf)
    for _ in range(n - 1):
        last = X[selected[-1]]
        dist = np.linalg.norm(X - last, axis=1)
        distances = np.minimum(distances, dist)
        selected.append(int(np.argmax(distances)))
    return selected

def midpoint_sampler(coordinates, starting_indices, starting_labels, min_dist):
    sampled_coords = coordinates[starting_indices]
    unique_classes = np.unique(starting_labels)
    class_points = {cls: sampled_coords[starting_labels == cls] for cls in unique_classes}
    all_midpoints, all_distances = [], []

    for (class_a, class_b) in combinations(unique_classes, 2):
        points_a = class_points[class_a]
        points_b = class_points[class_b]
        if len(points_a) == 0 or len(points_b) == 0: continue
        dist_matrix = distance_matrix(points_a, points_b)
        min_indices = np.unravel_index(np.argsort(dist_matrix, axis=None), dist_matrix.shape)
        closest_pairs = list(zip(min_indices[0], min_indices[1]))
        for i, j in closest_pairs:
            all_midpoints.append((points_a[i] + points_b[j]) / 2.0)
            all_distances.append(dist_matrix[i, j])

    if len(all_midpoints) == 0: return np.array([], dtype=int)
    all_midpoints, all_distances = np.array(all_midpoints), np.array(all_distances)

    mask = all_distances >= min_dist
    candidates = all_midpoints[mask]
    if len(candidates) == 0: return np.array([], dtype=int)

    dist_to_known = distance_matrix(sampled_coords, candidates)
    valid = np.all(dist_to_known >= min_dist, axis=0)
    candidates = candidates[valid]
    if len(candidates) == 0: return np.array([], dtype=int)

    dist_full = distance_matrix(coordinates, candidates)
    return np.unique(np.argmin(dist_full, axis=0))

@app.route('/api/suggest-experiments', methods=['POST'])
def suggest_experiments():
    payload = request.json
    config = payload.get('config', {})
    experiments = payload.get('experiments', [])

    n_suggestions = int(safe_float(payload.get('n_suggestions', config.get('numSuggestions', 96)), 96))
    start_id = int(safe_float(payload.get('start_id', 9000), 9000))
    strategy = config.get('strategy', 'safe')
    min_dist_factor = safe_float(config.get('minDistanceFactor'), 0.05)

    enable_d = bool(config.get('enableCompD', False))

    feature_cols = ["anion", "cation", "salt", "compD"] if enable_d else ["anion", "cation", "salt"]
    dedup_cols = feature_cols

    defaults = {"anion": (0.0, 6.0, 0.5), "cation": (0.0, 6.0, 0.5),
                "salt": (0.0, 200.0, 10.0), "compD": (0.0, 1.0, 0.1)}
    specs = [(k,) + axis_spec(config, k, *defaults[k]) for k in feature_cols]
    strides, warnings = fit_grid_to_budget(specs)

    X_space_min = np.array([lo for _, lo, _, _ in specs])
    X_space_max = np.array([hi for _, _, hi, _ in specs])
    denom = X_space_max - X_space_min
    denom[denom == 0] = 1.0

    # Selection distances are measured in the NORMALISED box, so the min-distance
    # filter treats every component alike. In raw units the axis with the biggest
    # numeric span (salt in mM, typically 0–200) swallowed the whole budget and the
    # suggestions barely moved in the 0–6 mM components.
    min_dist = min_dist_factor * math.sqrt(len(feature_cols))

    grids = [axis_grid(lo, hi, st, strides[k]) for k, lo, hi, st in specs]
    mesh = np.meshgrid(*grids, indexing="ij")
    points = np.column_stack([m.ravel() for m in mesh])
    df = pd.DataFrame(points, columns=feature_cols)

    n_full_grid = len(df)
    try:
        df, dep_notes = apply_dependencies(df, config, feature_cols)
    except LinkInfeasible as e:
        return jsonify({"error": str(e)}), 400
    if df.empty:
        return jsonify({"error": "No well satisfies the configured component links inside the given "
                                 "ranges. Loosen a link's factor/offset or widen the linked "
                                 "component's min/max."}), 400
    n_candidates = len(df)
    if dep_notes:
        warnings.append({
            "axis": "links",
            "message": f"{len(dep_notes)} component link(s) applied — "
                       f"{len(df):,} of {n_full_grid:,} grid wells are reachable",
            "detail": dep_notes,
        })

    df["phase"] = -1

    if experiments:
        exp_df = pd.DataFrame(experiments)
        # Ensure compD column exists when running in 4D (default to 0 if missing).
        if enable_d and 'compD' not in exp_df.columns:
            exp_df['compD'] = 0.0
        cols_needed = feature_cols + ["phase"]
        exp_df = exp_df[cols_needed].apply(pd.to_numeric, errors='coerce').dropna()
        exp_df = exp_df[exp_df['phase'] != -1]

        if not exp_df.empty:
            df = pd.concat([exp_df, df], ignore_index=True)
            df = df.drop_duplicates(subset=dedup_cols, keep="first")

    X_raw = df[feature_cols].values
    X = (X_raw - X_space_min) / denom
    y = df["phase"].values.astype(int)
    known_mask = y != -1
    unknown_idx = np.where(~known_mask)[0]

    if known_mask.sum() < 2 or len(np.unique(y[known_mask])) < 2:
        selected_local = fps_sampling(X[unknown_idx], n_suggestions)
        selected = unknown_idx[selected_local]
    else:
        if strategy == 'risky':
            midpoint_idx = midpoint_sampler(X, np.where(known_mask)[0], y[known_mask], min_dist)
            midpoint_idx = np.setdiff1d(midpoint_idx, np.where(known_mask)[0])
            if len(midpoint_idx) >= n_suggestions:
                selected = midpoint_idx[:n_suggestions]
            else:
                remaining = np.setdiff1d(unknown_idx, midpoint_idx)
                n_missing = n_suggestions - len(midpoint_idx)
                if len(remaining) > 0:
                    extra_local = fps_sampling(X[remaining], n_missing)
                    selected = np.concatenate([midpoint_idx, remaining[extra_local]])
                else:
                    selected = midpoint_idx
        else:
            try:
                clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
                clf.fit(X[known_mask], y[known_mask])
                proba = clf.predict_proba(X[unknown_idx])
                proba = np.nan_to_num(proba, nan=1e-6)

                entropy = -np.sum(proba * np.log(proba + 1e-12), axis=1)
                entropy = np.nan_to_num(entropy, nan=0.0)
                entropy += np.random.uniform(0, 1e-8, size=entropy.shape)

                sorted_indices = np.argsort(entropy)[::-1]
                selected_local = []

                if min_dist > 0:
                    for idx in sorted_indices:
                        pt = X[unknown_idx[idx]]
                        dist_ok = True
                        if len(selected_local) > 0:
                            if np.min(np.linalg.norm(X[unknown_idx[selected_local]] - pt, axis=1)) < min_dist:
                                dist_ok = False
                        if dist_ok and known_mask.sum() > 0:
                            if np.min(np.linalg.norm(X[known_mask] - pt, axis=1)) < min_dist:
                                dist_ok = False
                        if dist_ok:
                            selected_local.append(idx)
                        if len(selected_local) >= n_suggestions: break
                    if len(selected_local) < n_suggestions:
                        unused = [i for i in sorted_indices if i not in selected_local]
                        selected_local.extend(unused[:(n_suggestions - len(selected_local))])
                else:
                    selected_local = sorted_indices[:n_suggestions].tolist()

                selected = unknown_idx[selected_local]
            except Exception as e:
                # Absolute fallback if math entirely fails
                print(f"GP Optimization Failed: {e}")
                selected_local = fps_sampling(X[unknown_idx], n_suggestions)
                selected = unknown_idx[selected_local]

    suggested_df = df.iloc[selected].copy()
    # Round to the resolution the user's own step needs — a fixed 2 dp would merge
    # neighbouring wells on a 0.01 grid into one value.
    decimals = {k: value_decimals(st) for k, _, _, st in specs}
    suggestions = []
    current_id = start_id
    for idx, row in suggested_df.iterrows():
        sug = {"sampleId": current_id, "phase": -1}
        for k in feature_cols:
            sug[k] = round(float(row[k]), decimals[k])
        suggestions.append(sug)
        current_id += 1

    if len(suggestions) < n_suggestions:
        warnings.append({
            "axis": "count",
            "message": f"only {len(suggestions)} of the {n_suggestions} requested wells exist — "
                       f"{len(unknown_idx):,} untested well(s) fit the ranges and links",
        })

    return jsonify({
        "suggestions": suggestions,
        "warnings": warnings,
        "n_candidates": int(n_candidates),
    })

@app.route('/api/phase-boundary', methods=['POST'])
def phase_boundary():
    payload = request.json
    config = payload.get('config', {})
    experiments = payload.get('experiments', [])

    if not experiments: return jsonify({"error": "No data."}), 400

    enable_d = bool(config.get('enableCompD', False))

    n_received = len(experiments)
    exp_df = pd.DataFrame(experiments)
    if enable_d and 'compD' not in exp_df.columns:
        exp_df['compD'] = 0.0
    cols_needed = ["anion", "cation", "salt", "compD", "phase"] if enable_d else ["anion", "cation", "salt", "phase"]
    exp_df = exp_df[cols_needed].apply(pd.to_numeric, errors='coerce').dropna()
    # Keep only labeled rows; prefer the last entry for duplicate coordinates (covers phase updates)
    exp_df = exp_df[exp_df['phase'] != -1]
    feature_cols = ["anion", "cation", "salt", "compD"] if enable_d else ["anion", "cation", "salt"]
    exp_df = exp_df.drop_duplicates(subset=feature_cols, keep="last")

    n_labeled = len(exp_df)
    print(f"[phase-boundary] received={n_received}  labeled={n_labeled}  4D={enable_d}  phases={sorted(exp_df['phase'].unique().tolist())}")

    if n_labeled < 2 or len(np.unique(exp_df['phase'])) < 2:
        return jsonify({"error": f"Need at least 2 labeled points from 2 different phases (got {n_labeled} labeled)."}), 400

    X_known = exp_df[feature_cols].values
    y_known = exp_df["phase"].values.astype(int)

    anion_min, cation_min, salt_min = safe_float(config.get('anionMin'), 0.0), safe_float(config.get('cationMin'), 0.0), safe_float(config.get('saltMin'), 0.0)
    anion_max, cation_max, salt_max = safe_float(config.get('anionMax'), 6.0), safe_float(config.get('cationMax'), 6.0), safe_float(config.get('saltMax'), 200.0)

    if enable_d:
        d_min = safe_float(config.get('compDMin'), 0.0)
        d_max = safe_float(config.get('compDMax'), 1.0)
        X_space_min = np.array([anion_min, cation_min, salt_min, d_min])
        X_space_max = np.array([anion_max, cation_max, salt_max, d_max])
    else:
        X_space_min = np.array([anion_min, cation_min, salt_min])
        X_space_max = np.array([anion_max, cation_max, salt_max])

    denom = X_space_max - X_space_min
    denom[denom == 0] = 1.0

    X_known_scaled = (X_known - X_space_min) / denom

    clf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
    clf.fit(X_known_scaled, y_known)

    # 20^3 grid for the probability field (3D mode).
    # In 4D mode, slice the D axis at N_D values and stack 20^3 grids for each — the
    # frontend renders the slice closest to the current D slider value.
    N = 20
    anion_grid = np.linspace(anion_min, anion_max, N)
    cation_grid = np.linspace(cation_min, cation_max, N)
    salt_grid = np.linspace(salt_min, salt_max, N)

    if enable_d:
        N_D = 8  # number of D slices the frontend can interpolate between
        d_grid = np.linspace(d_min, d_max, N_D)
        mesh = np.meshgrid(anion_grid, cation_grid, salt_grid, d_grid, indexing="ij")
        grid_points = np.column_stack([m.ravel() for m in mesh])
        grid_scaled = (grid_points - X_space_min) / denom
        predicted_class = clf.predict(grid_scaled)
        prob_dict = {}
        for class_label in clf.classes_:
            field = (predicted_class == class_label).astype(float).reshape(N, N, N, N_D)
            # Smooth only the 3 spatial axes; keep D crisp.
            smoothed = gaussian_filter(field, sigma=(1.0, 1.0, 1.0, 0.0))
            prob_dict[str(class_label)] = smoothed.ravel().tolist()
        return jsonify({
            "x": grid_points[:, 0].tolist(),
            "y": grid_points[:, 1].tolist(),
            "z": grid_points[:, 2].tolist(),
            "d": grid_points[:, 3].tolist(),
            "probs": prob_dict,
            "n_received": n_received,
            "n_labeled": n_labeled,
            "phases_used": sorted([int(c) for c in clf.classes_]),
            "enable_d": True
        })

    mesh = np.meshgrid(anion_grid, cation_grid, salt_grid, indexing="ij")
    grid_points = np.column_stack([m.ravel() for m in mesh])
    grid_scaled = (grid_points - X_space_min) / denom

    predicted_class = clf.predict(grid_scaled)
    prob_dict = {}

    for class_label in clf.classes_:
        # Build a 0/1 indicator: 1 where RFC votes for this class, 0 elsewhere.
        # RFC memorises its training data, so all training points are in regions
        # where this field == 1 before smoothing → they stay inside the isosurface.
        # Light Gaussian smoothing (sigma=1.0) just anti-aliases the jagged voxel edges.
        field = (predicted_class == class_label).astype(float).reshape(N, N, N)
        smoothed = gaussian_filter(field, sigma=1.0)
        prob_dict[str(class_label)] = smoothed.ravel().tolist()

    return jsonify({
        "x": grid_points[:, 0].tolist(),
        "y": grid_points[:, 1].tolist(),
        "z": grid_points[:, 2].tolist(),
        "probs": prob_dict,
        "n_received": n_received,
        "n_labeled": n_labeled,
        "phases_used": sorted([int(c) for c in clf.classes_])
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
