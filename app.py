from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial import distance_matrix
from scipy.ndimage import gaussian_filter
from itertools import combinations
from lida_kinetics import lida_bp
import os

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.register_blueprint(lida_bp)
RANDOM_STATE = 42

def safe_float(val, fallback):
    try:
        if pd.isna(val) or val is None or str(val).strip() == '': return fallback
        return float(val)
    except (ValueError, TypeError):
        return fallback

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

    anion_min, cation_min, salt_min = safe_float(config.get('anionMin'), 0.0), safe_float(config.get('cationMin'), 0.0), safe_float(config.get('saltMin'), 0.0)
    anion_max, cation_max, salt_max = safe_float(config.get('anionMax'), 6.0), safe_float(config.get('cationMax'), 6.0), safe_float(config.get('saltMax'), 200.0)

    anion_step  = safe_float(config.get('anionStep'),  0.5)
    cation_step = safe_float(config.get('cationStep'), 0.5)
    salt_step   = safe_float(config.get('saltStep'),  10.0)
    if anion_step  <= 0: anion_step  = (anion_max  - anion_min)  / 20.0 or 1.0
    if cation_step <= 0: cation_step = (cation_max - cation_min) / 20.0 or 1.0
    if salt_step   <= 0: salt_step   = (salt_max   - salt_min)   / 20.0 or 1.0
    # Cap each axis at 100 steps to prevent accidental memory blowout; user step
    # is honoured as long as it produces ≤100 grid points per axis.
    MAX_PER_AXIS = 100
    anion_step  = max(anion_step,  (anion_max  - anion_min)  / MAX_PER_AXIS)
    cation_step = max(cation_step, (cation_max - cation_min) / MAX_PER_AXIS)
    salt_step   = max(salt_step,   (salt_max   - salt_min)   / MAX_PER_AXIS)

    X_space_min = np.array([anion_min, cation_min, salt_min])
    X_space_max = np.array([anion_max, cation_max, salt_max])
    denom = X_space_max - X_space_min
    denom[denom == 0] = 1.0

    min_dist = min_dist_factor * np.linalg.norm(X_space_max - X_space_min)

    anion_grid = np.arange(anion_min, anion_max + (anion_step*0.1), anion_step)
    cation_grid = np.arange(cation_min, cation_max + (cation_step*0.1), cation_step)
    salt_grid = np.arange(salt_min, salt_max + (salt_step*0.1), salt_step)
    mesh = np.meshgrid(anion_grid, cation_grid, salt_grid, indexing="ij")
    points = np.column_stack([m.ravel() for m in mesh])

    df = pd.DataFrame(points, columns=["anion", "cation", "salt"])
    df["phase"] = -1

    if experiments:
        exp_df = pd.DataFrame(experiments)
        exp_df = exp_df[["anion", "cation", "salt", "phase"]].apply(pd.to_numeric, errors='coerce').dropna()
        exp_df = exp_df[exp_df['phase'] != -1]

        if not exp_df.empty:
            df = pd.concat([exp_df, df], ignore_index=True)
            df = df.drop_duplicates(subset=["anion", "cation", "salt"], keep="first")

    X_raw = df[["anion", "cation", "salt"]].values
    X = (X_raw - X_space_min) / denom
    y = df["phase"].values.astype(int)
    known_mask = y != -1
    unknown_idx = np.where(~known_mask)[0]

    if known_mask.sum() < 2 or len(np.unique(y[known_mask])) < 2:
        selected_local = fps_sampling(X[unknown_idx], n_suggestions)
        selected = unknown_idx[selected_local]
    else:
        if strategy == 'risky':
            midpoint_idx = midpoint_sampler(X_raw, np.where(known_mask)[0], y[known_mask], min_dist)
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
                        pt_raw = X_raw[unknown_idx[idx]]
                        dist_ok = True
                        if len(selected_local) > 0:
                            if np.min(np.linalg.norm(X_raw[unknown_idx[selected_local]] - pt_raw, axis=1)) < min_dist:
                                dist_ok = False
                        if dist_ok and known_mask.sum() > 0:
                            if np.min(np.linalg.norm(X_raw[known_mask] - pt_raw, axis=1)) < min_dist:
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
    suggestions = []
    current_id = start_id
    for idx, row in suggested_df.iterrows():
        suggestions.append({
            "sampleId": current_id,
            "anion": round(row['anion'], 2),
            "cation": round(row['cation'], 2),
            "salt": round(row['salt'], 1),
            "phase": -1
        })
        current_id += 1

    return jsonify({"suggestions": suggestions})

@app.route('/api/phase-boundary', methods=['POST'])
def phase_boundary():
    payload = request.json
    config = payload.get('config', {})
    experiments = payload.get('experiments', [])

    if not experiments: return jsonify({"error": "No data."}), 400

    n_received = len(experiments)
    exp_df = pd.DataFrame(experiments)
    exp_df = exp_df[["anion", "cation", "salt", "phase"]].apply(pd.to_numeric, errors='coerce').dropna()
    # Keep only labeled rows; prefer the last entry for duplicate coordinates (covers phase updates)
    exp_df = exp_df[exp_df['phase'] != -1]
    exp_df = exp_df.drop_duplicates(subset=["anion", "cation", "salt"], keep="last")

    n_labeled = len(exp_df)
    print(f"[phase-boundary] received={n_received}  labeled={n_labeled}  phases={sorted(exp_df['phase'].unique().tolist())}")

    if n_labeled < 2 or len(np.unique(exp_df['phase'])) < 2:
        return jsonify({"error": f"Need at least 2 labeled points from 2 different phases (got {n_labeled} labeled)."}), 400

    X_known = exp_df[["anion", "cation", "salt"]].values
    y_known = exp_df["phase"].values.astype(int)

    anion_min, cation_min, salt_min = safe_float(config.get('anionMin'), 0.0), safe_float(config.get('cationMin'), 0.0), safe_float(config.get('saltMin'), 0.0)
    anion_max, cation_max, salt_max = safe_float(config.get('anionMax'), 6.0), safe_float(config.get('cationMax'), 6.0), safe_float(config.get('saltMax'), 200.0)

    X_space_min = np.array([anion_min, cation_min, salt_min])
    X_space_max = np.array([anion_max, cation_max, salt_max])
    denom = X_space_max - X_space_min
    denom[denom == 0] = 1.0

    X_known_scaled = (X_known - X_space_min) / denom

    clf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
    clf.fit(X_known_scaled, y_known)

    # 20^3 grid for the probability field
    N = 20
    anion_grid = np.linspace(anion_min, anion_max, N)
    cation_grid = np.linspace(cation_min, cation_max, N)
    salt_grid = np.linspace(salt_min, salt_max, N)
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
