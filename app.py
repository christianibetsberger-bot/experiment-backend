from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.spatial import distance_matrix
from itertools import combinations
import os

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "https://christianibetsberger-bot.github.io"}})

# Configuration
KERNEL = ConstantKernel(1.0) * RBF(length_scale=1.0)
N_RESTARTS = 5
RANDOM_STATE = 42

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
    
    all_midpoints = []
    all_distances = []
    
    for (class_a, class_b) in combinations(unique_classes, 2):
        points_a = class_points[class_a]
        points_b = class_points[class_b]
        if len(points_a) == 0 or len(points_b) == 0: continue
        
        dist_matrix = distance_matrix(points_a, points_b)
        min_indices = np.unravel_index(np.argsort(dist_matrix, axis=None), dist_matrix.shape)
        closest_pairs = list(zip(min_indices[0], min_indices[1]))
        
        for i, j in closest_pairs:
            midpoint = (points_a[i] + points_b[j]) / 2.0
            all_midpoints.append(midpoint)
            all_distances.append(dist_matrix[i, j])
            
    if len(all_midpoints) == 0: return np.array([], dtype=int)
    
    all_midpoints = np.array(all_midpoints)
    all_distances = np.array(all_distances)
    
    mask = all_distances >= min_dist
    candidates = all_midpoints[mask]
    if len(candidates) == 0: return np.array([], dtype=int)
    
    dist_to_known = distance_matrix(sampled_coords, candidates)
    valid = np.all(dist_to_known >= min_dist, axis=0)
    candidates = candidates[valid]
    if len(candidates) == 0: return np.array([], dtype=int)
    
    dist_full = distance_matrix(coordinates, candidates)
    closest_indices = np.argmin(dist_full, axis=0)
    return np.unique(closest_indices)

@app.route('/api/suggest-experiments', methods=['POST'])
def suggest_experiments():
    payload = request.json
    config = payload.get('config', {})
    experiments = payload.get('experiments', [])
    n_suggestions = payload.get('n_suggestions', 96)
    start_id = payload.get('start_id', 9000)
    strategy = config.get('strategy', 'safe')
    
    # 1. Unified UI Bounds
    anion_max = float(config.get('anionMax', 6))
    cation_max = float(config.get('cationMax', 6))
    salt_max = float(config.get('saltMax', 200))
    X_space_min = np.array([0.0, 0.0, 0.0])
    X_space_max = np.array([anion_max, cation_max, salt_max])
    denom = X_space_max - X_space_min
    denom[denom == 0] = 1.0

    # 2. Build Grid
    anion_grid = np.linspace(0, anion_max, 15)
    cation_grid = np.linspace(0, cation_max, 15)
    salt_grid = np.linspace(0, salt_max, 10)
    mesh = np.meshgrid(anion_grid, cation_grid, salt_grid, indexing="ij")
    points = np.column_stack([m.ravel() for m in mesh])
    
    df = pd.DataFrame(points, columns=["anion", "cation", "salt"])
    df["phase"] = -1
    
    if experiments:
        exp_df = pd.DataFrame(experiments)
        exp_df = exp_df[exp_df['phase'] != -1]
        if not exp_df.empty:
            exp_df = exp_df[["anion", "cation", "salt", "phase"]]
            df = pd.concat([exp_df, df], ignore_index=True)
            df = df.drop_duplicates(subset=["anion", "cation", "salt"], keep="first")
            
    X_raw = df[["anion", "cation", "salt"]].values
    X = (X_raw - X_space_min) / denom  # Use unified scaling
    y = df["phase"].values.astype(int)
    known_mask = y != -1
    unknown_idx = np.where(~known_mask)[0]
    
    if known_mask.sum() < 2 or len(np.unique(y[known_mask])) < 2:
        selected_local = fps_sampling(X[unknown_idx], n_suggestions)
        selected = unknown_idx[selected_local]
    else:
        if strategy == 'risky':
            space_range = np.linalg.norm(X.max(axis=0) - X.min(axis=0))
            min_dist = 0.05 * space_range
            midpoint_idx = midpoint_sampler(X, np.where(known_mask)[0], y[known_mask], min_dist)
            midpoint_idx = np.setdiff1d(midpoint_idx, np.where(known_mask)[0])
            
            if len(midpoint_idx) >= n_suggestions:
                selected = midpoint_idx[:n_suggestions]
            else:
                remaining = np.setdiff1d(unknown_idx, midpoint_idx)
                n_missing = n_suggestions - len(midpoint_idx)
                if len(remaining) > 0:
                    extra_local = fps_sampling(X[remaining], n_missing)
                    extra_idx = remaining[extra_local]
                    selected = np.concatenate([midpoint_idx, extra_idx])
                else:
                    selected = midpoint_idx
        else:
            clf = GaussianProcessClassifier(kernel=KERNEL, n_restarts_optimizer=N_RESTARTS, random_state=RANDOM_STATE)
            clf.fit(X[known_mask], y[known_mask])
            proba = clf.predict_proba(X[unknown_idx])
            entropy = -np.sum(proba * np.log(proba + 1e-12), axis=1)
            entropy += np.random.uniform(0, 1e-8, size=entropy.shape) 
            top_local = np.argsort(entropy)[::-1][:n_suggestions]
            selected = unknown_idx[top_local]
            
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
    
    if not experiments:
        return jsonify({"error": "No data to build boundary."}), 400
        
    exp_df = pd.DataFrame(experiments)
    exp_df = exp_df[exp_df['phase'] != -1]
    if len(exp_df) < 2 or len(np.unique(exp_df['phase'])) < 2:
        return jsonify({"error": "Need at least one Hit and one Clear."}), 400
        
    X_known = exp_df[["anion", "cation", "salt"]].values
    y_known = exp_df["phase"].values.astype(int)
    
    # Unified Scaling
    anion_max = float(config.get('anionMax', 6))
    cation_max = float(config.get('cationMax', 6))
    salt_max = float(config.get('saltMax', 200))
    X_space_min = np.array([0.0, 0.0, 0.0])
    X_space_max = np.array([anion_max, cation_max, salt_max])
    denom = X_space_max - X_space_min
    denom[denom == 0] = 1.0
    
    X_known_scaled = (X_known - X_space_min) / denom
    
    clf = GaussianProcessClassifier(kernel=KERNEL, n_restarts_optimizer=N_RESTARTS, random_state=RANDOM_STATE)
    clf.fit(X_known_scaled, y_known)
    
    # Smooth 3D Grid
    anion_grid = np.linspace(0, anion_max, 15)
    cation_grid = np.linspace(0, cation_max, 15)
    salt_grid = np.linspace(0, salt_max, 15)
    mesh = np.meshgrid(anion_grid, cation_grid, salt_grid, indexing="ij")
    grid_points = np.column_stack([m.ravel() for m in mesh])
    
    grid_scaled = (grid_points - X_space_min) / denom
    class_idx = list(clf.classes_).index(1) if 1 in clf.classes_ else 1
    proba = clf.predict_proba(grid_scaled)[:, class_idx] 
    
    return jsonify({
        "x": grid_points[:, 0].tolist(),
        "y": grid_points[:, 1].tolist(),
        "z": grid_points[:, 2].tolist(),
        "prob": proba.tolist()
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
