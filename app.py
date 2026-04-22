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
    all_midpoints, all_distances = [], []
    
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
    all_midpoints, all_distances = np.array(all_midpoints), np.array(all_distances)
    
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
    n_suggestions = int(payload.get('n_suggestions', config.get('numSuggestions', 96)))
    start_id = payload.get('start_id', 9000)
    strategy = config.get('strategy', 'safe')
    min_dist_factor = float(config.get('minDistanceFactor', 0.05))
    
    # Grid Boundaries
    anion_min = float(config.get('anionMin', 0))
    cation_min = float(config.get('cationMin', 0))
    salt_min = float(config.get('saltMin', 0))
    anion_max = float(config.get('anionMax', 6))
    cation_max = float(config.get('cationMax', 6))
    salt_max = float(config.get('saltMax', 200))
    
    # Step Enforcement (With mathematical failsafes to prevent memory crash)
    anion_step = max(float(config.get('anionStep', 0.5)), (anion_max - anion_min)/100.0)
    cation_step = max(float(config.get('cationStep', 0.5)), (cation_max - cation_min)/100.0)
    salt_step = max(float(config.get('saltStep', 10.0)), (salt_max - salt_min)/100.0)
    
    X_space_min = np.array([anion_min, cation_min, salt_min])
    X_space_max = np.array([anion_max, cation_max, salt_max])
    denom = X_space_max - X_space_min
    denom[denom == 0] = 1.0

    space_range = np.linalg.norm(X_space_max - X_space_min)
    min_dist = min_dist_factor * space_range

    # Build Step-Locked Grid
    anion_grid = np.arange(anion_min, anion_max + (anion_step*0.1), anion_step)
    cation_grid = np.arange(cation_min, cation_max + (cation_step*0.1), cation_step)
    salt_grid = np.arange(salt_min, salt_max + (salt_step*0.1), salt_step)
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
    X = (X_raw - X_space_min) / denom
    y = df["phase"].values.astype(int)
    known_mask = y != -1
    unknown_idx = np.where(~known_mask)[0]
    
    # Multi-class compatibility check
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
            # GP natively supports multi-class One-vs-Rest!
            clf = GaussianProcessClassifier(kernel=KERNEL, n_restarts_optimizer=N_RESTARTS, random_state=RANDOM_STATE)
            clf.fit(X[known_mask], y[known_mask])
            proba = clf.predict_proba(X[unknown_idx])
            
            # Shannon Entropy captures uncertainty across ALL 5 possible phases
            entropy = -np.sum(proba * np.log(proba + 1e-12), axis=1)
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
    exp_df = pd.DataFrame(experiments)
    exp_df = exp_df[exp_df['phase'] != -1]
    if len(exp_df) < 2 or len(np.unique(exp_df['phase'])) < 2: return jsonify({"error": "Need multiple phases."}), 400
        
    X_known = exp_df[["anion", "cation", "salt"]].values
    y_known = exp_df["phase"].values.astype(int)
    
    anion_min, cation_min, salt_min = float(config.get('anionMin', 0)), float(config.get('cationMin', 0)), float(config.get('saltMin', 0))
    anion_max, cation_max, salt_max = float(config.get('anionMax', 6)), float(config.get('cationMax', 6)), float(config.get('saltMax', 200))
    X_space_min, X_space_max = np.array([anion_min, cation_min, salt_min]), np.array([anion_max, cation_max, salt_max])
    denom = X_space_max - X_space_min
    denom[denom == 0] = 1.0
    
    X_known_scaled = (X_known - X_space_min) / denom
    clf = GaussianProcessClassifier(kernel=KERNEL, n_restarts_optimizer=N_RESTARTS, random_state=RANDOM_STATE)
    clf.fit(X_known_scaled, y_known)
    
    # Smooth 3D Grid for visual bounds
    anion_grid = np.linspace(anion_min, anion_max, 15)
    cation_grid = np.linspace(cation_min, cation_max, 15)
    salt_grid = np.linspace(salt_min, salt_max, 15)
    mesh = np.meshgrid(anion_grid, cation_grid, salt_grid, indexing="ij")
    grid_points = np.column_stack([m.ravel() for m in mesh])
    
    grid_scaled = (grid_points - X_space_min) / denom
    
    # MULTI-CLASS SHANNON ENTROPY: Defines the boundary regardless of how many phases exist
    proba = clf.predict_proba(grid_scaled)
    entropy = -np.sum(proba * np.log(proba + 1e-12), axis=1)
    
    # Normalize entropy so 1.0 is always the absolute boundary peak (highest uncertainty)
    e_min, e_max = np.min(entropy), np.max(entropy)
    norm_entropy = (entropy - e_min) / (e_max - e_min) if e_max > e_min else entropy
    
    return jsonify({
        "x": grid_points[:, 0].tolist(),
        "y": grid_points[:, 1].tolist(),
        "z": grid_points[:, 2].tolist(),
        "prob": norm_entropy.tolist() 
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
