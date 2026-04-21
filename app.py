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

def scale_features(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    denom = X_max - X_min
    denom[denom == 0] = 1
    return (X - X_min) / denom

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
    
    # 1. Build Dense Search Space (15x15x10 = 2250 points)
    param_dict = {
        "Anion": (0, float(config.get('anionMax', 6)), 15),
        "Cation": (0, float(config.get('cationMax', 6)), 15),
        "Salt": (0, float(config.get('saltMax', 200)), 10),
    }
    grids = [np.linspace(lo, hi, n) for col, (lo, hi, n) in param_dict.items()]
    mesh = np.meshgrid(*grids, indexing="ij")
    points = np.column_stack([m.ravel() for m in mesh])
    
    df = pd.DataFrame(points, columns=["anion", "cation", "salt"])
    df["phase"] = -1
    
    # 2. Integrate memory without deleting (Append knowns to map them out)
    if experiments:
        exp_df = pd.DataFrame(experiments)
        exp_df = exp_df[exp_df['phase'] != -1] # Only integrate logged results
        if not exp_df.empty:
            exp_df = exp_df[["anion", "cation", "salt", "phase"]]
            df = pd.concat([df, exp_df], ignore_index=True)
            
    X_raw = df[["anion", "cation", "salt"]].values
    X = scale_features(X_raw)
    y = df["phase"].values.astype(int)
    known_mask = y != -1
    unknown_idx = np.where(~known_mask)[0]
    
    # 3. Routing Engine: FPS vs Risky vs Safe
    if known_mask.sum() < 2 or len(np.unique(y[known_mask])) < 2:
        # COLD START: Farthest Point Sampling
        selected_local = fps_sampling(X[unknown_idx], n_suggestions)
        selected = unknown_idx[selected_local]
    else:
        if strategy == 'risky':
            space_range = np.linalg.norm(X.max(axis=0) - X.min(axis=0))
            min_dist = 0.05 * space_range
            midpoint_idx = midpoint_sampler(X, np.where(known_mask)[0], y[known_mask], min_dist)
            midpoint_idx = np.setdiff1d(midpoint_idx, np.where(known_mask)[0]) # Safety filter
            
            if len(midpoint_idx) >= n_suggestions:
                selected = midpoint_idx[:n_suggestions]
            else:
                # Pad remainder with FPS to guarantee 96 wells
                remaining = np.setdiff1d(unknown_idx, midpoint_idx)
                n_missing = n_suggestions - len(midpoint_idx)
                if len(remaining) > 0:
                    extra_local = fps_sampling(X[remaining], n_missing)
                    extra_idx = remaining[extra_local]
                    selected = np.concatenate([midpoint_idx, extra_idx])
                else:
                    selected = midpoint_idx
        else:
            # SAFE MODE: GP Entropy
            clf = GaussianProcessClassifier(kernel=KERNEL, n_restarts_optimizer=N_RESTARTS, random_state=RANDOM_STATE)
            clf.fit(X[known_mask], y[known_mask])
            proba = clf.predict_proba(X[unknown_idx])
            entropy = -np.sum(proba * np.log(proba + 1e-12), axis=1)
            top_local = np.argsort(entropy)[::-1][:n_suggestions]
            selected = unknown_idx[top_local]
            
    # 4. Format Output
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

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
