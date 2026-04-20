from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "https://christianibetsberger-bot.github.io"}}) # Allows Vue frontend to communicate with Python backend

# Configuration matching your original script
KERNEL = ConstantKernel(1.0) * RBF(length_scale=1.0)
N_RESTARTS = 5
RANDOM_STATE = 42

def scale_features(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    denom = X_max - X_min
    denom[denom == 0] = 1
    return (X - X_min) / denom

@app.route('/api/suggest-experiments', methods=['POST'])
def suggest_experiments():
    payload = request.json
    config = payload.get('config', {})
    experiments = payload.get('experiments', [])
    
    # 1. Build the Search Space dynamically based on UI Config
    param_dict = {
        "Anion": (0, float(config.get('anionMax', 6)), 4),
        "Cation": (0, float(config.get('cationMax', 6)), 4),
        "Salt": (0, float(config.get('saltMax', 200)), 3),
    }
    
    grids = [np.linspace(lo, hi, n) for col, (lo, hi, n) in param_dict.items()]
    mesh = np.meshgrid(*grids, indexing="ij")
    points = np.column_stack([m.ravel() for m in mesh])
    
    df = pd.DataFrame(points, columns=["anion", "cation", "salt"])
    df["phase"] = -1 # Unassigned
    
    # 2. Update the Search Space with Known Experiments from Vue
    for exp in experiments:
        if exp['phase'] != -1:
            # Find the closest point in the grid or append it
            # For exact integration, we just append knowns and drop duplicates
            df = pd.concat([df, pd.DataFrame([exp])], ignore_index=True)
            
    # Clean up and prepare features
    X_raw = df[["anion", "cation", "salt"]].values
    X = scale_features(X_raw)
    y = df["phase"].values.astype(int)
    known_mask = y != -1
    
    # 3. Handle Edge Cases (Fallback to random if < 2 classes known)
    if known_mask.sum() < 2 or len(np.unique(y[known_mask])) < 2:
        unknown_idx = np.where(~known_mask)[0]
        selected = np.random.choice(unknown_idx, size=min(3, len(unknown_idx)), replace=False)
    else:
        # 4. Run the Gaussian Process Entropy Sampling
        clf = GaussianProcessClassifier(kernel=KERNEL, n_restarts_optimizer=N_RESTARTS, random_state=RANDOM_STATE)
        clf.fit(X[known_mask], y[known_mask])
        
        unknown_mask = ~known_mask
        X_unknown = X[unknown_mask]
        unknown_idx = np.where(unknown_mask)[0]
        
        proba = clf.predict_proba(X_unknown)
        entropy = -np.sum(proba * np.log(proba + 1e-12), axis=1)
        top_local = np.argsort(entropy)[::-1][:3]
        selected = unknown_idx[top_local]
        
    # 5. Format and Return Suggestions
    suggested_df = df.iloc[selected].copy()
    
    # Map back to the format Vue expects
    suggestions = []
    base_id = 9000
    for idx, row in suggested_df.iterrows():
        suggestions.append({
            "sampleId": base_id,
            "anion": round(row['anion'], 2),
            "cation": round(row['cation'], 2),
            "salt": round(row['salt'], 1),
            "phase": -1
        })
        base_id += 1
        
    return jsonify({"suggestions": suggestions})

import os

if __name__ == '__main__':
    # Render provides a $PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)