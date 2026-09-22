"""
IO Utilities

Provides JSON encoding, file I/O, and parameter validation functions.
"""

import json
import os
import glob
import math
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd


# ==============================================================================
# JSON Encoding
# ==============================================================================

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, range):
            return list(obj)
        return super().default(obj)


# ==============================================================================
# File I/O
# ==============================================================================

def save_json(data: Dict[str, Any], folder: str, filename: str) -> None:
    """Save dictionary to JSON file."""
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    print(f"Saved: {filepath}")


def load_experiment(folder: str, filename: str) -> Optional[Dict[str, Any]]:
    """Load experiment results from JSON."""
    filepath = os.path.join(folder, filename)
    
    if not os.path.isfile(filepath):
        return None
    
    if os.stat(filepath).st_size == 0:
        print(f"Warning: Empty file: {filepath}")
        return None
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in {filepath}: {e}")
        return None


# ==============================================================================
# Results Loading - Unified Interface
# ==============================================================================

def load_fit_results(patient_folder: str) -> Optional[Dict[str, Any]]:
    """
    Load GMM fitting results for a patient.
    Handles both old and new JSON formats automatically.
    """
    result = load_experiment(patient_folder, "result_fit.json")
    if result is None:
        return None
    
    # Detect format
    if "results" in result:
        # New format
        r = result["results"]
        return {
            "means": np.array(r["means"]),
            "covariances": np.array(r["covariances"]),
            "weights": [np.array(w) for w in r["weights"]],
            "training_error": r.get("training_error"),
            "test_error": r.get("test_error"),
            "fit_time": r.get("fit_time"),
            "config": result.get("config", {})
        }
    else:
        # Old format
        means = result.get("avg_means") or result.get("best_means (bag 1)")
        covs = result.get("avg_covariances") or result.get("best_covariances (bag 1)")
        weights = result.get("avg_weights") or result.get("best_weights (bag 1)")
        
        if means is None or covs is None or weights is None:
            return None
        
        return {
            "means": np.array(means),
            "covariances": np.array(covs),
            "weights": [np.array(w) for w in weights] if isinstance(weights[0], list) else [np.array(weights)],
            "training_error": result.get("avg_training_error"),
            "test_error": result.get("avg_test_error"),
            "fit_time": result.get("fit_time"),
        }


def load_node_results(patient_folder: str) -> Optional[Dict[str, Any]]:
    """Load Neural ODE results for a patient."""
    # Find node folder
    node_folder = _find_latest_node_folder(patient_folder)
    if node_folder is None:
        return None
    
    result = load_experiment(node_folder, "result_node.json")
    if result is None:
        return None
    
    # Handle formats
    if "trajectory" in result:
        traj = np.array(result["trajectory"])
        if traj.ndim == 3:
            traj = traj[:, 0, :]
        return {
            "trajectory": traj,
            "node_config": result.get("node_config", {}),
            "loss_history": result.get("loss_history", [])
        }
    elif "traj_weights" in result:
        traj = np.array(result["traj_weights"])
        if traj.ndim == 3:
            traj = traj[:, 0, :]
        return {
            "trajectory": traj,
            "node_config": {},
            "loss_history": []
        }
    
    return None


def _find_latest_node_folder(patient_folder: str) -> Optional[str]:
    """Find the most recent node folder."""
    pattern = os.path.join(patient_folder, "node_*")
    folders = sorted(glob.glob(pattern), 
                     key=lambda f: int(os.path.basename(f).split('_')[-1]) if os.path.basename(f).split('_')[-1].isdigit() else 0)
    return folders[-1] if folders else None


# ==============================================================================
# Parameter Validation
# ==============================================================================

def validate_params(params_data: Dict[str, Any], params_uq: Dict[str, Any],
                    options: Dict[str, Any]) -> None:
    """Validate experiment parameters."""
    load_data = options.get("load_data", False)
    
    # Handle bagging and fix_mv interaction
    if params_uq["n_bags"] > 1:
        options["fix_mv"] = False
    if options.get("fix_mv", False):
        params_uq["n_bags"] = 1
    
    # Configure dimensionality for loaded data
    if load_data:
        representation = params_data.get("representation", "bivariate")
        if representation not in {"univariate", "bivariate"}:
            raise ValueError("representation must be 'univariate' or 'bivariate'")
        add_deriv = params_data.get("add_deriv", False)
        add_second_deriv = params_data.get("add_second_deriv", False)
        
        if add_second_deriv:
            params_data["d"] = 3
        elif add_deriv:
            params_data["d"] = 2
        else:
            params_data["d"] = 1


def set_base_folder(options: Dict[str, Any], params_data: Dict[str, Any],
                    params_optim: Dict[str, Any], params_uq: Dict[str, Any],
                    params_node: Dict[str, Any], 
                    ignore_keys: Optional[List] = None) -> Tuple[Dict, Dict]:
    """Set up base folder for experiment results."""
    
    ignore_keys = ignore_keys or []
    # Flatten nested ignore_keys
    flat_ignore = []
    for k in ignore_keys:
        if k is None:
            continue
        if isinstance(k, tuple):
            flat_ignore.extend(k)
        else:
            flat_ignore.append(k)
    
    # Build signature for folder naming
    all_params = {**params_data, **params_optim, **params_uq, **params_node}
    
    # Split into fixed vs grid parameters
    fixed_params = {}
    grid_params = {}
    
    # These lists describe one scientific configuration; they are not grids.
    list_valued_fixed_keys = {"global_dictionary_ids"}
    for key, value in all_params.items():
        if key in flat_ignore:
            continue
        if key in list_valued_fixed_keys:
            fixed_params[key] = value
        elif isinstance(value, list) and len(value) > 1:
            grid_params[key] = value
        else:
            fixed_params[key] = value[0] if isinstance(value, list) else value
    fixed_params["fix_mv"] = bool(options.get("fix_mv", False))
    
    # Create base folder
    base_dir = options.get("results_root", "Results")
    if options.get("load_data", False):
        base_dir = os.path.join(base_dir, "CGM")
    else:
        base_dir = os.path.join(base_dir, "Synthetic")
    
    # Find or create run folder
    run_folder = _find_or_create_run_folder(base_dir, fixed_params)
    
    options["base_folder"] = run_folder
    options["grid_params"] = grid_params
    
    return options, params_data


def _find_or_create_run_folder(parent_dir: str, fixed_params: Dict[str, Any]) -> str:
    """Find matching run folder or create new one."""
    os.makedirs(parent_dir, exist_ok=True)
    
    # Search existing
    for folder_name in os.listdir(parent_dir) if os.path.exists(parent_dir) else []:
        if not folder_name.startswith("run"):
            continue
        
        params_path = os.path.join(parent_dir, folder_name, "fixed_params.json")
        if not os.path.isfile(params_path):
            continue
        
        try:
            with open(params_path) as f:
                cached = json.load(f)
            
            expected = json.loads(json.dumps(fixed_params, cls=NumpyEncoder))
            if cached == expected:
                path = os.path.join(parent_dir, folder_name)
                print(f"[INFO] Using existing run: {path}")
                return path
        except Exception:
            continue
    
    # Create new
    idx = 1
    while os.path.exists(os.path.join(parent_dir, f"run{idx}")):
        idx += 1
    
    run_folder = os.path.join(parent_dir, f"run{idx}")
    os.makedirs(run_folder, exist_ok=True)
    
    with open(os.path.join(run_folder, "fixed_params.json"), 'w') as f:
        json.dump(fixed_params, f, indent=2, sort_keys=True, cls=NumpyEncoder)
    
    print(f"[INFO] Created new run: {run_folder}")
    return run_folder


def create_folder(options: Dict[str, Any], params: Dict[str, Any]) -> str:
    """Create experiment subfolder based on varying parameters."""
    parts = []
    
    for key, values in options.get("grid_params", {}).items():
        if isinstance(values, list) and len(values) > 1:
            parts.append(f"{key}={params.get(key)}")
    
    name = "_".join(parts) if parts else "default"
    return os.path.join(options["base_folder"], name)


def check_nodefolder(combo_folder: str, combo: Dict[str, Any],
                     options: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Find or create node folder with caching support."""
    existing = sorted(
        glob.glob(os.path.join(combo_folder, "node_*")),
        key=lambda f: int(os.path.basename(f).split('_')[-1]) if os.path.basename(f).split('_')[-1].isdigit() else 0
    )
    
    # Search for matching
    for folder in existing:
        result = load_experiment(folder, "result_node.json")
        if result is None:
            continue
        
        if _node_params_match(result, combo):
            if options.get("load_cached_node", False):
                print("Loading cached NODE results")
                return folder, result
            # A requested refit gets a new node folder; existing results are
            # never overwritten.
            continue
    
    # Create new
    idx = len(existing) + 1
    new_folder = os.path.join(combo_folder, f"node_{idx}")
    os.makedirs(new_folder, exist_ok=True)
    
    return new_folder, None


def _node_params_match(cached: Dict[str, Any], combo: Dict[str, Any]) -> bool:
    """Check if NODE parameters match."""
    config = cached.get("node_config", cached)
    
    # A cached clinical NODE is only reused for the same processed dataset.
    if config.get("dataset_fingerprint") != combo.get("dataset_fingerprint"):
        return False

    node_keys = [
        "T", "dt", "seed_node", "hidden_dim", "non_linearity", "N_layers",
        "method", "normalization", "lr", "loss_func",
        "trainable_initial",
        "time_fourier_order",
        "time_polynomial_degree",
        "max_epochs",
        "softmax_epsilon", "solver_rtol", "solver_atol",
        "early_stopping_min_delta", "early_stopping_patience"
    ]
    
    for key in node_keys:
        cached_val = config.get(key)
        new_val = combo.get(key)
        
        if cached_val is None or new_val is None:
            return False
        
        if isinstance(new_val, float):
            if not math.isclose(cached_val, new_val, rel_tol=1e-8):
                return False
        elif cached_val != new_val:
            return False
    
    array_keys = [
        "observation_times_weeks", "observation_weights",
        "dictionary_means", "dictionary_covariances"
    ]
    for key in array_keys:
        cached_value = np.asarray(config.get(key, []), dtype=float)
        new_value = np.asarray(combo.get(key, []), dtype=float)
        if cached_value.shape != new_value.shape or not np.allclose(cached_value, new_value):
            return False
    return True


# ==============================================================================
# Data Export
# ==============================================================================

def save_data_csv(experiments: Dict[str, Any], output_dir: str,
                  filename: str = "data.csv") -> None:
    """Save experiment data summary to CSV."""
    records = []
    seeds, n_points_vals = set(), set()
    
    for exp_info in experiments.values():
        combo = exp_info["combo"]
        seed = combo.get("seed")
        n_pts = combo.get("n_points")
        
        if seed is None or n_pts is None:
            continue
        
        seeds.add(seed)
        n_points_vals.add(n_pts)
        
        data = [d.tolist() for d in exp_info["data"]]
        records.append((seed, n_pts, data))
    
    if not records:
        return
    
    df = pd.DataFrame(
        index=sorted(seeds),
        columns=sorted(n_points_vals),
        dtype=object
    )
    
    for seed, n_pts, data in records:
        df.at[seed, n_pts] = json.dumps(data)
    
    os.makedirs(output_dir, exist_ok=True)
    df.index.name = "seed"
    df.to_csv(os.path.join(output_dir, filename))
