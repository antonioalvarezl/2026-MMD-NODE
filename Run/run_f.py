"""
MMD-GMM Fitting Runner

Orchestrates the fitting process.
Returns format compatible with Experiment.py expectations (avg_means, avg_covariances, avg_weights).
"""

import os
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from Utils.IO_utils import load_experiment, save_json
from Visualization.log import log_experiment
from Core.train_mmd import (
    MMD_FIT_ALGORITHM_VERSION,
    FitResult,
    mmd_gmm_fit,
)
from Visualization.plots import plot_fit, plot_mixture


# ==============================================================================
# Main Entry Point
# ==============================================================================

def run_fitting(combo: Dict[str, Any], options: Dict[str, Any], 
                results_folder: str, Xs: List[np.ndarray], 
                test_gamma: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Run MMD-GMM fitting with caching support.
    
    Returns:
        Results dictionary with keys:
        - avg_means, avg_covariances, avg_weights (for NODE training)
    """
    combo = dict(combo)
    if test_gamma is not None:
        combo["window_bandwidths"] = np.asarray(test_gamma, dtype=float).tolist()
    log_experiment(combo, phase=2, options=options)
    
    # Try loading cached results
    cached = _try_load_cached_fit(results_folder, options, combo)
    if cached is not None:
        return cached
    
    # Determine fitting mode
    n_bags = combo.get("n_bags", 1)
    
    if n_bags > 1:
        return _run_bagging_fit(combo, options, results_folder, Xs, test_gamma, n_bags)
    else:
        return _run_single_fit(combo, options, results_folder, Xs, test_gamma)


# ==============================================================================
# Caching
# ==============================================================================

def _fit_config(combo, options):
    """Scientific inputs that must agree before a fit cache is reused."""
    return {
        "fit_algorithm_version": MMD_FIT_ALGORITHM_VERSION,
        "K": combo["K"],
        "d": combo.get("d", 1),
        "t_steps": combo.get("t_steps"),
        # A clinical cache is only ever reused for the same processed dataset.
        "dataset_fingerprint": combo.get("dataset_fingerprint"),
        "nominal_week_indices": combo.get("nominal_week_indices"),
        "analysis_window_definition": combo.get("analysis_window_definition"),
        "max_iter": combo["max_iter"],
        "tol_rel_param": combo["tol_rel_param"],
        "tol_rel_err": combo["tol_rel_err"],
        "tol_abs": combo["tol_abs"],
        "lr": combo["lr"],
        "grad_steps": combo["grad_steps"],
        "lambda_ridge": combo.get("lambda_ridge", 0),
        "covariance_floor": combo.get("covariance_floor", options.get("covariance_floor", 1e-2)),
        "bandwidth_candidates": combo.get("bandwidth_candidates", options.get("bandwidth_candidates", [0.25, 0.5, 1.0, 2.0, 4.0])),
        "global_bandwidth_multiplier": combo.get("global_bandwidth_multiplier"),
        "bandwidth_selection": combo.get("bandwidth_selection"),
        "dictionary_seed": combo.get("dictionary_seed", 0),
        "fix_mv": options.get("fix_mv", False),
        "window_bandwidths": combo.get("window_bandwidths", []),
        "bandwidth_scope": combo.get("bandwidth_scope"),
        "patient_bandwidth_reference": combo.get("patient_bandwidth_reference"),
        "patient_bandwidth_reference_n_total": combo.get("patient_bandwidth_reference_n_total"),
        "patient_bandwidth_reference_n_sample": combo.get("patient_bandwidth_reference_n_sample"),
        "block_metadata": combo.get("block_metadata", []),
        "observation_times_weeks": combo.get("observation_times_weeks", []),
        "calendar_origin_date": combo.get("calendar_origin_date"),
        # PEDAP v2 calendar origin (randomisation date); legacy runs carry
        # ``calendar_origin_date`` (first CGM reading) instead.
        "rand_date": combo.get("rand_date"),
        "dictionary": combo.get("dictionary", {})
    }


def _try_load_cached_fit(results_folder: str, options: Dict[str, Any],
                         combo: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Load cached fit results if available and requested."""
    if not options.get("load_cached_fit", False):
        return None
    
    existing = load_experiment(results_folder, "result_fit.json")
    if existing is None:
        return None
    
    if "results" in existing and "means" in existing["results"]:
        cached_config = existing.get("config", {})
        expected_config = _fit_config(combo, options)
        if any(cached_config.get(key) != value for key, value in expected_config.items()):
            print("[FIT] Cached result does not match the current scientific inputs")
            return None
        r = existing["results"]
        print("[FIT] Using cached fit results")
        return {
            "avg_means": np.array(r["means"]),
            "avg_covariances": np.array(r["covariances"]),
            "avg_weights": [np.array(w) for w in r["weights"]],
            "training_error": r.get("training_error"),
            "training_objective": r.get("training_objective"),
            "test_error": r.get("test_error"),
            "fit_time": r.get("fit_time"),
        }
    
    return None


# ==============================================================================
# Single Fit Mode
# ==============================================================================

def _run_single_fit(combo: Dict[str, Any], options: Dict[str, Any],
                    results_folder: str, Xs: List[np.ndarray],
                    test_gamma: Optional[List[float]]) -> Dict[str, Any]:
    """Run single fit without bagging."""
    print("[FIT] Starting single fit mode")
    
    # Execute fitting
    fit_result = _execute_mmd_fit(Xs, combo, options, test_gamma)
    
    # Log results
    log_experiment(
        combo, phase=3,
        errors=fit_result.mmd_history,
        best_index=fit_result.best_index,
        conv_it=fit_result.convergence_iter,
        test_error=None
    )
    
    # Generate plots
    if options.get("plots", True):
        _plot_fit_results(fit_result, Xs, combo, options, results_folder)
    
    # Build compatible format for Experiment.py
    results = {
        "avg_means": fit_result.means,
        "avg_covariances": fit_result.covariances,
        "avg_weights": fit_result.weights,
        "training_error": float(fit_result.mmd_history[fit_result.best_index]),
        "training_objective": float(
            fit_result.objective_history[fit_result.best_index]
        ),
        "test_error": None,
        "fit_time": fit_result.fit_time,
        "best_index": fit_result.best_index,
        "convergence_iter": fit_result.convergence_iter,
    }
    
    # Save clean JSON
    _save_results(results_folder, combo, options, fit_result)
    
    print("[FIT] Single fit completed")
    return results


# ==============================================================================
# Bagging Mode
# ==============================================================================

def _run_bagging_fit(combo: Dict[str, Any], options: Dict[str, Any],
                     results_folder: str, Xs: List[np.ndarray],
                     test_gamma: Optional[List[float]], n_bags: int) -> Dict[str, Any]:
    """Run fitting with bootstrap aggregation."""
    print(f"[FIT] Starting bagging mode: {n_bags} bags")
    
    all_means, all_covs, all_weights = [], [], []
    all_errors, all_times = [], []
    bag_results_for_json = []
    
    for bag_idx in range(n_bags):
        print(f"[FIT] Processing bag {bag_idx + 1}/{n_bags}")
        
        seed = combo.get("seed_split", 0) + bag_idx
        X_train = _bootstrap_sample(Xs, seed)
        
        # Fit on bootstrap sample
        fit_result = _execute_mmd_fit(X_train, combo, options, test_gamma)
        
        # Log
        log_experiment(
            combo, phase=3,
            errors=fit_result.mmd_history,
            best_index=fit_result.best_index,
            conv_it=fit_result.convergence_iter,
            test_error=None
        )
        
        # Generate plots for this bag
        bag_folder = os.path.join(results_folder, f"bag_{bag_idx + 1}")
        os.makedirs(bag_folder, exist_ok=True)
        _plot_fit_results(fit_result, X_train, combo, options, bag_folder)
        
        # Collect
        all_means.append(fit_result.means)
        all_covs.append(fit_result.covariances)
        all_weights.append(np.array([w for w in fit_result.weights]))
        all_errors.append(fit_result.mmd_history[fit_result.best_index])
        all_times.append(fit_result.fit_time)
        
        bag_results_for_json.append({
            "seed": seed,
            "means": fit_result.means.tolist(),
            "covariances": fit_result.covariances.tolist(),
            "weights": [w.tolist() for w in fit_result.weights],
            "training_error": float(fit_result.mmd_history[fit_result.best_index]),
            "fit_time": fit_result.fit_time
        })
    
    # Compute averages
    avg_means = np.mean(all_means, axis=0)
    avg_covs = np.mean(all_covs, axis=0)
    avg_weights_arr = np.mean(all_weights, axis=0)
    avg_weights = [avg_weights_arr[i] for i in range(avg_weights_arr.shape[0])]
    
    # Save JSON with bagging info
    _save_results_bagging(results_folder, combo, options, avg_means, avg_covs, 
                          avg_weights, all_errors, bag_results_for_json)
    
    print("[FIT] Bagging completed")
    
    # Return compatible format
    return {
        "avg_means": avg_means,
        "avg_covariances": avg_covs,
        "avg_weights": avg_weights,
        "training_error": float(np.mean(all_errors)),
        "training_error_std": float(np.std(all_errors)),
        "test_error": None,
        "fit_time": float(np.mean(all_times)),
        "n_bags": n_bags
    }


def _bootstrap_sample(Xs: List[np.ndarray], seed: int) -> List[np.ndarray]:
    """Create bootstrap sample."""
    rng = np.random.RandomState(seed)
    X_train = []
    
    for arr in Xs:
        n = len(arr)
        indices = rng.choice(n, n, replace=True)
        X_train.append(arr[indices])
    
    return X_train


# ==============================================================================
# Core Fitting
# ==============================================================================

def _execute_mmd_fit(Xs: List[np.ndarray], combo: Dict[str, Any],
                     options: Dict[str, Any], 
                     test_gamma: Optional[List[float]]) -> FitResult:
    """Execute MMD-GMM fitting."""
    
    # Determine gamma values
    gamma = test_gamma if test_gamma is not None else combo.get("gamma", 1.0)
    
    # Handle fixed means/covs
    fixed_means = None
    fixed_covs = None
    
    if options.get("fix_mv", False):
        has_means = "means" in combo
        has_covariances = "covs" in combo
        if options.get("load_data", False) and not (has_means and has_covariances):
            raise ValueError("A frozen dictionary requires both means and covariances")
        if has_means and has_covariances:
            fixed_means = np.asarray(combo["means"], dtype=float)
            fixed_covs = np.asarray(combo["covs"], dtype=float)
        else:
            # Preserve the original synthetic path, where ``fix_mv`` was set
            # globally but no clinical dictionary was supplied.
            fixed_means = fixed_covs = None
    if fixed_means is not None:
        expected_mean_shape = (int(combo["K"]), int(combo.get("d", 1)))
        expected_cov_shape = expected_mean_shape + (expected_mean_shape[1],)
        if fixed_means.shape != expected_mean_shape or fixed_covs.shape != expected_cov_shape:
            raise ValueError("Frozen dictionary dimensions do not match K and d")
        if (not np.all(np.isfinite(fixed_means)) or
                not np.all(np.isfinite(fixed_covs)) or
                any(np.min(np.linalg.eigvalsh(covariance)) <= 0
                    for covariance in fixed_covs)):
            raise ValueError("Frozen dictionary parameters must be finite and positive definite")
    
    # Run fitting
    return mmd_gmm_fit(
        X=Xs,
        K=combo["K"],
        gamma=gamma,
        max_iter=combo["max_iter"],
        tol_rel_param=combo["tol_rel_param"],
        tol_rel_err=combo["tol_rel_err"],
        tol_abs=combo["tol_abs"],
        lr=combo["lr"],
        grad_steps=combo["grad_steps"],
        verbose=options.get("verbose", False),
        fixed_means=fixed_means,
        fixed_covs=fixed_covs,
        lambda_ridge=combo.get("lambda_ridge", 0),
        seed=combo.get("dictionary_seed", 0),
        covariance_floor=combo.get("covariance_floor", options.get("covariance_floor", 1e-2))
    )


# ==============================================================================
# Saving Results
# ==============================================================================

def _save_results(results_folder: str, combo: Dict[str, Any], 
                  options: Dict[str, Any], fit_result: FitResult) -> None:
    """Save results in clean JSON format."""
    
    json_data = {
        "config": {**_fit_config(combo, options), "device": fit_result.device},
        "results": {
            "means": fit_result.means.tolist(),
            "covariances": fit_result.covariances.tolist(),
            "weights": [w.tolist() for w in fit_result.weights],
            "training_error": float(fit_result.mmd_history[fit_result.best_index]),
            "training_objective": float(
                fit_result.objective_history[fit_result.best_index]
            ),
            "mmd_history": [float(value) for value in fit_result.mmd_history],
            "objective_history": [float(value) for value in fit_result.objective_history],
            "fit_time": fit_result.fit_time,
            "convergence_iter": fit_result.convergence_iter,
            "best_index": fit_result.best_index
        }
    }
    
    # Add init_info if available (contains patient IDs used for means/covs and data quality)
    if "init_info" in combo:
        json_data["init_info"] = combo["init_info"]
    
    os.makedirs(results_folder, exist_ok=True)
    save_json(json_data, results_folder, "result_fit.json")


def _save_results_bagging(results_folder: str, combo: Dict[str, Any],
                          options: Dict[str, Any], avg_means: np.ndarray,
                          avg_covs: np.ndarray, avg_weights: List[np.ndarray],
                          all_errors: List[float], bags: List[Dict]) -> None:
    """Save bagging results in clean JSON format."""
    
    json_data = {
        "config": {**_fit_config(combo, options), "n_bags": len(bags)},
        "results": {
            "means": avg_means.tolist(),
            "covariances": avg_covs.tolist(),
            "weights": [w.tolist() for w in avg_weights],
            "training_error": float(np.mean(all_errors)),
            "training_error_std": float(np.std(all_errors)),
        },
        "bags": bags
    }
    
    # Add init_info if available (contains patient IDs used for means/covs and data quality)
    if "init_info" in combo:
        json_data["init_info"] = combo["init_info"]
    
    os.makedirs(results_folder, exist_ok=True)
    save_json(json_data, results_folder, "result_fit.json")


# ==============================================================================
# Plotting
# ==============================================================================

def _plot_fit_results(fit_result: FitResult, X_data: List[np.ndarray],
                      combo: Dict[str, Any], options: Dict[str, Any],
                      folder: str) -> None:
    """Generate diagnostic plots in a 'plots' subfolder."""
    try:
        d = combo.get("d", 1)
        fix_mv = options.get("fix_mv", False)
        
        # Create plots subfolder to avoid cluttering the patient folder
        plots_folder = os.path.join(folder, "plots")
        os.makedirs(plots_folder, exist_ok=True)
        
        for i, data in enumerate(X_data):
            if i >= len(fit_result.weights):
                break
                
            weights = fit_result.weights[i]
            filename = f"ds{i + 1}.png"
            
            if d == 1 and not fix_mv:
                # Full diagnostic plot with error curves
                weights_errors = [err[i] if i < len(err) else 0 
                                  for err in fit_result.weights_error_history]
                plot_fit(
                    weights_errors,
                    fit_result.means_error_history,
                    fit_result.covariances_error_history,
                    fit_result.mmd_history,
                    data, weights,
                    fit_result.means,
                    fit_result.covariances,
                    fit_result.best_index,
                    folder=plots_folder,
                    filename=f"fit_{filename}",
                    log_scale=True
                )
            elif d == 1:
                # Just mixture plot (fixed means/covs)
                plot_mixture(
                    data, weights,
                    fit_result.means,
                    fit_result.covariances,
                    folder=plots_folder,
                    filename=f"mix_{filename}"
                )
    except Exception as e:
        print(f"[WARNING] Failed to generate plots: {e}")
