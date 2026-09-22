"""
MMD-GMM Fitting Module

Fits Gaussian Mixture Models using Maximum Mean Discrepancy (MMD) objective.
Supports time-series data with shared global means/covariances and local weights.

Key improvements over original:
- Components are sorted by mean value INTERNALLY before returning
- Fixed means/covs are truly frozen (grad_update skipped when both fixed)
- Returns FitResult dataclass for clean interface
"""

import numpy as np
import time
import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union

from Utils.MMD_utils import (
    init_params, grad_update, compute_mmd_closed_form,
    compute_empirical_kernel_mean, update_local_weights
)


# Increment whenever a numerical change makes old fit caches scientifically
# non-equivalent to fits produced by the current implementation.
MMD_FIT_ALGORITHM_VERSION = "mmd-gmm-alternating-v3-fp32"


@dataclass
class FitResult:
    """Container for MMD-GMM fitting results."""
    # Best iteration
    best_index: int

    # Final parameters (already sorted by mean)
    weights: List[np.ndarray]      # [t_steps] arrays of shape (K,)
    means: np.ndarray              # Shape (K, d)
    covariances: np.ndarray        # Shape (K, d, d)

    # History (all sorted consistently)
    weights_history: List[List[np.ndarray]]
    means_history: List[np.ndarray]
    covariances_history: List[np.ndarray]

    # Error histories
    mmd_history: List[float]
    objective_history: List[float]  # MMD plus the ridge penalty used in fitting
    weights_error_history: List[List[float]]
    means_error_history: List[float]
    covariances_error_history: List[float]

    # Metadata
    fit_time: float
    device: str
    convergence_iter: int
    covariance_floor: float = 1e-2


def _sort_components(means: np.ndarray, covs: np.ndarray, 
                     local_weights: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Sort GMM components by mean value (first dimension).
    
    This ensures consistent ordering regardless of initialization,
    making results reproducible and comparable across runs.
    """
    # Sort by first dimension of means
    # Stable sorting makes the canonical label deterministic even if two
    # components have the same first-coordinate mean. Covariances and every
    # time-specific weight vector are permuted by exactly the same order.
    order = np.argsort(means[:, 0], kind="stable")
    
    sorted_means = means[order]
    sorted_covs = covs[order]
    sorted_weights = [w[order] for w in local_weights]
    
    return sorted_means, sorted_covs, sorted_weights


def _ridge_penalty(local_weights: List[np.ndarray],
                   lambda_ridge: Union[float, List[float]], K: int) -> float:
    """Return the ridge term used by all local-weight subproblems."""
    values = np.asarray(lambda_ridge, dtype=float).reshape(-1)
    if values.size == 1:
        diagonal = np.repeat(values, K)
    elif values.size == K:
        diagonal = values
    else:
        raise ValueError(f"lambda_ridge must have length 1 or K={K}")
    if not np.all(np.isfinite(diagonal)) or np.any(diagonal < 0):
        raise ValueError("lambda_ridge must be finite and non-negative")
    return float(sum(np.sum(diagonal * np.asarray(weights, dtype=float) ** 2)
                     for weights in local_weights))


def mmd_gmm_fit(
    X: List[np.ndarray],
    K: int,
    gamma: Union[float, List[float]] = 1.0,
    max_iter: int = 100,
    tol_rel_param: float = 1e-6,
    tol_rel_err: float = 1e-5,
    tol_abs: float = 1e-4,
    lr: float = 0.01,
    grad_steps: int = 20,
    verbose: bool = False,
    init_weights: Optional[np.ndarray] = None,
    init_means: Optional[np.ndarray] = None,
    init_covs: Optional[np.ndarray] = None,
    fixed_means: Optional[np.ndarray] = None,
    fixed_covs: Optional[np.ndarray] = None,
    lambda_ridge: float = 0.0,
    seed: int = 0,
    covariance_floor: float = 1e-2
) -> FitResult:
    """
    Fit a Gaussian Mixture Model using MMD-based objective.
    
    Performs alternating optimization:
    1. Update local weights for each dataset via quadratic programming
    2. Update global means/covariances via gradient descent (if not fixed)
    
    Components are sorted by mean value before returning to ensure
    consistent ordering across runs and patients.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"[INFO] Using device: {device}")
    
    covariance_floor = float(covariance_floor)
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    start = time.time()
    
    # Get dimensions
    X_all = np.concatenate(X, axis=0)
    _, d = X_all.shape
    t_steps = len(X)
    
    # Initialize parameters
    local_weights, means, covs = init_params(
        X_all, K, init_weights, init_means, init_covs, t_steps, seed=seed
    )
    
    # Handle fixed parameters
    freeze_mu = fixed_means is not None
    freeze_cov = fixed_covs is not None
    
    if freeze_mu:
        means = np.asarray(fixed_means, dtype=float).copy()
    if freeze_cov:
        covs = np.asarray(fixed_covs, dtype=float).copy()
    
    # Pre-sort fixed parameters for consistency
    if freeze_mu and freeze_cov:
        means, covs, local_weights = _sort_components(means, covs, local_weights)
    
    # Ensure gamma is a list
    if not isinstance(gamma, (list, tuple, np.ndarray)):
        gamma = [gamma] * t_steps
    gamma = np.asarray(gamma, dtype=float).reshape(-1)
    if len(gamma) != t_steps:
        raise ValueError("gamma must contain one bandwidth per time window")
    if not np.all(np.isfinite(gamma)) or np.any(gamma <= 0):
        raise ValueError("All bandwidths must be positive and finite")
    
    # Precompute Gamma matrices
    Gamma_matrices = [(g**2) * np.eye(d) for g in gamma]
    empirical_terms = [
        compute_empirical_kernel_mean(X_i, gamma_i)
        for X_i, gamma_i in zip(X, gamma)
    ]
    
    # Initialize histories
    hist_means, hist_covs = [], []
    hist_errmeans, hist_errcovs, hist_errweights = [], [], []
    mmd_list, objective_list = [], []
    history_ws = []
    
    best_index, best_objective, prev_objective = -1, np.inf, None
    
    # Main optimization loop
    convergence_iter = max_iter
    for it in range(max_iter):
        # Store old parameters
        means_old = means.copy()
        covs_old = covs.copy()
        local_weights_old = [lw.copy() for lw in local_weights]
        
        # Step 1: Update local weights for each time instant
        for i, X_i in enumerate(X):
            local_weights[i] = update_local_weights(
                X_i, means, covs, Gamma_matrices[i], lambda_ridge, local_weights[i]
            )
        
        # Step 2: Update global means/covariances (ONLY if not fixed)
        if not (freeze_mu and freeze_cov):
            new_means, new_covs = grad_update(
                X, local_weights, means, covs, 
                gamma, grad_steps, lr, device,
                eps_val=covariance_floor,
                freeze_means=freeze_mu,
                freeze_covariances=freeze_cov,
            )

            means = new_means
            covs = new_covs

            # The global update changes the geometry of every component.  The
            # weights optimized immediately before it are consequently stale;
            # solve their convex subproblems again before evaluating/storing a
            # checkpoint.
            for i, X_i in enumerate(X):
                local_weights[i] = update_local_weights(
                    X_i, means, covs, Gamma_matrices[i],
                    lambda_ridge, local_weights[i]
                )
        
        # Compute parameter changes
        eps = 1e-12
        diff_means = (0.0 if freeze_mu else
                      np.linalg.norm(means - means_old) / (np.linalg.norm(means_old) + eps))
        diff_cov = (0.0 if freeze_cov else
                    np.linalg.norm(covs - covs_old) / (np.linalg.norm(covs_old) + eps))
        diff_w = [np.linalg.norm(lw - lw_old) / (np.linalg.norm(lw_old) + eps)
                  for lw, lw_old in zip(local_weights, local_weights_old)]
        
        # Compute total MMD error
        total_sq_mmd = sum(
            compute_mmd_closed_form(
                X[i], local_weights[i], means, covs, gamma[i], empirical_terms[i]
            )
            for i in range(t_steps)
        )
        total_objective = (
            total_sq_mmd + _ridge_penalty(local_weights, lambda_ridge, K)
        )
        
        # Store histories
        hist_errmeans.append(diff_means)
        hist_errcovs.append(diff_cov)
        hist_errweights.append(diff_w)
        mmd_list.append(total_sq_mmd)
        objective_list.append(total_objective)
        hist_means.append(means.copy())
        hist_covs.append(covs.copy())
        history_ws.append([lw.copy() for lw in local_weights])
        
        # Track best
        if total_objective < best_objective:
            best_objective = total_objective
            best_index = it
        
        if verbose and it % 10 == 0:
            avg_diff_w = np.max(diff_w) if diff_w else 0
            print(f"Iter {it+1}: Δmeans={diff_means:.6f}, Δcovs={diff_cov:.6f}, "
                  f"Δweights={avg_diff_w:.6f}, MMD²={total_sq_mmd:.6f}, "
                  f"objective={total_objective:.6f}")
        
        # Check convergence
        avg_diff_w = np.max(diff_w) if diff_w else 0
        
        if diff_means < tol_rel_param and diff_cov < tol_rel_param and avg_diff_w < tol_rel_param:
            if verbose:
                print(f"[INFO] Converged at iteration {it}: parameter tolerance reached")
            convergence_iter = it
            break
            
        if (prev_objective is not None and
                abs(total_objective - prev_objective) /
                (abs(prev_objective) + eps) < tol_rel_err):
            if verbose:
                print(f"[INFO] Converged at iteration {it}: relative objective tolerance reached")
            convergence_iter = it
            break
            
        # ``tol_abs`` measures the unpenalized fitting discrepancy.  With a
        # positive ridge, the full objective has a positive lower bound and
        # could never satisfy a small absolute tolerance.
        if total_sq_mmd < tol_abs:
            if verbose:
                print(f"[INFO] Converged at iteration {it}: absolute MMD tolerance reached")
            convergence_iter = it
            break
        
        prev_objective = total_objective
    
    fit_time = time.time() - start
    if verbose:
        print(f"[INFO] Fitting completed in {fit_time:.4f} sec")
    
    # Get best parameters
    best_means = hist_means[best_index]
    best_covs = hist_covs[best_index]
    best_weights = history_ws[best_index]
    
    # Sort all results by component mean for consistent ordering
    best_means, best_covs, best_weights = _sort_components(
        best_means, best_covs, best_weights
    )
    
    # Sort histories as well (for plotting consistency)
    sorted_hist_means = []
    sorted_hist_covs = []
    sorted_history_ws = []
    
    for m, c, w in zip(hist_means, hist_covs, history_ws):
        sm, sc, sw = _sort_components(m, c, w)
        sorted_hist_means.append(sm)
        sorted_hist_covs.append(sc)
        sorted_history_ws.append(sw)
    
    return FitResult(
        best_index=best_index,
        weights=best_weights,
        means=best_means,
        covariances=best_covs,
        weights_history=sorted_history_ws,
        means_history=sorted_hist_means,
        covariances_history=sorted_hist_covs,
        mmd_history=mmd_list,
        objective_history=objective_list,
        weights_error_history=hist_errweights,
        means_error_history=hist_errmeans,
        covariances_error_history=hist_errcovs,
        fit_time=fit_time,
        device=str(device),
        convergence_iter=convergence_iter,
        covariance_floor=covariance_floor
    )
