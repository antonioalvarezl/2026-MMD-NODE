"""
MMD Utilities - Optimized Version

Key improvements:
- Vectorized J computation (O(n) instead of O(n²) Python loops)
- Cleaner matrix operations
- Better numerical stability
"""

import numpy as np
from numpy.linalg import inv, det, cholesky, LinAlgError
from scipy.optimize import minimize
from typing import List, Tuple, Optional, Union


def backtransform_gmm_parameters(means, covariances, center, scale):
    """Return Gaussian parameters in the original glucose/slope units."""
    means = np.asarray(means, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    center = np.asarray(center, dtype=float)
    scale = np.asarray(scale, dtype=float)
    if means.shape[1] != len(center) or len(center) != len(scale):
        raise ValueError("Scaling dimensions do not match Gaussian parameters")
    transform = np.diag(scale)
    means_original = means * scale + center
    covariances_original = np.asarray([
        transform @ covariance @ transform for covariance in covariances
    ])
    return means_original, covariances_original


# ==============================================================================
# Kernel Functions
# ==============================================================================

def gaussian_kernel(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    """
    Compute Gaussian (RBF) kernel matrix between two point sets.
    
    K(x, y) = exp(-||x - y||² / (2γ²))
    
    Args:
        X: (n, d) array
        Y: (m, d) array
        gamma: Kernel bandwidth
        
    Returns:
        (n, m) kernel matrix
    """
    # Use broadcasting for efficiency
    diff = X[:, None, :] - Y[None, :, :]  # (n, m, d)
    sq_dists = np.sum(diff**2, axis=2)     # (n, m)
    return np.exp(-0.5 * sq_dists / (gamma**2))


# ==============================================================================
# Closed-Form MMD Terms
# ==============================================================================

def compute_I(mu_k: np.ndarray, Sigma_k: np.ndarray, 
              mu_l: np.ndarray, Sigma_l: np.ndarray, 
              Gamma: np.ndarray) -> float:
    """
    Compute I_{k,l} = E[k(X, X')] where X ~ N(μ_k, Σ_k), X' ~ N(μ_l, Σ_l).
    
    Closed form for Gaussian kernel with Gaussian distributions.
    """
    Sigma_sum = Sigma_k + Sigma_l + Gamma
    
    try:
        det_ratio = np.sqrt(det(Gamma) / det(Sigma_sum))
        diff = mu_k - mu_l
        exp_term = np.exp(-0.5 * diff @ inv(Sigma_sum) @ diff)
        return det_ratio * exp_term
    except LinAlgError:
        # Fallback for singular matrices
        return 0.0


def compute_I_matrix(means: np.ndarray, covs: np.ndarray, 
                     Gamma: np.ndarray) -> np.ndarray:
    """
    Compute full I matrix for all component pairs.
    
    Args:
        means: (K, d) array of component means
        covs: (K, d, d) array of component covariances
        Gamma: (d, d) kernel matrix
        
    Returns:
        (K, K) matrix where I[k,l] = I_{k,l}
    """
    K = len(means)
    I_mat = np.zeros((K, K))
    
    for k in range(K):
        for l in range(k, K):  # Exploit symmetry
            I_kl = compute_I(means[k], covs[k], means[l], covs[l], Gamma)
            I_mat[k, l] = I_kl
            I_mat[l, k] = I_kl
    
    return I_mat


def compute_J_vectorized(means: np.ndarray, covs: np.ndarray, 
                         X: np.ndarray, Gamma: np.ndarray) -> np.ndarray:
    """
    Compute J matrix for all data points and components - VECTORIZED.
    
    J[i,k] = E[k(X, x_i)] where X ~ N(μ_k, Σ_k)
    
    Args:
        means: (K, d) array of component means
        covs: (K, d, d) array of component covariances
        X: (n, d) array of data points
        Gamma: (d, d) kernel matrix
        
    Returns:
        (n, K) matrix where J[i,k] = J_{k,i}
    """
    K, d = means.shape
    n = X.shape[0]
    
    J = np.zeros((n, K))
    det_Gamma = det(Gamma)
    
    for k in range(K):
        Sigma_sum = covs[k] + Gamma
        
        try:
            det_Sigma_sum = det(Sigma_sum)
            det_ratio = np.sqrt(det_Gamma / det_Sigma_sum)
            inv_Sigma = inv(Sigma_sum)
            
            # Vectorized computation for all data points
            diff = X - means[k]  # (n, d)
            
            # Efficient quadratic form: diff @ inv_Sigma @ diff.T diagonal
            # Using einsum for efficiency
            quad_form = np.einsum('ij,jk,ik->i', diff, inv_Sigma, diff)
            
            J[:, k] = det_ratio * np.exp(-0.5 * quad_form)
            
        except LinAlgError:
            J[:, k] = 0.0
    
    return J


def compute_J(mu_k: np.ndarray, Sigma_k: np.ndarray, 
              x_i: np.ndarray, Gamma: np.ndarray) -> float:
    """
    Compute single J value (kept for backward compatibility).
    
    J_{k,i} = E[k(X, x_i)] where X ~ N(μ_k, Σ_k)
    """
    Sigma_sum = Sigma_k + Gamma
    
    try:
        det_ratio = np.sqrt(det(Gamma) / det(Sigma_sum))
        diff = x_i - mu_k
        exp_term = np.exp(-0.5 * diff @ inv(Sigma_sum) @ diff)
        return det_ratio * exp_term
    except LinAlgError:
        return 0.0


# ==============================================================================
# MMD Computation
# ==============================================================================

def compute_empirical_kernel_mean(X, gamma, block_size=512):
    """Compute the empirical RBF V-term without allocating an n by n matrix."""
    X = np.asarray(X, dtype=float)
    squared_norms = np.sum(X**2, axis=1)
    total = 0.0
    for start in range(0, len(X), block_size):
        block = X[start:start + block_size]
        squared = (np.sum(block**2, axis=1)[:, None] + squared_norms[None, :]
                   - 2.0 * block @ X.T)
        np.maximum(squared, 0.0, out=squared)
        total += np.exp(-0.5 * squared / float(gamma)**2).sum()
    return float(total / (len(X) * len(X)))

def compute_mmd_closed_form(X: np.ndarray, weights: np.ndarray, 
                            means: np.ndarray, covariances: np.ndarray, 
                            gamma: float, empirical_term=None) -> float:
    """
    Compute squared MMD between empirical distribution and GMM.
    
    MMD²(P_X, Q) = E_{Q×Q}[k] - 2E_{P×Q}[k] + E_{P×P}[k]
    
    Uses closed-form expressions for Gaussian components.
    
    Args:
        X: (n, d) data array
        weights: (K,) mixture weights
        means: (K, d) component means
        covariances: (K, d, d) component covariances
        gamma: Kernel bandwidth
        
    Returns:
        Squared MMD value
    """
    means = np.asarray(means)
    covariances = np.asarray(covariances)
    weights = np.asarray(weights)
    
    n, d = X.shape
    K = len(weights)
    Gamma = (gamma**2) * np.eye(d)
    
    # Term 3: E_{P×P}[k] - Empirical kernel mean
    term3 = (compute_empirical_kernel_mean(X, gamma)
             if empirical_term is None else float(empirical_term))
    
    # Term 1: E_{Q×Q}[k] - GMM self-interaction
    I_mat = compute_I_matrix(means, covariances, Gamma)
    term1 = weights @ I_mat @ weights
    
    # Term 2: E_{P×Q}[k] - Cross term (VECTORIZED)
    J_mat = compute_J_vectorized(means, covariances, X, Gamma)  # (n, K)
    term2 = 2.0 * np.mean(J_mat @ weights)
    
    return term1 - term2 + term3


# ==============================================================================
# Parameter Initialization
# ==============================================================================

def KMeans(X: np.ndarray, n: int, d: int, K: int,
           seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize GMM parameters via k-means clustering.
    
    Returns:
        (weights, means, covariances)
    """
    try:
        from sklearn.cluster import KMeans as SKKMeans
        kmeans = SKKMeans(n_clusters=K, random_state=seed, n_init=10).fit(X)
        labels = kmeans.labels_
        means = kmeans.cluster_centers_
    except ImportError:
        # Fallback: random initialization
        indices = np.random.default_rng(seed).choice(n, K, replace=False)
        means = X[indices].copy()
        labels = np.zeros(n, dtype=int)
    
    weights = np.zeros(K)
    covariances = np.zeros((K, d, d))
    
    for k in range(K):
        cluster_data = X[labels == k]
        
        if cluster_data.shape[0] < 2:
            weights[k] = 1.0 / K
            covariances[k] = np.eye(d)
        else:
            weights[k] = cluster_data.shape[0] / n
            cov = np.cov(cluster_data, rowvar=False, ddof=0)
            
            # Ensure proper shape and positive definiteness
            if d == 1:
                covariances[k] = np.array([[cov]]) if np.isscalar(cov) else cov.reshape(1, 1)
            else:
                covariances[k] = cov
            
            # Add regularization for numerical stability
            covariances[k] += 1e-6 * np.eye(d)
    
    return weights, means, covariances


def init_params(X: np.ndarray, K: int, 
                init_w: Optional[np.ndarray] = None,
                init_means: Optional[np.ndarray] = None,
                init_covs: Optional[np.ndarray] = None,
                t_steps: int = 1,
                seed: int = 0) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Initialize GMM parameters.
    
    If all initial values provided, use them.
    Otherwise, use k-means clustering.
    
    Returns:
        (local_weights, means, covariances)
        - local_weights: list of (K,) arrays, one per time step
        - means: (K, d) array
        - covariances: (K, d, d) array
    """
    if init_means is not None and init_w is not None and init_covs is not None:
        # Use provided initialization
        w_list = [np.asarray(init_w).copy() for _ in range(t_steps)]
        return w_list, np.asarray(init_means).copy(), np.asarray(init_covs).copy()
    
    # K-means initialization
    n, d = X.shape
    w, means, covs = KMeans(X, n, d, K, seed=seed)
    w_list = [w.copy() for _ in range(t_steps)]
    
    return w_list, means, covs


# ==============================================================================
# Weight Optimization
# ==============================================================================

def update_local_weights(X: np.ndarray, means: np.ndarray, covs: np.ndarray,
                         Gamma: np.ndarray, lambda_ridge: Union[float, List[float]],
                         w_old: np.ndarray) -> np.ndarray:
    """
    Update local weights via quadratic programming.
    
    Minimizes: w'Iw - 2w'J + λ||w||²
    Subject to: w >= 0, sum(w) = 1
    
    Args:
        X: (n, d) data array
        means: (K, d) component means
        covs: (K, d, d) component covariances
        Gamma: (d, d) kernel matrix
        lambda_ridge: Regularization (scalar or per-component)
        w_old: (K,) initial weights
        
    Returns:
        (K,) optimized weights
    """
    K = len(means)
    # Ensure w_old is a feasible starting point.  SLSQP is substantially more
    # reliable when its initial point satisfies the simplex constraint exactly.
    w_old = np.asarray(w_old, dtype=float).flatten()
    if w_old.size == 1:
        w_old = np.ones(K) / K
    elif w_old.size != K:
        raise ValueError(f"Expected weights of length {K}, got {w_old.size}")
    if not np.all(np.isfinite(w_old)):
        raise ValueError("Initial weights must be finite")
    w_old = np.maximum(w_old, 0.0)
    if w_old.sum() <= 0:
        w_old = np.ones(K) / K
    else:
        w_old /= w_old.sum()
    
    # Compute I matrix (K x K)
    I_mat = compute_I_matrix(means, covs, Gamma)
    I_mat = 0.5 * (I_mat + I_mat.T)
    
    # Compute J vector (K,) - using vectorized version
    J_mat = compute_J_vectorized(means, covs, X, Gamma)  # (n, K)
    J_vec = J_mat.mean(axis=0)  # Average over data points
    
    # Handle lambda_ridge
    if isinstance(lambda_ridge, (list, np.ndarray)):
        lambda_ridge = np.asarray(lambda_ridge, dtype=float).reshape(-1)
        if len(lambda_ridge) == 1:
            lambda_diag = lambda_ridge[0] * np.ones(K)
        elif len(lambda_ridge) == K:
            lambda_diag = lambda_ridge
        else:
            raise ValueError(f"lambda_ridge must have length 1 or K={K}")
    else:
        lambda_diag = float(lambda_ridge) * np.ones(K)

    lambda_diag = np.asarray(lambda_diag, dtype=float)
    if not np.all(np.isfinite(lambda_diag)) or np.any(lambda_diag < 0):
        raise ValueError("lambda_ridge must be finite and non-negative")
    
    # Objective function
    def objective(w):
        reg = np.sum(lambda_diag * w**2)
        return w @ I_mat @ w - 2 * (w @ J_vec) + reg

    def objective_jacobian(w):
        return 2.0 * (I_mat @ w - J_vec + lambda_diag * w)
    
    # Constraints
    constraints = [{
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1.0,
        'jac': lambda weights: np.ones_like(weights),
    }]
    bounds = [(0.0, 1.0)] * K
    
    # The problem is convex up to numerical roundoff in the kernel matrix.  A
    # second feasible start is a cheap safeguard for rare SLSQP line-search
    # failures.  If both attempts fail, surface the actual solver diagnostics
    # instead of silently retaining stale weights.
    diagnostics = []
    starts = [w_old]
    uniform = np.ones(K) / K
    if not np.allclose(w_old, uniform):
        starts.append(uniform)
    for start in starts:
        result = minimize(
            objective,
            start,
            jac=objective_jacobian,
            method='SLSQP',
            constraints=constraints,
            bounds=bounds,
            options={'ftol': 1e-12, 'maxiter': 1000, 'disp': False},
        )
        candidate = np.asarray(result.x, dtype=float)
        feasible = (
            np.all(np.isfinite(candidate))
            and np.min(candidate) >= -1e-8
            and abs(candidate.sum() - 1.0) <= 1e-8
        )
        diagnostics.append(
            f"success={result.success}, status={result.status}, "
            f"message={result.message!s}, constraint={candidate.sum() - 1.0:.3e}"
        )
        if result.success and feasible:
            candidate = np.maximum(candidate, 0.0)
            candidate /= candidate.sum()
            return candidate

    raise RuntimeError(
        "Local-weight SLSQP optimization failed from all feasible starts: "
        + " | ".join(diagnostics)
    )


# ==============================================================================
# Gradient-Based Parameter Update
# ==============================================================================

def _validate_window_bandwidths(gamma, n_windows):
    values = np.asarray(gamma, dtype=float).reshape(-1)
    if values.size == 1:
        values = np.repeat(values, n_windows)
    if values.size != n_windows:
        raise ValueError("gamma must contain one bandwidth per window")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("All window bandwidths must be positive and finite")
    return values


def _closed_form_I_torch(mu_k, cov_k, mu_l, cov_l, Gamma):
    import torch
    Sigma_sum = cov_k + cov_l + Gamma
    L = torch.linalg.cholesky(Sigma_sum)
    inverse = torch.cholesky_inverse(L)
    determinant = torch.prod(torch.diag(L)) ** 2
    difference = mu_k - mu_l
    return (torch.sqrt(torch.det(Gamma) / determinant) *
            torch.exp(-0.5 * difference @ inverse @ difference))


def _closed_form_J_torch(mu, cov, X_t, Gamma):
    import torch
    Sigma_sum = cov + Gamma
    L = torch.linalg.cholesky(Sigma_sum)
    inverse = torch.cholesky_inverse(L)
    determinant = torch.prod(torch.diag(L)) ** 2
    difference = X_t - mu
    exponent = -0.5 * torch.einsum("ij,jk,ik->i", difference, inverse, difference)
    return torch.sqrt(torch.det(Gamma) / determinant) * torch.exp(exponent)


def _mmd_objective_torch(X_tensors, w_tensors, means_t,
                         covariance_tensors, gamma, term3_constants=None):
    """Exact sum differentiated by ``grad_update``, with its own gamma_i."""
    import torch
    bandwidths = _validate_window_bandwidths(gamma, len(X_tensors))
    K, d = means_t.shape
    eye = torch.eye(d, dtype=means_t.dtype, device=means_t.device)
    total = means_t.new_zeros(())
    for index, (X_i, w_i, gamma_i) in enumerate(
            zip(X_tensors, w_tensors, bandwidths)):
        Gamma_i = float(gamma_i) ** 2 * eye
        I_matrix = torch.stack([
            torch.stack([
                _closed_form_I_torch(means_t[k], covariance_tensors[k],
                                     means_t[l], covariance_tensors[l], Gamma_i)
                for l in range(K)
            ]) for k in range(K)
        ])
        term1 = w_i @ I_matrix @ w_i
        J_sums = torch.stack([
            _closed_form_J_torch(means_t[k], covariance_tensors[k], X_i, Gamma_i).sum()
            for k in range(K)
        ])
        term2 = 2.0 / len(X_i) * (w_i @ J_sums)
        if term3_constants is None:
            term3 = _empirical_kernel_mean_torch(X_i, gamma_i)
        else:
            term3 = term3_constants[index]
        total = total + term1 - term2 + term3
    return total


def _empirical_kernel_mean_torch(X, gamma, block_size=512):
    """Torch counterpart of ``compute_empirical_kernel_mean``."""
    import torch
    squared_norms = (X**2).sum(dim=1)
    total = X.new_zeros(())
    for start in range(0, len(X), block_size):
        block = X[start:start + block_size]
        squared = ((block**2).sum(dim=1)[:, None] + squared_norms[None, :]
                   - 2.0 * block @ X.T)
        total = total + torch.exp(
            -0.5 * torch.clamp(squared, min=0.0) / float(gamma)**2
        ).sum()
    return total / (len(X) * len(X))

def grad_update(X: List[np.ndarray], w: List[np.ndarray], 
                means: np.ndarray, covs: np.ndarray,
                gamma: Union[float, List[float]], grad_steps: int, lr: float,
                device, eps_val: float = 1e-2,
                freeze_means: bool = False,
                freeze_covariances: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update global means and covariances via gradient descent.
    
    Uses PyTorch for automatic differentiation.
    
    Args:
        X: List of (n_i, d) data arrays (one per time instant)
        w: List of (K,) weight arrays
        means: (K, d) current means
        covs: (K, d, d) current covariances
        gamma: One kernel bandwidth for every time window
        grad_steps: Number of gradient steps
        lr: Learning rate
        device: PyTorch device
        eps_val: Regularization for covariance matrices
        freeze_means: Keep the supplied means exactly fixed
        freeze_covariances: Keep the supplied covariances exactly fixed
        
    Returns:
        (new_means, new_covs)
    """
    import torch

    d = X[0].shape[1]
    K = len(means)

    if not np.isfinite(eps_val) or eps_val < 0:
        raise ValueError("eps_val must be finite and non-negative")

    bandwidths = _validate_window_bandwidths(gamma, len(X))
    
    if grad_steps < 0:
        raise ValueError("grad_steps must be non-negative")
    if not np.isfinite(lr) or lr < 0:
        raise ValueError("lr must be finite and non-negative")

    # Keep the gradient-based geometry update consistent with the FP32 NODE.
    # The separate SciPy SLSQP solve for simplex weights remains float64.
    dtype = torch.float32

    # Initialize mean parameters.  Partial freezing must happen inside the
    # optimizer; optimizing a parameter and discarding it later changes the
    # conditional update of the parameter that is retained.
    means_t = torch.tensor(
        means, dtype=dtype, device=device, requires_grad=not freeze_means
    )
    
    # Initialize Cholesky factors for covariances (ensures positive definiteness)
    A_list = []
    fixed_covariance_tensors = []
    identity = np.eye(d)
    # The model covariance is A A^T + eps_val I.  Initialize A from the same
    # parameterization so a zero learning rate leaves the input covariance
    # unchanged (unless it genuinely lies below the requested eigenvalue floor).
    eigenvalue_floor = float(eps_val) + 1e-6
    for k in range(K):
        cov_k = np.asarray(covs[k], dtype=float)
        cov_k = 0.5 * (cov_k + cov_k.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_k)
        if not np.all(np.isfinite(eigenvalues)):
            raise ValueError("Covariances must be finite")
        if freeze_covariances:
            if np.min(eigenvalues) <= 0:
                raise ValueError("Fixed covariances must be positive definite")
            fixed_covariance_tensors.append(
                torch.tensor(cov_k, dtype=dtype, device=device)
            )
        else:
            eigenvalues = np.maximum(eigenvalues, eigenvalue_floor)
            cov_k = (eigenvectors * eigenvalues) @ eigenvectors.T
            L = cholesky(cov_k - float(eps_val) * identity)

            A_k = torch.tensor(L, dtype=dtype, device=device, requires_grad=True)
            A_list.append(A_k)

    def covariance_tensors():
        if freeze_covariances:
            return fixed_covariance_tensors
        eye = torch.eye(d, dtype=dtype, device=device)
        return [A @ A.T + eps_val * eye for A in A_list]
    
    # Optimizer
    params = ([] if freeze_means else [means_t])
    if not freeze_covariances:
        params.extend(A_list)
    if not params:
        return (
            np.asarray(means, dtype=np.float32).copy(),
            np.asarray(covs, dtype=np.float32).copy(),
        )
    optimizer = torch.optim.Adam(params, lr=lr)
    
    X_tensors = [torch.tensor(X_i, dtype=dtype, device=device) for X_i in X]
    w_tensors = [torch.tensor(w_i, dtype=dtype, device=device) for w_i in w]
    term3_constants = []
    with torch.no_grad():
        for X_i, gamma_i in zip(X_tensors, bandwidths):
            term3_constants.append(_empirical_kernel_mean_torch(X_i, gamma_i))

    best_loss = np.inf
    best_means = means_t.detach().clone()
    best_factors = [A.detach().clone() for A in A_list]

    def retain_if_best(loss):
        nonlocal best_loss, best_means, best_factors
        value = float(loss.detach().cpu())
        if np.isfinite(value) and value < best_loss:
            best_loss = value
            best_means = means_t.detach().clone()
            best_factors = [A.detach().clone() for A in A_list]
        return np.isfinite(value)

    # Retain the unmodified input as a valid candidate.  This makes an inner
    # Adam call non-worsening even when a large learning rate overshoots.
    with torch.no_grad():
        initial_loss = _mmd_objective_torch(
            X_tensors, w_tensors, means_t, covariance_tensors(),
            bandwidths, term3_constants=term3_constants
        )
        retain_if_best(initial_loss)

    for step in range(grad_steps):
        optimizer.zero_grad()
        total_loss = _mmd_objective_torch(
            X_tensors, w_tensors, means_t, covariance_tensors(),
            bandwidths, term3_constants=term3_constants
        )
        if not retain_if_best(total_loss):
            break
        total_loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = _mmd_objective_torch(
            X_tensors, w_tensors, means_t, covariance_tensors(),
            bandwidths, term3_constants=term3_constants
        )
        retain_if_best(final_loss)
    
    # Extract the best inner state, not merely the last Adam iterate.
    new_means = (np.asarray(means, dtype=np.float32).copy() if freeze_means
                 else best_means.cpu().numpy())
    if freeze_covariances:
        new_covs = np.asarray(covs, dtype=np.float32).copy()
    else:
        new_covs = np.array([
            (factor.cpu().numpy() @ factor.cpu().numpy().T
             + np.float32(eps_val) * np.eye(d, dtype=np.float32))
            for factor in best_factors
        ], dtype=np.float32)
    
    return new_means, new_covs
