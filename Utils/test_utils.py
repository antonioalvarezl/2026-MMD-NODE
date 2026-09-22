import numpy as np
from scipy.stats import multivariate_normal

def compute_integration_bounds(mus_gt, vars_gt):
    """
    Compute integration bounds for a hyperrectangle using all ground truth parameters.

    For each coordinate i:
        lower bound = (min over all clusters and time of m_i) - 3 * (max over time of sqrt(var))
        upper bound = (max over all clusters and time of m_i) + 3 * (max over time of sqrt(var))

    Parameters:
        mus_gt: List (length t_steps) of ground truth means. Each element is a list of length n_clusters,
                where each cluster is represented as a list of d numbers.
        vars_gt: List (length t_steps) of scalar variances.

    Returns:
        bounds: List of (lower, upper) tuples for each dimension.
    """
    # Flatten ground truth means over time: convert each time step's list to a numpy array, then stack.
    all_means = np.vstack([np.atleast_2d(time_mus) for time_mus in mus_gt])  # Shape: (t_steps*n_clusters, d)
    # Use the maximum standard deviation among all time steps.
    max_std = np.sqrt(max(vars_gt))
    d = all_means.shape[1]
    bounds = []
    for i in range(d):
        lower = np.min(all_means[:, i]) - 3 * max_std
        upper = np.max(all_means[:, i]) + 3 * max_std
        bounds.append((lower, upper))
    return bounds

def sample_uniform(bounds, n):
    """
    Uniformly sample n points from within the hyper-rectangle defined by bounds.
    
    Parameters:
        bounds: List of (lower, upper) tuples for each dimension.
        n: Number of sample points.
    
    Returns:
        samples: Array of shape (n, d) with uniformly sampled points.
    """
    seed = 42
    np.random.seed(seed)
    d = len(bounds)
    samples = np.empty((n, d))
    for j, (lower, upper) in enumerate(bounds):
        samples[:, j] = np.random.uniform(lower, upper, size=n)
    return samples

def compute_gmm_density_mc(points, weights, mus, covs):
    """
    Evaluate the GMM density at given sample points using Monte Carlo integration.
    
    All inputs are standardized:
      - 'points' is reshaped to (n, d),
      - each mean is converted to a 1D array of length d,
      - each covariance is converted to a (d, d) matrix if it isn't already.
    
    Parameters:
        points: Evaluation points; if 1D, reshaped to (n, d).
        weights: List/array of mixture weights (length K).
        mus: List/array of mixture means (each convertible to a length-d vector); expected shape (K, d).
        covs: List/array of covariances. For 1D, a scalar or 1D array is converted to a (d, d) matrix;
              for d>1, they are expected to have shape (d,d) or to be convertible.
    
    Returns:
        density: 1D array with density evaluated at each sample point.
    """
    points = np.asarray(points)
    if points.ndim == 1:
        # Assume points comes as (n,) and reshape to (n, d)
        d = len(np.atleast_1d(mus[0]))
        points = points.reshape(-1, d)
    else:
        d = points.shape[1]
    
    density = np.zeros(points.shape[0])
    for w, mu, cov in zip(weights, mus, covs):
        mu = np.asarray(mu).flatten()
        # Convert covariance to (d,d) matrix.
        if np.isscalar(cov):
            cov_matrix = np.eye(d) * cov
        else:
            cov = np.asarray(cov)
            if cov.ndim == 1:
                cov_matrix = np.eye(d) * cov[0]
            elif cov.ndim == 2:
                if cov.shape != (d, d):
                    if cov.size == d*d:
                        cov_matrix = cov.reshape(d, d)
                    else:
                        raise ValueError(f"Covariance shape {cov.shape} does not match expected {(d, d)}.")
                else:
                    cov_matrix = cov
            else:
                raise ValueError("Covariance array has too many dimensions.")
        density += w * multivariate_normal.pdf(points, mean=mu, cov=cov_matrix)
    return density


def compute_gt_density_mc(points, gt_means, gt_var, d):
    """
    Evaluate the ground truth density at given sample points.
    
    The ground truth density is computed as the average over the densities
    of all components, each defined by the same scalar variance.

    Parameters:
        points: Array-like of evaluation points. If 1D, reshaped to (n, d).
        gt_means: List or array of ground truth means for the current time instant,
                  expected shape (K, d) (for the current time instant).
        gt_var: Scalar variance for the ground truth.
        d: Data dimension.
    
    Returns:
        density: 1D array with the averaged density evaluated at the sample points.
    """
    points = np.asarray(points)
    if points.ndim == 1:
        points = points.reshape(-1, d)
    n_samples = points.shape[0]
    
    # Ensure ground truth means is a 2D array.
    gt_means = np.atleast_2d(gt_means)
    n_components = gt_means.shape[0]
    
    # Build covariance matrix from scalar variance.
    cov_matrix = gt_var * np.eye(d)
    
    # Initialize density array
    density = np.zeros(n_samples)
    
    # Add contribution from each component
    for mu in gt_means:
        density += multivariate_normal.pdf(points, mean=np.asarray(mu).flatten(), cov=cov_matrix)
    
    # Normalize by number of components (for mixture model)
    density = density / n_components
    
    return density


