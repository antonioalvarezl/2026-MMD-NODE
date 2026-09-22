import numpy as np
from scipy.stats import multivariate_normal, norm

# Optimized Gaussian Mixture Generator using np.random.default_rng and list comprehensions.
def generate_gmm_data(d=2, n_clusters=3, points_per_cluster=300, means=None, var=1, seed=42):
    """
    Generate a dataset from a Gaussian mixture.
    
    Parameters:
      d (int): Data dimension.
      n_clusters (int): Number of clusters.
      points_per_cluster (int): Number of points to generate per cluster.
      means: List of cluster centers. Each element is a list of d coordinates.
             If None, cluster centers are generated using a linspace between -5 and 5,
             repeated for each coordinate.
      var (float): Scalar variance used for all clusters.
      seed (int): Random seed.
      
    Returns:
      data: A numpy array of shape (n_clusters * points_per_cluster, d).
    """
    rng = np.random.default_rng(seed)
    # Generate default means if not provided.
    if means is None:
        means_scalar = np.linspace(-5, 5, n_clusters)
        means = [[val] * d for val in means_scalar]
    
    data = []
    # Ensure points_per_cluster is a list; if it's a scalar, replicate it.
    if not isinstance(points_per_cluster, (list, tuple, np.ndarray)):
        points_per_cluster = [points_per_cluster] * n_clusters
    
    for i in range(n_clusters):
        pts = points_per_cluster[i]
        if d == 1:
            # For 1D, convert the cluster center to a scalar.
            mu_scalar = np.array(means[i]).item()
            cluster = rng.normal(loc=mu_scalar, scale=np.sqrt(var), size=(pts, 1))
        else:
            # For higher dimensions, use a diagonal covariance matrix.
            cov = var * np.eye(d)
            cluster = rng.multivariate_normal(mean=means[i], cov=cov, size=pts)
        data.append(cluster)
    
    return np.vstack(data)

# Optimized Poisson data generation using default_rng.
def generate_poisson_data(d=1, n_points=1000, lam=5, seed=42):
    """
    Generate a dataset where each coordinate is sampled from a Poisson distribution with parameter lam.
    For d > 1, returns an array of shape (n_points, d).
    """
    rng = np.random.default_rng(seed)
    return rng.poisson(lam, size=(n_points, d)).astype(float)

# Optimized Student's t data generation.
def generate_student_t_data(d=1, n_points=1000, df=5, seed=42, loc=None, scale=None):
    """
    Generate a dataset from a multivariate Student's t-distribution.

    Parameters:
      d (int): Dimensionality.
      n_points (int): Number of samples.
      df (float): Degrees of freedom.
      seed (int): Random seed.
      loc (np.ndarray): Mean vector (default zero vector).
      scale (np.ndarray): Scale (covariance) matrix (default identity).

    Returns:
      np.ndarray: Samples of shape (n_points, d).
    """
    rng = np.random.default_rng(seed)
    if loc is None:
        loc = np.zeros(d)
    if scale is None:
        scale = np.eye(d)
    # Use Cholesky factorization of the scale matrix.
    L = np.linalg.cholesky(scale)
    g = rng.standard_normal((n_points, d))
    z = g @ L.T
    chi2_samples = rng.chisquare(df, size=n_points)
    factors = np.sqrt(df / chi2_samples)
    return loc + z * factors[:, None]

# Optimized uniform data generation.
def generate_uniform_data(d=2, n_points=1000, low=0.0, high=1.0, seed=42):
    """
    Generate a dataset uniformly distributed in [low, high]^d.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(n_points, d))

def generate_data(distribution, d, params, seed=None):
    """
    Generate synthetic data based on specified distribution.
    
    Args:
    the total number of points equals params["n_points"]. Points are allocated 
    among clusters (which may receive an unequal number of points if needed).
    
    Args:
        distribution (str): Distribution type ('gaussian', 'student_t', 'uniform', 'poisson').
        d (int): Data dimension.
        params (dict): Parameters for data generation.
            For 'gaussian', expected keys include:
              - "means": List of cluster means; each element is a list of d numbers.
              - "vars": Scalar variance.
              - "n_points": Total number of data points.
              - "seed": (Optional) Seed for random number generator.
        seed (int, optional): Random seed. If not provided, uses params["seed"].
    
    Returns:
        ndarray: Generated synthetic data.
    """
    if seed is None:
        seed = params["seed"]
        
    if distribution == 'gaussian':
        # Extract parameters for the Gaussian mixture
        means = params["means"]    # Each element is a list of d numbers
        var = params["vars"]       # Scalar variance
        n_clusters = len(means)
        total_points = params["n_points"]
        
        # Initialize points per cluster as a list
        base_points = total_points // n_clusters
        points_per_cluster = [base_points] * n_clusters
        
        # Distribute remainder points among first 'remainder' clusters
        remainder = total_points % n_clusters
        for i in range(remainder):
            points_per_cluster[i] += 1
        
        # Generate data using generate_gmm_data
        return generate_gmm_data(
            d=d,
            n_clusters=n_clusters,
            points_per_cluster=points_per_cluster,
            means=means,
            var=var,
            seed=seed
        )
        
    elif distribution == 'student_t':
        return generate_student_t_data(d, n_points=params['n_points'], df=params['df'], seed=seed)
    elif distribution == 'uniform':
        return generate_uniform_data(d, n_points=params['n_points'], low=0.0, high=1.0, seed=seed)
    elif distribution == 'poisson':
        return generate_poisson_data(d, n_points=params['n_points'], lam=5, seed=seed)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")



def compute_gmm_density(x_grid, weights, mus, covs, M=False):
    """
    Compute the density of a Gaussian Mixture Model on the given grid.
    
    Parameters:
        x_grid: For 1D, a 1D numpy array. For d>=2, a tuple of 1D arrays (one per dimension).
        weights: Mixture weights as an array (shape: (K,)).
        mus: Mixture means (shape: (K, d)).
        covs: Covariance matrices, shape: (K, d) for 1D or (K, d, d) for d>=2.
        M (int): Number of points to sample (used for d>2 to avoid full meshgrid).
    
    Returns:
        For d==1 or 2: a density array with shape matching the grid (or reshaped grid).
        For d>2: a tuple (density, sample_points), where density is evaluated on M sampled points.
    """

    d = np.array(mus).shape[1]
    
    if d == 1:
        density = np.zeros_like(x_grid)
        for w, mu, cov in zip(weights, mus, covs):
            density += w * norm.pdf(x_grid, loc=mu[0], scale=np.sqrt(cov[0]))
        return density
    elif d == 2:
        if isinstance(x_grid, tuple):
            X_grid, Y_grid = x_grid
            grid_points = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
        else:
            grid_points = np.array(x_grid).reshape(-1, d)
        density = np.zeros(grid_points.shape[0])
        for w, mu, cov in zip(weights, mus, covs):
            density += w * multivariate_normal.pdf(grid_points, mean=mu, cov=cov)
        return density.reshape(X_grid.shape)
    else:
        # For d > 2, sample a fixed number M of points from the hyper-rectangle defined by x_grid.
        if M:
            sample_points = np.column_stack([np.random.choice(arr, size=M, replace=True) for arr in x_grid])
            density = np.zeros(M)
        else:
            grid_points = np.column_stack([X.ravel() for X in x_grid])
            sample_points = grid_points
            density = np.zeros(grid_points.shape[0])
        for w, mu, cov in zip(weights, mus, covs):
            density += w * multivariate_normal.pdf(sample_points, mean=mu, cov=cov)
        return density, sample_points

def compute_gammaopt(X, sample_size=1000, seed=None):
    """
    Compute the optimal Gaussian kernel bandwidth gamma based on 
    the median of pairwise distances between points in X.
    
    Args:
        X (ndarray): Data matrix of shape (N, d) where N is the number of samples 
                     and d is the data dimension.
        sample_size (int): Number of samples to use for approximating the median
                           pairwise distance when N is large.
    
    Returns:
        float: The optimal gamma (bandwidth) value.
    """
    from scipy.spatial.distance import pdist

    X = np.atleast_2d(X)
    
    # Handle edge cases where we can't compute distances
    if X.shape[0] <= 1:
        return 1.0  # Default value when insufficient data points
    
    n = X.shape[0]
    
    # For large datasets, use sampling to approximate the median distance.
    if n > 5000:
        rng = np.random.default_rng(seed)
        indices = rng.choice(n, size=min(sample_size, n), replace=False)
        sample = X[indices]
        dists = pdist(sample)
    else:
        # Compute exact pairwise distances for smaller datasets.
        dists = pdist(X)
    
    # Compute the median of distances, ensuring a positive value.
    positive = dists[np.isfinite(dists) & (dists > 0)]
    gamma_opt = max(np.median(positive), 1e-6) if len(positive) else 1.0
    return gamma_opt

def get_distrparams(params, T, t_steps):
    """
    Define and get distribution parameters that evolve over time.
    
    Args:
        params (dict): Dictionary containing:
            - n_clusters (int): Number of clusters.
            - d (int): Dimension of the space.
            
    Returns:
        tuple: (means_list, var_list) where:
            - means_list is a list of length t_steps. Each element is a list of length n_clusters,
              and each cluster is represented as a list of d coordinates.
            - var_list is a list of scalars (one for each time instant).
    """
    # Define time values
    t_vals = np.linspace(0, T, t_steps) if T != 0 else np.array([0])
    d = params["d"]
    n_clusters = params.get("n_clusters", 3)
    
    # Initialize lists
    means_list = []
    var_list = []
    
    # For each time step, create the means for all clusters
    for t in t_vals:
        # Calculate positions for each cluster at time t
        if t <= 0.5:
            m1 = -2 + 20*t
            m2 = 16*t
            m3 = 5 + 6*t
        else:
            m1 = m2 = m3 = 8
        
        # Create means for this time step
        means_at_t = []
        if n_clusters >= 1:
            means_at_t.append([m1] * d)
        if n_clusters >= 2:
            means_at_t.append([m2] * d)
        if n_clusters >= 3:
            means_at_t.append([m3] * d)
            
        if n_clusters > 3:
            raise ValueError("n_clusters must be 1, 2, or 3.")
        
        means_list.append(means_at_t)
        var_list.append(1 + t)
    
    return means_list, var_list
