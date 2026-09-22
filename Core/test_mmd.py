import numpy as np
import os
from Utils.test_utils import compute_integration_bounds, sample_uniform, compute_gmm_density_mc, compute_gt_density_mc
from Visualization.plots import plot_L2err, plot_L2MMDerr
from Utils.MMD_utils import compute_mmd_closed_form
from scipy.stats import gaussian_kde
from tqdm import tqdm
from data import compute_gammaopt
from Visualization.log import log_error


def test(ws, mus, covs, X_test, gamma):
    """Compute MMD error on test set."""
    test_error = []
    for i, X in enumerate(X_test):
        test_error.append(compute_mmd_closed_form(X, ws[i], mus, covs, gamma[i]))
    return test_error


def _compute_L2_statistics_and_plot(L2errs, L2errs_ts, t_grid, t_i, folder, plot_name, plot=True):
    """Compute L2 statistics and generate plots when plotting is enabled."""
    if plot:
        plot_path = os.path.join(folder, plot_name)
        plot_L2err(L2errs, L2errs_ts, t_grid, t_i, plot_path)
    
    avgL2_ts, stdL2_ts = np.mean(L2errs_ts), np.std(L2errs_ts)
    avgL2, stdL2 = np.mean(L2errs), np.std(L2errs)
    
    return avgL2, stdL2, avgL2_ts, stdL2_ts


def _compute_mc_L2_error(sample_points, weights, mus, covs, gt_means, gt_var, volume, n_MC, d):
    """Compute L2 error using Monte Carlo integration."""
    gt_density = compute_gt_density_mc(sample_points, gt_means, gt_var, d)
    gmm_density = compute_gmm_density_mc(sample_points, weights, mus, covs)
    error = gt_density - gmm_density
    L2_err = np.sqrt(volume / n_MC * np.sum(error**2))
    return L2_err, error


def _compute_adaptive_bounds(gt_means, gt_var, expansion_factor=3.0):
    """Compute adaptive integration bounds for specific time step."""
    gt_means = np.atleast_2d(gt_means)
    std = np.sqrt(gt_var)
    d = gt_means.shape[1]
    
    bounds = []
    for i in range(d):
        lower = np.min(gt_means[:, i]) - expansion_factor * std
        upper = np.max(gt_means[:, i]) + expansion_factor * std
        bounds.append((lower, upper))
    
    return bounds


def _importance_sample(weights, mus, covs, gt_means, gt_var, n_samples, d):
    """Generate samples from mixture of model GMM and ground truth for importance sampling."""
    weights = np.asarray(weights)
    gt_means = np.atleast_2d(gt_means)
    
    K_model = len(weights)
    K_gt = len(gt_means)
    
    # Create combined mixture (50% model, 50% ground truth)
    proposal_weights = np.zeros(K_model + K_gt)
    proposal_weights[:K_model] = weights * 0.5
    proposal_weights[K_model:] = np.ones(K_gt) * 0.5 / K_gt
    proposal_weights = proposal_weights / np.sum(proposal_weights)
    
    # Combine parameters
    proposal_mus = list(mus) + [np.array(mu) for mu in gt_means]
    proposal_covs = list(covs) + [np.eye(d) * gt_var for _ in range(K_gt)]
    
    # Sample component indices and generate samples
    component_indices = np.random.choice(len(proposal_weights), size=n_samples, p=proposal_weights)
    samples = np.zeros((n_samples, d))
    
    for k in range(len(proposal_weights)):
        mask = (component_indices == k)
        count = np.sum(mask)
        
        if count > 0:
            mu = np.asarray(proposal_mus[k]).flatten()
            cov = proposal_covs[k]
            if np.isscalar(cov):
                cov = np.eye(d) * cov
            samples[mask] = np.random.multivariate_normal(mean=mu, cov=cov, size=count)
    
    # Calculate proposal density
    proposal_density = compute_gmm_density_mc(samples, proposal_weights, proposal_mus, proposal_covs)
    
    return samples, proposal_density


def _gaussian_product_integral(mu1, cov1, mu2, cov2):
    """Compute integral of product of two Gaussian PDFs."""
    d = len(mu1)
    sum_cov = cov1 + cov2
    diff_mu = mu1 - mu2
    
    det_sum_cov = np.linalg.det(sum_cov)
    quad_form = diff_mu.T @ np.linalg.solve(sum_cov, diff_mu)
    
    integral = (2 * np.pi) ** (-d / 2) * np.sqrt(1 / det_sum_cov) * np.exp(-0.5 * quad_form)
    return integral


def _compute_L2_error_analytical(weights1, means1, covs1, weights2, means2, covs2):
    """Compute L2 error between two GMMs analytically using L2² = ∫p² + ∫q² - 2∫pq."""
    weights1, weights2 = np.asarray(weights1), np.asarray(weights2)
    means1 = [np.asarray(mu).flatten() for mu in means1]
    means2 = [np.asarray(mu).flatten() for mu in means2]
    
    K1, K2 = len(weights1), len(weights2)
    d = len(means1[0])
    
    # Process covariances to ensure proper matrix format
    def process_covs(covs):
        processed = []
        for cov in covs:
            if np.isscalar(cov):
                processed.append(np.eye(d) * cov)
            else:
                cov = np.asarray(cov)
                if cov.ndim == 1:
                    processed.append(np.eye(d) * cov[0])
                elif cov.ndim == 2 and cov.shape == (d, d):
                    processed.append(cov)
                else:
                    raise ValueError(f"Invalid covariance shape: {cov.shape}")
        return processed
    
    processed_covs1 = process_covs(covs1)
    processed_covs2 = process_covs(covs2)
    
    # Calculate ∫p², ∫q², and cross term
    p_squared = sum(weights1[i] * weights1[j] * _gaussian_product_integral(
        means1[i], processed_covs1[i], means1[j], processed_covs1[j])
        for i in range(K1) for j in range(K1))
    
    q_squared = sum(weights2[i] * weights2[j] * _gaussian_product_integral(
        means2[i], processed_covs2[i], means2[j], processed_covs2[j])
        for i in range(K2) for j in range(K2))
    
    cross_term = -2 * sum(weights1[i] * weights2[j] * _gaussian_product_integral(
        means1[i], processed_covs1[i], means2[j], processed_covs2[j])
        for i in range(K1) for j in range(K2))
    
    L2_squared = max(0, p_squared + q_squared + cross_term)
    return np.sqrt(L2_squared)


def testL2(ws, ws_ts, mus, covs, mus_gt, vars_gt, mus_gt_ts, vars_gt_ts, t_grid, t_i, folder, combo):
    """
    Compute L2 errors via Monte Carlo integration using global bounds.
    
    Returns: (L2errs, L2errs_ts, avgL2, stdL2, avgL2_ts, stdL2_ts)
    """
    L2errs, L2errs_ts, history_errs = [], [], []
    n_MC, d = combo["n_MC"], combo["d"]

    # Global integration bounds and sampling
    bounds = compute_integration_bounds(mus_gt, vars_gt)
    volume = np.prod([upper - lower for lower, upper in bounds])
    sample_points = sample_uniform(bounds, n_MC)
    
    print(f"Integration volume: {volume:.2e}, MC points: {n_MC}")
    
    # Compute errors for coarse grid (MMD fitting)
    for i, weights_ts in tqdm(enumerate(ws_ts), desc="Computing L2 errors for MMD fitting", total=len(ws_ts)):
        L2_err_ts, _ = _compute_mc_L2_error(sample_points, weights_ts, mus, covs, 
                                          mus_gt_ts[i], vars_gt_ts[i], volume, n_MC, d)
        L2errs_ts.append(L2_err_ts)
    
    # Compute errors for fine grid (NODE trajectory)
    for i, w in tqdm(enumerate(ws), desc="Computing L2 errors for NODE trajectory", total=len(ws)):
        L2_err, error = _compute_mc_L2_error(sample_points, w, mus, covs, 
                                           mus_gt[i], vars_gt[i], volume, n_MC, d)
        L2errs.append(L2_err)
        history_errs.append(error)
    
    avgL2, stdL2, avgL2_ts, stdL2_ts = _compute_L2_statistics_and_plot(
        L2errs, L2errs_ts, t_grid, t_i, folder, "L2_errors.png", plot=combo.get("plots", True))
    
    log_error("synthetic", avgL2_ts, stdL2_ts, avgL2, stdL2, 0, 0, 0, 0)
    
    return (L2errs, L2errs_ts, avgL2, stdL2, avgL2_ts, stdL2_ts)


def testL2_adaptive(ws, ws_ts, mus, covs, mus_gt, vars_gt, mus_gt_ts, vars_gt_ts, t_grid, t_i, folder, combo):
    """
    Compute L2 errors via Monte Carlo integration with adaptive bounds per time step.
    
    Returns: (L2errs, L2errs_ts, avgL2, stdL2, avgL2_ts, stdL2_ts)
    """
    L2errs, L2errs_ts, history_errs = [], [], []
    n_MC, d = combo["n_MC"], combo["d"]
    
    # Compute errors for coarse grid with adaptive bounds
    for i, weights_ts in tqdm(enumerate(ws_ts), desc="Computing adaptive L2 errors for MMD fitting", total=len(ws_ts)):
        bounds_ts = _compute_adaptive_bounds(mus_gt_ts[i], vars_gt_ts[i])
        volume_ts = np.prod([upper - lower for lower, upper in bounds_ts])
        sample_points_ts = sample_uniform(bounds_ts, n_MC)
        
        L2_err_ts, _ = _compute_mc_L2_error(sample_points_ts, weights_ts, mus, covs,
                                          mus_gt_ts[i], vars_gt_ts[i], volume_ts, n_MC, d)
        L2errs_ts.append(L2_err_ts)
    
    # Compute errors for fine grid with adaptive bounds
    for i, w in tqdm(enumerate(ws), desc="Computing adaptive L2 errors for NODE trajectory", total=len(ws)):
        bounds = _compute_adaptive_bounds(mus_gt[i], vars_gt[i])
        volume = np.prod([upper - lower for lower, upper in bounds])
        sample_points = sample_uniform(bounds, n_MC)
        
        L2_err, error = _compute_mc_L2_error(sample_points, w, mus, covs,
                                           mus_gt[i], vars_gt[i], volume, n_MC, d)
        L2errs.append(L2_err)
        history_errs.append(error)
    
    avgL2, stdL2, avgL2_ts, stdL2_ts = _compute_L2_statistics_and_plot(
        L2errs, L2errs_ts, t_grid, t_i, folder, "L2_errors_adaptive.png", plot=combo.get("plots", True))
    
    log_error("synthetic_adaptive", avgL2_ts, stdL2_ts, avgL2, stdL2, 0, 0, 0, 0)
    
    return (L2errs, L2errs_ts, avgL2, stdL2, avgL2_ts, stdL2_ts)


def testL2_importance(ws, ws_ts, mus, covs, mus_gt, vars_gt, mus_gt_ts, vars_gt_ts, t_grid, t_i, folder, combo):
    """
    Compute L2 errors via Monte Carlo integration with importance sampling.
    
    Returns: (L2errs, L2errs_ts, avgL2, stdL2, avgL2_ts, stdL2_ts)
    """
    L2errs, L2errs_ts, history_errs = [], [], []
    n_MC, d = combo["n_MC"], combo["d"]
    
    # Compute errors for coarse grid with importance sampling
    for i, weights_ts in tqdm(enumerate(ws_ts), desc="Computing importance L2 errors for MMD fitting", total=len(ws_ts)):
        bounds_ts = _compute_adaptive_bounds(mus_gt_ts[i], vars_gt_ts[i])
        volume_ts = np.prod([upper - lower for lower, upper in bounds_ts])
        
        proposal_samples, proposal_density = _importance_sample(
            weights_ts, mus, covs, mus_gt_ts[i], vars_gt_ts[i], n_MC, d)
        
        gt_density_ts = compute_gt_density_mc(proposal_samples, mus_gt_ts[i], vars_gt_ts[i], d)
        gmm_density_ts = compute_gmm_density_mc(proposal_samples, weights_ts, mus, covs)
        
        error_ts = gt_density_ts - gmm_density_ts
        importance_weights_ts = 1.0 / np.maximum(proposal_density, 1e-12)
        importance_weights_ts = importance_weights_ts / np.sum(importance_weights_ts) * n_MC
        
        L2_err_ts = np.sqrt(volume_ts / n_MC * np.sum(error_ts**2 * importance_weights_ts))
        L2errs_ts.append(L2_err_ts)
    
    # Compute errors for fine grid with importance sampling
    for i, w in tqdm(enumerate(ws), desc="Computing importance L2 errors for NODE trajectory", total=len(ws)):
        bounds = _compute_adaptive_bounds(mus_gt[i], vars_gt[i])
        volume = np.prod([upper - lower for lower, upper in bounds])
        
        proposal_samples, proposal_density = _importance_sample(
            w, mus, covs, mus_gt[i], vars_gt[i], n_MC, d)
        
        gt_density = compute_gt_density_mc(proposal_samples, mus_gt[i], vars_gt[i], d)
        gmm_density = compute_gmm_density_mc(proposal_samples, w, mus, covs)
        
        error = gt_density - gmm_density
        importance_weights = 1.0 / np.maximum(proposal_density, 1e-12)
        importance_weights = importance_weights / np.sum(importance_weights) * n_MC
        
        history_errs.append(error * np.sqrt(importance_weights))
        
        L2_err = np.sqrt(volume / n_MC * np.sum(error**2 * importance_weights))
        L2errs.append(L2_err)
    
    avgL2, stdL2, avgL2_ts, stdL2_ts = _compute_L2_statistics_and_plot(
        L2errs, L2errs_ts, t_grid, t_i, folder, "L2_errors_importance.png", plot=combo.get("plots", True))
    
    log_error("synthetic_importance", avgL2_ts, stdL2_ts, avgL2, stdL2, 0, 0, 0, 0)
    
    return (L2errs, L2errs_ts, avgL2, stdL2, avgL2_ts, stdL2_ts)

def testL2_analytical(ws, ws_ts, mus, covs, mus_gt, vars_gt, mus_gt_ts, vars_gt_ts, t_grid, t_i, folder, combo):
    """
    Compute L2 errors analytically using closed-form expressions.
    
    Returns: (L2errs, L2errs_ts, avgL2, stdL2, avgL2_ts, stdL2_ts, None, None)
    """
    L2errs, L2errs_ts = [], []
    d = combo["d"]
    
    # Compute errors for coarse grid analytically
    for i, weights_ts in tqdm(enumerate(ws_ts), desc="Computing analytical L2 errors for MMD fitting", total=len(ws_ts)):
        gt_weights_ts = np.ones(len(mus_gt_ts[i])) / len(mus_gt_ts[i])
        gt_covs_ts = [np.eye(d) * vars_gt_ts[i]] * len(mus_gt_ts[i])
        
        L2_err_ts = _compute_L2_error_analytical(weights_ts, mus, covs, 
                                               gt_weights_ts, mus_gt_ts[i], gt_covs_ts)
        L2errs_ts.append(L2_err_ts)
    
    # Compute errors for fine grid analytically
    for i, w in tqdm(enumerate(ws), desc="Computing analytical L2 errors for NODE trajectory", total=len(ws)):
        gt_weights = np.ones(len(mus_gt[i])) / len(mus_gt[i])
        gt_covs = [np.eye(d) * vars_gt[i]] * len(mus_gt[i])
        
        L2_err = _compute_L2_error_analytical(w, mus, covs, gt_weights, mus_gt[i], gt_covs)
        L2errs.append(L2_err)
    
    avgL2, stdL2, avgL2_ts, stdL2_ts = _compute_L2_statistics_and_plot(
        L2errs, L2errs_ts, t_grid, t_i, folder, "L2_errors_analytical.png", plot=combo.get("plots", True))
    
    log_error("synthetic_analytical", avgL2_ts, stdL2_ts, avgL2, stdL2, 0, 0, 0, 0)
    
    return (L2errs, L2errs_ts, avgL2, stdL2, avgL2_ts, stdL2_ts)


def testL2MMDemp(ws, ws_ts, mus, covs, data, t_grid, t_i, folder, combo):
    """
    Compute empirical L2 and MMD errors for real data using KDE density estimation.
    
    Returns: (L2_NODE, MMD_NODE, L2_FIT, MMD_FIT, avgL2, stdL2, avgMMD, stdMMD, 
              avgL2_ts, stdL2_ts, avgMMD_ts, stdMMD_ts)
    """
    os.makedirs(folder, exist_ok=True)
    
    n_MC = combo["n_MC"]
    
    # Global bounding box for integration
    all_pts = np.vstack(data)
    mu_all = all_pts.mean(0)
    std_all = all_pts.std(0) + 1e-12
    bounds = [(mu_all[j] - 4 * std_all[j], mu_all[j] + 4 * std_all[j])
              for j in range(combo["d"])]
    volume = np.prod([u - l for l, u in bounds])
    
    # Map fine grid times to coarse grid indices
    coarse_idx_for_fine = np.searchsorted(t_i, t_grid, side="right") - 1
    coarse_idx_for_fine = np.clip(coarse_idx_for_fine, 0, len(t_i) - 1)
    
    sample_points = sample_uniform(bounds, n_MC)
    
    L2_NODE, L2_FIT = [], []
    MMD_NODE, MMD_FIT = [], []
    
    print(f"Integration volume: {volume:.2e}, MC points: {n_MC}")
    
    # Process coarse grid (MMD fitting)
    for k, w_fit in tqdm(enumerate(ws_ts), desc="Computing L2 and MMD errors on coarse grid", total=len(ws_ts)):
        X_tk = data[k]
        gamma_tk = compute_gammaopt(X_tk)
        bw_kde = gamma_tk / all_pts.std()
        
        kde = gaussian_kde(X_tk.T, bw_method=bw_kde)
        emp_dens = kde(sample_points.T)
        gmm_dens = compute_gmm_density_mc(sample_points, w_fit, mus, covs)
        
        diff = emp_dens - gmm_dens
        L2_FIT.append(np.sqrt(volume / n_MC * np.sum(diff ** 2)))
        MMD_FIT.append(np.sqrt(compute_mmd_closed_form(X_tk, w_fit, mus, covs, gamma_tk)))
    
    # Process fine grid (NODE trajectory)
    for j, (w, k_coarse) in tqdm(enumerate(zip(ws, coarse_idx_for_fine)), 
                                desc="Computing L2 and MMD errors on fine grid", total=len(ws)):
        X_t = data[k_coarse]
        gamma_t = compute_gammaopt(X_t)
        bw_kde = gamma_t / all_pts.std()
        
        kde = gaussian_kde(X_t.T, bw_method=bw_kde)
        emp_dens = kde(sample_points.T)
        gmm_dens = compute_gmm_density_mc(sample_points, w, mus, covs)
        
        diff = emp_dens - gmm_dens
        L2_NODE.append(np.sqrt(volume / n_MC * np.sum(diff ** 2)))
        MMD_NODE.append(np.sqrt(compute_mmd_closed_form(X_t, w, mus, covs, gamma_t)))
    
    # Compute statistics
    avgL2, stdL2 = np.mean(L2_NODE), np.std(L2_NODE)
    avgMMD, stdMMD = np.mean(MMD_NODE), np.std(MMD_NODE)
    avgL2_ts, stdL2_ts = np.mean(L2_FIT), np.std(L2_FIT)
    avgMMD_ts, stdMMD_ts = np.mean(MMD_FIT), np.std(MMD_FIT)
    
    # Generate plots
    plot_L2MMDerr(L2_NODE, MMD_NODE, L2_FIT, MMD_FIT, t_grid, t_i,
                  os.path.join(folder, "L2_MMD_empirical.png"))
    
    log_error("load", avgL2_ts, stdL2_ts, avgL2, stdL2, 
              avgMMD_ts, stdMMD_ts, avgMMD, stdMMD)

    return (L2_NODE, MMD_NODE, L2_FIT, MMD_FIT,
            avgL2, stdL2, avgMMD, stdMMD, avgL2_ts, stdL2_ts, avgMMD_ts, stdMMD_ts)

