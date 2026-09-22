
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import imageio
from tqdm import tqdm

# =====================================================================
# Funciones para visualización GIF
# =====================================================================

def gif_error(history_weights, history_means, history_covariances,
              means_gt, vars_gt, t_grid=None, fps=10,
              folder="Results", filename="error_evolution.gif"):
    """
    Animated visualization of pointwise error between predicted GMM and ground truth GMM.
    
    Shows at each time step:
    - Top panel: Predicted GMM (blue), GT GMM (gray), and pointwise error (red)
    - Bottom panel: Evolution of integrated absolute error over time
    
    Parameters:
    -----------
    history_weights : list of arrays
        Predicted weights at each time step
    history_means : array (K, d) or list
        Fitted means (fixed across time)
    history_covariances : array (K, d, d) or list
        Fitted covariances (fixed across time)
    means_gt : list
        Ground truth means at each time step
    vars_gt : list
        Ground truth variances at each time step
    t_grid : array, optional
        Time grid
    fps : int
        Frames per second
    folder : str
        Output folder
    filename : str
        Output filename
    """
    os.makedirs(folder, exist_ok=True)
    full_path = os.path.join(folder, filename)
    
    if os.path.exists(full_path):
        print(f"Error GIF already exists → {full_path}")
        return
    
    n_frames = len(history_weights)
    K = len(history_weights[0])
    
    # Handle means format - could be list or array
    if isinstance(history_means, np.ndarray):
        means_arr = history_means
    else:
        means_arr = np.array(history_means[0])  # Take first if list of lists
    
    if isinstance(history_covariances, np.ndarray):
        covs_arr = history_covariances
    else:
        covs_arr = np.array(history_covariances[0])
    
    # Determine x-range from GT means
    all_gt_means = []
    for t_means in means_gt:
        for cluster_mean in t_means:
            if isinstance(cluster_mean, list):
                all_gt_means.append(cluster_mean[0])
            else:
                all_gt_means.append(cluster_mean)
    
    max_std = np.sqrt(max(vars_gt)) if vars_gt else 2.0
    x_min = min(all_gt_means) - 5 * max_std
    x_max = max(all_gt_means) + 5 * max_std
    x_grid = np.linspace(x_min, x_max, 400)
    
    # Precompute all PDFs and errors
    pred_pdfs = []
    gt_pdfs = []
    errors = []
    integrated_errors = []
    
    for f in range(n_frames):
        # Predicted GMM
        pred_pdf = np.zeros_like(x_grid)
        for k in range(K):
            if means_arr.ndim == 1:
                mu = means_arr[k]
            else:
                mu = means_arr[k, 0] if means_arr.shape[1] > 0 else means_arr[k]
            
            if covs_arr.ndim == 1:
                sigma2 = covs_arr[k]
            elif covs_arr.ndim == 2:
                sigma2 = covs_arr[k, 0]
            else:
                sigma2 = covs_arr[k, 0, 0]
            
            pred_pdf += history_weights[f][k] * norm.pdf(x_grid, loc=mu, scale=np.sqrt(sigma2))
        
        # GT GMM (equal weights)
        gt_pdf = np.zeros_like(x_grid)
        frame_idx = min(f, len(means_gt) - 1)
        n_clusters_gt = len(means_gt[frame_idx])
        weight_gt = 1.0 / n_clusters_gt
        
        for k_gt in range(n_clusters_gt):
            if isinstance(means_gt[frame_idx][k_gt], list):
                mu_gt = means_gt[frame_idx][k_gt][0]
            else:
                mu_gt = means_gt[frame_idx][k_gt]
            gt_pdf += weight_gt * norm.pdf(x_grid, mu_gt, np.sqrt(vars_gt[frame_idx]))
        
        pred_pdfs.append(pred_pdf)
        gt_pdfs.append(gt_pdf)
        
        # Pointwise error
        err = pred_pdf - gt_pdf
        errors.append(err)
        
        # Integrated absolute error
        dx = x_grid[1] - x_grid[0]
        integrated_errors.append(np.sum(np.abs(err)) * dx)
    
    # Y-limits
    max_pdf = max(max(p.max() for p in pred_pdfs), max(g.max() for g in gt_pdfs))
    max_err = max(np.abs(e).max() for e in errors)
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])
    
    frames_img = []
    t_axis = t_grid if t_grid is not None else np.arange(n_frames)
    
    for f in tqdm(range(n_frames), desc="Building Error GIF"):
        ax_top.clear()
        ax_bot.clear()
        
        # Top panel: PDFs and error
        ax_top.fill_between(x_grid, 0, gt_pdfs[f], color='lightgray', alpha=0.5, label='Ground Truth')
        ax_top.plot(x_grid, gt_pdfs[f], 'k-', lw=2)
        ax_top.fill_between(x_grid, 0, pred_pdfs[f], color='steelblue', alpha=0.3, label='Predicted')
        ax_top.plot(x_grid, pred_pdfs[f], 'b-', lw=2)
        
        # Error as filled area (positive=red, negative=blue)
        err = errors[f]
        ax_top.fill_between(x_grid, 0, err, where=(err > 0), color='red', alpha=0.4)
        ax_top.fill_between(x_grid, 0, err, where=(err < 0), color='blue', alpha=0.4)
        
        ax_top.set_xlim(x_min, x_max)
        ax_top.set_ylim(-max_err * 1.1, max_pdf * 1.1)
        ax_top.axhline(0, color='k', lw=0.5, ls='--')
        ax_top.set_ylabel('Density / Error')
        t_val = t_axis[f] if f < len(t_axis) else f
        ax_top.set_title(f't = {t_val:.2f}')
        ax_top.legend(loc='upper right', fontsize=9)
        
        # Bottom panel: integrated error over time
        ax_bot.plot(t_axis[:f+1], integrated_errors[:f+1], 'r-', lw=2)
        ax_bot.scatter([t_axis[f]], [integrated_errors[f]], s=80, c='red', zorder=5)
        ax_bot.set_xlim(t_axis[0], t_axis[-1])
        ax_bot.set_ylim(0, max(integrated_errors) * 1.2 + 0.01)
        ax_bot.set_xlabel('Time')
        ax_bot.set_ylabel('∫|error| dx')
        ax_bot.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.canvas.draw()
        
        # Handle retina displays (get actual buffer size)
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3].copy()
        frames_img.append(img)
    
    plt.close(fig)
    imageio.mimsave(full_path, frames_img, fps=fps)
    print(f"[INFO] Pointwise Error GIF saved → {full_path}")

def gif(history_weights,
            history_means,
            history_covariances,
            history_X=None,
            history_X_times=None,
            means_gt=None, 
            vars_gt=None,
            t_grid=None, 
            fps=10,
            evol_weights='path',      # 'path' | 'barplot'
            bg_mode='histogram',      # 'histogram' | 'gaussian'
            folder="Results",
            filename="gmm_evolution.gif"):
    """
    Animated visualisation of a time-evolving GMM.

    Parameters:
    -----------
    history_weights : list
        List of weight arrays for each time step
    history_means : list
        List of mean arrays for each time step
    history_covariances : list
        List of covariance arrays for each time step
    history_X : list, optional
        List of data arrays for each time step (needed if bg_mode == 'histogram')
    history_X_times : array, optional
        Observation times associated with ``history_X``. When the animation
        grid is finer, each frame uses the temporally nearest observed block.
    means_gt : list, optional
        Ground truth means (needed if bg_mode == 'gaussian')
    vars_gt : list, optional
        Ground truth variances (needed if bg_mode == 'gaussian')
    t_grid : array, optional
        Time grid for evolution plots
    fps : int, optional
        Frames per second for the GIF
    evol_weights : str, optional
        Mode for visualizing weight evolution ('path' | 'barplot')
    bg_mode : str, optional
        Background visualization mode ('histogram' | 'gaussian')
    folder : str, optional
        Folder to save the GIF
    filename : str, optional
        Filename for the GIF
    """
    # -----------------------------------------------------------------
    # 0)  Early exit & basic constants
    # -----------------------------------------------------------------
    os.makedirs(folder, exist_ok=True)
    full_path = os.path.join(folder, filename)
    if os.path.exists(full_path) and (t_grid is not None and len(t_grid) > 1e5):
        print("GIF already exists →", full_path)
        return

    n_frames = len(history_weights)
    K = len(history_weights[0])
    cmap = plt.cm.get_cmap('viridis', K)

    histogram_indices = None
    history_X_times_array = None
    if bg_mode == 'histogram':
        if history_X is None or len(history_X) == 0:
            raise ValueError("history_X is required for histogram backgrounds")
        if history_X_times is None:
            if len(history_X) != n_frames:
                raise ValueError(
                    "Unequal histogram and frame counts require history_X_times"
                )
            histogram_indices = np.arange(n_frames, dtype=int)
        else:
            if t_grid is None:
                raise ValueError("t_grid is required when history_X_times is provided")
            history_X_times_array = np.asarray(history_X_times, dtype=float)
            frame_times = np.asarray(t_grid, dtype=float)
            if (history_X_times_array.ndim != 1 or frame_times.ndim != 1 or
                    len(history_X_times_array) != len(history_X) or
                    len(frame_times) != n_frames):
                raise ValueError("Histogram times must align with data and frame times")
            if (not np.all(np.isfinite(history_X_times_array)) or
                    not np.all(np.isfinite(frame_times)) or
                    np.any(np.diff(history_X_times_array) <= 0) or
                    np.any(np.diff(frame_times) <= 0)):
                raise ValueError("Histogram and frame times must be finite and increasing")
            if (frame_times[0] < history_X_times_array[0] or
                    frame_times[-1] > history_X_times_array[-1]):
                raise ValueError("Animation frames cannot exceed the observed support")

            positions = np.searchsorted(history_X_times_array, frame_times)
            right = np.clip(positions, 0, len(history_X_times_array) - 1)
            left = np.clip(positions - 1, 0, len(history_X_times_array) - 1)
            use_left = (
                np.abs(frame_times - history_X_times_array[left]) <=
                np.abs(history_X_times_array[right] - frame_times)
            )
            histogram_indices = np.where(use_left, left, right)

    # -----------------------------------------------------------------
    # 1)  x-axis limits
    # -----------------------------------------------------------------
    if bg_mode == 'histogram' and history_X is not None:
        all_X = np.vstack(history_X)
        x_min = all_X[:, 0].min()
        x_max = all_X[:, 0].max()
        margin = 0.1 * (x_max - x_min)
        top_xlim = (x_min - margin, x_max + margin)

    elif bg_mode == 'gaussian' and means_gt is not None:
        # FIX: Handle the structure [[m1]*d, [m2]*d, [m3]*d]
        all_means = []
        for t_means in means_gt:  # For each time
            for cluster_mean in t_means:  # For each cluster
                if isinstance(cluster_mean, list):
                    all_means.append(cluster_mean[0])  # Get first element
                else:
                    all_means.append(cluster_mean)
        
        max_std = np.sqrt(max(vars_gt))
        top_xlim = (min(all_means) - 5 * max_std,
                    max(all_means) + 5 * max_std)
    else:
        raise ValueError("Inconsistent background setup.")

    # -----------------------------------------------------------------
    # 2)  y-axis limits - Evaluate every mixture on the same fine grid
    # -----------------------------------------------------------------
    x_grid = np.linspace(top_xlim[0], top_xlim[1], 400)
    global_max = 0.0
    for f in range(n_frames):
        pdf_f = np.zeros_like(x_grid)
        for k in range(K):
            # Extract mean properly, handling both float and array/list cases
            if hasattr(history_means[f][k], '__len__'):
                μ = history_means[f][k][0]  # If it's a list/array, get first element
            else:
                μ = history_means[f][k]     # If it's a float, use directly
            
            # Extract variance properly, handling different formats
            if hasattr(history_covariances[f][k], '__len__'):
                if hasattr(history_covariances[f][k][0], '__len__'):
                    σ2 = history_covariances[f][k][0][0]  # For 2D array format
                else:
                    σ2 = history_covariances[f][k][0]     # For 1D array format
            else:
                σ2 = history_covariances[f][k]            # For scalar format
                
            pdf_f += history_weights[f][k] * norm.pdf(x_grid, loc=μ, scale=np.sqrt(σ2))
        global_max = max(global_max, pdf_f.max())

    top_ylim = (0.0, global_max * 1.10)   # 10% head-room

    # -----------------------------------------------------------------
    # 3)  bottom-panel limits
    # -----------------------------------------------------------------
    if evol_weights == 'barplot':
        bottom_xlim = (-0.5, K - 0.5)
    else:
        bottom_xlim = (t_grid[0], t_grid[-1]) if t_grid is not None else (0, n_frames - 1)
    bottom_ylim = (0, 1)

    # -----------------------------------------------------------------
    # 4)  Prepare figure
    # -----------------------------------------------------------------
    fig = plt.figure(figsize=(8, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
    axT = fig.add_subplot(gs[0])          # top panel
    axB = fig.add_subplot(gs[1])          # bottom panel
    frames = []

    # -----------------------------------------------------------------
    # 5)  Build each frame
    # -----------------------------------------------------------------
    for f in tqdm(range(n_frames), desc="Building GMM GIF"):
        axT.clear(); axB.clear()
        axT.set_xlim(top_xlim); axT.set_ylim(top_ylim)
        axB.set_xlim(bottom_xlim); axB.set_ylim(bottom_ylim)

        # ---------- background ----------
        if bg_mode == 'histogram':
            histogram_index = int(histogram_indices[f])
            Xf = np.asarray(history_X[histogram_index])[:, 0]
            axT.hist(Xf, bins=50, density=True, color='lightgray', alpha=0.8)
            if history_X_times_array is not None:
                axT.set_title(
                    f"Week {t_grid[f]:.2f}; nearest observed block "
                    f"{history_X_times_array[histogram_index]:.2f}"
                )
        else:   # gaussian GT
            # FIX: Calculate GT as proper GMM, not average
            frame_idx = min(f, len(means_gt) - 1)  # Handle shorter GT
            pdf = np.zeros_like(x_grid)
            n_clusters_gt = len(means_gt[frame_idx])
            
            # Assume equal weights for GT (as in data generation)
            weight_gt = 1.0 / n_clusters_gt
            
            for k in range(n_clusters_gt):
                # Extract mean for cluster k
                if isinstance(means_gt[frame_idx][k], list):
                    μ_gt = means_gt[frame_idx][k][0]  # First dimension
                else:
                    μ_gt = means_gt[frame_idx][k]
                
                # Add component to mixture
                pdf += weight_gt * norm.pdf(x_grid, μ_gt, np.sqrt(vars_gt[frame_idx]))
            
            axT.fill_between(x_grid, 0, pdf, color='lightgray', alpha=0.4)
            axT.plot(x_grid, pdf, color='k', lw=2)

        # ---------- draw mixture components ----------
        for k in range(K):
            # Extract mean
            if hasattr(history_means[f][k], '__len__'):
                μ = history_means[f][k][0]
            else:
                μ = history_means[f][k]
            
            # Extract variance
            if hasattr(history_covariances[f][k], '__len__'):
                if hasattr(history_covariances[f][k][0], '__len__'):
                    σ2 = history_covariances[f][k][0][0]
                else:
                    σ2 = history_covariances[f][k][0]
            else:
                σ2 = history_covariances[f][k]
            
            pdf_k = history_weights[f][k] * norm.pdf(x_grid, μ, np.sqrt(σ2))
            axT.fill_between(x_grid, 0, pdf_k, color=cmap(k), alpha=0.3)
            axT.plot(x_grid, pdf_k, color=cmap(k), lw=1)
            axT.scatter([μ], [0], s=70*history_weights[f][k], color=cmap(k), marker='s')

        # ---------- bottom panel ----------
        if evol_weights == 'barplot':
            axB.bar(np.arange(K), history_weights[f], color=[cmap(k) for k in range(K)])
        else:   # evol_weights == 'path'
            x_axis = np.arange(f+1) if t_grid is None else t_grid[:f+1]
            for k in range(K):
                traj = [history_weights[i][k] for i in range(f+1)]
                axB.plot(x_axis, traj, color=cmap(k), marker='o', ms=1)

        plt.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        frames.append(img)

    plt.close(fig)
    imageio.mimsave(full_path, frames, fps=fps)
    print(f"[INFO] GMM GIF saved → {full_path}")
