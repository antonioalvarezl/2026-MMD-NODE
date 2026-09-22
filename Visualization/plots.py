import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import os
from scipy.optimize import linear_sum_assignment
from scipy.stats import multivariate_normal
from tqdm import tqdm

from Utils.IO_utils import load_experiment
from Utils.test_utils import compute_gmm_density_mc, compute_gt_density_mc
from data import compute_gmm_density


# Canonical component palette. Component labels are ordered by the shared
# Gaussian dictionary, so alpha_k has the same colour in every trajectory plot.
COMPONENT_COLORS = (
    "#17becf",  # component 1: cyan
    "#ff7f0e",  # component 2: orange
    "#2ca02c",  # component 3: green
    "#d62728",  # component 4: red
    "#9467bd",  # component 5: purple
    "#8c564b",  # component 6: brown
    "#e377c2",  # component 7: pink
    "#7f7f7f",  # component 8: grey
    "#bcbd22",  # component 9: olive
    "#1f77b4",  # component 10: blue
)


def component_colors(n_components):
    """Return the canonical, stable colour assigned to each component."""
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    return [COMPONENT_COLORS[index % len(COMPONENT_COLORS)]
            for index in range(n_components)]


def _load_patient_results(patient_dir, include_covs=False):
    """
    Load means from result_fit.json and traj_weights from result_node.json.
    Handles both old and new JSON formats.
    
    Args:
        patient_dir: Path to patient directory
        include_covs: If True, also load covariances
    
    Returns:
        dict with 'means', 'traj_weights', and optionally 'covs', or None if loading fails
    """
    # Load means from result_fit.json
    fit_res = load_experiment(patient_dir, "result_fit.json")
    if not fit_res:
        return None
    
    # Extract means - handle both old and new format
    if "results" in fit_res and "means" in fit_res["results"]:
        means = fit_res["results"]["means"]
        covs = fit_res["results"].get("covariances")
        fit_weights = fit_res["results"].get("weights")
    elif "avg_means" in fit_res:
        means = fit_res["avg_means"]
        covs = fit_res.get("avg_covariances")
        fit_weights = fit_res.get("avg_weights")
    elif "best_means" in fit_res:
        means = fit_res["best_means"]
        covs = fit_res.get("best_covariances")
        fit_weights = fit_res.get("best_weights")
    elif "means" in fit_res:
        means = fit_res["means"]
        covs = fit_res.get("covs", fit_res.get("covariances"))
        fit_weights = fit_res.get("weights")
    else:
        return None

    # Load traj_weights from result_node.json
    nodes = sorted(
        (d for d in os.listdir(patient_dir) if d.startswith("node_")),
        key=lambda name: int(name.split("_")[-1]) if name.split("_")[-1].isdigit() else -1
    )
    if not nodes:
        return None
    node_res = None
    legacy_node_res = None
    for node_name in reversed(nodes):
        candidate = load_experiment(
            os.path.join(patient_dir, node_name), "result_node.json"
        )
        if not candidate:
            continue
        config = candidate.get("node_config", {})
        if not config:
            if legacy_node_res is None:
                legacy_node_res = candidate
            continue

        comparisons = [
            (config.get("dictionary_means"), means),
            (config.get("dictionary_covariances"), covs),
            (config.get("observation_weights"), fit_weights),
        ]
        compatible = True
        for saved, expected in comparisons:
            if saved is None or expected is None:
                compatible = False
                break
            saved_array = np.asarray(saved, dtype=float)
            expected_array = np.asarray(expected, dtype=float)
            if (saved_array.shape != expected_array.shape or
                    not np.allclose(saved_array, expected_array)):
                compatible = False
                break
        if compatible:
            node_res = candidate
            break

    # Old result files did not record enough identity information to match a
    # NODE to its fit. Retain the historical latest-node fallback only when no
    # modern, matchable result exists.
    if node_res is None:
        node_res = legacy_node_res
    if node_res is None:
        return None
    
    # Extract traj_weights - handle both old and new format
    if "trajectory" in node_res:
        traj_weights = node_res["trajectory"]
    elif "traj_weights" in node_res:
        traj_weights = node_res["traj_weights"]
    else:
        return None

    result = {
        "means": means,
        "traj_weights": traj_weights,
        "trajectory_time_grid_weeks": node_res.get("trajectory_time_grid_weeks"),
        "temporal_support_weeks": node_res.get("temporal_support_weeks"),
        "node_config": node_res.get("node_config", {})
    }
    if include_covs and covs is not None:
        result["covs"] = covs

    dictionary = fit_res.get("config", {}).get("dictionary", {})
    if dictionary:
        if "means_original_units" in dictionary:
            result["means_original_units"] = dictionary["means_original_units"]
        if "covariances_original_units" in dictionary:
            result["covs_original_units"] = dictionary["covariances_original_units"]
        if "scaling" in dictionary:
            result["scaling"] = dictionary["scaling"]
    
    return result


def _get_patient_dir(root_folder, pid):
    """Find patient directory, supporting both naming conventions."""
    for fmt in [f"patient_{pid}", f"ids={pid}"]:
        path = os.path.join(root_folder, fmt)
        if os.path.isdir(path):
            return path
    return None

def plot_fit(diff_ws, diff_mus, diff_covs, mmds,
             X, final_weights, final_means, final_covariances,
             best_index, folder="Results", filename='fit.png',
             log_scale=False):
    """
    Plot the evolution of parameter differences and the final Gaussian mixture.
    Colors of components are assigned in increasing order of their means.
    """
    # create folder if missing
    os.makedirs(folder, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # --- Left panel: error evolution ---
    iters = np.arange(len(diff_mus))
    ax1 = axes[0]
    ax1.plot(iters, diff_ws, label='Weights diff')
    ax1.plot(iters, diff_mus, label='Means diff')
    ax1.plot(iters, diff_covs, label='Covariance diff')
    ax1.plot(iters, mmds, label='Squared MMD error')
    ax1.set(xlabel='Iteration', ylabel='Error (log scale)' if log_scale else 'Error',
            title='Error Evolution')
    ax1.legend(); ax1.grid(True)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    if log_scale:
        ax1.set_yscale('log')

    # --- Right panel: final mixture ---
    ax2 = axes[1]
    K = len(final_weights)
    if K == 3:
        base_colors = ["tab:blue", "tab:green", "tab:orange"][:K]
    else:
        cmap = cm.get_cmap("viridis", K)
        base_colors = [cmap(i) for i in range(K)]

    # only for 1D data
    if X.shape[1] == 1:
        ax2.hist(X[:, 0], bins=50, density=True,
                 alpha=0.5, label='Empirical Data')
        # grid for density
        x_grid = np.linspace(-8,30, 5000)

        # sort components by their means
        means_1d = np.array([m if np.isscalar(m) else m[0]
                              for m in final_means])
        order = np.argsort(means_1d)

        total_density = np.zeros_like(x_grid)
        for idx, k in enumerate(order):
            w = final_weights[k]
            mu = final_means[k]
            cov = final_covariances[k]
            # compute density
            comp = w * multivariate_normal.pdf(x_grid, mean=mu, cov=cov)
            total_density += comp
            color = base_colors[idx]
            # fill, line, and marker
            ax2.fill_between(x_grid, 0, comp, color=color, alpha=0.3)
            ax2.plot(x_grid, comp, color=color, lw=max(1, 2*w))
            ax2.scatter(mu, 0, s=int(70*w), color=color, marker='s')
        # mixture density
        ax2.plot(x_grid, total_density, 'r--', lw=2, label='Mixture')
        ax2.set(xlabel='x', ylabel='Density')

    ax2.set_title(f'Optimal Gaussian Mixture (Iter {best_index})')
    fig.tight_layout()
    out_path = os.path.join(folder, filename)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    
def _grid_1d(X: np.ndarray,
             means: np.ndarray,
             covs: np.ndarray,
             n: int = 400) -> np.ndarray:
    """
    Build a 1-D grid from the minimum data point to the maximum mean,
    extended by ±4·σ_max on both sides.
    """
    x_min = X.min()
    x_max = X.max()
    max_sigma = np.sqrt(np.max(covs))
    lo = min(x_min, means.min()) - 4 * max_sigma
    hi = max(x_max, means.max()) + 4 * max_sigma
    return np.linspace(lo, hi, n)


def plot_mixture(X: np.ndarray,
                    weights: np.ndarray,
                    means: np.ndarray,
                    covs: np.ndarray,
                    folder: str = "Results",
                    filename: str = "mixture.png",
                    figsize: tuple = (10, 5)) -> None:
    """
    Figure that shows the 1-D Gaussian mixture
    (components + total density + data histogram).

    """
    if X.shape[1] != 1:
        print("plot_mixture_1d is for d = 1 only.")
        return

    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)

    K = len(weights)
    # --- special colour mapping when K == 3 -------------------------
    if K == 3:
        order = np.argsort(means.ravel())          # left → right
        base_colors = ["tab:blue", "tab:green", "tab:orange"]
        colours = {k: base_colors[i] for i, k in enumerate(order)}
        cmap_fn = lambda k: colours[k]
    else:
        cmap = cm.get_cmap("viridis", K)
        cmap_fn = lambda k: cmap(k)

    # --- grid & densities ------------------------------------------
    x_grid = _grid_1d(X, means, covs)
    total  = np.zeros_like(x_grid)

    # --- figure ----------------------------------------------------
    plt.figure(figsize=figsize)
    plt.hist(X[:, 0], bins=50, density=True,
             alpha=0.5, label="Empirical data", color="lightgray")

    for k in range(K):
        mu  = means[k]
        var = covs[k] if np.ndim(covs[k]) == 0 else covs[k][0]
        pdf_k = weights[k] * multivariate_normal.pdf(x_grid,
                                                     mean=mu,
                                                     cov=var)
        total += pdf_k
        lw = max(1, 2 * weights[k])            
        plt.fill_between(x_grid, 0, pdf_k,
                         color=cmap_fn(k), alpha=0.5)
        plt.plot(x_grid, pdf_k, color=cmap_fn(k), lw=lw)
        plt.scatter(mu, 0, s=80 * weights[k],
                    color=cmap_fn(k), marker='s')

    plt.plot(x_grid, total, 'k--', lw=2, label="Mixture density")
    plt.xlabel('x')
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
   
# Plotting functions for NODE training loss and trajectory

def plot_training(loss_history, log = True, folder=None):
    """
    Plot the evolution of the training loss over epochs.
    
    Parameters:
      loss_history (list): List of loss values per epoch.
    log (bool): If True, plot on a logarithmic scale.
      folder (str): Directory to save the plot.
    """
    print(f"Plotting training loss history.")
    # Simulate a progress bar while "processing" loss history.
    for _ in tqdm(loss_history, desc="Processing training loss history", leave=False):
        pass  # This loop only serves to display a progress bar.
    
    plt.figure()
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if log:
        plt.yscale("log")
    plt.legend()
    plt.title("Training Loss Evolution")
    if folder is not None:
        filename = os.path.join(folder, "node_training.png")
        plt.savefig(filename)
        print(f"Training plot saved.")
        plt.close()
    else:
        plt.show()
        plt.close()
        
def add_multicolored_title(ax, base_text="Weight Trajectory - Components",
                          comp_colors=['#1a659e', '#ffa62b', '#7ae582']):
    """
    Add a title to a plot with different colored component numbers,
    placed above the axes so they never overlap the data.
    Adjusted vertical placement to match other plots.
    """
    
    # Get figure and adjust top margin to leave space for title
    fig = ax.get_figure()  

    # Title lower down
    fig.subplots_adjust(top=0.75)

    # 1) Draw base text with fig.text, track Text instance for bbox
    #    Use y=0.94 for vertical placement
    t = fig.text(0.5, 0.9, base_text,
                 ha='center', va='bottom', fontsize=22)

    # 2) Force draw to populate the renderer and get bounding box
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bb = t.get_window_extent(renderer)

    # 3) Transform the right edge of base text to figure coords
    inv = fig.transFigure.inverted()
    x_base_end, _ = inv.transform((bb.x1, bb.y0))

    # 4) Define labels and small horizontal shift
    labels = ['1', '/', '2', '/', '3']
    colors = [comp_colors[0], 'black', comp_colors[1], 'black', comp_colors[2]]

    # 5) Place each symbol in figure coords, just to the right of base text
    gap = 0.007
    dx = 0.03
    for i, (lab, col) in enumerate(zip(labels, colors)):
        fig.text(x_base_end + gap + i*dx, 0.9, lab,
                 ha='left', va='bottom', fontsize=22, color=col)



def plot_traj(t_grid, t_i, pred_traj, sample_data=None, K=3,
            folder=None, filename=None):
    """
    Plot predicted trajectories against observed data points.
    
    Args:
        t_grid (ndarray): Time grid for predictions (length T).
        t_i (ndarray): Observation times (length n_i).
        pred_traj (ndarray): Predicted trajectories, shape (T, K).
        sample_data (ndarray): Observed data, shape (n_i, K).
        K (int): Number of mixture components.
        legend (bool): Whether to add a legend.
        folder (str): Directory to save the plot (if provided).
        filename (str): Base filename without extension.
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend

    # Ensure t_i is an array
    if t_i.ndim == 0:
        t_i = np.array([t_i])
        
    if sample_data is not None:
        # If sample_data has one row but multiple t_i, repeat the row
        if sample_data.shape[0] == 1 and pred_traj.shape[0] > 1:
            sample_data = np.repeat(sample_data, len(t_i), axis=0)

    # 1) Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    # 2) Plot each component with consistent colors
    colors = component_colors(K)
    for k in range(K):
        ax.plot(t_grid, pred_traj[:, k],
                color=colors[k], label=f'pred w_{k+1}')
        if sample_data is not None:
            ax.scatter(t_i, sample_data[:, k],
                    color=colors[k], marker='o', s=24, alpha=0.9,
                    edgecolors='white', linewidths=0.35,
                    label=f'data w_{k+1}')

    # 3) Set axis labels and tick formatting
    ax.set_xlabel('Time', fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=20)

    # 4) Add the multicolored title if there are exactly 3 components
    if K == 3:
        add_multicolored_title(ax,
            base_text="Weight Trajectory - Components",
            comp_colors=colors)

    # 5) Adjust layout to leave space for the title
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    # 6) Save or display the figure
    if folder is not None:
        fname = 'trajectories.png' if filename is None else f"{filename}.png"
        fig.savefig(os.path.join(folder, fname),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


            
def plot_L2err(L2errs, L2errs_ts, t_grid, t_i, path):
    """
    Plot L2 errors over time for synthetic data (solo L2, sin KL).
    
    Parameters:
      L2errs: List/array of L2 error values corresponding to t_grid.
      L2errs_ts: List/array of L2 error values at specific time instants (t_i).
      t_grid: 1D array of all time points.
      t_i: 1D array of specific time instants where markers should be plotted.
      path: Path to save the plot.
    """
    plt.figure(figsize=(10, 4))
    
    # Plot L2 error
    plt.plot(t_grid, L2errs, 'b-', label='L2 Error (NODE)', linewidth=2)
    # Add large black diamond markers at specific time instants.
    plt.plot(t_i, L2errs_ts, 'D', markersize=6, color='black', label='L2 Error (MMD fitting)')
    
    plt.title('L2 Error Evolution', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('L2 Error', fontsize=14)
    plt.yscale('log')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(path, dpi=600)
    plt.close()
    print(f"L2 error plot saved: {path}")
    
# ------------------------------------------------------------------
# Plot   L2   +   MMD   empirical errors
# ------------------------------------------------------------------
def plot_L2MMDerr(L2_node, MMD_node,
                  L2_fit,  MMD_fit,
                  t_grid,  t_i,
                  path):
    """
    Two-panel log-scale plot.

    * Top  : L2 distance KDE ↔ GMM
    * Bottom : MMD distance   X ↔ GMM

    Black diamonds show the coarse instants *t_i* (weights from the
    MMD-fitting step); continuous lines show the NODE trajectory
    evaluated on the fine grid *t_grid*.

    Parameters
    ----------
    L2_node , MMD_node : list / ndarray, len == len(t_grid)
    L2_fit  , MMD_fit  : list / ndarray, len == len(t_i)
    t_grid  : 1-D array of fine time instants (NODE)
    t_i     : 1-D array of coarse time instants (MMD-fit)
    path    : full file-name where the PNG is saved (600 dpi)
    """

    plt.figure(figsize=(10, 6))

    # -------- L2 panel -------------------------------------------
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(t_grid, L2_node, 'b-',  label='L2 (NODE)')
    ax1.plot(t_i,    L2_fit,  'Dk',  ms=4, label='L2 (MMD fit)')
    ax1.set_yscale('log')
    ax1.set_ylabel('L2 distance')
    ax1.set_title('Empirical L2 and MMD errors')
    ax1.legend()
    ax1.grid(alpha=.4, which='both')

    # -------- MMD panel ------------------------------------------
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(t_grid, MMD_node, 'r-',  label='MMD (NODE)')
    ax2.plot(t_i,    MMD_fit,  'Dk',  ms=4, label='MMD (MMD fit)')
    ax2.set_yscale('log')
    ax2.set_xlabel('time')
    ax2.set_ylabel('MMD distance')
    ax2.legend()
    ax2.grid(alpha=.4, which='both')

    plt.tight_layout()

    # high-resolution save
    plt.savefig(path, dpi=600)
    plt.close()

def plot_sumweights(t_grid, traj_weights, folder=None,
                            obs_weights=None, obs_times=None):
    """
    Extended version with individual component plots and diagnostic subplots.
    
    Parameters:
        Same as plot_sumweights
    """
    
    # Ensure arrays are properly shaped
    t_grid = np.array(t_grid).flatten()
    
    # Check if we have multi-component data
    traj_weights = np.asarray(traj_weights, dtype=float)
    if traj_weights.ndim == 1:
        traj_weights = traj_weights[:, None]
    if obs_weights is not None:
        obs_weights = np.asarray(obs_weights, dtype=float)
        if obs_weights.ndim == 1:
            obs_weights = obs_weights[:, None]
    
    n_components = traj_weights.shape[1]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Subplot 1: Sum of weights (main plot)
    ax1 = fig.add_subplot(gs[0, :])
    sum_weights_pred = np.sum(traj_weights, axis=1)
    ax1.plot(t_grid, sum_weights_pred, 
            label='Sum (NODE)', color='blue', linewidth=2)
    
    if obs_weights is not None and obs_times is not None:
        obs_times = np.array(obs_times).flatten()
        if obs_weights.ndim > 1:
            sum_weights_obs = np.sum(obs_weights, axis=1)
        else:
            sum_weights_obs = obs_weights
        
        ax1.scatter(obs_times, sum_weights_obs, 
                   label='Sum (MMD targets)', 
                   color='red', s=100, marker='o', zorder=5)
        ax1.plot(obs_times, sum_weights_obs, 
                color='red', linestyle='--', alpha=0.5)
    
    ax1.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Ideal')
    ax1.set_ylabel('Sum of Weights')
    ax1.set_title('Sum of All Components', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.9, 1.1)
    
    # Subplot 2: Individual weight trajectories
    ax2 = fig.add_subplot(gs[1, :])
    colors = component_colors(n_components)
    
    for k in range(n_components):
        ax2.plot(t_grid, traj_weights[:, k], 
                label=f'Component {k+1}', 
                color=colors[k], linewidth=1.5, alpha=0.8)
        
        if obs_weights is not None and obs_times is not None and obs_weights.ndim > 1:
            ax2.scatter(obs_times, obs_weights[:, k], 
                       color=colors[k], s=50, marker='o', zorder=5)
    
    ax2.set_ylabel('Individual Weights')
    ax2.set_title('Individual Component Trajectories', fontweight='bold')
    ax2.legend(ncol=min(n_components, 5), fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)
    
    # Subplot 3: Deviation from simplex (sum - 1)
    ax3 = fig.add_subplot(gs[2, 0])
    deviation_pred = sum_weights_pred - 1.0
    ax3.plot(t_grid, deviation_pred, color='blue', linewidth=2)
    ax3.axhline(y=0, color='green', linestyle='--', alpha=0.7)
    ax3.fill_between(t_grid, deviation_pred, 0, alpha=0.3, color='blue')
    
    if obs_weights is not None and obs_times is not None:
        deviation_obs = sum_weights_obs - 1.0
        ax3.scatter(obs_times, deviation_obs, color='red', s=80, marker='o', zorder=5)
    
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Deviation from 1')
    ax3.set_title('Simplex Constraint Violation (Σw - 1)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Statistics table
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')
    
    stats_data = []
    stats_data.append(['Metric', 'NODE Pred', 'MMD Target'])
    stats_data.append(['─'*15, '─'*12, '─'*12])
    stats_data.append(['Mean sum', f'{sum_weights_pred.mean():.6f}', 
                      f'{sum_weights_obs.mean():.6f}' if obs_weights is not None else 'N/A'])
    stats_data.append(['Std sum', f'{sum_weights_pred.std():.6f}',
                      f'{sum_weights_obs.std():.6f}' if obs_weights is not None else 'N/A'])
    stats_data.append(['Max deviation', f'{abs(deviation_pred).max():.6f}',
                      f'{abs(deviation_obs).max():.6f}' if obs_weights is not None else 'N/A'])
    stats_data.append(['Min weight', f'{traj_weights.min():.6f}',
                      f'{obs_weights.min():.6f}' if obs_weights is not None else 'N/A'])
    stats_data.append(['Max weight', f'{traj_weights.max():.6f}',
                      f'{obs_weights.max():.6f}' if obs_weights is not None else 'N/A'])
    
    table = ax4.table(cellText=stats_data, cellLoc='left',
                     bbox=[0, 0, 1, 1], edges='horizontal')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle('Weight Trajectory Diagnostics', 
                fontsize=16, fontweight='bold', y=0.995)
    
    if folder is not None:
        filename = os.path.join(folder, "sum_weights_detailed.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Detailed sum of weights plot saved to {filename}")
    
    plt.close()

    
def _read_csv_simple(path):
    """Load a CSV, fill NaNs by linear interpolation row-wise, return a 2D ndarray."""
    data = np.genfromtxt(path, delimiter=",", dtype=float)
    data = np.atleast_2d(data)
    if np.isnan(data).any():
        x = np.arange(data.shape[1])
        for r in range(data.shape[0]):
            mask = ~np.isnan(data[r])
            if mask.sum() >= 2:
                data[r, ~mask] = np.interp(x[~mask], x[mask], data[r, mask])
    return data

def _percentile_band(arr: np.ndarray, pct: float):
    """Return lower/upper percentile trajectories that include pct% of samples."""
    lower = np.percentile(arr, (100 - pct) / 2, axis=0)
    upper = np.percentile(arr, 100 - (100 - pct) / 2, axis=0)
    return lower, upper


def plot_all_trajs(patient_ids, root_folder, name, alpha=0.8, dist="L2",
                   bands=False, band_pct=90.0):
    """
    Plot weight trajectories for each mixture component across multiple patients,
    after aligning components by matching their fitted means.
    Adds mean trajectory (dashed line) and standard deviation band.

    patient_ids : list of patient IDs
    root_folder : base directory containing 'ids={pid}' subfolders
    name        : prefix for saved image files
    alpha       : fraction of trajectories to plot (0 < alpha <= 1)
    dist        : 'L2' or 'L1' distance metric for selection
    bands       : if True, plot mean + percentile band instead of individual trajectories
    band_pct    : percentile width for uncertainty band (e.g., 90 → 5th–95th)
    """
    

    # 1) Load all results
    results = []
    for pid in patient_ids:
        patient_dir = _get_patient_dir(root_folder, pid)
        if patient_dir is None:
            continue

        patient_result = _load_patient_results(patient_dir)
        if patient_result is None:
            print(f"[WARN] No fit-compatible NODE result in {patient_dir}")
            continue

        results.append(patient_result)

    if not results:
        raise RuntimeError("No valid results found.")

    # 2) Use first patient's means as reference
    ref_means = np.array(results[0]["means"])
    if ref_means.ndim == 1:
        ref_means = ref_means[:, None]
    K = ref_means.shape[0]

    # 3) Align all trajectories to reference
    aligned_trajs = []
    for res in results:
        means = np.array(res["means"])
        if means.ndim == 1:
            means = means[:, None]

        traj = np.array(res["traj_weights"])
        if traj.ndim == 3 and traj.shape[1] == 1:
            traj = traj[:, 0, :]

        if traj.ndim != 2 or traj.shape[1] != K:
            print(f"[WARN] Skipping reordering for shape mismatch: traj {traj.shape}, expected K={K}")
            aligned_trajs.append(traj)
            continue

        if means.shape[1] == 1:
            order = np.argsort(means.flatten())
        else:
            cost = np.linalg.norm(ref_means[:, None, :] - means[None, :, :], axis=2)
            _, order = linear_sum_assignment(cost)

        aligned_trajs.append(traj[:, order])

    # 4) Prepare time grid.
    # These per-run diagnostics plot the NODE grid INDEX, not clinical time:
    # participants' fitted supports differ in weeks since randomization, so a
    # shared 0-1 axis aligns different clinical times. Use the paper figures
    # (Analysis/paper_figures.py) for anything on a clinical axis.
    T = aligned_trajs[0].shape[0]
    t_grid = np.linspace(0, 1.0, T)

    # Component identity, rather than cohort, determines trajectory colour.
    comp_colors = component_colors(K)

    # Plot each component separately
    for k in range(K):
        comp_trajs = np.stack([
            traj[:, k].reshape(-1)
            for traj in aligned_trajs
            if traj.ndim == 2 and traj.shape[1] > k and traj.shape[0] == T
        ])

        if comp_trajs.shape[0] == 0:
            print(f"[WARN] No valid trajectories for component {k}")
            continue

        mean_traj = comp_trajs.mean(axis=0)
        if bands:
            low, up = _percentile_band(comp_trajs, band_pct)
            m = len(comp_trajs)
        else:
            if dist == "L2":
                dists = np.linalg.norm(comp_trajs - mean_traj, axis=1)
            else:
                dists = np.abs(comp_trajs - mean_traj).sum(axis=1)
            m = int(np.ceil(alpha * len(comp_trajs)))
            sel = np.argsort(dists)[:m]
            sel_trajs = comp_trajs[sel]

        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        # Plot individual trajectories
        if bands:
            ax.fill_between(t_grid, low, up, color=comp_colors[k], alpha=0.25)
        else:
            for tr in sel_trajs:
                ax.plot(t_grid, tr, lw=1, color=comp_colors[k])
        # Plot mean trajectory (darker shade)
        darker_color = tuple(max(0, c - 0.3) for c in plt.cm.colors.to_rgb(comp_colors[k]))
        ax.plot(t_grid, mean_traj, lw=3, color=darker_color)

        ax.set_xlabel("Time", fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=20)
        title = f"Component {k+1} ({name})"
        if bands:
            title += f" – {band_pct:.0f}% band"
        ax.set_title(title, fontsize=22)
        ax.set_ylim(0.0, 0.8)
        plt.tight_layout()

        out_file = os.path.join(root_folder, f"{name}_comp{k+1}.png")
        fig.savefig(out_file)
        plt.close(fig)
        print(f"Saved {out_file} ({m}/{len(comp_trajs)} trajectories)")



def _load_aligned_trajs(patient_ids, root_folder, ref_means=None):
    """
    Helper to load and align trajectories given a list of patients.
    If ref_means is provided, use it for alignment; otherwise take the
    first patient's means as reference.
    """
    results = []
    for pid in patient_ids:
        patient_dir = _get_patient_dir(root_folder, pid)
        if patient_dir is None:
            continue
        res = _load_patient_results(patient_dir)
        if res is None:
            continue
        results.append(res)
    if not results:
        raise RuntimeError("No valid results found.")

    if ref_means is None:
        ref_means = np.array(results[0]["means"])
        if ref_means.ndim == 1:
            ref_means = ref_means[:, None]
    K = ref_means.shape[0]

    aligned = []
    for res in results:
        means = np.array(res["means"])
        if means.ndim == 1:
            means = means[:, None]

        traj = np.array(res["traj_weights"])
        if traj.ndim == 3 and traj.shape[1] == 1:
            traj = traj[:, 0, :]

        if traj.ndim != 2 or traj.shape[1] != K:
            continue

        if means.shape[1] == 1:
            order = np.argsort(means.flatten())
        else:
            cost = np.linalg.norm(ref_means[:, None, :] - means[None, :, :], axis=2)
            _, order = linear_sum_assignment(cost)

        aligned.append(traj[:, order])
    return np.stack(aligned), ref_means


def plot_group_trajs_bands(patient_ids, root_folder, name, band_pct=90.0, ref_means=None):
    """
    Plot mean + percentile bands for all components of a group in one figure.
    Colors use the same canonical component palette as ``plot_traj``.
    Returns the reference means used for alignment.
    """
    aligned_trajs, ref_means = _load_aligned_trajs(patient_ids, root_folder, ref_means)
    T = aligned_trajs.shape[1]
    K = aligned_trajs.shape[2]
    # Normalized NODE-grid position, not clinical time: see the note in
    # _load_aligned_trajs. Clinical-time group figures live in
    # Analysis/paper_figures.py, which interpolates on a documented common
    # support instead.
    t_grid = np.linspace(0, 1.0, T)

    colors = component_colors(K)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    for k in range(K):
        comp_trajs = aligned_trajs[:, :, k]
        mean_traj = comp_trajs.mean(axis=0)
        low, up = _percentile_band(comp_trajs, band_pct)
        ax.fill_between(t_grid, low, up, color=colors[k], alpha=0.25)
        darker_color = tuple(max(0, c - 0.3) for c in plt.cm.colors.to_rgb(colors[k]))
        ax.plot(t_grid, mean_traj, lw=3, color=darker_color, label=f"w{k+1}")

    ax.set_xlabel("Time", fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylim(0.0, 0.8)
    ax.set_title(f"{name} – {band_pct:.0f}% bands", fontsize=22)
    plt.tight_layout()

    out_file = os.path.join(root_folder, f"{name}_bands.png")
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved {out_file}")
    return ref_means


def plot_trajs_bands_two_groups(treatment_ids, control_ids, root_folder, band_pct=90.0):
    """
    Convenience wrapper: plots bands for Treatment and Control (two figures),
    ensuring consistent component ordering via shared reference means.
    """
    ref_means = plot_group_trajs_bands(treatment_ids, root_folder, "Treatment", band_pct=band_pct)
    plot_group_trajs_bands(control_ids, root_folder, "Control", band_pct=band_pct, ref_means=ref_means)


def plot_timesignal(csv_files, out_png="timesignal.png"):
    """
    Plots every row from the given CSV files as a thin, semi-transparent blue line.
    csv_files can be either a list of paths or a single string.
    """
    
    # allow a single string
    if isinstance(csv_files, str):
        csv_files = [csv_files]

    rows = []
    for fp in csv_files:
        if not os.path.isfile(fp):
            print(f"[WARN] file not found: {fp}")
            continue
        rows.extend(_read_csv_simple(fp))

    if not rows:
        raise RuntimeError("No valid data found in the provided CSVs.")

    plt.figure(figsize=(10, 5))
    for row in rows:
        plt.plot(row, color="#1f77b4", alpha=0.25, lw=0.6)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.title(f"Time signal", fontsize=22)  # Changed from set_title to title
    plt.xlabel("Number of measurements", fontsize=22)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_boxplots(treatment_ids, control_ids, root_folder, alpha=0.8, dist="L2", compute_ks=False):
    """
    Plot paired boxplots of weight differences (before-after) for each component,
    comparing Treatment vs Control in the same figure.

    treatment_ids : list of patient IDs for Treatment group
    control_ids   : list of patient IDs for Control group
    root_folder   : base directory containing patient subfolders
    alpha, dist   : as before
    compute_ks    : if True, compute Kolmogorov-Smirnov test and show p-value
    """
    from scipy.stats import ks_2samp
    
    def load_aligned(patient_ids):
        from scipy.optimize import linear_sum_assignment
        results = []
        for pid in patient_ids:
            patient_dir = _get_patient_dir(root_folder, pid)
            if patient_dir is None:
                continue
            res = _load_patient_results(patient_dir)
            if res is None:
                continue
            results.append(res)
        if not results:
            raise RuntimeError("No valid results found.")
        ref_means = np.array(results[0]["means"])
        if ref_means.ndim == 1:
            ref_means = ref_means[:, None]
        K = ref_means.shape[0]
        aligned = []
        for res in results:
            means = np.array(res["means"])
            if means.ndim == 1:
                means = means[:, None]
            traj = np.array(res["traj_weights"])
            if traj.ndim == 3 and traj.shape[1] == 1:
                traj = traj[:, 0, :]
            if traj.ndim != 2 or traj.shape[1] != K:
                continue
            if means.shape[1] == 1:
                order = np.argsort(means.flatten())
            else:
                cost = np.linalg.norm(ref_means[:, None, :] - means[None, :, :], axis=2)
                _, order = linear_sum_assignment(cost)
            aligned.append(traj[:, order])
        return np.stack(aligned)

    traj_t = load_aligned(treatment_ids)
    traj_c = load_aligned(control_ids)
    K = traj_t.shape[2] if traj_t.ndim == 3 else 1
    cols = {"Treatment": "#588157", "Control": "#ad2831"}
    
    ks_results = []

    for k in range(K):
        dif_t = traj_t[:, 0, k] - traj_t[:, -1, k] if traj_t.ndim == 3 else traj_t[:, 0] - traj_t[:, -1]
        dif_c = traj_c[:, 0, k] - traj_c[:, -1, k] if traj_c.ndim == 3 else traj_c[:, 0] - traj_c[:, -1]
        
        # KS test if requested
        ks_stat, p_value = None, None
        if compute_ks:
            ks_stat, p_value = ks_2samp(dif_t, dif_c)
            ks_results.append({'component': k+1, 'ks_stat': ks_stat, 'p_value': p_value})
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        bp = ax.boxplot([dif_t, dif_c], positions=[1, 2], widths=0.6, patch_artist=True,
                        medianprops=dict(color='black', linewidth=2))
        
        for idx, grp in enumerate(["Treatment", "Control"]):
            color = cols[grp]
            bp['boxes'][idx].set_facecolor(color)
            bp['boxes'][idx].set_alpha(0.6)
            for element in ['whiskers', 'caps', 'fliers']:
                for obj in bp[element][2*idx:2*idx+2]:
                    obj.set_color(color)
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Treatment", "Control"], fontsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.set_ylim(-0.5, 0.5)
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        
        # Title with or without p-value
        if compute_ks:
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            ax.set_title(f"Weight Difference - Component {k+1}\nKS p-value: {p_value:.4f} {sig}", fontsize=20)
        else:
            ax.set_title(f"Weight Difference - Component {k+1}", fontsize=22)
        
        plt.tight_layout()
        out_file = os.path.join(root_folder, f"boxplot_comp{k+1}.png")
        fig.savefig(out_file)
        plt.close(fig)
        
        if compute_ks:
            print(f"Saved {out_file} | KS stat: {ks_stat:.4f}, p-value: {p_value:.4f}")
        else:
            print(f"Saved {out_file}")
    
    if compute_ks:
        return ks_results
        

def plot_quantilecurves(treatment_ids, control_ids, root_folder, quantiles=None, y_limits=None):
    """
    Plot quantile curves (difference from initial) for each component,
    comparing Treatment vs Control in same figure.

    treatment_ids, control_ids : lists of patient IDs
    root_folder : results directory
    quantiles : list of quantile levels (0-1). Default is [0.25, 0.5, 0.75]
    y_limits : optional tuple (ymin, ymax); if None, auto-scales with padding
    """
    
    from scipy.optimize import linear_sum_assignment

    if quantiles is None:
        quantiles = [0.25, 0.5, 0.75]
    quantiles = sorted(float(q) for q in quantiles)

    def load_diff(ids, k):
        # load and align trajectories, then return matrix of difference-from-initial (n_patients x T)
        results = []
        for pid in ids:
            pd_dir = _get_patient_dir(root_folder, pid)
            if pd_dir is None:
                continue
            res = _load_patient_results(pd_dir)
            if res is None:
                continue
            results.append(res)
        if not results:
            raise RuntimeError("No valid results found.")
        
        # set up reference component means
        ref = np.array(results[0]["means"])
        if ref.ndim == 1:
            ref = ref[:, None]
        K = ref.shape[0]
        
        diffs = []
        for res in results:
            means = np.array(res["means"])
            traj = np.array(res["traj_weights"])
            if traj.ndim == 3 and traj.shape[1] == 1:
                traj = traj[:, 0, :]
            # ensure means has shape (K,1) when needed
            if means.ndim == 1:
                means = means[:, None]
            # compute optimal matching to reference
            if means.shape[1] > 1:
                cost = np.linalg.norm(ref[:, None, :] - means[None, :, :], axis=2)
                _, order = linear_sum_assignment(cost)
                traj = traj[:, order]
            # compute difference from initial time point
            diff = traj[:, k] - traj[0, k]
            diffs.append(diff)
        return np.stack(diffs), K

    # get number of components from treatment group
    sample, K = load_diff(treatment_ids, 0)
    T = sample.shape[1]
    # Normalized NODE-grid position, not clinical time: see _load_aligned_trajs.
    t_grid = np.linspace(0, 1.0, T)
    cols = {"Treatment": "#588157", "Control": "#ad2831"}

    # loop over each component
    for k in range(K):
        dif_t, _ = load_diff(treatment_ids, k)
        dif_c, _ = load_diff(control_ids,   k)
        # compute quantiles across patients at each time point
        q_t = np.quantile(dif_t, quantiles, axis=0)
        q_c = np.quantile(dif_c, quantiles, axis=0)

        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

        if 0.25 in quantiles and 0.75 in quantiles:
            q25_t = np.quantile(dif_t, 0.25, axis=0)
            q75_t = np.quantile(dif_t, 0.75, axis=0)
            q25_c = np.quantile(dif_c, 0.25, axis=0)
            q75_c = np.quantile(dif_c, 0.75, axis=0)
            ax.fill_between(t_grid, q25_t, q75_t, color=cols['Treatment'], alpha=0.18)
            ax.fill_between(t_grid, q25_c, q75_c, color=cols['Control'], alpha=0.18)

        for i, q in enumerate(quantiles):
            if q == 0.5:
                ax.plot(
                    t_grid, q_t[i],
                    lw=3,
                    linestyle='-',
                    alpha=0.95,
                    color=cols['Treatment'],
                    label=f"Treatment median" if k == 0 else None
                )
                ax.plot(
                    t_grid, q_c[i],
                    lw=3,
                    linestyle='-',
                    alpha=0.95,
                    color=cols['Control'],
                    label=f"Control median" if k == 0 else None
                )
            elif q not in {0.25, 0.75}:
                ax.plot(
                    t_grid, q_t[i],
                    lw=2,
                    linestyle='--',
                    alpha=0.9,
                    color=cols['Treatment'],
                    label=f"T {int(q*100)}th" if k == 0 else None
                )
                ax.plot(
                    t_grid, q_c[i],
                    lw=2,
                    linestyle='--',
                    alpha=0.9,
                    color=cols['Control'],
                    label=f"C {int(q*100)}th" if k == 0 else None
                )

        ax.set_xlabel("Time", fontsize=22)
        ax.tick_params(labelsize=20)

        component_y_limits = y_limits
        if component_y_limits is None:
            all_vals = np.concatenate([dif_t, dif_c])
            y_min = np.nanmin(all_vals)
            y_max = np.nanmax(all_vals)
            pad = 0.1 * max(abs(y_min), abs(y_max), 1e-6)
            component_y_limits = (y_min - pad, y_max + pad)
        ax.set_ylim(*component_y_limits)
        ax.set_title(f"Weight Difference - Quantile Curves - Component {k+1}", fontsize=22)
        
        # Add legend with color explanation
        import matplotlib.patches as mpatches
        handles = [
            mpatches.Patch(color=cols['Treatment'], label="Treatment"),
            mpatches.Patch(color=cols['Control'], label="Control"),
        ]
        ax.legend(handles=handles, loc="upper left", frameon=False)
        
        plt.tight_layout()
        fig.savefig(os.path.join(root_folder, f"quantile_curves_comp{k+1}.png"), dpi=300)
        plt.close(fig)
        print(f"Saved file with quantile curves for component {k+1}")

def plot_gmm_heatmap(means, covs, weights, output_file, cmap, grid_size=400, title="Predicted Density Heatmap"):
    # --- debug prints ---
    x_axis, y_axis = compute_grid(means, means, covs, n=grid_size) 
    X, Y = np.meshgrid(x_axis, y_axis)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])

    density = np.zeros(grid_points.shape[0], dtype=float)

    for w, mu, cov in zip(weights, means, covs):
        density += w * multivariate_normal.pdf(grid_points, mean=mu, cov=cov)

    Z = density.reshape(X.shape)

    fig, ax = plt.subplots(figsize=(10,5), dpi=300)
    contour = plt.contourf(X, Y, Z, levels=300, cmap=cmap)
    plt.title(title, fontsize=22)
    
    # Formatear colorbar con notación científica y 2 cifras significativas
    from matplotlib.ticker import ScalarFormatter
    cbar = plt.colorbar(contour)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    cbar.formatter = formatter
    cbar.update_ticks()
    
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

 
def compute_grid(X, means, covariances, n=400):
    """
    Compute a grid for plotting density functions.
    
    For 1D: returns a 1D numpy array.
    For d>=2: returns a tuple of 1D arrays (one per dimension).
    
    Parameters:
        X (np.ndarray): Data array of shape (n_samples, d) or (n_samples,) for 1D.
        means (array-like): Mixture means, shape: (K, d) or (K,) for 1D.
        covariances (array-like): Covariance matrices. For 1D: (K, d); for d>=2: (K, d, d).
        n (int): Number of points per dimension for constructing the grid.
    
    Returns:
        For 1D: a numpy array.
        For d>=2: a tuple of 1D arrays.
    """
    X = np.atleast_2d(X)         # Ensure X is 2D.
    means = np.atleast_2d(means)   # Ensure means is 2D.
    d = X.shape[1]               # Determine the dimensionality.
    
    if d == 1:
        # For 1D, compute min and max using both data and means, and add a margin.
        x_min = min(X[:, 0].min(), means[:, 0].min())
        x_max = max(X[:, 0].max(), means[:, 0].max())
        margin = 0.1 * (x_max - x_min)
        return np.linspace(x_min - margin, x_max + margin, n)
    else:
        grid_list = []
        cov = np.array(covariances)
        for j in range(d):
            col_data = X[:, j]
            col_means = means[:, j]
            # If covariance is 3-dimensional, extract the j-th diagonal element.
            if cov.ndim == 3:
                col_cov = cov[:, j, j]
            else:
                col_cov = cov[:, j]
            # Define lower and upper bounds for the grid in this dimension.
            lower = min(col_data.min(), col_means.min()) - 3 * np.sqrt(col_cov.max())
            upper = max(col_data.max(), col_means.max()) + 3 * np.sqrt(col_cov.max())
            points = np.linspace(lower, upper, n)
            grid_list.append(points)
        return tuple(grid_list)

def plot_model_vs_gt(ws_ts, mus, covs, mus_gt_ts, vars_gt_ts, t_i, folder, combo, n_points=1000):
    """Generate 1D comparative plots between ground truth and GMM model across time points."""
    d = combo["d"]
    
    if d != 1:
        print(f"Skipping visualization for d={d} (only 1D supported)")
        return
    
    fig = plt.figure(figsize=(15, 10))
    n_rows, n_cols = 3, 4
    gs = GridSpec(n_rows, n_cols, figure=fig)
    
    for i, t in enumerate(t_i):
        ax = fig.add_subplot(gs[i // n_cols, i % n_cols])
        
        current_mus_gt = mus_gt_ts[i]
        current_var_gt = vars_gt_ts[i]
        current_weights = ws_ts[i]
        
        # Determine plot range
        all_means = np.concatenate([np.atleast_2d(current_mus_gt), np.atleast_2d(mus)])
        min_mean, max_mean = np.min(all_means), np.max(all_means)
        max_std = max(np.sqrt(current_var_gt), np.sqrt(np.max([np.max(cov) for cov in covs])))
        x_min, x_max = min_mean - 4 * max_std, max_mean + 4 * max_std
        
        x_points = np.linspace(x_min, x_max, n_points).reshape(-1, 1)
        
        # Compute and plot densities
        gt_density = compute_gt_density_mc(x_points, current_mus_gt, current_var_gt, d)
        gmm_density = compute_gmm_density_mc(x_points, current_weights, mus, covs)
        
        ax.plot(x_points, gt_density, 'r-', linewidth=2, label='Ground Truth')
        ax.plot(x_points, gmm_density, 'b--', linewidth=2, label='GMM Model')
        
        # Plot individual components
        gt_weights = np.ones(len(current_mus_gt)) / len(current_mus_gt)
        for j, mu in enumerate(current_mus_gt):
            component_density = compute_gmm_density_mc(
                x_points, [1.0], [mu], [np.eye(d) * current_var_gt]) * gt_weights[j]
            ax.plot(x_points, component_density, 'r:', alpha=0.5)
        
        for j, (w, mu, cov) in enumerate(zip(current_weights, mus, covs)):
            component_density = compute_gmm_density_mc(x_points, [1.0], [mu], [cov]) * w
            ax.plot(x_points, component_density, 'b:', alpha=0.5)
        
        ax.set_title(f't = {t:.2f}')
        if i >= (n_rows-1) * n_cols:
            ax.set_xlabel('x')
        if i % n_cols == 0:
            ax.set_ylabel('Density')
        if i == 0:
            ax.legend()
    
    fig.suptitle('Ground Truth vs GMM Model Comparison (1D)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(folder, "model_vs_ground_truth_1d.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate time evolution summary
    _plot_time_evolution(ws_ts, mus, mus_gt_ts, t_i, folder)
    print(f"1D plots saved to {folder}")

def _plot_time_evolution(ws_ts, mus, mus_gt_ts, t_i, folder):
    """Generate time evolution summary plot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Determine number of ground truth clusters from first time step
    n_clusters_gt = len(mus_gt_ts[0]) if isinstance(mus_gt_ts[0], list) else 1
    
    # Plot ground truth means evolution (first dimension)
    for cluster_idx in range(n_clusters_gt):
        cluster_means = []
        for t_idx in range(len(t_i)):  # Iterate only over actual time indices
            means_at_t = mus_gt_ts[t_idx]
            # Extract the mean for this cluster at this time
            if isinstance(means_at_t[cluster_idx], list):
                # If it's [[m1]*d, [m2]*d, [m3]*d] structure
                cluster_means.append(means_at_t[cluster_idx][0])
            else:
                # If it's a scalar
                cluster_means.append(means_at_t[cluster_idx])
        
        ax.plot(t_i, cluster_means, 'r-o', 
               label=f'GT Cluster {cluster_idx+1}' if cluster_idx == 0 else None)
    
    # Plot model means as horizontal lines
    for k, mu in enumerate(mus):
        # Handle the case where mu might be a list/array or a scalar
        mu_val = mu[0] if hasattr(mu, '__len__') else mu
        ax.axhline(y=mu_val, color='b', linestyle='--', 
                  label=f'Model Component {k+1}' if k == 0 else None)
    
    # Plot weights as scatter points
    max_weight = max([max(w) for w in ws_ts])
    for k, mu in enumerate(mus):
        # Handle the case where mu might be a list/array or a scalar
        mu_val = mu[0] if hasattr(mu, '__len__') else mu
        weight_sizes = [500 * w[k] / max_weight for w in ws_ts]
        
        ax.scatter(t_i, [mu_val] * len(t_i), s=weight_sizes, 
                  alpha=0.5, color='blue', edgecolors='black')
    
    ax.legend(loc='upper left')
    ax.set_xlabel('Time')
    ax.set_ylabel('First Dimension Mean Value')
    ax.set_title('Evolution of Means and Weights Over Time')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(os.path.join(folder, "time_evolution_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()


def participant_level_permutation_test_curves(
        treatment_ids, control_ids, root_folder, n_permutations=9999,
    seed=2025, interval=(0.0, 20.0), n_grid=101,
        include_pointwise=False, output_folder=None,
        trajectory_transform="raw"):
    """Participant-level permutation MMD on intact weight trajectories.

    The global full-trajectory contrast is primary. Component contrasts use
    Holm correction. Optional time-by-component contrasts are exploratory and
    use a maxT correction based on the same participant-label permutations.
    """
    from scipy.spatial.distance import pdist, squareform
    from Utils.IO_utils import save_json

    treatment_ids = [int(pid) for pid in treatment_ids]
    control_ids = [int(pid) for pid in control_ids]
    if len(set(treatment_ids)) != len(treatment_ids) or len(set(control_ids)) != len(control_ids):
        raise ValueError("Participant IDs must be unique within each arm")
    if set(treatment_ids) & set(control_ids):
        raise ValueError("Treatment and control IDs must be disjoint")
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    valid_transforms = {"raw", "change_from_interval_start"}
    if trajectory_transform not in valid_transforms:
        raise ValueError(
            f"trajectory_transform must be one of {sorted(valid_transforms)}"
        )

    common_grid = np.linspace(float(interval[0]), float(interval[1]), int(n_grid))
    requested = [(pid, 1) for pid in treatment_ids] + [(pid, 0) for pid in control_ids]
    included_ids, labels, trajectories = [], [], []
    excluded = []
    reference_means = reference_covs = None

    for pid, label in requested:
        arm = "treatment" if label == 1 else "control"
        patient_dir = _get_patient_dir(root_folder, pid)
        if patient_dir is None:
            excluded.append({"patient_id": pid, "arm": arm, "reason": "result folder not found"})
            continue
        result = _load_patient_results(patient_dir, include_covs=True)
        if result is None or result.get("trajectory_time_grid_weeks") is None:
            excluded.append({"patient_id": pid, "arm": arm,
                             "reason": "calendar-time NODE result not found"})
            continue

        trajectory = np.asarray(result["traj_weights"], dtype=float)
        if trajectory.ndim == 3 and trajectory.shape[1] == 1:
            trajectory = trajectory[:, 0, :]
        time_grid = np.asarray(result["trajectory_time_grid_weeks"], dtype=float)
        if (trajectory.ndim != 2 or len(time_grid) != len(trajectory) or
                np.any(np.diff(time_grid) <= 0)):
            excluded.append({"patient_id": pid, "arm": arm,
                             "reason": "invalid trajectory time grid"})
            continue
        if time_grid[0] > interval[0] or time_grid[-1] < interval[1]:
            excluded.append({"patient_id": pid, "arm": arm,
                             "reason": f"incomplete support for [{interval[0]},{interval[1]}]"})
            continue
        if (not np.all(np.isfinite(trajectory)) or np.any(trajectory < -1e-10) or
                not np.allclose(trajectory.sum(axis=1), 1.0, atol=1e-5)):
            excluded.append({"patient_id": pid, "arm": arm,
                             "reason": "invalid mixture weights"})
            continue

        means = np.asarray(result["means"], dtype=float)
        covs = np.asarray(result.get("covs"), dtype=float)
        if reference_means is None:
            reference_means, reference_covs = means, covs
        elif not (np.allclose(means, reference_means) and np.allclose(covs, reference_covs)):
            raise RuntimeError("All included participants must use the same Gaussian dictionary")

        # Interpolation occurs only after full support has been established.
        interpolated = np.column_stack([
            np.interp(common_grid, time_grid, trajectory[:, component])
            for component in range(trajectory.shape[1])
        ])
        included_ids.append(pid)
        labels.append(label)
        trajectories.append(interpolated)

    trajectories = np.asarray(trajectories, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if trajectories.ndim != 3 or len(np.unique(labels)) != 2:
        raise RuntimeError("Both groups need included participants with complete support")
    n_treatment = int(labels.sum())
    n_control = int(len(labels) - n_treatment)
    if min(n_treatment, n_control) < 2:
        raise RuntimeError("At least two included participants per group are required")

    if trajectory_transform == "change_from_interval_start":
        trajectories = trajectories - trajectories[:, :1, :]

    rng = np.random.default_rng(seed)
    permuted_labels = np.empty((n_permutations, len(labels)), dtype=bool)
    for index in range(n_permutations):
        # Draw uniformly with replacement from the complete set of assignments
        # having the observed group sizes. The trajectory rows never move.
        permuted_labels[index] = labels[rng.permutation(len(labels))].astype(bool)
    assert np.all(permuted_labels.sum(axis=1) == n_treatment)

    observed_coefficients = np.where(labels == 1, 1.0 / n_treatment, -1.0 / n_control)
    permutation_coefficients = np.where(
        permuted_labels, 1.0 / n_treatment, -1.0 / n_control
    )

    def fixed_bandwidth(observations):
        distances = pdist(np.asarray(observations, dtype=float))
        positive = distances[np.isfinite(distances) & (distances > 0)]
        return float(np.median(positive)) if len(positive) else 1.0

    def kernel_matrix(observations, sigma):
        squared = squareform(pdist(np.asarray(observations, dtype=float), metric="sqeuclidean"))
        return np.exp(-squared / (2.0 * sigma**2))

    def statistic_from_kernel(kernel, coefficients):
        return np.maximum(0.0, np.einsum(
            "...i,ij,...j->...", coefficients, kernel, coefficients,
            optimize=True
        ))

    def p_value(observed, null):
        return float((1 + np.sum(null >= observed)) / (n_permutations + 1))

    def holm(raw):
        raw = np.asarray(raw, dtype=float)
        order = np.argsort(raw)
        adjusted_sorted = np.maximum.accumulate(
            np.minimum(1.0, (len(raw) - np.arange(len(raw))) * raw[order])
        )
        adjusted = np.empty_like(raw)
        adjusted[order] = adjusted_sorted
        return adjusted

    pooled_full = trajectories.reshape(len(trajectories), -1)
    global_sigma = fixed_bandwidth(pooled_full)
    global_kernel = kernel_matrix(pooled_full, global_sigma)
    global_statistic = float(statistic_from_kernel(global_kernel, observed_coefficients))
    global_null = statistic_from_kernel(global_kernel, permutation_coefficients)
    global_p = p_value(global_statistic, global_null)

    component_results = []
    component_raw_p = []
    for component in range(trajectories.shape[2]):
        pooled_component = trajectories[:, :, component]
        sigma = fixed_bandwidth(pooled_component)
        kernel = kernel_matrix(pooled_component, sigma)
        observed = float(statistic_from_kernel(kernel, observed_coefficients))
        null = statistic_from_kernel(kernel, permutation_coefficients)
        raw_p = p_value(observed, null)
        component_raw_p.append(raw_p)
        component_results.append({
            "component": component + 1,
            "statistic": observed,
            "bandwidth": sigma,
            "p_value_raw": raw_p
        })
    component_holm = holm(component_raw_p)
    for result, adjusted in zip(component_results, component_holm):
        result["p_value_holm"] = float(adjusted)

    pointwise = None
    if include_pointwise:
        shape = (len(common_grid), trajectories.shape[2])
        observed_pointwise = np.empty(shape)
        raw_pointwise = np.empty(shape)
        bandwidth_pointwise = np.empty(shape)
        max_null = np.full(n_permutations, -np.inf)
        for time_index in range(shape[0]):
            for component in range(shape[1]):
                observations = trajectories[:, time_index, component][:, None]
                sigma = fixed_bandwidth(observations)
                kernel = kernel_matrix(observations, sigma)
                observed = float(statistic_from_kernel(kernel, observed_coefficients))
                null = statistic_from_kernel(kernel, permutation_coefficients)
                observed_pointwise[time_index, component] = observed
                raw_pointwise[time_index, component] = p_value(observed, null)
                bandwidth_pointwise[time_index, component] = sigma
                max_null = np.maximum(max_null, null)
        maxT = np.empty(shape)
        for time_index in range(shape[0]):
            for component in range(shape[1]):
                observed = observed_pointwise[time_index, component]
                maxT[time_index, component] = p_value(observed, max_null)
        if np.any(maxT + 1e-15 < raw_pointwise):
            raise RuntimeError("maxT p-values cannot be smaller than raw p-values")
        pointwise = {
            "analysis_role": "secondary/exploratory",
            "statistics": observed_pointwise.tolist(),
            "bandwidths": bandwidth_pointwise.tolist(),
            "p_values_raw": raw_pointwise.tolist(),
            "p_values_maxT": maxT.tolist(),
            "significant_maxT_0.05": (maxT < 0.05).tolist()
        }

    result = {
        "method": "participant-level permutation test",
        "permutation_scheme": ("unstratified permutation of whole-participant labels; "
                               "group sizes fixed; Monte Carlo draws with replacement from "
                               "the complete assignment space"),
        "primary_global_test": {
            "estimand": (
                "raw full trajectory"
                if trajectory_transform == "raw"
                else "full trajectory of within-participant changes from interval start"
            ),
            "mmd_v_statistic": global_statistic,
            "p_value": global_p,
            "bandwidth": global_sigma
        },
        "component_tests": component_results,
        "pointwise_tests": pointwise,
        "n_permutations": int(n_permutations),
        "seed": int(seed),
        "trajectory_transform": trajectory_transform,
        "p_value_formula": "(1 + number of permuted statistics >= observed) / (B + 1)",
        "common_time_grid_weeks": common_grid.tolist(),
        "included_subject_ids": included_ids,
        "included_labels": labels.tolist(),
        "strata": None,
        "stratification_used": False,
        "included_counts": {"treatment": n_treatment, "control": n_control},
        "excluded_subjects": excluded,
        "requested_treatment_ids": treatment_ids,
        "requested_control_ids": control_ids,
        "global_null_statistics": global_null.tolist()
    }

    if output_folder is not None:
        suffix = ("raw" if trajectory_transform == "raw" else "change")
        save_json(
            result, output_folder,
            f"participant_permutation_mmd_{suffix}.json"
        )
        if pointwise is not None:
            adjusted = np.asarray(pointwise["p_values_maxT"])
            fig, axes = plt.subplots(trajectories.shape[2], 1, figsize=(9, 2.5 * trajectories.shape[2]),
                                     sharex=True, squeeze=False)
            for component, axis in enumerate(axes[:, 0]):
                axis.plot(common_grid, adjusted[:, component], marker="o", markersize=2)
                axis.axhline(0.05, color="black", linestyle="--", linewidth=1)
                axis.set_ylabel(f"Comp. {component + 1}\nmaxT p")
                axis.set_ylim(0, 1)
            axes[-1, 0].set_xlabel("Elapsed calendar time (weeks)")
            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    output_folder,
                    f"participant_permutation_pointwise_maxT_{suffix}.png"
                ),
                dpi=300
            )
            plt.close(fig)

    print(f"Primary global MMD V-statistic: {global_statistic:.6g}; p={global_p:.6g}")
    return result


def mmd_permutation_test_curves(*args, **kwargs):
    """Deprecated compatibility name for the participant-level procedure."""
    import warnings
    warnings.warn(
        "Use participant_level_permutation_test_curves; the participant-level "
        "permutation test is now the active clinical procedure.",
        DeprecationWarning, stacklevel=2
    )
    return participant_level_permutation_test_curves(*args, **kwargs)


def compute_frechet_means(treatment_ids, control_ids, root_folder):
    """
    Compute Fréchet means (= arithmetic mean) for Treatment and Control groups
    at initial, middle, and final time points. Prints summary table.
    """
    def load_aligned(patient_ids):
        results = []
        for pid in patient_ids:
            patient_dir = _get_patient_dir(root_folder, pid)
            if patient_dir is None:
                continue
            res = _load_patient_results(patient_dir)
            if res is None:
                continue
            results.append(res)
        if not results:
            raise RuntimeError("No valid results found.")
        ref_means = np.array(results[0]["means"])
        if ref_means.ndim == 1:
            ref_means = ref_means[:, None]
        K = ref_means.shape[0]
        aligned = []
        for res in results:
            means = np.array(res["means"])
            if means.ndim == 1:
                means = means[:, None]
            traj = np.array(res["traj_weights"])
            if traj.ndim == 3 and traj.shape[1] == 1:
                traj = traj[:, 0, :]
            if traj.ndim != 2 or traj.shape[1] != K:
                continue
            if means.shape[1] == 1:
                order = np.argsort(means.flatten())
            else:
                cost = np.linalg.norm(ref_means[:, None, :] - means[None, :, :], axis=2)
                _, order = linear_sum_assignment(cost)
            aligned.append(traj[:, order])
        return np.stack(aligned)

    traj_t = load_aligned(treatment_ids)
    traj_c = load_aligned(control_ids)
    T, K = traj_t.shape[1], traj_t.shape[2]
    
    t_idx = {'initial': 0, 'middle': T // 2, 'final': T - 1}
    frechet = {'Treatment': {}, 'Control': {}}
    
    for name, idx in t_idx.items():
        frechet['Treatment'][name] = np.mean(traj_t[:, idx, :], axis=0)
        frechet['Control'][name] = np.mean(traj_c[:, idx, :], axis=0)
    
    print("\n" + "="*70)
    print("FRÉCHET MEANS")
    print("="*70)
    print(f"{'Time':<10} {'Group':<12} " + " ".join([f"Comp {k+1:>6}" for k in range(K)]))
    print("-"*70)
    for name in ['initial', 'middle', 'final']:
        for grp in ['Treatment', 'Control']:
            vals = " ".join([f"{v:>8.4f}" for v in frechet[grp][name]])
            print(f"{name:<10} {grp:<12} {vals}")
        print("-"*70)
    
    return frechet


def plot_density_initial_final(treatment_ids, control_ids, root_folder, d=1, output_folder=None):
    """
    Plot predicted density distributions at initial and final time points,
    comparing Treatment vs Control groups with uncertainty bands.
    
    The bands represent variability ACROSS PATIENTS within each group.
    """
    def load_patient_gmm(patient_ids):
        """Load GMM parameters for all patients."""
        patients_data = []
        for pid in patient_ids:
            patient_dir = _get_patient_dir(root_folder, pid)
            if patient_dir is None:
                continue
            res = _load_patient_results(patient_dir, include_covs=True)
            if res is None or "covs" not in res:
                continue
            
            means = np.array(res["means"])
            covs = np.array(res["covs"])
            traj = np.array(res["traj_weights"])
            
            if covs.ndim == 3 and covs.shape[2] == 1:
                K = covs.shape[0]
                covs = np.array([np.diag(covs[k].flatten()) for k in range(K)])
            if traj.ndim == 3:
                traj = traj.squeeze(axis=1)
            
            patients_data.append({
                'means': means,
                'covs': covs,
                'weights_initial': traj[0],
                'weights_final': traj[-1]
            })
        return patients_data
    
    def compute_density_1d(x_grid, means, covs, weights):
        """Compute 1D GMM density."""
        density = np.zeros_like(x_grid)
        for k in range(len(weights)):
            mu = means[k, 0] if means.ndim > 1 else means[k]
            var = covs[k, 0, 0] if covs.ndim > 2 else covs[k, 0] if covs.ndim > 1 else covs[k]
            sigma = np.sqrt(var)
            density += weights[k] * (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_grid - mu) / sigma) ** 2)
        return density
    
    # Load data
    treat_data = load_patient_gmm(treatment_ids)
    control_data = load_patient_gmm(control_ids)
    
    if not treat_data or not control_data:
        print("No data found for one or both groups")
        return
    
    # Determine grid range from all patients
    all_means = np.vstack([p['means'] for p in treat_data + control_data])
    all_covs = np.vstack([p['covs'] for p in treat_data + control_data])
    save_dir = output_folder if output_folder is not None else root_folder
    os.makedirs(save_dir, exist_ok=True)
    
    if d == 1:
        mu_min, mu_max = all_means[:, 0].min(), all_means[:, 0].max()
        std_max = np.sqrt(all_covs[:, 0, 0].max())
        x_grid = np.linspace(mu_min - 4*std_max, mu_max + 4*std_max, 500)
        
        # Compute densities for each patient at initial and final
        colors = {"Treatment": "#588157", "Control": "#ad2831"}
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
        
        for ax, time_label, weight_key in zip(axes, ["Initial", "Final"], 
                                               ["weights_initial", "weights_final"]):
            for group_name, group_data, color in [("Treatment", treat_data, colors["Treatment"]),
                                                   ("Control", control_data, colors["Control"])]:
                # Compute density for each patient
                densities = []
                for p in group_data:
                    dens = compute_density_1d(x_grid, p['means'], p['covs'], p[weight_key])
                    densities.append(dens)
                
                densities = np.array(densities)
                mean_dens = densities.mean(axis=0)
                std_dens = densities.std(axis=0)
                
                # Plot mean and band
                ax.plot(x_grid, mean_dens, color=color, lw=2, label=f'{group_name} (n={len(group_data)})')
                ax.fill_between(x_grid, mean_dens - std_dens, mean_dens + std_dens, 
                               color=color, alpha=0.3)
            
            ax.set_xlabel('Glucose (mg/dL)', fontsize=14)
            ax.set_ylabel('Density', fontsize=14)
            ax.set_title(f'{time_label} Distribution', fontsize=16)
            ax.legend(fontsize=12)
            ax.grid(alpha=0.3)
        
        plt.suptitle('Predicted Density: Treatment vs Control', fontsize=18, y=1.02)
        plt.tight_layout()
        
        out_file = os.path.join(save_dir, "density_initial_final_comparison.png")
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {out_file}")
    
    elif d == 2:
        # 2D case: contour plots
        mu_x_min, mu_x_max = all_means[:, 0].min(), all_means[:, 0].max()
        mu_y_min, mu_y_max = all_means[:, 1].min(), all_means[:, 1].max()
        std_x = np.sqrt(all_covs[:, 0, 0].max())
        std_y = np.sqrt(all_covs[:, 1, 1].max())
        
        x_grid = np.linspace(mu_x_min - 3*std_x, mu_x_max + 3*std_x, 100)
        y_grid = np.linspace(mu_y_min - 3*std_y, mu_y_max + 3*std_y, 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        pos = np.dstack((X, Y))
        
        def compute_density_2d(pos, means, covs, weights):
            """Compute 2D GMM density on grid."""
            Z = np.zeros(pos.shape[:2])
            for k in range(len(weights)):
                Z += weights[k] * multivariate_normal(means[k], covs[k]).pdf(pos)
            return Z
        
        colors = {"Treatment": "#588157", "Control": "#ad2831"}
        cmaps = {"Treatment": "Greens", "Control": "Reds"}
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=150)
        
        for col, (time_label, weight_key) in enumerate([("Initial", "weights_initial"), 
                                                         ("Final", "weights_final")]):
            for row, (group_name, group_data) in enumerate([("Treatment", treat_data),
                                                             ("Control", control_data)]):
                ax = axes[row, col]
                
                # Compute mean density across patients
                densities = []
                for p in group_data:
                    dens = compute_density_2d(pos, p['means'], p['covs'], p[weight_key])
                    densities.append(dens)
                
                mean_dens = np.mean(densities, axis=0)
                
                # Plot contour
                im = ax.contourf(X, Y, mean_dens, levels=15, cmap=cmaps[group_name], alpha=0.8)
                plt.colorbar(im, ax=ax, label='Density')
                
                ax.set_xlabel('Glucose (mg/dL)', fontsize=12)
                ax.set_ylabel('Glucose Derivative', fontsize=12)
                ax.set_title(f'{group_name} - {time_label} (n={len(group_data)})', fontsize=14)
                ax.grid(alpha=0.3)
        
        plt.suptitle('Predicted Density: Treatment vs Control', fontsize=18)
        plt.tight_layout()
        
        out_file = os.path.join(save_dir, "density_initial_final_comparison.png")
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {out_file}")
    
    else:
        print(f"plot_density_initial_final supports d=1 or d=2, got d={d}")


def plot_timesignal_groups(treatment_ids, control_ids, cgm_csv, root_folder, n_patients=5, seed=42, output_folder=None):
    """
    Plot CGM time signals for sample patients from Treatment and Control groups.
    All plots use the same x and y axis scales for comparability.
    """
    import pandas as pd
    np.random.seed(seed)
    
    cgm = pd.read_csv(cgm_csv)
    time_cols = [c for c in cgm.columns if c not in ['PtID', 'date', 'n_obs', 'n_slots', 'n_missing', 'prop_missing']]
    
    # Sample patients from both groups
    samples = {}
    for group_name, patient_ids in [("Treatment", treatment_ids), ("Control", control_ids)]:
        available = [pid for pid in patient_ids if pid in cgm['PtID'].unique()]
        samples[group_name] = np.random.choice(available, min(n_patients, len(available)), replace=False)
    
    # Compute global axis limits across all sampled patients
    all_pids = list(samples["Treatment"]) + list(samples["Control"])
    y_min, y_max = np.inf, -np.inf
    x_max = 0
    
    for pid in all_pids:
        patient_data = cgm[cgm['PtID'] == pid][time_cols].values
        for row in patient_data:
            row_numeric = pd.to_numeric(pd.Series(row), errors='coerce').values
            valid = row_numeric[~np.isnan(row_numeric)]
            if len(valid) > 0:
                y_min = min(y_min, np.min(valid))
                y_max = max(y_max, np.max(valid))
        x_max = max(x_max, len(time_cols))
    
    # Add margin to y limits
    y_margin = 0.05 * (y_max - y_min)
    y_min, y_max = max(0, y_min - y_margin), y_max + y_margin
    
    save_dir = output_folder if output_folder is not None else root_folder
    os.makedirs(save_dir, exist_ok=True)

    # Plot with consistent scales
    for group_name, color in [("Treatment", "#588157"), ("Control", "#ad2831")]:
        for pid in samples[group_name]:
            patient_data = cgm[cgm['PtID'] == pid][time_cols].values
            
            fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
            for row in patient_data:
                row_numeric = pd.to_numeric(pd.Series(row), errors='coerce').values
                ax.plot(row_numeric, color=color, alpha=0.3, lw=0.6)
            
            ax.set_xlabel("Time slot (5-min intervals)", fontsize=14)
            ax.set_ylabel("Glucose (mg/dL)", fontsize=14)
            ax.set_title(f"Patient {pid} ({group_name}) - {len(patient_data)} days", fontsize=16)
            ax.tick_params(labelsize=12)
            ax.grid(alpha=0.3)
            ax.set_xlim(0, x_max)
            ax.set_ylim(y_min, y_max)
            plt.tight_layout()
            
            out_file = os.path.join(save_dir, f"timesignal_{group_name}_patient{pid}.png")
            fig.savefig(out_file)
            plt.close(fig)
            print(f"Saved {out_file}")


def plot_gmm_vs_empirical_1d(means, covs, weights, empirical_data, title="", save_path=None):
    """
    Plot 1D GMM prediction. Generates two plots:
    - One with empirical histogram
    - One without (GMM only)
    """
    means = np.atleast_2d(np.asarray(means, dtype=float))
    covs = np.asarray(covs, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    empirical_data = np.asarray(empirical_data, dtype=float)

    # Normalise shapes
    if covs.ndim == 1:
        covs = covs[:, None, None]
    elif covs.ndim == 2 and covs.shape[1] == 1:
        covs = covs[:, :, None]
    K = means.shape[0]
    if weights.size > K:
        weights = weights[:K]
    elif weights.size < K:
        pad = np.zeros(K)
        pad[:weights.size] = weights
        weights = pad

    mu_min, mu_max = means.min(), means.max()
    std_max = np.sqrt(np.maximum(covs.max(), 1e-12))
    data_min, data_max = empirical_data.min(), empirical_data.max()
    
    x_min = min(mu_min - 3*std_max, data_min)
    x_max = max(mu_max + 3*std_max, data_max)
    x = np.linspace(x_min, x_max, 1000)
    
    gmm_density = np.zeros_like(x)
    for k in range(K):
        mu, sigma, w = means[k, 0], np.sqrt(covs[k, 0, 0] + 1e-9), weights[k]
        gmm_density += w * (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
    # Filter zeros from empirical data (CGM sensor errors)
    empirical_data_clean = empirical_data[empirical_data != 0]
    
    # Plot WITH empirical data
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.hist(empirical_data_clean.flatten(), bins=50, density=True, alpha=0.5, 
            color='steelblue', edgecolor='darkblue', linewidth=0.8, label='Empirical data')
    ax.plot(x, gmm_density, '-', color='darkorange', linewidth=3.5, label='GMM Prediction', alpha=0.95)
    ax.fill_between(x, gmm_density, alpha=0.25, color='orange')
    ax.set_xlabel('Glucose (mg/dL)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot WITHOUT empirical data (GMM only)
        fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=100)
        ax2.plot(x, gmm_density, '-', color='darkorange', linewidth=3.5, label='GMM Prediction', alpha=0.95)
        ax2.fill_between(x, gmm_density, alpha=0.25, color='orange')
        ax2.set_xlabel('Glucose (mg/dL)', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title(title + " (GMM only)", fontsize=13, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)
        ax2.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # Save with different name
        base, ext = os.path.splitext(save_path)
        plt.savefig(f"{base}_gmm_only{ext}", dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_gmm_vs_empirical_2d(means, covs, weights, empirical_data, title="", save_path=None):
    """
    Plot 2D GMM prediction as contour. Generates two plots:
    - One with empirical scatter
    - One without (GMM only)
    """
    means = np.asarray(means, dtype=float)
    covs = np.asarray(covs, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    empirical_data = np.asarray(empirical_data, dtype=float)

    # Project to first two dimensions if necessary
    if means.ndim == 1:
        means = means[:, None]
    if means.shape[1] > 2:
        means = means[:, :2]
    if covs.ndim == 3 and covs.shape[1] > 2:
        covs = covs[:, :2, :2]
    if empirical_data.ndim > 1 and empirical_data.shape[1] > 2:
        empirical_data = empirical_data[:, :2]

    # Regularise covariances slightly to avoid singular matrices
    if covs.ndim == 3:
        jitter = 1e-6 * np.eye(2)
        covs = covs + jitter

    K = means.shape[0]
    if weights.size > K:
        weights = weights[:K]
    elif weights.size < K:
        pad = np.zeros(K)
        pad[:weights.size] = weights
        weights = pad

    mu_x_min, mu_x_max = means[:, 0].min(), means[:, 0].max()
    std_x = np.sqrt(np.maximum(covs[:, 0, 0].max(), 1e-12))
    data_x_min, data_x_max = empirical_data[:, 0].min(), empirical_data[:, 0].max()
    mu_y_min, mu_y_max = means[:, 1].min(), means[:, 1].max()
    std_y = np.sqrt(np.maximum(covs[:, 1, 1].max(), 1e-12))
    data_y_min, data_y_max = empirical_data[:, 1].min(), empirical_data[:, 1].max()
    
    x_min, x_max = min(mu_x_min - 3*std_x, data_x_min), max(mu_x_max + 3*std_x, data_x_max)
    y_min, y_max = min(mu_y_min - 3*std_y, data_y_min), max(mu_y_max + 3*std_y, data_y_max)
    
    x, y = np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    Z = np.zeros_like(X)
    for k in range(K):
        Z += weights[k] * multivariate_normal(means[k], covs[k], allow_singular=True).pdf(pos)
    
    # Plot WITH empirical data
    fig, ax = plt.subplots(figsize=(11, 8), dpi=100)
    im = ax.contourf(X, Y, Z, levels=15, cmap='Oranges', alpha=0.85)
    plt.colorbar(im, ax=ax, label='Density')
    ax.scatter(empirical_data[:, 0], empirical_data[:, 1], s=4, alpha=0.35, 
               color='black', label='Empirical data', rasterized=True)
    ax.set_xlabel('Glucose (mg/dL)', fontsize=12)
    ax.set_ylabel('Glucose Derivative (mg/dL/min)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot WITHOUT empirical data (GMM only)
        fig2, ax2 = plt.subplots(figsize=(11, 8), dpi=100)
        im2 = ax2.contourf(X, Y, Z, levels=15, cmap='Oranges', alpha=0.85)
        plt.colorbar(im2, ax=ax2, label='Density')
        ax2.set_xlabel('Glucose (mg/dL)', fontsize=12)
        ax2.set_ylabel('Glucose Derivative (mg/dL/min)', fontsize=12)
        ax2.set_title(title + " (GMM only)", fontsize=13, fontweight='bold')
        ax2.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        base, ext = os.path.splitext(save_path)
        plt.savefig(f"{base}_gmm_only{ext}", dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
