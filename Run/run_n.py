"""
Neural ODE Runner

Trains Neural ODE to model weight trajectory evolution.
Loads GMM parameters from result_fit.json and saves trajectory to result_node.json.

Key improvements:
- Clean JSON output: only NODE-specific results
- References result_fit.json for GMM parameters (no duplication)
- Proper handling of fixed vs fitted parameters
"""

import os
import numpy as np
import torch
from torch import optim
from typing import Dict, Any, List, Tuple, Optional

from Utils.IO_utils import load_experiment, save_json
from Visualization.gifs import gif, gif_error
from Visualization.plots import plot_training, plot_traj, plot_sumweights
from Utils.NODE_utils import build_time_grids, time_transformation


def _validate_component_identity(combo, means, covariances):
    """Ensure the fitted dictionary reaching the NODE retains canonical labels."""
    dictionary = combo.get("dictionary", {})
    if not dictionary or "means" not in dictionary or "covariances" not in dictionary:
        return
    source_means = np.asarray(dictionary["means"], dtype=float)
    source_covariances = np.asarray(dictionary["covariances"], dtype=float)
    means = np.asarray(means, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    if source_means.shape != means.shape or source_covariances.shape != covariances.shape:
        raise ValueError("Fitted and global dictionaries have incompatible dimensions")
    order = np.argsort(source_means[:, 0], kind="stable")
    if (not np.allclose(means, source_means[order]) or
            not np.allclose(covariances, source_covariances[order])):
        raise ValueError(
            "Gaussian component identity changed between the global dictionary "
            "and NODE training"
        )


def run_node(combo: Dict[str, Any], data: List[np.ndarray],
             best_means: np.ndarray, best_covs: np.ndarray,
             ts_weights: List[np.ndarray], combo_folder: str,
             options: Dict[str, Any]) -> Tuple[np.ndarray, str]:
    """
    Train or load Neural ODE for weight trajectory prediction.
    
    Args:
        combo: Experiment parameters
        data: List of datasets (one per time instant)
        best_means: GMM means from fitting, shape (K, d)
        best_covs: GMM covariances from fitting, shape (K, d, d)
        ts_weights: Fitted weights per time instant, list of (K,) arrays
        combo_folder: Parent folder for results
        options: Runtime options
        
    Returns:
        Tuple of (predicted_trajectory, node_folder_path)
    """
    from Utils.IO_utils import check_nodefolder
    combo = dict(combo)
    K = combo["K"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(combo["seed_node"])

    _validate_component_identity(combo, best_means, best_covs)

    weights_array = np.stack(ts_weights).astype(float)
    if weights_array.shape != (int(combo["t_steps"]), int(K)):
        raise ValueError("Fitted weights must have shape (t_steps, K)")
    if (not np.all(np.isfinite(weights_array)) or np.any(weights_array < -1e-10) or
            not np.allclose(weights_array.sum(axis=1), 1.0, atol=1e-6)):
        raise ValueError("Fitted weights must be finite, non-negative and sum to one")
    combo["observation_weights"] = weights_array.tolist()
    combo["dictionary_means"] = np.asarray(best_means, dtype=float).tolist()
    combo["dictionary_covariances"] = np.asarray(best_covs, dtype=float).tolist()

    # The cache identity includes the observations and frozen dictionary.
    node_folder, cached = check_nodefolder(combo_folder, combo, options)

    # Prepare observation tensor: (T, 1, K)
    obs_w = torch.tensor(
        weights_array, dtype=torch.float32, device=device
    ).unsqueeze(1)
    
    t_i, t_grid, real_times, real_grid = build_time_grids(combo, device=device)
    
    # Train or load
    if cached is not None:
        print(">> Using cached NODE trajectory")
        pred_traj = np.asarray(cached["trajectory"])
        loss_history = cached.get("loss_history", None)  # Load from cache
        training_diagnostics = cached.get("training_diagnostics")
    else:
        pred_traj, loss_history, training_diagnostics = _train_node(
            combo, obs_w, t_i, t_grid, device
        )

        # Save results
        _save_node_results(
            node_folder, combo, pred_traj, loss_history, training_diagnostics,
            t_i.detach().cpu().numpy(), t_grid.detach().cpu().numpy(),
            real_times, real_grid
        )
    
    # Generate diagnostic plots
    if options.get("plots", True):
        _generate_plots(
            pred_traj, obs_w,
            torch.as_tensor(real_grid, dtype=torch.float32, device=device),
            torch.as_tensor(real_times, dtype=torch.float32, device=device),
            K, node_folder,
            combo, options, data, best_means, best_covs, loss_history
        )
    
    # Return trajectory without batch dimension: (T_grid, K)
    pred_np = pred_traj[:, 0, :] if pred_traj.ndim == 3 else pred_traj
    
    return pred_np, node_folder


def _train_node(combo: Dict[str, Any], obs_w: torch.Tensor,
                t_i: torch.Tensor, t_grid: torch.Tensor,
                device: torch.device) -> Tuple[np.ndarray, List[float], Dict[str, Any]]:
    """Train Neural ODE model."""
    from Core.train_node import Trainer
    
    K = combo["K"]
    normalization = combo.get("normalization", "softmax")
    from Core.node import NeuralODE
    model = NeuralODE(
        K, combo["hidden_dim"],
        non_linearity=combo["non_linearity"],
        N_layers=combo["N_layers"],
        T=combo["T"],
        step_size=combo["dt"],
        method=combo["method"],
        normalization=normalization,
        softmax_epsilon=combo.get("softmax_epsilon", 1e-8),
        solver_rtol=combo.get("solver_rtol", 1e-7),
        solver_atol=combo.get("solver_atol", 1e-9),
        time_fourier_order=int(combo.get("time_fourier_order", 0)),
        time_polynomial_degree=int(combo.get("time_polynomial_degree", 1)),
    ).to(device)
    trainable_initial = bool(combo.get("trainable_initial", True))
    model.configure_initial_state(obs_w[0], trainable=trainable_initial)
    print(f">> Using Neural ODE with {normalization} normalization")
    print(f">> Trainable initial state: {trainable_initial}")
    
    # Setup optimizer
    opt = optim.Adam(model.parameters(), lr=combo["lr"])
    
    # Setup trainer
    trainer = Trainer(
        model, opt, device,
        loss_str=combo["loss_func"],
        print_freq=combo["print_freq"],
        max_epochs=combo["max_epochs"],
        early_stopping_min_delta=combo.get("early_stopping_min_delta", 1e-5),
        early_stopping_patience=combo.get("early_stopping_patience", 200)
    )
    
    # Train
    trainer.train(t_i, obs_w, obs_w[0])
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        pred_traj = model(obs_w[0], eval_times=t_grid).cpu().numpy()
    
    diagnostics = {
        "epochs_run": trainer.epochs_run,
        "best_epoch": trainer.best_epoch,
        "best_objective": trainer.best_loss,
        "stop_reason": trainer.stop_reason,
        "stop_reason_is_inferred": False,
        "observed_initial_weights": obs_w[0].detach().cpu().numpy().tolist(),
        "fitted_initial_weights": model.fitted_initial_weights().detach().cpu().numpy().tolist(),
        "fitted_initial_latent_state": model.initial_latent_state().detach().cpu().numpy().tolist(),
    }
    return pred_traj, trainer.loss_history, diagnostics


def _save_node_results(node_folder: str, combo: Dict[str, Any],
                       pred_traj: np.ndarray, loss_history: Optional[List[float]],
                       training_diagnostics: Optional[Dict[str, Any]],
                       observation_times_normalized,
                       trajectory_time_grid_normalized,
                       observation_times_weeks, trajectory_time_grid_weeks) -> None:
    """
    Save NODE results to JSON.
    
    Clean format - only NODE-specific data:
    {
        "node_config": { ... NODE parameters ... },
        "trajectory": [...],
        "loss_history": [...]
    }
    """
    # NODE-specific config
    node_config = {
        "T": combo["T"],
        "dt": combo["dt"],
        "seed_node": combo["seed_node"],
        "hidden_dim": combo["hidden_dim"],
        "non_linearity": combo["non_linearity"],
        "N_layers": combo["N_layers"],
        "method": combo["method"],
        "normalization": combo.get("normalization", "softmax"),
        "time_fourier_order": int(combo.get("time_fourier_order", 0)),
        "time_polynomial_degree": int(combo.get("time_polynomial_degree", 1)),
        "trainable_initial": bool(combo.get("trainable_initial", True)),
        "softmax_epsilon": combo.get("softmax_epsilon", 1e-8),
        "solver_rtol": combo.get("solver_rtol", 1e-7),
        "solver_atol": combo.get("solver_atol", 1e-9),
        "early_stopping_min_delta": combo.get("early_stopping_min_delta", 1e-5),
        "early_stopping_patience": combo.get("early_stopping_patience", 200),
        "lr": combo["lr"],
        "loss_func": combo["loss_func"],
        "max_epochs": combo["max_epochs"],
        "dataset_fingerprint": combo.get("dataset_fingerprint"),
        "dataset_version": combo.get("dataset_version"),
        "analysis_window_definition": combo.get("analysis_window_definition"),
        "nominal_week_indices": combo.get("nominal_week_indices"),
        "observed_support_weeks": combo.get("observed_support_weeks"),
        "treatment_group": combo.get("treatment_group"),
        "analysis_end_source": combo.get("analysis_end_source"),
        "is_dropout_before_week13": combo.get("is_dropout_before_week13"),
        "observation_weights": combo.get("observation_weights", []),
        "dictionary_means": combo.get("dictionary_means", []),
        "dictionary_covariances": combo.get("dictionary_covariances", []),
        "observation_times_weeks": np.asarray(observation_times_weeks).tolist()
    }
    
    result = {
        "node_config": node_config,
        "trajectory": pred_traj.tolist(),
        "loss_history": loss_history if loss_history else [],
        "training_diagnostics": training_diagnostics,
        "observation_times_normalized": np.asarray(observation_times_normalized).tolist(),
        "trajectory_time_grid_normalized": np.asarray(trajectory_time_grid_normalized).tolist(),
        "observation_times_weeks": np.asarray(observation_times_weeks).tolist(),
        "trajectory_time_grid_weeks": np.asarray(trajectory_time_grid_weeks).tolist(),
        "temporal_support_weeks": [float(observation_times_weeks[0]),
                                   float(observation_times_weeks[-1])],
        "time_transformation": time_transformation(combo),
        "nominal_week_indices": combo.get("nominal_week_indices"),
        "omitted_blocks": combo.get("omitted_blocks", []),
        "block_metadata": combo.get("block_metadata", [])
    }
    
    save_json(result, node_folder, "result_node.json")


def _generate_plots(pred_traj: np.ndarray, obs_w: torch.Tensor,
                    t_grid: torch.Tensor, t_i: torch.Tensor,
                    K: int, node_folder: str, combo: Dict[str, Any],
                    options: Dict[str, Any], data: List[np.ndarray],
                    best_means: np.ndarray, best_covs: np.ndarray,
                    loss_history: Optional[List[float]]) -> None:
    """Generate diagnostic plots."""
    pred_np = pred_traj[:, 0, :] if pred_traj.ndim == 3 else pred_traj
    t_grid_np = t_grid.cpu().numpy()
    t_i_np = t_i.cpu().numpy()
    obs_np = obs_w.cpu().numpy()[:, 0, :]
    
    # Loss history
    if loss_history:
        plot_training(loss_history, folder=node_folder)
    
    # Trajectory plot
    plot_traj(t_grid_np, t_i_np, pred_np, obs_np, K, folder=node_folder)
    
    # Sum of weights (sanity check)
    plot_sumweights(t_grid_np, pred_np, folder=node_folder,
                    obs_weights=obs_np, obs_times=t_i_np)
    
    # Optional GIF. For bivariate CGM this displays the first-coordinate
    # (glucose) marginal; the weight panel still contains all K components.
    if options.get("gif", False) and combo.get("d", 1) in {1, 2}:
        if options.get("load_data", False):
            # Real data: use histogram background
            gif_filename = ("gmm_evolution.gif" if combo.get("d", 1) == 1
                            else "gmm_glucose_marginal_evolution.gif")
            gif(pred_np, [best_means]*len(pred_np), [best_covs]*len(pred_np),
                history_X=data, history_X_times=t_i_np,
                bg_mode="histogram", evol_weights="path",
                t_grid=t_grid_np,
                fps=min(30, combo["T"] / combo["dt"]),
                folder=node_folder, filename=gif_filename)
        else:
            # Synthetic data: use ground truth
            from data import get_distrparams
            means_gt, vars_gt = get_distrparams(combo, combo["T"], len(t_grid_np))
            gif(pred_np, [best_means]*len(pred_np), [best_covs]*len(pred_np),
                means_gt=means_gt, vars_gt=vars_gt, bg_mode="gaussian",
                t_grid=t_grid_np,
                fps=min(30, combo["T"] / combo["dt"]),
                folder=node_folder, filename="gmm_evolution.gif")

    # Optional Pointwise Error GIF (only for synthetic d=1)
    if options.get("gif_error", False) and combo.get("d", 1) == 1:
        if not options.get("load_data", False):
            from data import get_distrparams
            means_gt, vars_gt = get_distrparams(combo, combo["T"], len(t_grid_np))
            gif_error(pred_np, best_means, best_covs,
                      means_gt=means_gt, vars_gt=vars_gt,
                      t_grid=t_grid_np,
                      fps=min(30, combo["T"] / combo["dt"]),
                      folder=node_folder, filename="error_evolution.gif")
