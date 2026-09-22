import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def init_weights(m, activation_type):
    """Initialize weights based on activation type."""
    if isinstance(m, nn.Linear):
        if activation_type.lower() in ['relu', 'leakyrelu']:
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        else:
            nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def normalize(out, eps=1e-8):
    """
    L1 normalization: ReLU + division.
    
    Properties:
    - If input already positive and sums to 1, acts as approximate identity
    - Can lose gradient information if components go negative
    """
    out = F.relu(out)
    sums = out.sum(dim=-1, keepdim=True)
    
    # Handle edge case: if sum is too small, use uniform distribution
    mask_zero = (sums < eps)
    safe_sums = torch.where(mask_zero, torch.ones_like(sums), sums)
    out_normalized = out / safe_sums
    uniform = torch.ones_like(out_normalized) / out.shape[-1]
    out_final = torch.where(mask_zero, uniform, out_normalized)
    
    return out_final


def softmax_normalize(out):
    """
    Softmax normalization.
    
    Properties:
    - Always produces valid probabilities
    - Better gradient flow than ReLU + division
    - Changes values even if they already sum to 1
    """
    return F.softmax(out, dim=-1)


def softplus_normalize(out, eps=1e-8):
    """Smooth positive normalization based on ``softplus``."""
    positive = F.softplus(out) + eps
    return positive / positive.sum(dim=-1, keepdim=True)


def apply_normalization(out, method='softmax', eps=1e-8):
    """
    Apply the specified normalization method.
    
    Parameters:
        out: Tensor to normalize (..., K)
        method: One of 'softmax', 'l1' or 'softplus'
    
    Returns:
        Normalized tensor
    """
    if method == 'l1':
        return normalize(out, eps=eps)
    elif method == 'softmax':
        return softmax_normalize(out)
    elif method == 'softplus':
        return softplus_normalize(out, eps=eps)
    else:
        raise ValueError("normalization must be 'softmax', 'l1' or 'softplus'")


def build_time_grids(combo, device=None):
    """Build NODE times while preserving the observed calendar spacing."""
    T = float(combo["T"])
    dt = float(combo["dt"])
    if not np.isfinite(T) or not np.isfinite(dt) or T <= 0 or dt <= 0:
        raise ValueError("NODE horizon and step size must be positive and finite")
    n_grid = int(round(T / dt)) + 1
    real_times = combo.get("observation_times_weeks")

    if real_times is not None:
        real_times = np.asarray(real_times, dtype=float)
        if not np.all(np.isfinite(real_times)):
            raise ValueError("Clinical block-centre times must be finite")
        if len(real_times) != int(combo["t_steps"]):
            raise ValueError("Calendar times and fitted weight windows are misaligned")
        if len(real_times) < 2 or np.any(np.diff(real_times) <= 0):
            raise ValueError("Clinical block-centre times must be strictly increasing")
        span = real_times[-1] - real_times[0]
        normalized = T * (real_times - real_times[0]) / span
        normalized_grid = np.linspace(0.0, T, n_grid)
        real_grid = real_times[0] + normalized_grid * span / T
    else:
        normalized = np.linspace(0.0, T, int(combo["t_steps"]))
        normalized_grid = np.linspace(0.0, T, n_grid)
        real_times = normalized.copy()
        real_grid = normalized_grid.copy()

    t_obs = torch.as_tensor(normalized, dtype=torch.float32, device=device)
    t_grid = torch.as_tensor(normalized_grid, dtype=torch.float32, device=device)
    return t_obs, t_grid, real_times, real_grid


def time_transformation(combo):
    """The exact, reversible map between clinical time and NODE internal time.

    Clinical time is weeks since randomization; the NODE integrates on [0, T].
    Recording origin and scale here lets any result be mapped back without
    guessing, and makes explicit that the NODE grid spans only the observed
    support (interpolation), never beyond it (extrapolation).
    """
    real_times = combo.get("observation_times_weeks")
    if real_times is None:
        return {"kind": "identity", "clinical_units": "normalized"}
    real_times = np.asarray(real_times, dtype=float)
    T = float(combo["T"])
    span = float(real_times[-1] - real_times[0])
    return {
        "kind": "affine",
        "clinical_units": "weeks since randomization",
        "internal_units": f"[0, {T}]",
        "origin_weeks": float(real_times[0]),
        "span_weeks": span,
        "forward": "internal = T * (weeks - origin_weeks) / span_weeks",
        "inverse": "weeks = origin_weeks + internal * span_weeks / T",
        "observed_support_weeks": [float(real_times[0]), float(real_times[-1])],
        "grid_covers": "observed support only (interpolation, never extrapolation)",
    }
