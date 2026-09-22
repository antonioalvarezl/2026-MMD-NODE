"""Participant-level functional MMD with wild-bootstrap inference for CGM trajectories.

Primary test: each participant contributes the complete trajectory fitted only on its
configured calendar-week analysis window as one observation in L2([0,1]; R^K). The wild bootstrap uses one Rademacher multiplier per
participant, constant across that participant's complete trajectory.

Secondary analyses reuse the same participant-level multipliers:

* bandwidth sensitivity: the same global wild test at fixed multiples of the
  median bandwidth h_med, never used to choose a bandwidth;
* component-functional: one scalar L2 functional MMD per mixture component and
  a joint max-T over components;
* weekly pointwise: one scalar Gaussian-RBF MMD per component at each clinical
  week centre t_j = 0.5, 1.5, ... inside the common clinical support, plus a per-component
  cumulative Cauchy/ACAT combination of those weekly p-values.

The global null-distribution figure is formatted in Medical_DataAnalysis.ipynb;
the weekly pointwise and cumulative-ACAT figures are saved by this module.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist

# Allow direct execution via `python Analysis/wild_bootstrap_mmd.py`.
PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from Analysis.paper_config import (
    COMPONENT_COLORS,
    CONTROL_IDS,
    INFERENCE_SEED,
    PAPER_INTERVAL,
    PAPER_N_GRID,
    N_NOMINAL_WEEKS,
    TREATMENT_IDS,
    dataset_fingerprint,
)
from Analysis.paper_figures import (
    _load_interval_data,
    _locked_architecture,
    _save_figure,
)
from Analysis.clinical_support import validate_rct_run, select_inference_window, inference_window_sensitivity


ESTIMAND_ABSOLUTE = "absolute"
ESTIMAND_CHANGE_FROM_BASELINE = "change_from_baseline"

ESTIMANDS = (
    ESTIMAND_ABSOLUTE,
    ESTIMAND_CHANGE_FROM_BASELINE,
)

ESTIMAND_ANCHORS = {
    ESTIMAND_ABSOLUTE: "none (absolute mixture weights)",
    ESTIMAND_CHANGE_FROM_BASELINE: (
        "the participant's PRE-RANDOMIZATION distribution (last two weeks "
        "before RandDate, >= 168 h), projected onto the same treatment-blind "
        "dictionary with the same patient bandwidth and ridge"
    ),
}
# Pre-specified multiples of h_med; 1.0 reproduces the primary test.
BANDWIDTH_SENSITIVITY_MULTIPLIERS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)
# Memory of the cumulative weekly ACAT: exactly one week, expressed in days.
TAU_DAYS = 7.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--interval-start", type=float, default=PAPER_INTERVAL[0])
    parser.add_argument("--interval-end", type=float, default=PAPER_INTERVAL[1])
    parser.add_argument("--n-grid", type=int, default=PAPER_N_GRID)
    parser.add_argument("--n-bootstrap", type=int, default=9999)
    parser.add_argument("--n-permutations", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=INFERENCE_SEED)
    parser.add_argument("--estimand", choices=ESTIMANDS, action="append", default=None)
    parser.add_argument("--skip-permutation", action="store_true")
    parser.add_argument("--selection-json", type=Path, default=None)
    parser.add_argument("--strata-json", type=Path, default=None)
    return parser.parse_args()


def functional_squared_distance_matrix(
    trajectories: np.ndarray,
    time_grid: np.ndarray,
) -> np.ndarray:
    """Pairwise squared L2 distances on the common interval mapped to [0,1]."""
    trajectories = np.asarray(trajectories, dtype=float)
    time_grid = np.asarray(time_grid, dtype=float)

    if trajectories.ndim != 3:
        raise ValueError("trajectories must have shape (participants, times, components)")
    if time_grid.ndim != 1 or trajectories.shape[1] != len(time_grid):
        raise ValueError("time_grid must align with trajectories")
    if len(time_grid) < 2 or np.any(np.diff(time_grid) <= 0):
        raise ValueError("time_grid must be strictly increasing")
    if not np.all(np.isfinite(trajectories)):
        raise ValueError("trajectories must be finite")

    normalized = (time_grid - time_grid[0]) / (time_grid[-1] - time_grid[0])
    pointwise_sq = np.sum(
        (trajectories[:, None, :, :] - trajectories[None, :, :, :]) ** 2,
        axis=-1,
    )
    widths = np.diff(normalized)
    distances = np.sum(
        0.5 * (pointwise_sq[:, :, :-1] + pointwise_sq[:, :, 1:])
        * widths[None, None, :],
        axis=-1,
    )
    distances = np.maximum(0.0, 0.5 * (distances + distances.T))
    np.fill_diagonal(distances, 0.0)
    return distances


def functional_gaussian_kernel(
    trajectories: np.ndarray,
    time_grid: np.ndarray,
    bandwidth: float | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Gaussian kernel on complete trajectories with treatment-blind bandwidth."""
    squared_distances = functional_squared_distance_matrix(trajectories, time_grid)

    if bandwidth is None:
        upper = np.sqrt(
            squared_distances[np.triu_indices(len(squared_distances), k=1)]
        )
        positive = upper[np.isfinite(upper) & (upper > 0)]
        if len(positive) == 0:
            raise ValueError("participant trajectories have no positive L2 distance")
        bandwidth = float(np.median(positive))
    else:
        bandwidth = float(bandwidth)

    if bandwidth <= 0 or not np.isfinite(bandwidth):
        raise ValueError("functional kernel bandwidth must be positive and finite")

    kernel = np.exp(-0.5 * squared_distances / bandwidth**2)
    return kernel, bandwidth, squared_distances


def scaled_mmd_v_from_kernels(
    kxx: np.ndarray,
    kyy: np.ndarray,
    kxy: np.ndarray,
) -> float:
    """Scaled biased V-statistic used for observed and resampled statistics."""
    nx = int(kxx.shape[0])
    ny = int(kyy.shape[0])
    scale = nx * ny / (nx + ny)
    return float(
        scale
        * (
            np.sum(kxx) / nx**2
            + np.sum(kyy) / ny**2
            - 2.0 * np.sum(kxy) / (nx * ny)
        )
    )


def centered_rademacher_multipliers(
    nx: int,
    ny: int,
    n_bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """One independent raw Rademacher multiplier per participant, centered by arm."""
    if nx < 2 or ny < 2 or n_bootstrap < 1:
        raise ValueError("both arms need at least two participants and B must be positive")
    rng = np.random.default_rng(seed)
    wx = rng.choice(np.array([-1.0, 1.0]), size=(n_bootstrap, nx))
    wy = rng.choice(np.array([-1.0, 1.0]), size=(n_bootstrap, ny))
    wx -= wx.mean(axis=1, keepdims=True)
    wy -= wy.mean(axis=1, keepdims=True)
    return wx, wy


def wild_statistics_from_kernels(
    kxx: np.ndarray,
    kyy: np.ndarray,
    kxy: np.ndarray,
    wx: np.ndarray,
    wy: np.ndarray,
) -> np.ndarray:
    """Wild-bootstrap scaled MMD statistics."""
    nx = kxx.shape[0]
    ny = kyy.shape[0]
    scale = nx * ny / (nx + ny)

    xx = np.einsum("bi,ij,bj->b", wx, kxx, wx, optimize=True) / nx**2
    yy = np.einsum("bi,ij,bj->b", wy, kyy, wy, optimize=True) / ny**2
    xy = np.einsum("bi,ij,bj->b", wx, kxy, wy, optimize=True) / (nx * ny)
    return scale * (xx + yy - 2.0 * xy)


def pvalue_plus_one(observed: float, replicates: np.ndarray) -> float:
    replicates = np.asarray(replicates, dtype=float)
    return float(
        (1 + np.count_nonzero(replicates >= observed))
        / (len(replicates) + 1)
    )


def _permutation_coefficients(
    labels: np.ndarray,
    n_permutations: int,
    seed: int,
    strata: np.ndarray | None = None,
) -> np.ndarray:
    """Whole-participant randomization coefficients, optionally within strata."""
    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    nx = int(np.sum(labels == 1))
    ny = n - nx
    rng = np.random.default_rng(seed)
    assignments = np.empty((n_permutations, n), dtype=np.int8)

    if strata is None:
        for b in range(n_permutations):
            treated = rng.choice(n, size=nx, replace=False)
            assignments[b] = 0
            assignments[b, treated] = 1
    else:
        strata = np.asarray(strata)
        if strata.shape != (n,):
            raise ValueError("strata must align with labels")
        for b in range(n_permutations):
            assignment = np.zeros(n, dtype=np.int8)
            for value in np.unique(strata):
                members = np.flatnonzero(strata == value)
                n_treated = int(np.sum(labels[members] == 1))
                if n_treated:
                    chosen = rng.choice(members, size=n_treated, replace=False)
                    assignment[chosen] = 1
            assignments[b] = assignment

    return np.where(assignments == 1, 1.0 / nx, -1.0 / ny)


def global_functional_mmd_test(
    trajectories: np.ndarray,
    labels: np.ndarray,
    time_grid: np.ndarray,
    *,
    n_bootstrap: int = 9999,
    n_permutations: int = 9999,
    seed: int = INFERENCE_SEED,
    bandwidth: float | None = None,
    strata: np.ndarray | None = None,
) -> dict:
    """One global subject-level wild test plus participant-label permutation validation."""
    trajectories = np.asarray(trajectories, dtype=float)
    labels = np.asarray(labels, dtype=int)

    if trajectories.ndim != 3 or labels.shape != (len(trajectories),):
        raise ValueError("labels must align with participant trajectories")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("labels must contain treatment=1 and control=0")

    treatment_index = np.flatnonzero(labels == 1)
    control_index = np.flatnonzero(labels == 0)
    nx, ny = len(treatment_index), len(control_index)
    if min(nx, ny) < 2:
        raise ValueError("both arms need at least two participants")

    kernel, bandwidth, squared_distances = functional_gaussian_kernel(
        trajectories,
        time_grid,
        bandwidth=bandwidth,
    )
    kxx = kernel[np.ix_(treatment_index, treatment_index)]
    kyy = kernel[np.ix_(control_index, control_index)]
    kxy = kernel[np.ix_(treatment_index, control_index)]

    observed = scaled_mmd_v_from_kernels(kxx, kyy, kxy)
    scale = nx * ny / (nx + ny)

    wx, wy = centered_rademacher_multipliers(nx, ny, n_bootstrap, seed)
    bootstrap = wild_statistics_from_kernels(kxx, kyy, kxy, wx, wy)

    permutation = None
    if n_permutations:
        coefficients = _permutation_coefficients(
            labels,
            n_permutations,
            seed,
            strata=strata,
        )
        permuted = scale * np.einsum(
            "bi,ij,bj->b",
            coefficients,
            kernel,
            coefficients,
            optimize=True,
        )
        permutation = {
            "pvalue": pvalue_plus_one(observed, permuted),
            "n_permutations": int(n_permutations),
            "quantiles": {
                "0.90": float(np.quantile(permuted, 0.90)),
                "0.95": float(np.quantile(permuted, 0.95)),
                "0.99": float(np.quantile(permuted, 0.99)),
            },
            "_replicates": permuted,
        }

    return {
        "mmd_squared": float(observed / scale),
        "scaled_mmd_v_statistic": float(observed),
        "kernel_bandwidth_h": float(bandwidth),
        "bandwidth_rule": (
            "median strictly positive pairwise L2 distance across all included "
            "participants, ignoring treatment labels"
        ),
        "n_treatment": int(nx),
        "n_control": int(ny),
        "wild_bootstrap": {
            "pvalue": pvalue_plus_one(observed, bootstrap),
            "n_bootstrap": int(n_bootstrap),
            "mean": float(np.mean(bootstrap)),
            "standard_deviation": float(np.std(bootstrap, ddof=0)),
            "quantiles": {
                "0.90": float(np.quantile(bootstrap, 0.90)),
                "0.95": float(np.quantile(bootstrap, 0.95)),
                "0.99": float(np.quantile(bootstrap, 0.99)),
            },
            "_replicates": bootstrap,
        },
        "permutation_validation": permutation,
        "distance_diagnostics": {
            "median_pairwise_distance": float(bandwidth),
            "maximum_squared_distance": float(np.max(squared_distances)),
        },
    }


def bandwidth_sensitivity(
    trajectories: np.ndarray,
    labels: np.ndarray,
    time_grid: np.ndarray,
    *,
    n_bootstrap: int = 9999,
    seed: int = INFERENCE_SEED,
    multipliers: tuple[float, ...] = BANDWIDTH_SENSITIVITY_MULTIPLIERS,
) -> dict:
    """Global wild test repeated at fixed multiples of the median bandwidth.

    Sensitivity analysis only: the primary test keeps h_med and no bandwidth is
    selected from these p-values. Every scale runs the primary test with the same
    cohort, statistic, B and seed, so ``centered_rademacher_multipliers`` returns
    the identical participant-level multiplier matrices at every scale.
    """
    _, median_bandwidth, _ = functional_gaussian_kernel(trajectories, time_grid)

    rows = []
    for multiplier in multipliers:
        result = global_functional_mmd_test(
            trajectories,
            labels,
            time_grid,
            n_bootstrap=n_bootstrap,
            n_permutations=0,
            seed=seed,
            bandwidth=float(multiplier) * median_bandwidth,
        )
        rows.append({
            "bandwidth_multiplier": float(multiplier),
            "effective_bandwidth": result["kernel_bandwidth_h"],
            "mmd_squared": result["mmd_squared"],
            "scaled_mmd_v_statistic": result["scaled_mmd_v_statistic"],
            "wild_pvalue": result["wild_bootstrap"]["pvalue"],
        })

    return {
        "purpose": (
            "sensitivity analysis only; no bandwidth is selected from these p-values"
        ),
        "median_bandwidth_h_med": float(median_bandwidth),
        "effective_bandwidth_rule": "bandwidth_multiplier * h_med",
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "same_rademacher_multipliers_at_every_bandwidth": True,
        "rows": rows,
    }


def component_functional_tests(
    trajectories: np.ndarray,
    labels: np.ndarray,
    time_grid: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict:
    """K functional tests plus one joint max-component wild-bootstrap test.

    Each participant contributes one complete scalar L2 trajectory per component.
    The same participant-level Rademacher multipliers are reused for every component.
    """
    trajectories = np.asarray(trajectories, dtype=float)
    labels = np.asarray(labels, dtype=int)
    treat_idx = np.flatnonzero(labels == 1)
    ctrl_idx = np.flatnonzero(labels == 0)
    nt, nc = len(treat_idx), len(ctrl_idx)

    wt, wc = centered_rademacher_multipliers(nt, nc, n_bootstrap, seed)

    K = trajectories.shape[2]
    observed = np.zeros(K, dtype=float)
    bandwidths = np.zeros(K, dtype=float)
    bootstrap_matrix = np.zeros((n_bootstrap, K), dtype=float)

    for k in range(K):
        scalar_functions = trajectories[:, :, k:k+1]
        kernel, h, _ = functional_gaussian_kernel(
            scalar_functions, time_grid, bandwidth=None
        )
        ktt = kernel[np.ix_(treat_idx, treat_idx)]
        kcc = kernel[np.ix_(ctrl_idx, ctrl_idx)]
        ktc = kernel[np.ix_(treat_idx, ctrl_idx)]

        observed[k] = scaled_mmd_v_from_kernels(ktt, kcc, ktc)
        bootstrap_matrix[:, k] = wild_statistics_from_kernels(
            ktt, kcc, ktc, wt, wc
        )
        bandwidths[k] = h

    raw_p = np.asarray([
        pvalue_plus_one(observed[k], bootstrap_matrix[:, k])
        for k in range(K)
    ])

    # Studentize before taking the maximum because component-specific MMD
    # statistics can have different null scales.
    boot_mean = bootstrap_matrix.mean(axis=0)
    boot_sd = bootstrap_matrix.std(axis=0, ddof=0)
    boot_sd = np.where(boot_sd > 0, boot_sd, 1.0)
    observed_z = (observed - boot_mean) / boot_sd
    bootstrap_z = (bootstrap_matrix - boot_mean[None, :]) / boot_sd[None, :]
    max_bootstrap_z = bootstrap_z.max(axis=1)
    max_observed_z = float(observed_z.max())
    max_global_p = pvalue_plus_one(max_observed_z, max_bootstrap_z)

    rows = []
    for k in range(K):
        rows.append({
            "component": k + 1,
            "scaled_mmd_v": float(observed[k]),
            "functional_bandwidth": float(bandwidths[k]),
            "wild_pvalue_raw": float(raw_p[k]),
            "studentized_statistic": float(observed_z[k]),
            "maxT_pvalue_across_components": float(
                pvalue_plus_one(observed_z[k], max_bootstrap_z)
            ),
        })

    return {
        "rows": rows,
        "max_component_global": {
            "statistic": max_observed_z,
            "wild_pvalue": float(max_global_p),
            "selected_component": int(np.argmax(observed_z) + 1),
            "calibration": (
                "maximum of studentized component-functional MMD statistics; "
                "same participant-level multipliers reused across all components"
            ),
        },
    }


def _rbf_scalar(x: np.ndarray, y: np.ndarray, h: float) -> np.ndarray:
    d = np.asarray(x, float)[:, None] - np.asarray(y, float)[None, :]
    return np.exp(-0.5 * d**2 / h**2)


def _median_positive_abs_distance(x: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=float)
    d = np.abs(x[:, None] - x[None, :])
    d = d[np.triu_indices(len(x), k=1)]
    d = d[np.isfinite(d) & (d > 0)]
    return None if len(d) == 0 else float(np.median(d))


def pointwise_tests(
    trajectories: np.ndarray,
    labels: np.ndarray,
    time_grid: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict:
    """Exploratory component-by-time MMD tests with subject-level multipliers.

    Cells whose pooled values have no positive distance (e.g. change from
    baseline at the baseline time) are not testable and stay NaN.
    """
    trajectories = np.asarray(trajectories, dtype=float)
    labels = np.asarray(labels, dtype=int)
    treat = trajectories[labels == 1]
    ctrl = trajectories[labels == 0]
    nt, nc = len(treat), len(ctrl)

    # Generate ONCE: same subject multipliers for every time and component.
    wt, wc = centered_rademacher_multipliers(nt, nc, n_bootstrap, seed)

    m, K = trajectories.shape[1:]
    pvalues = np.full((m, K), np.nan, dtype=float)
    statistics = np.full((m, K), np.nan, dtype=float)
    bandwidths = np.full((m, K), np.nan, dtype=float)

    for j in range(m):
        for k in range(K):
            x = treat[:, j, k]
            y = ctrl[:, j, k]
            pooled = np.concatenate([x, y])
            h = _median_positive_abs_distance(pooled)
            if h is None:
                continue

            ktt = _rbf_scalar(x, x, h)
            kcc = _rbf_scalar(y, y, h)
            ktc = _rbf_scalar(x, y, h)
            observed = scaled_mmd_v_from_kernels(ktt, kcc, ktc)
            boot = wild_statistics_from_kernels(ktt, kcc, ktc, wt, wc)

            statistics[j, k] = observed
            bandwidths[j, k] = h
            pvalues[j, k] = pvalue_plus_one(observed, boot)

    return {
        "time_grid": np.asarray(time_grid, dtype=float),
        "scaled_mmd_v": statistics,
        "bandwidth": bandwidths,
        "wild_pvalue": pvalues,
    }


def cumulative_acat(
    pvalues: np.ndarray,
    weeks: np.ndarray,
    tau_days: float = TAU_DAYS,
) -> np.ndarray:
    """Cumulative Cauchy/ACAT combination of weekly p-values for each component.

    T_k(t_m) = sum_{j<=m} w_mj tan[pi(1/2 - p_k(t_j))] and
    p_k,hist(t_m) = 1/2 - arctan(T_k(t_m)) / pi, where w_mj is proportional to
    exp(-lag_days / tau_days) with lag_days = 7 (t_m - t_j) because t is in weeks,
    normalized to sum to one. Untestable (NaN) weeks receive no weight; a time
    with no testable week up to it stays NaN.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    weeks = np.asarray(weeks, dtype=float)

    lag_days = 7.0 * (weeks[:, None] - weeks[None, :])
    memory = np.where(lag_days >= 0.0, np.exp(-lag_days / tau_days), 0.0)
    testable = np.isfinite(pvalues)
    weights = memory[:, :, None] * testable[None, :, :]
    cauchy = np.tan(np.pi * (0.5 - np.where(testable, pvalues, 0.5)))

    with np.errstate(invalid="ignore"):
        statistic = (
            np.einsum("mjk,jk->mk", weights, cauchy) / weights.sum(axis=1)
        )
    return 0.5 - np.arctan(statistic) / np.pi


def _save_weekly_pvalue_figure(
    weeks: np.ndarray,
    pvalues: np.ndarray,
    output_dir: Path,
    stem: str,
    *,
    ylabel: str,
    n_bootstrap: int,
) -> list[str]:
    """One panel per component, identified by its title; no legend."""
    n_components = pvalues.shape[1]
    fig, axes = plt.subplots(
        1, n_components,
        figsize=(2.4 * n_components, 2.6),
        sharex=True, sharey=True,
        squeeze=False,
        layout="constrained",
    )
    for k, ax in enumerate(axes[0]):
        color = COMPONENT_COLORS[k] if k < len(COMPONENT_COLORS) else None
        # clip_on=False keeps markers at p = 1 whole on the top spine.
        ax.plot(
            weeks, pvalues[:, k],
            color=color, linewidth=1.6, marker="o", markersize=3.5, clip_on=False,
        )
        ax.axhline(0.05, color="black", linestyle="--", linewidth=1.0)
        ax.set_yscale("log")
        ax.set_title(f"Component {k + 1}", fontsize=10)
        ax.grid(alpha=0.15, linestyle="--", linewidth=0.6)

    # Both p-values are >= 1/(B+1); keep the smallest attainable one inside the axes.
    axes[0, 0].set_ylim(0.5 / (n_bootstrap + 1), 1.0)
    # The x axis is the trial's nominal span in weeks, NOT the number of
    # testable week centres: those differ once the complete-case inference
    # window starts after week 0, and using the count put the last centres
    # outside the axes (with clip_on=False they then drew over the figure).
    # A fixed 0..N_NOMINAL_WEEKS axis also keeps panels comparable across runs
    # and makes the untested span visible rather than hidden.
    axes[0, 0].set_xlim(0.0, float(N_NOMINAL_WEEKS))
    axes[0, 0].set_ylabel(ylabel)
    fig.supxlabel("Study week", fontsize=10)
    return _save_figure(fig, output_dir, stem)


def _load_strata(
    path: Path | None,
    patient_ids: list[int],
) -> tuple[np.ndarray | None, dict]:
    if path is None:
        return None, {"available": False, "reason": "no strata JSON supplied"}

    with path.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)

    missing = [pid for pid in patient_ids if str(pid) not in mapping]
    if missing:
        raise ValueError(f"strata JSON is missing included patients: {missing}")

    values = np.asarray([mapping[str(pid)] for pid in patient_ids])
    return values, {"available": True, "path": str(path.resolve())}


def _json_clean(value):
    """Remove internal arrays before serializing while keeping them in returned results."""
    if isinstance(value, dict):
        return {
            key: _json_clean(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, list):
        return [_json_clean(item) for item in value]
    return value


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def run_analysis(args: argparse.Namespace, estimand: str) -> dict:
    if estimand not in ESTIMANDS:
        raise ValueError(f"estimand must be one of {ESTIMANDS}")

    expected_architecture = _locked_architecture(args.selection_json)
    params = validate_rct_run(args.run)

    # Step 1: read every participant's real fitted support, restricting nothing.
    # The window must be chosen from the supports themselves, because taking the
    # intersection of the WHOLE cohort lets one or two extreme participants
    # dictate a tiny mid-trial slice.
    survey = _load_interval_data(
        TREATMENT_IDS,
        CONTROL_IDS,
        str(args.run.resolve()),
        interval=(args.interval_start, args.interval_end),
        n_grid=args.n_grid,
        expected_architecture=expected_architecture,
        common_grid_mode="available_case",
        expected_fingerprint=params["dataset_fingerprint"],
    )
    window_low, window_high, keep_mask, window_report = select_inference_window(
        survey["fitted_node_supports_original_week_coordinates"],
        survey["labels"],
        interval=(args.interval_start, args.interval_end),
    )
    survey_ids = np.asarray(survey["included_ids"], dtype=int)
    survey_labels = np.asarray(survey["labels"], dtype=int)
    inference_treatment_ids = tuple(
        int(pid) for pid in survey_ids[keep_mask & (survey_labels == 1)]
    )
    inference_control_ids = tuple(
        int(pid) for pid in survey_ids[keep_mask & (survey_labels == 0)]
    )
    window_report["dropped_participant_ids"] = [
        int(pid) for pid in survey_ids[~keep_mask]
    ]

    # Step 2: the actual inference, COMPLETE-CASE on that window.  Every
    # retained participant covers all of it, so the array has no NaN and the
    # wild bootstrap's one-multiplier-per-subject scheme is well defined.
    loaded = _load_interval_data(
        inference_treatment_ids,
        inference_control_ids,
        str(args.run.resolve()),
        interval=(window_low, window_high),
        n_grid=args.n_grid,
        expected_architecture=expected_architecture,
        common_grid_mode="intersection",
        expected_fingerprint=params["dataset_fingerprint"],
    )

    # Weekly pointwise inference uses the nominal RCT week centres
    # t_j = 0.5, 1.5, ..., that lie inside the COMMON clinical support of the
    # included participants.  Participants' supports genuinely differ (a
    # truncated week 0, a real 13-week visit date, dropouts), so the set of
    # testable centres is determined by the data rather than assumed to be
    # 0.5, ..., 12.5.  Every tested point is interpolated inside each
    # participant's own support; nothing is extrapolated.
    support_low, support_high = loaded["common_grid_support_weeks"]
    all_centers = np.arange(N_NOMINAL_WEEKS, dtype=float) + 0.5
    week_centers = all_centers[
        (all_centers >= support_low) & (all_centers <= support_high)
    ]
    if len(week_centers) < 2:
        raise RuntimeError(
            "fewer than two nominal week centres lie inside the common "
            f"clinical support [{support_low:.3f}, {support_high:.3f}] weeks; "
            "weekly pointwise tests are not defined for this cohort"
        )
    dropped = [float(c) for c in all_centers if c not in set(week_centers.tolist())]
    if dropped:
        print(
            f"[weekly] nominal week centres outside the common support and "
            f"therefore not tested: {dropped}"
        )
    # Equally spaced centres one week apart: a linspace over the same endpoints
    # with this many points lands exactly on them.
    weekly = _load_interval_data(
        inference_treatment_ids,
        inference_control_ids,
        str(args.run.resolve()),
        interval=(float(week_centers[0]), float(week_centers[-1])),
        n_grid=len(week_centers),
        expected_architecture=expected_architecture,
        common_grid_mode="intersection",
        expected_fingerprint=params["dataset_fingerprint"],
    )
    if not np.allclose(weekly["common_grid"], week_centers):
        raise RuntimeError(
            "weekly grid does not land on the nominal week centres: "
            f"{weekly['common_grid']} vs {week_centers}"
        )
    if list(weekly["included_ids"]) != list(loaded["included_ids"]):
        raise RuntimeError("weekly and functional analyses must use the same cohort")

    trajectories = np.asarray(loaded["trajectories"], dtype=float)
    weekly_trajectories = np.asarray(weekly["trajectories"], dtype=float)
    labels = np.asarray(loaded["labels"], dtype=int)
    baseline_info = None
    if estimand == ESTIMAND_CHANGE_FROM_BASELINE:
        # REAL anchor: the pre-randomization distribution, projected onto the
        # same dictionary.  A participant without a >= 168 h baseline has no
        # anchor and is DROPPED here rather than imputed; it remains available
        # to the absolute estimand.
        from Analysis.baseline_projection import baseline_weights_for_run

        included = [int(p) for p in loaded["included_ids"]]
        baseline_weights, baseline_excluded, baseline_meta = (
            baseline_weights_for_run(args.run, included)
        )
        keep = np.array([p in baseline_weights for p in included], dtype=bool)
        if int((labels[keep] == 1).sum()) < 2 or int((labels[keep] == 0).sum()) < 2:
            raise RuntimeError(
                "change_from_baseline needs at least two participants per arm "
                "with a pre-randomization baseline"
            )
        anchor = np.array(
            [baseline_weights[p] for p in included if p in baseline_weights],
            dtype=float,
        )[:, None, :]
        trajectories = trajectories[keep] - anchor
        weekly_trajectories = weekly_trajectories[keep] - anchor
        labels = labels[keep]
        kept_ids = [p for p in included if p in baseline_weights]
        loaded = dict(loaded)
        loaded["included_ids"] = kept_ids
        weekly = dict(weekly)
        weekly["included_ids"] = kept_ids
        baseline_info = {
            **baseline_meta,
            "n_with_baseline": int(keep.sum()),
            "n_without_baseline": int((~keep).sum()),
            "n_with_baseline_clc": int((labels == 1).sum()),
            "n_with_baseline_sc": int((labels == 0).sum()),
            "dropped_participant_ids": [
                p for p in included if p not in baseline_weights
            ],
            "anchor": ESTIMAND_ANCHORS[ESTIMAND_CHANGE_FROM_BASELINE],
        }
        print(
            f"[baseline anchor] {int(keep.sum())}/{len(included)} participants "
            f"(CLC {baseline_info['n_with_baseline_clc']}, "
            f"SC {baseline_info['n_with_baseline_sc']}); "
            f"{int((~keep).sum())} dropped for lack of a baseline"
        )

    strata, strata_metadata = _load_strata(
        args.strata_json,
        list(loaded["included_ids"]),
    )

    time_grid = np.asarray(loaded["common_grid"], dtype=float)
    result = global_functional_mmd_test(
        trajectories,
        labels,
        time_grid,
        n_bootstrap=args.n_bootstrap,
        n_permutations=(0 if args.skip_permutation else args.n_permutations),
        seed=args.seed,
        strata=strata,
    )
    sensitivity = bandwidth_sensitivity(
        trajectories,
        labels,
        time_grid,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    component = component_functional_tests(
        trajectories,
        labels,
        time_grid,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    pointwise = pointwise_tests(
        weekly_trajectories,
        labels,
        week_centers,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    acat = cumulative_acat(pointwise["wild_pvalue"], week_centers)
    weekly_rows = [
        {
            "week": float(week),
            "component": k + 1,
            "pointwise_pvalue": _finite_or_none(pointwise["wild_pvalue"][j, k]),
            "cumulative_acat_pvalue": _finite_or_none(acat[j, k]),
            "bandwidth": _finite_or_none(pointwise["bandwidth"][j, k]),
        }
        for j, week in enumerate(week_centers)
        for k in range(weekly_trajectories.shape[2])
    ]

    payload = {
        "analysis": "global participant-level functional Gaussian-kernel MMD",
        "estimand": estimand,
        "estimand_anchor": ESTIMAND_ANCHORS[estimand],
        "baseline_anchor": baseline_info,
        "configuration": {
            "interval": [args.interval_start, args.interval_end],
            "dataset_fingerprint": params["dataset_fingerprint"],
            "grid_mode": "intersection (fixed inference cohort)",
            "inference_window_selection": window_report,
            "inference_window_sensitivity": inference_window_sensitivity(
                survey["fitted_node_supports_original_week_coordinates"],
                survey["labels"], interval=(args.interval_start, args.interval_end), verbose=False),
            "functional_domain": (
                "NODE trajectories on clinical time (weeks since "
                "randomization), restricted to the selected complete-case "
                "inference window (see inference_window_selection): every "
                "retained participant covers all of it, and nothing is "
                "extrapolated. Mapped linearly to [0,1] for the functional "
                "statistic only"
            ),
            "common_grid_support_weeks": loaded["common_grid_support_weeks"],
            "weekly_centers_tested": [float(c) for c in week_centers],
            "n_grid": args.n_grid,
            "n_components": int(trajectories.shape[2]),
            "n_bootstrap": args.n_bootstrap,
            "n_permutations": 0 if args.skip_permutation else args.n_permutations,
            "seed": args.seed,
            "one_multiplier_per_complete_participant": True,
            "multipliers_vary_over_time": False,
            "kernel": "exp(-functional_squared_distance / (2*h^2))",
        },
        "population": {
            "included_ids": list(loaded["included_ids"]),
            "treatment_n": int(np.sum(labels == 1)),
            "control_n": int(np.sum(labels == 0)),
            "excluded": loaded["excluded"],
        },
        "global": result,
        "bandwidth_sensitivity": sensitivity,
        "component_functional": component,
        "weekly_pointwise": {
            "week_centers": week_centers.tolist(),
            "pointwise_test": (
                "scalar Gaussian-RBF MMD per component at each clinical week centre "
                "(saved NODE trajectory, linearly interpolated at the centre); "
                "bandwidth = median positive pooled absolute difference, ignoring "
                "treatment labels"
            ),
            "cumulative_acat": (
                "T_k(t_m) = sum_{j<=m} w_mj tan[pi(1/2 - p_k(t_j))], "
                "p_hist = 1/2 - arctan(T)/pi; w_mj proportional to "
                "exp(-7 (t_m - t_j) / tau_days), normalized over testable weeks j <= m"
            ),
            "tau_days": TAU_DAYS,
            "untestable": "null when all pooled values coincide (change estimand, week 0.5)",
            "rows": weekly_rows,
        },
        "permutation_strata": strata_metadata,
        "common_time_grid_weeks": time_grid.tolist(),
    }

    output_dir = (
        args.output_dir.resolve()
        if getattr(args, "output_dir", None) is not None
        else Path(__file__).resolve().parents[1] / "figures_paper" /
        f"d{params['d']}" / args.run.name / "wild_bootstrap"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    sensitivity_csv = (
        output_dir / f"wild_bootstrap_mmd_{estimand}_bandwidth_sensitivity.csv"
    )
    with sensitivity_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sensitivity["rows"][0]))
        writer.writeheader()
        writer.writerows(sensitivity["rows"])
    sensitivity["csv"] = str(sensitivity_csv)

    weekly_csv = output_dir / f"wild_bootstrap_mmd_{estimand}_weekly_pointwise_acat.csv"
    with weekly_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(weekly_rows[0]))
        writer.writeheader()
        writer.writerows(weekly_rows)
    payload["weekly_pointwise"]["csv"] = str(weekly_csv)
    payload["weekly_pointwise"]["figures"] = _save_weekly_pvalue_figure(
        week_centers,
        pointwise["wild_pvalue"],
        output_dir,
        f"wild_bootstrap_mmd_{estimand}_weekly_pointwise_pvalues",
        ylabel="Pointwise p-value",
        n_bootstrap=args.n_bootstrap,
    ) + _save_weekly_pvalue_figure(
        week_centers,
        acat,
        output_dir,
        f"wild_bootstrap_mmd_{estimand}_weekly_cumulative_acat_pvalues",
        ylabel="Cumulative ACAT p-value",
        n_bootstrap=args.n_bootstrap,
    )

    json_path = output_dir / f"wild_bootstrap_mmd_{estimand}.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_clean(payload), handle, indent=2, allow_nan=False)

    print(
        f"[wild] {estimand}: MMD2={result['mmd_squared']:.6g}, "
        f"p={result['wild_bootstrap']['pvalue']:.6g}"
    )
    for row in sensitivity["rows"]:
        print(
            f"[wild] {estimand} bandwidth sensitivity "
            f"{row['bandwidth_multiplier']:g} x h_med "
            f"(h={row['effective_bandwidth']:.4g}): "
            f"MMD2={row['mmd_squared']:.6g}, p={row['wild_pvalue']:.6g}"
        )
    max_result = component["max_component_global"]
    print(
        f"[wild] {estimand} component-functional: raw p = "
        + ", ".join(f"{row['wild_pvalue_raw']:.4g}" for row in component["rows"])
        + f"; joint max-T p={max_result['wild_pvalue']:.4g}"
    )
    print(f"[wild] {estimand} weekly pointwise + cumulative ACAT: {weekly_csv}")
    return payload


def run_global_functional_mmd_analyses(
    run: str | Path,
    *,
    interval: tuple[float, float] = PAPER_INTERVAL,
    n_grid: int = PAPER_N_GRID,
    n_bootstrap: int = 9999,
    n_permutations: int = 9999,
    seed: int = INFERENCE_SEED,
    selection_json: str | Path | None = None,
    strata_json: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, dict]:
    """Run absolute and baseline-change global participant-level tests."""
    args = argparse.Namespace(
        run=Path(run),
        interval_start=float(interval[0]),
        interval_end=float(interval[1]),
        n_grid=int(n_grid),
        n_bootstrap=int(n_bootstrap),
        n_permutations=int(n_permutations),
        seed=int(seed),
        selection_json=(Path(selection_json) if selection_json else None),
        strata_json=(Path(strata_json) if strata_json else None),
        output_dir=(Path(output_dir) if output_dir else None),
        skip_permutation=False,
    )
    validate_rct_run(args.run)
    return {
        estimand: run_analysis(args, estimand)
        for estimand in ESTIMANDS
    }


def main() -> None:
    args = _parse_args()
    estimands = args.estimand or list(ESTIMANDS)
    for estimand in estimands:
        run_analysis(args, estimand)


if __name__ == "__main__":
    main()
