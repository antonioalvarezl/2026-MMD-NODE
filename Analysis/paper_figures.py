"""Generate reproducible clinical figures from the final saved CGM runs.

The participant is the sampling unit. Only participants whose fitted NODE
trajectory covers the complete requested interval are included. All curves
are interpolated to one elapsed-calendar-time grid before arm summaries are
computed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# paper_figures.py lives in <PROJECT_ROOT>/Analysis/.
# Add the project root so the shared Analysis configuration can be imported
# when the file is executed directly as a script.
PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, multivariate_normal, norm

from Analysis.paper_config import (
    ARM_COLORS,
    ARM_NAMES,
    ANALYSIS_WINDOW_DEFINITION,
    CONTROL_LABEL,
    TREATMENT_LABEL,
    N_NOMINAL_WEEKS,
    dataset_fingerprint,
    CONTROL_IDS,
    FIGURES_ROOT,
    PAPER_BAND_PCT,
    PAPER_INTERVAL,
    PAPER_N_GRID,
    PROJECT_ROOT,
    TREATMENT_IDS,
)
from Utils.IO_utils import load_experiment, save_json
from Visualization.plots import _get_patient_dir, _load_patient_results



def _load_interval_data(
    treatment_ids,
    control_ids,
    root_folder,
    interval=PAPER_INTERVAL,
    n_grid=PAPER_N_GRID,
    expected_architecture=None,
    common_grid_mode=None,
    expected_fingerprint=None,
):
    """Load fitted trajectories onto a common CLINICAL time grid.

    Available-case curves are descriptive; intersection grids are used for
    fixed-cohort inference. Both use the primary PEDAP dataset.

    The x axis is weeks since randomization.  Participants' observed supports
    genuinely differ (a truncated week 0, a real 13-week visit date, gaps,
    dropouts), so no participant's support is stretched onto a shared interval.
    ``common_grid_mode`` decides how that heterogeneity is handled:

    ``"available_case"`` (default, descriptive figures)
        The grid spans the UNION of the supports.  Each participant
        contributes only where its own support reaches; elsewhere its value
        is NaN.  Group curves must use nan-aware reductions, and
        ``n_available_per_grid_point`` says how many participants back each
        point -- the tails are backed by fewer participants, so read them
        with that in mind.

    ``"intersection"`` (complete-case, inference)
        The grid is the INTERSECTION of every included participant's support,
        so the array has no NaN and every participant contributes at every
        point.  This is what the wild-bootstrap / ACAT inference uses, whose
        subject-level multipliers are drawn once for a fixed cohort and are
        not defined under time-varying membership.

    In both modes every value on that grid is interpolation inside a
    participant's own support; nothing is ever extrapolated.
    """
    root_folder = Path(root_folder).resolve()
    if not root_folder.is_dir():
        raise FileNotFoundError(f"Results root does not exist: {root_folder}")

    from Analysis.clinical_support import validate_rct_run
    validate_rct_run(root_folder)
    if common_grid_mode is None:
        common_grid_mode = "available_case"
    if expected_fingerprint is None:
        expected_fingerprint = dataset_fingerprint()

    requested = (
        [(int(pid), "Treatment", 1) for pid in treatment_ids]
        + [(int(pid), "Control", 0) for pid in control_ids]
    )
    included_ids = []
    labels = []
    trajectories = []
    excluded = []
    reference_means = None
    reference_covs = None
    means_original = None
    covs_original = None
    reference_scaling_mean = None
    reference_scaling_std = None
    reference_scaling_method = None
    node_architectures = set()
    fitted_supports_weeks = []

    for patient_id, arm, label in requested:
        patient_dir = _get_patient_dir(root_folder, patient_id)
        if patient_dir is None:
            excluded.append({
                "patient_id": patient_id,
                "arm": arm,
                "reason": "result folder not found",
            })
            continue

        fit_payload = load_experiment(patient_dir, "result_fit.json")
        fit_config = (fit_payload or {}).get("config", {})
        if fit_config.get("dataset_fingerprint") != expected_fingerprint:
            excluded.append({
                "patient_id": patient_id,
                "arm": arm,
                "reason": (
                    "result was fitted on PEDAP dataset "
                    f"{fit_config.get('dataset_fingerprint')!r}, not "
                    f"{expected_fingerprint!r}"
                ),
            })
            continue

        result = _load_patient_results(patient_dir, include_covs=True)
        if result is None or result.get("trajectory_time_grid_weeks") is None:
            excluded.append({
                "patient_id": patient_id,
                "arm": arm,
                "reason": "calendar-time NODE result not found",
            })
            continue

        node_config = result.get("node_config", {})
        architecture = (
            int(node_config.get("hidden_dim", -1)),
            int(node_config.get("N_layers", -1)),
        )
        if expected_architecture is not None and architecture != tuple(expected_architecture):
            raise RuntimeError(
                f"patient {patient_id}: loaded NODE {architecture} but locked "
                f"architecture is {tuple(expected_architecture)}"
            )
        node_architectures.add(architecture)

        trajectory = np.asarray(result["traj_weights"], dtype=float)
        if trajectory.ndim == 3 and trajectory.shape[1] == 1:
            trajectory = trajectory[:, 0, :]
        time_grid = np.asarray(result["trajectory_time_grid_weeks"], dtype=float)
        if (
            trajectory.ndim != 2
            or len(trajectory) != len(time_grid)
            or np.any(np.diff(time_grid) <= 0)
        ):
            excluded.append({
                "patient_id": patient_id,
                "arm": arm,
                "reason": "invalid trajectory time grid",
            })
            continue
        model_span = float(time_grid[-1] - time_grid[0])
        if len(time_grid) < 2 or not np.isfinite(model_span) or model_span <= 0:
            excluded.append({
                "patient_id": patient_id,
                "arm": arm,
                "reason": "invalid fitted NODE support",
            })
            continue

        # Clinical time is used as it is.  Rescaling each participant's
        # support onto a shared interval would align different clinical times
        # with one another, which is exactly what must not happen once the
        # supports genuinely differ between participants.
        analysis_time_grid = time_grid
        if (
            not np.all(np.isfinite(trajectory))
            or np.any(trajectory < -1e-10)
            or not np.allclose(trajectory.sum(axis=1), 1.0, atol=1e-5)
        ):
            excluded.append({
                "patient_id": patient_id,
                "arm": arm,
                "reason": "invalid mixture weights",
            })
            continue

        # ``means`` / ``covs`` are the dictionary in the analysis space.
        # For d=2 this is the globally z-scored space used by MMD-GMM.
        means = np.asarray(result["means"], dtype=float)
        covs = np.asarray(result["covs"], dtype=float)

        if (
            result.get("means_original_units") is None
            or result.get("covs_original_units") is None
        ):
            raise RuntimeError(
                "Paper density figures require dictionary parameters in original units"
            )

        participant_means_original = np.asarray(
            result["means_original_units"], dtype=float
        )
        participant_covs_original = np.asarray(
            result["covs_original_units"], dtype=float
        )

        scaling = result.get("scaling")
        if scaling is None:
            raise RuntimeError(
                "Paper density figures require the dictionary scaling metadata"
            )
        scaling_mean = np.asarray(scaling.get("mean"), dtype=float)
        scaling_std = np.asarray(scaling.get("std"), dtype=float)
        scaling_method = str(scaling.get("method", "unknown"))

        dimension = means.shape[1]
        if (
            scaling_mean.shape != (dimension,)
            or scaling_std.shape != (dimension,)
            or not np.all(np.isfinite(scaling_mean))
            or not np.all(np.isfinite(scaling_std))
            or np.any(scaling_std <= 0)
        ):
            raise RuntimeError(
                f"patient {patient_id}: invalid dictionary scaling metadata"
            )

        if reference_means is None:
            reference_means = means
            reference_covs = covs
            means_original = participant_means_original
            covs_original = participant_covs_original
            reference_scaling_mean = scaling_mean
            reference_scaling_std = scaling_std
            reference_scaling_method = scaling_method
        elif not (
            np.allclose(means, reference_means)
            and np.allclose(covs, reference_covs)
            and np.allclose(participant_means_original, means_original)
            and np.allclose(participant_covs_original, covs_original)
            and np.allclose(scaling_mean, reference_scaling_mean)
            and np.allclose(scaling_std, reference_scaling_std)
            and scaling_method == reference_scaling_method
        ):
            raise RuntimeError(
                "All included participants must use the same Gaussian dictionary "
                "and the same global scaling"
            )

        included_ids.append(patient_id)
        labels.append(label)
        trajectories.append((analysis_time_grid, trajectory))
        fitted_supports_weeks.append([float(time_grid[0]), float(time_grid[-1])])

    coverage = None
    if trajectories:
        if common_grid_mode == "intersection":
            # Every participant covers every grid point: no NaN, so complete-case
            # statistics (the inference path) are valid on this grid.
            support_low = max(grid[0] for grid, _ in trajectories)
            support_high = min(grid[-1] for grid, _ in trajectories)
            if interval is not None:
                support_low = max(support_low, float(interval[0]))
                support_high = min(support_high, float(interval[1]))
            if not np.isfinite(support_low) or support_high <= support_low:
                raise RuntimeError(
                    "The included participants share no common clinical time "
                    f"support (intersection [{support_low}, {support_high}]). "
                    "Narrow the cohort, or use common_grid_mode='available_case'."
                )
        elif common_grid_mode == "available_case":
            # The grid spans the UNION of the supports, and each participant
            # contributes only where its own support actually reaches; outside
            # it the value is NaN rather than an extrapolation.  Group curves
            # must therefore be computed with nan-aware reductions over the
            # participants available at each time point.
            support_low = min(grid[0] for grid, _ in trajectories)
            support_high = max(grid[-1] for grid, _ in trajectories)
            if interval is not None:
                support_low = max(support_low, float(interval[0]))
                support_high = min(support_high, float(interval[1]))
            if not np.isfinite(support_low) or support_high <= support_low:
                raise RuntimeError(
                    "The included participants cover no clinical time at all "
                    f"within the requested interval ([{support_low}, {support_high}])."
                )
        else:
            raise ValueError(
                "common_grid_mode must be 'intersection' or 'available_case'"
            )

        common_grid = np.linspace(support_low, support_high, int(n_grid))
        stacked = []
        covered = []
        for grid, traj in trajectories:
            values = np.column_stack([
                np.interp(common_grid, grid, traj[:, component])
                for component in range(traj.shape[1])
            ])
            inside = (common_grid >= grid[0]) & (common_grid <= grid[-1])
            if common_grid_mode == "available_case":
                # np.interp clamps outside the support; blank those out so no
                # extrapolated value can reach a figure or a statistic.
                values[~inside, :] = np.nan
            stacked.append(values)
            covered.append(inside)
        trajectories = np.asarray(stacked, dtype=float)
        coverage = np.asarray(covered, dtype=bool)
    else:
        common_grid = np.linspace(float(interval[0]), float(interval[1]), int(n_grid))
        trajectories = np.asarray(trajectories, dtype=float)
    labels = np.asarray(labels, dtype=int)

    n_treatment = int(np.sum(labels == 1)) if labels.size else 0
    n_control = int(np.sum(labels == 0)) if labels.size else 0
    if trajectories.ndim != 3 or n_treatment == 0 or n_control == 0:
        reason_counts = {}
        for item in excluded:
            reason = item["reason"]
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        details = "; ".join(
            f"{reason}: {count}"
            for reason, count in sorted(
                reason_counts.items(), key=lambda pair: (-pair[1], pair[0])
            )
        )
        raise RuntimeError(
            "Could not form both treatment and control samples on the requested "
            f"interval {tuple(map(float, interval))}. "
            f"Included Treatment={n_treatment}, Control={n_control}. "
            f"Results root={Path(root_folder).resolve()}. "
            f"Exclusions: {details or 'none recorded'}"
        )

    if len(node_architectures) != 1:
        raise RuntimeError(f"included participants use mixed NODE architectures: {node_architectures}")

    return {
        "common_grid": common_grid,
        "trajectories": trajectories,
        "labels": labels,
        "included_ids": included_ids,
        "excluded": excluded,
        "means_standardized": reference_means,
        "covs_standardized": reference_covs,
        "means_original_units": means_original,
        "covs_original_units": covs_original,
        "scaling": {
            "method": reference_scaling_method,
            "mean": reference_scaling_mean,
            "std": reference_scaling_std,
        },
        "node_architecture": list(next(iter(node_architectures))),
        "dataset_fingerprint": expected_fingerprint,
        "n_nominal_weeks": int(N_NOMINAL_WEEKS),
        "analysis_window_definition": ANALYSIS_WINDOW_DEFINITION,
        "fitted_node_supports_original_week_coordinates": fitted_supports_weeks,
        "common_grid_units": "weeks since randomization",
        "common_grid_rule": (
            (
                "intersection of every included participant's fitted NODE "
                "support (itself the observed support), intersected with the "
                "requested interval; every participant contributes at every "
                "grid point"
            )
            if common_grid_mode == "intersection" else
            (
                "union of the included participants' fitted NODE supports, "
                "intersected with the requested interval; each participant "
                "contributes only where its own support reaches, and is NaN "
                "elsewhere (available-case)"
            )
        ) + "; interpolation only, never extrapolation",
        "common_grid_support_weeks": [float(common_grid[0]), float(common_grid[-1])],
        "common_grid_mode": common_grid_mode,
        "coverage_mask": coverage,
        "n_available_per_grid_point": (
            coverage.sum(axis=0).astype(int) if coverage is not None else None
        ),
        "n_available_per_grid_point_by_arm": (
            {
                "CLC": coverage[labels == 1].sum(axis=0).astype(int),
                "SC": coverage[labels == 0].sum(axis=0).astype(int),
            }
            if coverage is not None and labels.size else None
        ),
    }


def _save_figure(fig, output_folder, stem):
    """Save vector PDF plus high-resolution PNG with minimal padding."""
    files = []
    for extension, dpi in (("pdf", None), ("png", 400)):
        path = os.path.join(output_folder, f"{stem}.{extension}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
        files.append(path)
    plt.close(fig)
    return files


def _display_time(common_grid):
    """Return elapsed-calendar time in weeks for plotting."""
    grid = np.asarray(common_grid, dtype=float)
    if grid.ndim != 1 or len(grid) < 2 or np.any(np.diff(grid) <= 0):
        raise ValueError("common time grid must be a strictly increasing 1-D array")
    return grid


def _paper_axis(axis, *, xlabel=None, ylabel=None, xlim=None, ylim=None):
    if xlabel is not None:
        axis.set_xlabel(xlabel, fontsize=15)
    if ylabel is not None:
        axis.set_ylabel(ylabel, fontsize=15)
    if xlim is not None:
        axis.set_xlim(*xlim)
    if ylim is not None:
        axis.set_ylim(*ylim)
    axis.tick_params(axis="both", which="major", labelsize=13)
    axis.grid(alpha=0.16, linestyle="--", linewidth=0.6)


def _trajectory_display_limits(trajectories):
    ymax = float(np.nanmax(trajectories))
    # Preserve the historical 0--0.8 scale when possible, but never clip data.
    upper = max(0.8, np.ceil(min(1.0, 1.04 * ymax) / 0.05) * 0.05)
    return 0.0, min(1.0, upper)


def _change_display_limits(trajectories, labels):
    """Fixed paper display range for change-from-baseline quantile panels."""
    return -0.2, 0.2


def _plot_trajectory_figures(data, output_folder, band_pct=90.0):
    """Historical paper style with the corrected common-support data pipeline."""
    t = _display_time(data["common_grid"])
    trajectories = np.asarray(data["trajectories"], dtype=float)
    labels = np.asarray(data["labels"], dtype=int)
    colors = ARM_COLORS
    names = ARM_NAMES
    lower_q = (100.0 - float(band_pct)) / 2.0
    upper_q = 100.0 - lower_q
    mixed_ylim = _trajectory_display_limits(trajectories)
    generated = []

    for component in range(trajectories.shape[2]):
        # -------------------------------------------------------------
        # Absolute trajectories: faint subject curves + solid mean +
        # dashed outer percentile envelope. This matches the old paper
        # much more closely than fill_between.
        # -------------------------------------------------------------
        fig, axis = plt.subplots(figsize=(10.0, 5.0))
        for label in (1, 0):
            arm = trajectories[labels == label, :, component]
            color = colors[label]
            for curve in arm:
                # 0.10 alone left arm colors indistinguishable in this faint
                # background once ~90 curves overlap; 0.22 keeps the curves
                # subordinate to the mean/percentile lines while still
                # reading as Control-red vs Treatment-green.
                axis.plot(t, curve, color=color, alpha=0.22, linewidth=0.65, zorder=1)
            # Wider and fully opaque now that the background curves above are
            # more visible at alpha=0.22: at the old 1.25/2.5 widths, the
            # percentile and mean lines no longer stood out from that busier
            # background.
            # nan-aware: under available_case a participant is NaN outside
            # its own support, so each point uses only the participants
            # actually covering it.
            lower, upper = np.nanpercentile(arm, [lower_q, upper_q], axis=0)
            axis.plot(t, lower, color=color, linestyle="--", linewidth=2.0, zorder=2)
            axis.plot(t, upper, color=color, linestyle="--", linewidth=2.0, zorder=2)
            axis.plot(t, np.nanmean(arm, axis=0), color=color, linewidth=3.2,
                      label=names[label], zorder=4)

        _paper_axis(axis, xlabel="Study follow-up week", ylabel="Mixture weight",
                    xlim=(float(t[0]), float(t[-1])), ylim=mixed_ylim)
        # No legend: the manuscript/caption carries the arm identification.
        fig.tight_layout()
        generated.extend(_save_figure(
            fig, output_folder, f"Mixed_comp{component + 1}_bands"
        ))

    return generated



def _regularize_covariance(covariance, min_eigenvalue=1e-8):
    """Symmetrise and make a covariance numerically positive definite."""
    covariance = np.asarray(covariance, dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    eigvals = np.linalg.eigvalsh(covariance)
    minimum = float(np.min(eigvals))
    if minimum < min_eigenvalue:
        covariance = covariance + (min_eigenvalue - minimum) * np.eye(covariance.shape[0])
    return covariance


def _mixture_cdf_1d(x, means, covs, weights):
    """CDF of a one-dimensional Gaussian mixture."""
    x = np.asarray(x, dtype=float)
    means = np.asarray(means, dtype=float).reshape(-1)
    covs = np.asarray(covs, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    weights = np.clip(weights, 0.0, None)
    weights = weights / weights.sum()

    variances = covs[:, 0, 0] if covs.ndim == 3 else covs.reshape(-1)
    scales = np.sqrt(np.maximum(variances, 1e-12))

    z = (x[:, None] - means[None, :]) / scales[None, :]
    return np.sum(weights[None, :] * norm.cdf(z), axis=1)



def _mixture_pdf_1d(x, means, covs, weights):
    """PDF of a one-dimensional Gaussian mixture."""
    x = np.asarray(x, dtype=float)
    means = np.asarray(means, dtype=float).reshape(-1)
    covs = np.asarray(covs, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    weights = np.clip(weights, 0.0, None)
    weights = weights / weights.sum()

    variances = covs[:, 0, 0] if covs.ndim == 3 else covs.reshape(-1)
    scales = np.sqrt(np.maximum(variances, 1e-12))
    z = (x[:, None] - means[None, :]) / scales[None, :]
    component_pdf = np.exp(-0.5 * z**2) / (scales[None, :] * np.sqrt(2.0 * np.pi))
    return np.sum(weights[None, :] * component_pdf, axis=1)

def _wasserstein_barycenter_density_1d(
    subject_weights,
    means,
    covs,
    plot_grid,
    n_quantiles=6000,
    inversion_grid_size=16000,
    kde_bw=None,
):
    """Equal-weight W2 barycenter density of one-dimensional GMMs.

    In one dimension the W2 Fréchet mean is exact in quantile coordinates:

        Q_bar(u) = (1/n) sum_i Q_i(u).

    No KDE is used here.  Once the subject quantiles are known, use

        Q_i'(u) = 1 / f_i(Q_i(u))

    and hence

        f_bar(Q_bar(u)) = 1 / mean_i[1 / f_i(Q_i(u))].

    This avoids the arbitrary extra smoothing that made the previous W2
    implementation look much smoother than the historical paper figures.
    """
    subject_weights = np.asarray(subject_weights, dtype=float)
    means = np.asarray(means, dtype=float)
    covs = np.asarray(covs, dtype=float)
    plot_grid = np.asarray(plot_grid, dtype=float)

    component_means = means[:, 0]
    component_sd = np.sqrt(np.maximum(covs[:, 0, 0], 1e-12))

    inverse_lower = float(np.min(component_means - 9.0 * component_sd))
    inverse_upper = float(np.max(component_means + 9.0 * component_sd))
    inverse_grid = np.linspace(inverse_lower, inverse_upper, int(inversion_grid_size))

    probabilities = np.linspace(5e-4, 1.0 - 5e-4, int(n_quantiles))
    quantile_functions = []
    quantile_densities = []

    for weights in subject_weights:
        cdf = _mixture_cdf_1d(inverse_grid, means, covs, weights)
        cdf = np.maximum.accumulate(cdf)
        cdf_unique, unique_index = np.unique(cdf, return_index=True)
        x_unique = inverse_grid[unique_index]

        if cdf_unique[0] > probabilities[0] or cdf_unique[-1] < probabilities[-1]:
            raise RuntimeError(
                "1-D GMM inversion grid is too narrow for the requested quantiles"
            )

        q = np.interp(probabilities, cdf_unique, x_unique)
        fq = _mixture_pdf_1d(q, means, covs, weights)
        fq = np.maximum(fq, 1e-14)

        quantile_functions.append(q)
        quantile_densities.append(fq)

    quantile_functions = np.stack(quantile_functions, axis=0)
    quantile_densities = np.stack(quantile_densities, axis=0)

    barycenter_quantiles = np.mean(quantile_functions, axis=0)
    barycenter_density_at_quantiles = 1.0 / np.mean(1.0 / quantile_densities, axis=0)

    # Q_bar is monotone. Remove accidental numerical duplicates before
    # interpolation of f_bar from quantile locations to the display grid.
    q_unique, unique_index = np.unique(barycenter_quantiles, return_index=True)
    f_unique = barycenter_density_at_quantiles[unique_index]

    density = np.interp(
        plot_grid,
        q_unique,
        f_unique,
        left=0.0,
        right=0.0,
    )
    density = np.maximum(density, 0.0)

    integral = np.trapezoid(density, plot_grid)
    if integral > 0:
        density /= integral

    return density, barycenter_quantiles

def _sample_from_gmm(means, covs, weights, n_samples, rng):
    """Draw reproducible samples from a Gaussian mixture."""
    means = np.asarray(means, dtype=float)
    covs = np.asarray(covs, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)

    weights = np.clip(weights, 0.0, None)
    total = weights.sum()
    if total <= 0:
        raise ValueError("GMM weights must contain positive mass")
    weights = weights / total

    n_components, dimension = means.shape
    components = rng.choice(n_components, size=int(n_samples), p=weights)
    sample = np.empty((int(n_samples), dimension), dtype=float)

    for component in range(n_components):
        positions = np.flatnonzero(components == component)
        if positions.size == 0:
            continue
        covariance = _regularize_covariance(covs[component])
        sample[positions] = rng.multivariate_normal(
            means[component],
            covariance,
            size=positions.size,
        )

    return sample


def _wasserstein_barycenter_support_2d(
    subject_samples,
    n_support=250,
    max_iter=100,
    stop_thr=1e-7,
    seed=42,
):
    """Empirical free-support W2 barycenter of bivariate subject distributions.

    This uses POT's exact discrete optimal-transport solver inside
    ``ot.lp.free_support_barycenter``.  Each participant receives equal
    Fréchet weight, irrespective of treatment-arm sample size.
    """
    try:
        import ot
    except ImportError as exc:
        raise ImportError(
            "The d=2 Wasserstein barycenter requires POT (Python Optimal "
            "Transport). Install it in the project environment with `pip install POT`."
        ) from exc

    subject_samples = [np.asarray(sample, dtype=float) for sample in subject_samples]
    if not subject_samples:
        raise ValueError("No subject samples supplied to Wasserstein barycenter")
    if any(sample.ndim != 2 or sample.shape[1] != 2 for sample in subject_samples):
        raise ValueError("All d=2 subject samples must have shape (n, 2)")

    rng = np.random.default_rng(int(seed))
    pooled = np.vstack(subject_samples)
    n_support = int(min(int(n_support), len(pooled)))
    if n_support < 2:
        raise ValueError("Wasserstein barycenter needs at least two support points")

    init_index = rng.choice(len(pooled), size=n_support, replace=False)
    x_init = pooled[init_index].copy()

    measures_locations = subject_samples
    measures_weights = [
        np.full(len(sample), 1.0 / len(sample), dtype=float)
        for sample in subject_samples
    ]
    barycenter_weights = np.full(n_support, 1.0 / n_support, dtype=float)
    participant_weights = np.full(
        len(subject_samples),
        1.0 / len(subject_samples),
        dtype=float,
    )

    support = ot.lp.free_support_barycenter(
        measures_locations,
        measures_weights,
        X_init=x_init,
        b=barycenter_weights,
        weights=participant_weights,
        numItermax=int(max_iter),
        stopThr=float(stop_thr),
        verbose=False,
    )
    return np.asarray(support, dtype=float), barycenter_weights


def _kde_density_2d_from_standardized_support(
    support_standardized,
    support_weights,
    x_grid_original,
    y_grid_original,
    scaling_mean,
    scaling_std,
    kde_bw="scott",
):
    """Visualise a standardized-space W2 barycenter in clinical units.

    The Wasserstein barycenter is computed in the globally standardized
    analysis space.  KDE is also performed in that space.  The evaluation
    grid is mapped from original clinical units to standardized coordinates,
    and the density is transformed back using the affine Jacobian

        f_X(x) = f_Z((x - mean) / std) / prod(std).

    This avoids reintroducing the original unit imbalance through the
    visualisation smoother.
    """
    support_standardized = np.asarray(support_standardized, dtype=float)
    support_weights = np.asarray(support_weights, dtype=float)
    scaling_mean = np.asarray(scaling_mean, dtype=float)
    scaling_std = np.asarray(scaling_std, dtype=float)

    if support_standardized.ndim != 2 or support_standardized.shape[1] != 2:
        raise ValueError("standardized barycenter support must have shape (n, 2)")
    if scaling_mean.shape != (2,) or scaling_std.shape != (2,):
        raise ValueError("d=2 scaling mean/std must each have shape (2,)")
    if np.any(scaling_std <= 0):
        raise ValueError("scaling standard deviations must be positive")

    kde = gaussian_kde(
        support_standardized.T,
        weights=support_weights,
        bw_method=kde_bw,
    )

    xx, yy = np.meshgrid(x_grid_original, y_grid_original)
    original_positions = np.column_stack([xx.ravel(), yy.ravel()])
    standardized_positions = (
        original_positions - scaling_mean[None, :]
    ) / scaling_std[None, :]

    density_standardized = kde(standardized_positions.T).reshape(xx.shape)

    jacobian = float(np.prod(scaling_std))
    density_original = density_standardized / jacobian
    density_original = np.maximum(density_original, 0.0)

    # Numerical grid truncation can lose a negligible amount of mass.
    # Renormalise only for plotting on the displayed finite grid.
    dx = float(x_grid_original[1] - x_grid_original[0])
    dy = float(y_grid_original[1] - y_grid_original[0])
    integral = float(np.sum(density_original) * dx * dy)
    if integral > 0:
        density_original /= integral

    return xx, yy, density_original

def _blue_white_green_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "blue_white_green",
        ["#2c7fb8", "#f7f7f7", "#31a354"],
        N=256,
    )


def _blue_white_red_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "blue_white_red",
        ["#2c7fb8", "#f7f7f7", "#d62728"],
        N=256,
    )


def _format_scientific_colorbar(colorbar):
    from matplotlib.ticker import ScalarFormatter
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    colorbar.formatter = formatter
    colorbar.update_ticks()


def _silverman_bandwidth_1d(samples, n_effective):
    """Robust common bandwidth for DISPLAY of 1-D W2 barycenter densities."""
    samples = np.asarray(samples, dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if len(samples) < 2:
        raise ValueError("at least two finite barycenter quantiles are required")
    std = float(np.std(samples, ddof=1))
    q25, q75 = np.quantile(samples, [0.25, 0.75])
    robust = float((q75 - q25) / 1.349)
    positive = [v for v in (std, robust) if np.isfinite(v) and v > 0]
    scale = min(positive) if positive else max(std, 1.0)
    h = 0.9 * scale * float(max(int(n_effective), 2)) ** (-0.2)
    return max(float(h), 1e-6)


def _fixed_bandwidth_kde_1d(samples, grid, bandwidth, chunk_size=600):
    """Smooth the barycenter measure for plotting using one fixed bandwidth."""
    samples = np.asarray(samples, dtype=float).reshape(-1)
    grid = np.asarray(grid, dtype=float).reshape(-1)
    bandwidth = float(bandwidth)
    if bandwidth <= 0 or not np.isfinite(bandwidth):
        raise ValueError("display KDE bandwidth must be positive and finite")
    density = np.zeros_like(grid, dtype=float)
    normalizer = bandwidth * np.sqrt(2.0 * np.pi)
    for start in range(0, len(samples), int(chunk_size)):
        chunk = samples[start:start + int(chunk_size)]
        z = (grid[:, None] - chunk[None, :]) / bandwidth
        density += np.exp(-0.5 * z**2).sum(axis=1) / normalizer
    density /= len(samples)
    density = np.maximum(density, 0.0)
    integral = np.trapezoid(density, grid)
    if integral > 0:
        density /= integral
    return density


def _plot_signed_density_difference_1d(
    grid,
    difference,
    output_folder,
    stem,
    positive_color,
    bound=None,
):
    """Historical d=1 difference plot: signed fill + value-coloured curve."""
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.cm import ScalarMappable

    grid = np.asarray(grid, dtype=float)
    difference = np.asarray(difference, dtype=float)

    if bound is None:
        bound = float(np.max(np.abs(difference)))
    bound = float(bound)
    if bound <= 0:
        bound = 1e-12

    cmap = LinearSegmentedColormap.from_list(
        f"blue_white_{stem}",
        ["#2c7fb8", "#f7f7f7", positive_color],
        N=256,
    )
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)

    fig, axis = plt.subplots(figsize=(10.0, 5.0))

    axis.fill_between(
        grid, 0.0, difference,
        where=difference >= 0.0,
        color=positive_color,
        alpha=0.16,
        interpolate=True,
    )
    axis.fill_between(
        grid, 0.0, difference,
        where=difference < 0.0,
        color="#2c7fb8",
        alpha=0.13,
        interpolate=True,
    )

    points = np.column_stack([grid, difference]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    segment_values = 0.5 * (difference[:-1] + difference[1:])
    collection = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidth=2.1,
        alpha=0.95,
    )
    collection.set_array(segment_values)
    axis.add_collection(collection)

    axis.axhline(0.0, color="black", linewidth=0.75, alpha=0.7)
    axis.set_xlim(float(grid[0]), float(grid[-1]))
    axis.set_ylim(-1.08 * bound, 1.08 * bound)
    axis.set_xticks([0, 100, 200, 300, 400])
    axis.tick_params(axis="both", which="major", labelsize=13)
    axis.grid(alpha=0.16, linestyle="--", linewidth=0.6)
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.set_title("")

    # Match the historical standalone figures: no title/legend/axis labels.
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    colorbar = fig.colorbar(mappable, ax=axis, fraction=0.040, pad=0.025)
    _format_scientific_colorbar(colorbar)

    fig.tight_layout()
    return _save_figure(fig, output_folder, stem)


def _plot_density_figure(
    data,
    output_folder,
    *,
    wasserstein_samples=250,
    wasserstein_support=250,
    wasserstein_seed=42,
    wasserstein_max_iter=100,
    wasserstein_stop_thr=1e-7,
    kde_bw="scott",
    d1_display_bandwidth=None,
    expected_dimension=None,
):
    """Endpoint group Fréchet means under the W2 metric.

    d=1:
        exact 1-D W2 barycenter in quantile coordinates; its density is
        reconstructed from the quantile derivative identity, with no KDE.

    d=2:
        empirical free-support W2 barycenter of samples drawn from each
        participant GMM, followed by KDE for display.

    The Fréchet objective gives every participant equal weight within arm.
    """
    trajectories = np.asarray(data["trajectories"], dtype=float)
    labels = np.asarray(data["labels"], dtype=int)
    means = np.asarray(data["means_original_units"], dtype=float)
    covs = np.asarray(data["covs_original_units"], dtype=float)
    means_standardized = np.asarray(data["means_standardized"], dtype=float)
    covs_standardized = np.asarray(data["covs_standardized"], dtype=float)
    scaling_mean = np.asarray(data["scaling"]["mean"], dtype=float)
    scaling_std = np.asarray(data["scaling"]["std"], dtype=float)
    scaling_method = str(data["scaling"]["method"])
    dimension = means.shape[1]

    arm_names = ARM_NAMES
    arm_colors = ARM_COLORS
    generated = []

    # Each descriptive endpoint uses only participants covering that time.
    endpoint_subject_weights = {}
    endpoint_subject_counts = {}
    for label in (0, 1):
        arm = trajectories[labels == label]
        for when, column in (("initial", 0), ("final", -1)):
            values = arm[:, column, :]
            covered = np.isfinite(values).all(axis=1)
            endpoint_subject_weights[(label, when)] = values[covered]
            endpoint_subject_counts[(label, when)] = (
                int(covered.sum()), int(values.shape[0])
            )
            if not covered.any():
                raise RuntimeError(
                    f"No participant in arm {ARM_NAMES[label]} covers the "
                    f"{when} endpoint of the common grid; narrow the interval "
                    "or use common_grid_mode='intersection'."
                )
    print(
        "[density endpoints] participants covering each endpoint: "
        + ", ".join(
            f"{ARM_NAMES[label]}/{when} {endpoint_subject_counts[(label, when)][0]}"
            f"/{endpoint_subject_counts[(label, when)][1]}"
            for label in (0, 1) for when in ("initial", "final")
        )
    )

    # -----------------------------------------------------------------
    # d = 1: true 1-D W2 barycenter via mean quantile function
    # -----------------------------------------------------------------
    if dimension == 1:
        component_sd = np.sqrt(np.maximum(covs[:, 0, 0], 1e-12))

        # Preserve the visual range of the historical d=1 paper panels while
        # expanding automatically if the current dictionary requires it.
        lower = min(-40.0, float(np.min(means[:, 0] - 5.0 * component_sd)))
        upper = max(450.0, float(np.max(means[:, 0] + 5.0 * component_sd)))
        grid = np.linspace(lower, upper, 1200)

        exact_densities = {}
        barycenter_quantiles = {}
        for label in (0, 1):
            for when in ("initial", "final"):
                exact_density, quantiles = _wasserstein_barycenter_density_1d(
                    endpoint_subject_weights[(label, when)],
                    means,
                    covs,
                    grid,
                    n_quantiles=6000,
                    inversion_grid_size=16000,
                    kde_bw=kde_bw,
                )
                exact_densities[(label, when)] = exact_density
                barycenter_quantiles[(label, when)] = quantiles

        # Q_bar is the actual W2 barycenter.  Its numerical derivative can look
        # artificially jagged after CDF inversion.  For the PAPER FIGURE ONLY,
        # render the same barycenter measure by KDE on Q_bar(u), with ONE shared
        # bandwidth for Treatment/Control and initial/final.  This keeps the
        # Fréchet/Wasserstein object unchanged while avoiding subtraction of
        # four differently-smoothed curves.
        pooled_q = np.concatenate([
            barycenter_quantiles[(label, when)]
            for label in (0, 1) for when in ("initial", "final")
        ])
        if d1_display_bandwidth is None:
            display_bandwidth = _silverman_bandwidth_1d(
                pooled_q,
                n_effective=len(next(iter(barycenter_quantiles.values()))),
            )
        else:
            display_bandwidth = float(d1_display_bandwidth)
            if display_bandwidth <= 0 or not np.isfinite(display_bandwidth):
                raise ValueError("d1_display_bandwidth must be positive and finite")

        densities = {
            key: _fixed_bandwidth_kde_1d(q, grid, display_bandwidth)
            for key, q in barycenter_quantiles.items()
        }

        # Historical standalone initial/final comparison:
        # Treatment green, Control red, no internal legend/title/axis labels.
        # Use the SAME vertical scale for initial and final.
        global_density_max = max(
            float(np.max(density))
            for density in densities.values()
        )
        density_ymin = -0.00015 * max(1.0, global_density_max / 0.008)
        density_ymax = 1.06 * global_density_max

        for when in ("initial", "final"):
            fig, axis = plt.subplots(figsize=(10.0, 5.0))
            axis.plot(
                grid,
                densities[(1, when)],
                color=arm_colors[1],
                linewidth=2.25,
            )
            axis.plot(
                grid,
                densities[(0, when)],
                color=arm_colors[0],
                linewidth=2.25,
            )
            axis.set_xlim(float(grid[0]), float(grid[-1]))
            axis.set_ylim(density_ymin, density_ymax)
            axis.set_xticks([0, 100, 200, 300, 400])
            axis.tick_params(axis="both", which="major", labelsize=13)
            axis.grid(alpha=0.16, linestyle="--", linewidth=0.6)
            axis.set_xlabel("")
            axis.set_ylabel("")
            axis.set_title("")
            fig.tight_layout()
            generated.extend(
                _save_figure(
                    fig,
                    output_folder,
                    f"density_{when}_comparison_frechet",
                )
            )

        # Historical signed final-minus-initial panels.
        # Treatment and Control use the SAME y-range and colour scale.
        treatment_difference = densities[(1, "final")] - densities[(1, "initial")]
        control_difference = densities[(0, "final")] - densities[(0, "initial")]
        global_diff_bound = max(
            float(np.max(np.abs(treatment_difference))),
            float(np.max(np.abs(control_difference))),
            1e-12,
        )

        generated.extend(
            _plot_signed_density_difference_1d(
                grid,
                treatment_difference,
                output_folder,
                "density_frechet_difference_measure_treatment",
                positive_color="#31a354",
                bound=global_diff_bound,
            )
        )
        generated.extend(
            _plot_signed_density_difference_1d(
                grid,
                control_difference,
                output_folder,
                "density_frechet_difference_measure_control",
                positive_color="#d62728",
                bound=global_diff_bound,
            )
        )

        # Optional compact composite matching the same quantities.
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6), sharex=True)

        for col, when in enumerate(("initial", "final")):
            ax = axes[0, col]
            ax.plot(grid, densities[(1, when)], color=arm_colors[1], linewidth=2.0)
            ax.plot(grid, densities[(0, when)], color=arm_colors[0], linewidth=2.0)
            ax.set_title(when.capitalize(), fontsize=12)
            ax.grid(alpha=0.16, linestyle="--", linewidth=0.6)
            ax.tick_params(labelsize=11)

        for col, (label, difference) in enumerate(
            ((0, control_difference), (1, treatment_difference))
        ):
            ax = axes[1, col]
            positive_color = "#d62728" if label == 0 else "#31a354"
            ax.fill_between(
                grid, 0.0, difference,
                where=difference >= 0.0,
                color=positive_color,
                alpha=0.20,
                interpolate=True,
            )
            ax.fill_between(
                grid, 0.0, difference,
                where=difference < 0.0,
                color="#2c7fb8",
                alpha=0.16,
                interpolate=True,
            )
            ax.plot(grid, difference, color=arm_colors[label], linewidth=1.7)
            ax.axhline(0.0, color="black", linewidth=0.65, alpha=0.7)
            ax.set_ylim(-1.05 * global_diff_bound, 1.05 * global_diff_bound)
            ax.set_title(f"Difference: {arm_names[label]}", fontsize=12)
            ax.grid(alpha=0.16, linestyle="--", linewidth=0.6)
            ax.tick_params(labelsize=11)

        for ax in axes[-1]:
            ax.set_xlabel("Glucose concentration (mg/dL)")
        axes[0, 0].set_ylabel("Density")
        axes[1, 0].set_ylabel("Density difference")
        fig.tight_layout()
        generated.extend(
            _save_figure(fig, output_folder, "density_initial_final_difference")
        )
        return generated

    # -----------------------------------------------------------------
    # d = 2: empirical free-support W2 barycenter
    # -----------------------------------------------------------------
    if dimension != 2:
        raise ValueError(
            f"Paper density figure supports d=1 or d=2, got d={dimension}"
        )

    marginal_sd = np.sqrt(np.maximum(covs[:, [0, 1], [0, 1]], 1e-12))
    x_lower = max(20.0, float(np.min(means[:, 0] - 3.5 * marginal_sd[:, 0])))
    x_upper = min(400.0, float(np.max(means[:, 0] + 3.5 * marginal_sd[:, 0])))
    y_lower = float(np.min(means[:, 1] - 3.5 * marginal_sd[:, 1]))
    y_upper = float(np.max(means[:, 1] + 3.5 * marginal_sd[:, 1]))
    x_grid = np.linspace(x_lower, x_upper, 280)
    y_grid = np.linspace(y_lower, y_upper, 240)

    densities = {}
    xx = yy = None

    for label in (0, 1):
        for when_index, when in enumerate(("initial", "final")):
            # Independent but reproducible RNG stream for each arm/time cell.
            local_seed = (
                int(wasserstein_seed)
                + 1000 * int(label)
                + 100 * int(when_index)
            )
            rng = np.random.default_rng(local_seed)

            if scaling_method != "global_zscore":
                raise RuntimeError(
                    "d=2 Wasserstein barycenters are expected to use the "
                    f"global_zscore analysis space, got {scaling_method!r}"
                )

            # IMPORTANT: W2 is computed in the SAME globally standardized
            # space in which the d=2 MMD-GMM dictionary was fitted.
            # Therefore glucose and rate-of-change contribute on comparable
            # scales instead of mg/dL numerically dominating the transport.
            subject_samples_standardized = [
                _sample_from_gmm(
                    means_standardized,
                    covs_standardized,
                    subject_weight,
                    n_samples=wasserstein_samples,
                    rng=rng,
                )
                for subject_weight in endpoint_subject_weights[(label, when)]
            ]

            barycenter_support_standardized, barycenter_weights = (
                _wasserstein_barycenter_support_2d(
                    subject_samples_standardized,
                    n_support=wasserstein_support,
                    max_iter=wasserstein_max_iter,
                    stop_thr=wasserstein_stop_thr,
                    seed=local_seed + 17,
                )
            )

            # Only the DISPLAY is brought back to clinical units.
            # KDE is evaluated in standardized coordinates and transformed
            # with the correct Jacobian.
            xx, yy, density = _kde_density_2d_from_standardized_support(
                barycenter_support_standardized,
                barycenter_weights,
                x_grid,
                y_grid,
                scaling_mean=scaling_mean,
                scaling_std=scaling_std,
                kde_bw=kde_bw,
            )
            densities[(label, when)] = density

    absolute_max = max(float(np.max(value)) for value in densities.values())
    absolute_levels = np.linspace(0.0, absolute_max, 301)

    difference_max = max(
        float(
            np.max(
                np.abs(
                    densities[(label, "final")]
                    - densities[(label, "initial")]
                )
            )
        )
        for label in (0, 1)
    )
    difference_max = max(difference_max, 1e-12)
    difference_levels = np.linspace(-difference_max, difference_max, 301)

    # Six historical-style standalone d=2 files.
    for label in (0, 1):
        cmap = "Greens" if label == 1 else "Reds"
        arm = arm_names[label].lower()

        for when in ("initial", "final"):
            fig, axis = plt.subplots(figsize=(10.0, 5.0))
            axis.contourf(
                xx,
                yy,
                densities[(label, when)],
                levels=absolute_levels,
                cmap=cmap,
                extend="max",
                alpha=0.92,
            )
            _paper_axis(
                axis,
                xlabel="Glucose concentration (mg/dL)",
                ylabel="Glucose rate of change (mg/dL/time)",
                xlim=(x_lower, x_upper),
                ylim=(y_lower, y_upper),
            )
            fig.tight_layout()
            generated.extend(
                _save_figure(
                    fig,
                    output_folder,
                    f"density_{when}_{arm}_frechet",
                )
            )

        difference = densities[(label, "final")] - densities[(label, "initial")]
        difference_cmap = (
            _blue_white_green_cmap()
            if label == 1
            else _blue_white_red_cmap()
        )
        fig, axis = plt.subplots(figsize=(10.0, 5.0))
        contour = axis.contourf(
            xx,
            yy,
            difference,
            levels=difference_levels,
            cmap=difference_cmap,
            extend="both",
            alpha=0.95,
        )
        colorbar = fig.colorbar(
            contour,
            ax=axis,
            fraction=0.045,
            pad=0.035,
        )
        _format_scientific_colorbar(colorbar)
        _paper_axis(
            axis,
            xlabel="Glucose concentration (mg/dL)",
            ylabel="Glucose rate of change (mg/dL/time)",
            xlim=(x_lower, x_upper),
            ylim=(y_lower, y_upper),
        )
        fig.tight_layout()
        generated.extend(
            _save_figure(
                fig,
                output_folder,
                f"density_frechet_difference_measure_{arm}",
            )
        )

    # 2x3 manuscript composite: Control row, then Treatment row.
    fig, axes = plt.subplots(
        2, 3,
        figsize=(14.0, 7.3),
        sharex=True,
        sharey=True,
    )
    for row, label in enumerate((0, 1)):
        cmap = "Reds" if label == 0 else "Greens"
        for col, when in enumerate(("initial", "final")):
            axes[row, col].contourf(
                xx,
                yy,
                densities[(label, when)],
                levels=absolute_levels,
                cmap=cmap,
                extend="max",
                alpha=0.92,
            )
            axes[row, col].set_title(
                f"{arm_names[label]}: {when.capitalize()}",
                fontsize=12,
            )

        difference = densities[(label, "final")] - densities[(label, "initial")]
        difference_cmap = (
            _blue_white_green_cmap()
            if label == 1
            else _blue_white_red_cmap()
        )
        contour = axes[row, 2].contourf(
            xx,
            yy,
            difference,
            levels=difference_levels,
            cmap=difference_cmap,
            extend="both",
            alpha=0.95,
        )
        axes[row, 2].set_title(
            f"{arm_names[label]}: Difference",
            fontsize=12,
        )
        colorbar = fig.colorbar(
            contour,
            ax=axes[row, 2],
            fraction=0.046,
            pad=0.04,
        )
        _format_scientific_colorbar(colorbar)

    for col in range(3):
        _paper_axis(
            axes[1, col],
            xlabel="Glucose concentration (mg/dL)",
            xlim=(x_lower, x_upper),
            ylim=(y_lower, y_upper),
        )
    for row in range(2):
        _paper_axis(
            axes[row, 0],
            ylabel="Glucose rate of change (mg/dL/time)",
            xlim=(x_lower, x_upper),
            ylim=(y_lower, y_upper),
        )

    fig.tight_layout()
    generated.extend(
        _save_figure(
            fig,
            output_folder,
            "density_initial_final_difference",
        )
    )
    return generated

def _plot_window_sensitivity(data, output_folder):
    """Display support-based window choices independently of outcomes."""
    from Analysis.clinical_support import select_inference_window, inference_window_sensitivity
    supports = data["fitted_node_supports_original_week_coordinates"]
    labels = data["labels"]
    reports = inference_window_sensitivity(supports, labels, verbose=False)
    reports["0.90"] = select_inference_window(supports, labels, verbose=False)[-1]
    save_json(reports, output_folder, "window_sensitivity.json")
    fig, ax = plt.subplots(figsize=(7, 2.7), layout="constrained")
    for index, key in enumerate(sorted(reports)):
        report = reports[key]
        if "error" in report:
            continue
        low, high = report["window_weeks"]
        ax.plot([low, high], [index, index], marker="|", markersize=12, linewidth=3)
        text = (f"CLC {report['n_retained_clc']}/{report['n_total_clc']}; "
                f"SC {report['n_retained_sc']}/{report['n_total_sc']}")
        ax.text((low + high) / 2, index + 0.15, text, ha="center", fontsize=9)
    ax.set(yticks=range(3), yticklabels=["85%", "90% (primary)", "95%"],
           xlabel="Weeks since randomization", ylabel="Minimum coverage per arm",
           xlim=(0, 13), ylim=(-0.4, 2.5))
    ax.spines[["top", "right"]].set_visible(False)
    return _save_figure(fig, output_folder, "inference_window_sensitivity")


def generate_paper_figures(
    treatment_ids, control_ids, root_folder, interval=PAPER_INTERVAL,
    n_grid=PAPER_N_GRID, band_pct=PAPER_BAND_PCT, output_folder=None,
    expected_architecture=None, expected_dimension=None,
):
    """Generate available-case group trajectories with coverage counts."""
    from Analysis.clinical_support import (
        GROUP_CURVE_RULE, coverage_table, validate_rct_run, verify_interval_data,
    )
    params = validate_rct_run(root_folder)
    dimension = int(params["d"])
    if output_folder is None:
        output_folder = FIGURES_ROOT / f"d{dimension}" / Path(root_folder).name
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    data = _load_interval_data(
        treatment_ids, control_ids, root_folder, interval=interval,
        n_grid=n_grid, expected_architecture=expected_architecture,
    )
    actual_dimension = int(data["means_original_units"].shape[1])
    if expected_dimension is not None and actual_dimension != int(expected_dimension):
        raise RuntimeError("Selected run has the wrong dimension")
    report = verify_interval_data(data)
    files = _plot_trajectory_figures(data, output_folder, band_pct=band_pct)
    files += _plot_window_sensitivity(data, output_folder)
    coverage_csv = Path(output_folder) / "group_curve_coverage.csv"
    coverage_table(data).to_csv(coverage_csv, index=False)
    labels = np.asarray(data["labels"])
    metadata = {
        "dataset_version": "pedap_v2",
        "dataset_fingerprint": params["dataset_fingerprint"],
        "source_results": str(Path(root_folder).resolve()),
        "dimension": actual_dimension,
        "n_components": int(data["trajectories"].shape[2]),
        "group_curve_rule": GROUP_CURVE_RULE,
        "common_grid_mode": "available_case",
        "coverage_report": report,
        "coverage_table_csv": str(coverage_csv.resolve()),
        "included_counts": {"treatment": int((labels == 1).sum()),
                            "control": int((labels == 0).sum())},
        "included_subject_ids": data["included_ids"],
        "excluded_subjects": data["excluded"],
        "figure_files": [str(Path(p).resolve()) for p in files],
    }
    save_json(metadata, output_folder, "figure_metadata.json")
    return metadata


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", choices=("d1", "d2"), required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-folder", type=Path, default=None)
    parser.add_argument("--interval-start", type=float, default=PAPER_INTERVAL[0])
    parser.add_argument("--interval-end", type=float, default=PAPER_INTERVAL[1])
    parser.add_argument("--n-grid", type=int, default=PAPER_N_GRID)
    parser.add_argument("--band-pct", type=float, default=PAPER_BAND_PCT)
    parser.add_argument("--selection-json", type=Path, default=None)
    return parser.parse_args()


def _locked_architecture(selection_json):
    if selection_json is None:
        return None
    with selection_json.open("r", encoding="utf-8") as handle:
        selection = json.load(handle)
    name = selection.get("decision", {}).get("selected_architecture")
    matches = [
        item for item in selection.get("configuration", {}).get("architectures", [])
        if item.get("name") == name
    ]
    if len(matches) != 1:
        raise RuntimeError("selection JSON has no unique selected architecture")
    return int(matches[0]["hidden_dim"]), int(matches[0]["n_layers"])


def main():
    args = _parse_args()
    run_root = args.run.resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(f"Results folder does not exist: {run_root}")

    expected_dimension = 1 if args.dimension == "d1" else 2
    output_folder = (
        args.output_folder.resolve()
        if args.output_folder is not None
        else FIGURES_ROOT / args.dimension / run_root.name
    )
    expected_architecture = _locked_architecture(args.selection_json)
    metadata = generate_paper_figures(
        TREATMENT_IDS,
        CONTROL_IDS,
        run_root,
        interval=(args.interval_start, args.interval_end),
        n_grid=args.n_grid,
        band_pct=args.band_pct,
        output_folder=output_folder,
        expected_architecture=expected_architecture,
        expected_dimension=expected_dimension,
    )
    counts = metadata["included_counts"]
    print(
        f"{args.dimension}: generated {len(metadata['figure_files'])} files; "
        f"Treatment n={counts['treatment']}, Control n={counts['control']}; "
        f"saved to {Path(output_folder).resolve()}"
    )


if __name__ == "__main__":
    main()
