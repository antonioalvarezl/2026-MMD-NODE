"""Bivariate Gaussian-mixture Rosenblatt diagnostics for saved CGM fits.

The transform is evaluated twice at every observed block: with the fitted
discrete MMD weights and with the saved NODE evaluated at that exact block
time.  Reported discrepancy measures are descriptive.  In particular, no
i.i.d. KS p-values are attached to five-minute CGM observations.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Allow direct execution via `python Analysis/<script>.py`.
PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp, ndtr
from scipy.stats import rankdata

from Analysis.paper_config import (
    DISPLAY_PATIENT_IDS,
    N_NOMINAL_WEEKS,
    PEDAP_DATA_DIR,
    dataset_fingerprint,
)
from load import load_patient_weeks


DEFAULT_DISPLAY_IDS = DISPLAY_PATIENT_IDS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--pedap-data-dir", type=Path, default=PEDAP_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--display-patient", type=int, action="append", default=None)
    parser.add_argument("--n-display-times", type=int, default=10)
    parser.add_argument("--variance-floor", type=float, default=1e-10)
    return parser.parse_args()


def normalize_weights(weights: np.ndarray, epsilon: float = 1e-15) -> np.ndarray:
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if not np.all(np.isfinite(weights)):
        raise ValueError("mixture weights must be finite")
    weights = np.maximum(weights, 0.0)
    total = float(weights.sum())
    if total <= epsilon:
        raise ValueError("mixture weights must have positive sum")
    return weights / total


def _validate_geometry(
    means: np.ndarray, covariances: np.ndarray, variance_floor: float
) -> tuple[np.ndarray, np.ndarray]:
    means = np.asarray(means, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    if means.ndim != 2 or means.shape[1] != 2:
        raise ValueError("means must have shape (K, 2)")
    if covariances.shape != (len(means), 2, 2):
        raise ValueError("covariances must have shape (K, 2, 2)")
    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(covariances)):
        raise ValueError("mixture geometry must be finite")
    if variance_floor <= 0 or not np.isfinite(variance_floor):
        raise ValueError("variance_floor must be positive and finite")
    covariances = 0.5 * (covariances + np.swapaxes(covariances, 1, 2))
    for index, covariance in enumerate(covariances):
        if np.min(np.linalg.eigvalsh(covariance)) <= 0:
            raise ValueError(f"covariance {index} is not positive definite")
    return means, covariances


def rosenblatt_transform_bivariate_gmm(
    observations: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    *,
    variance_floor: float = 1e-10,
    probability_clip: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the Rosenblatt coordinates ``(U1, U2)``.

    Coordinate order is glucose first and velocity conditional on glucose.
    Posterior component probabilities are evaluated in log space.
    """
    x = np.asarray(observations, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2 or not np.all(np.isfinite(x)):
        raise ValueError("observations must be a finite array of shape (n, 2)")
    means, covariances = _validate_geometry(means, covariances, variance_floor)
    alpha = normalize_weights(weights)
    if len(alpha) != len(means):
        raise ValueError("weights and geometry have different component counts")

    mu_g = means[:, 0]
    mu_v = means[:, 1]
    var_g = np.maximum(covariances[:, 0, 0], variance_floor)
    var_v = np.maximum(covariances[:, 1, 1], variance_floor)
    cov_vg = covariances[:, 1, 0]
    sigma_g = np.sqrt(var_g)
    g = x[:, 0][:, None]
    v = x[:, 1][:, None]

    standardized_g = (g - mu_g[None, :]) / sigma_g[None, :]
    u1 = np.sum(alpha[None, :] * ndtr(standardized_g), axis=1)

    conditional_mean = (
        mu_v[None, :]
        + (cov_vg / var_g)[None, :] * (g - mu_g[None, :])
    )
    conditional_variance_raw = var_v - cov_vg**2 / var_g
    conditional_variance = np.maximum(conditional_variance_raw, variance_floor)
    conditional_sigma = np.sqrt(conditional_variance)

    log_marginal_g = (
        -0.5 * standardized_g**2
        - np.log(sigma_g)[None, :]
        - 0.5 * np.log(2.0 * np.pi)
    )
    log_alpha = np.full_like(alpha, -np.inf)
    positive_alpha = alpha > 0.0
    log_alpha[positive_alpha] = np.log(alpha[positive_alpha])
    log_unnormalized = log_alpha[None, :] + log_marginal_g
    log_denominator = logsumexp(log_unnormalized, axis=1, keepdims=True)
    posterior = np.exp(log_unnormalized - log_denominator)
    standardized_conditional = (v - conditional_mean) / conditional_sigma[None, :]
    u2 = np.sum(posterior * ndtr(standardized_conditional), axis=1)

    if not np.all(np.isfinite(u1)) or not np.all(np.isfinite(u2)):
        raise FloatingPointError("non-finite Rosenblatt coordinate")
    lower = float(probability_clip)
    upper = 1.0 - lower
    return np.clip(u1, lower, upper), np.clip(u2, lower, upper)


def uniform_qq_coordinates(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.sort(np.asarray(values, dtype=float).reshape(-1))
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("QQ values must be finite and non-empty")
    theoretical = (np.arange(len(values), dtype=float) + 0.5) / len(values)
    return theoretical, values


def descriptive_uniform_metrics(values: np.ndarray) -> dict[str, float]:
    theoretical, empirical = uniform_qq_coordinates(values)
    difference = empirical - theoretical
    # Classical Cramer--von Mises discrepancy against Uniform(0,1), reported
    # descriptively only because the five-minute values are serially dependent.
    cramer_von_mises = 1.0 / (12.0 * len(empirical)) + np.sum(difference**2)
    return {
        "n": int(len(empirical)),
        "mean": float(np.mean(empirical)),
        "variance": float(np.var(empirical, ddof=0)),
        "cramer_von_mises_uniform_descriptive": float(cramer_von_mises),
        "qq_mean_squared_deviation": float(np.mean(difference**2)),
        "qq_mean_absolute_deviation": float(np.mean(np.abs(difference))),
        "qq_max_absolute_deviation": float(np.max(np.abs(difference))),
        "q10_minus_uniform": float(np.quantile(empirical, 0.10) - 0.10),
        "median_minus_uniform": float(np.median(empirical) - 0.50),
        "q90_minus_uniform": float(np.quantile(empirical, 0.90) - 0.90),
    }


def descriptive_rank_correlation(u1: np.ndarray, u2: np.ndarray) -> float:
    r1 = rankdata(u1)
    r2 = rankdata(u2)
    if np.std(r1) == 0 or np.std(r2) == 0:
        return 0.0
    return float(np.corrcoef(r1, r2)[0, 1])


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _matching_node(patient_dir: Path, weights: np.ndarray) -> tuple[Path, dict]:
    matches = []
    for path in sorted(patient_dir.glob("node_*/result_node.json")):
        node = _read_json(path)
        stored = np.asarray(
            node.get("node_config", {}).get("observation_weights", []), dtype=float
        )
        if stored.shape == weights.shape and np.allclose(stored, weights, atol=1e-10):
            matches.append((path, node))
    if not matches:
        raise FileNotFoundError("no NODE result matches fitted weights")
    return matches[-1]


def _node_weights_at_observations(node: dict) -> np.ndarray:
    trajectory = np.asarray(node["trajectory"], dtype=float)
    if trajectory.ndim == 3 and trajectory.shape[1] == 1:
        trajectory = trajectory[:, 0, :]
    grid = np.asarray(node["trajectory_time_grid_weeks"], dtype=float)
    times = np.asarray(node["observation_times_weeks"], dtype=float)
    if trajectory.ndim != 2 or len(grid) != len(trajectory):
        raise ValueError("invalid NODE trajectory")
    predicted = np.column_stack([
        np.interp(times, grid, trajectory[:, component])
        for component in range(trajectory.shape[1])
    ])
    predicted = np.maximum(predicted, 0.0)
    return predicted / predicted.sum(axis=1, keepdims=True)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _select_display_indices(n_times: int, requested: int) -> np.ndarray:
    n_display = min(n_times, requested)
    return np.unique(np.linspace(0, n_times - 1, n_display, dtype=int))


def _save_3d_qq(
    curves: list[dict], patient_id: int, coordinate: str, output_stem: Path
) -> None:
    fig = plt.figure(figsize=(9.4, 7.4))
    axis = fig.add_subplot(111, projection="3d")
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(curves)))
    key = coordinate.lower()
    for color, item in zip(colors, curves):
        elapsed = float(item["elapsed_weeks"])
        theoretical, discrete = uniform_qq_coordinates(item[f"{key}_discrete"])
        _, node = uniform_qq_coordinates(item[f"{key}_node"])
        z = np.full_like(theoretical, elapsed)
        axis.plot(theoretical, discrete, z, color=color, linewidth=2.0, alpha=0.90)
        axis.plot(theoretical, node, z, color=color, linewidth=1.3, linestyle=":", alpha=0.95)
        axis.plot(theoretical, theoretical, z, color="black", linewidth=0.7, linestyle="--", alpha=0.55)
    elapsed_values = [float(item["elapsed_weeks"]) for item in curves]
    axis.set(
        xlabel="Uniform quantiles",
        ylabel="Empirical quantiles",
        zlabel="Study follow-up week",
        xlim=(0, 1),
        ylim=(0, 1),
        zticks=elapsed_values,
        title=f"Patient {patient_id}: {coordinate} Rosenblatt QQ diagnostic",
    )
    axis.view_init(elev=24, azim=-62)
    # Minimal style legend independent of the time-specific colors.
    axis.plot([], [], [], color="black", linewidth=2.0, label="Discrete MMD weights")
    axis.plot([], [], [], color="black", linewidth=1.3, linestyle=":", label="NODE at same time")
    axis.plot([], [], [], color="black", linewidth=0.7, linestyle="--", label="Uniform reference")
    axis.legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_patient_summary(rows: list[dict], patient_id: int, output: Path) -> None:
    patient_rows = [row for row in rows if row["patient_id"] == patient_id]
    times = sorted({row["elapsed_weeks"] for row in patient_rows})
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), sharex=True)
    for axis, coordinate in zip(axes, ("u1", "u2")):
        for mode, linestyle in (("discrete_mmd", "-"), ("node_at_observed_time", ":")):
            values = [
                next(
                    row["qq_mean_squared_deviation"] for row in patient_rows
                    if row["elapsed_weeks"] == time
                    and row["coordinate"] == coordinate
                    and row["weight_source"] == mode
                )
                for time in times
            ]
            axis.plot(times, values, linestyle=linestyle, linewidth=1.8, label=mode)
        axis.set(
            xlabel="Study follow-up week",
            ylabel="QQ mean squared deviation",
            title=coordinate.upper(),
        )
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Patient {patient_id}: descriptive Rosenblatt discrepancies")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_diagnostics(args: argparse.Namespace) -> dict:
    run_dir = args.run.resolve()
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir / "rosenblatt_diagnostics"
    )
    output.mkdir(parents=True, exist_ok=True)
    # The run's own record decides which processed dataset its fits belong to,
    from Analysis.clinical_support import validate_rct_run
    params = validate_rct_run(run_dir)
    expected_fingerprint = params["dataset_fingerprint"]
    dictionary = _read_json(run_dir / "global_dictionary.json")
    if int(dictionary.get("d", 0)) != 2:
        raise ValueError("Rosenblatt diagnostics require the bivariate representation")
    scaling = dictionary["scaling"]
    scaling_mean = np.asarray(scaling["mean"], dtype=float)
    scaling_std = np.asarray(scaling["std"], dtype=float)
    display_ids = tuple(args.display_patient or DEFAULT_DISPLAY_IDS)
    rows = []
    patient_curves: dict[int, list[dict]] = {}
    provenance = []

    patient_dirs = sorted(
        run_dir.glob("patient_*"), key=lambda path: int(path.name.split("_", 1)[1])
    )
    for patient_dir in patient_dirs:
        fit_path = patient_dir / "result_fit.json"
        if not fit_path.exists():
            continue
        patient_id = int(patient_dir.name.split("_", 1)[1])
        fit = _read_json(fit_path)
        fit_config = fit.get("config", {})
        # A result is only read back against the dataset it was fitted on.
        if fit_config.get("dataset_fingerprint") != expected_fingerprint:
            raise RuntimeError(
                f"patient {patient_id}: result was fitted on PEDAP dataset "
                f"{fit_config.get('dataset_fingerprint')!r}, not "
                f"{expected_fingerprint!r}"
            )
        means = np.asarray(fit["results"]["means"], dtype=float)
        covariances = np.asarray(fit["results"]["covariances"], dtype=float)
        discrete_weights = np.asarray(fit["results"]["weights"], dtype=float)
        node_path, node = _matching_node(patient_dir, discrete_weights)
        node_weights = _node_weights_at_observations(node)
        times = np.asarray(fit["config"]["observation_times_weeks"], dtype=float)
        # A participant contributes the usable weekly distributions it has;
        # 13 is the nominal structure of the trial, not a requirement.
        if len(times) != len(discrete_weights) or len(times) > int(N_NOMINAL_WEEKS):
            raise RuntimeError(
                f"patient {patient_id}: {len(times)} observation times against "
                f"{len(discrete_weights)} fitted weekly states"
            )
        # The z axis is clinical time: weeks since randomization.
        display_times = times

        patient_data, metadata = load_patient_weeks(
            patient_id=patient_id,
            data_dir=str(args.pedap_data_dir),
            representation="bivariate",
            return_metadata=True,
        )
        if patient_id not in patient_data:
            raise RuntimeError(f"patient {patient_id} cannot be reconstructed")
        blocks = [(block - scaling_mean) / scaling_std for block in patient_data[patient_id]]
        if not (len(blocks) == len(times) == len(discrete_weights) == len(node_weights)):
            raise RuntimeError(f"patient {patient_id}: block/weight/time mismatch")
        if fit["config"].get("block_metadata") != metadata["blocks"]:
            raise RuntimeError(f"patient {patient_id}: preprocessing metadata changed")

        curves = []
        for block_index, (block, elapsed) in enumerate(zip(blocks, display_times)):
            transformed = {}
            for source, weights in (
                ("discrete_mmd", discrete_weights[block_index]),
                ("node_at_observed_time", node_weights[block_index]),
            ):
                u1, u2 = rosenblatt_transform_bivariate_gmm(
                    block, weights, means, covariances,
                    variance_floor=args.variance_floor,
                )
                transformed[source] = (u1, u2)
                rank_correlation = descriptive_rank_correlation(u1, u2)
                for coordinate, values in (("u1", u1), ("u2", u2)):
                    rows.append({
                        "patient_id": patient_id,
                        "block_index": block_index,
                        "elapsed_weeks": float(elapsed),
                        # v2 block metadata names these actual_start/actual_end
                        # (the real window of the weekly block); start_date /
                        # end_date belonged to the retired RDS-era schema.
                        "start_date": metadata["blocks"][block_index]["actual_start"],
                        "end_date": metadata["blocks"][block_index]["actual_end"],
                        "weight_source": source,
                        "coordinate": coordinate,
                        **descriptive_uniform_metrics(values),
                        "u1_u2_spearman_rho_descriptive": rank_correlation,
                    })
            curves.append({
                "block_index": block_index,
                "elapsed_weeks": float(elapsed),
                "u1_discrete": transformed["discrete_mmd"][0],
                "u2_discrete": transformed["discrete_mmd"][1],
                "u1_node": transformed["node_at_observed_time"][0],
                "u2_node": transformed["node_at_observed_time"][1],
            })
        patient_curves[patient_id] = curves
        provenance.append({
            "patient_id": patient_id,
            "n_blocks": len(blocks),
            "fit_result": str(fit_path.resolve()),
            "node_result": str(node_path.resolve()),
        })
        _save_patient_summary(
            rows, patient_id, output / "participants" / f"patient_{patient_id}_summary.png"
        )
        print(f"[Rosenblatt] patient {patient_id}: {len(blocks)} blocks", flush=True)

    for patient_id in display_ids:
        if patient_id not in patient_curves:
            raise RuntimeError(f"display patient {patient_id} is unavailable")
        curves = patient_curves[patient_id]
        display = [curves[index] for index in _select_display_indices(len(curves), args.n_display_times)]
        _save_3d_qq(
            display, patient_id, "U1", output / f"patient_{patient_id}_qq_u1_3d"
        )
        _save_3d_qq(
            display, patient_id, "U2", output / f"patient_{patient_id}_qq_u2_3d"
        )

    _write_csv(output / "rosenblatt_block_diagnostics.csv", rows)
    aggregate_rows = []
    for patient_id in sorted(patient_curves):
        for coordinate in ("u1", "u2"):
            by_source = {}
            for source in ("discrete_mmd", "node_at_observed_time"):
                subset = [
                    row for row in rows
                    if row["patient_id"] == patient_id
                    and row["weight_source"] == source
                    and row["coordinate"] == coordinate
                ]
                by_source[source] = {
                    "n_blocks": len(subset),
                    "mean_qq_msd": float(np.mean([
                        row["qq_mean_squared_deviation"] for row in subset
                    ])),
                    "median_qq_msd": float(np.median([
                        row["qq_mean_squared_deviation"] for row in subset
                    ])),
                    "mean_cvm": float(np.mean([
                        row["cramer_von_mises_uniform_descriptive"] for row in subset
                    ])),
                    "maximum_qq_absolute_deviation": float(max(
                        row["qq_max_absolute_deviation"] for row in subset
                    )),
                    "mean_absolute_rank_correlation": float(np.mean([
                        abs(row["u1_u2_spearman_rho_descriptive"]) for row in subset
                    ])),
                }
            mmd = by_source["discrete_mmd"]
            node = by_source["node_at_observed_time"]
            aggregate_rows.append({
                "patient_id": patient_id,
                "coordinate": coordinate,
                "n_blocks": mmd["n_blocks"],
                "mmd_mean_qq_msd": mmd["mean_qq_msd"],
                "node_mean_qq_msd": node["mean_qq_msd"],
                "node_minus_mmd_mean_qq_msd": node["mean_qq_msd"] - mmd["mean_qq_msd"],
                "mmd_median_qq_msd": mmd["median_qq_msd"],
                "node_median_qq_msd": node["median_qq_msd"],
                "mmd_mean_cvm_descriptive": mmd["mean_cvm"],
                "node_mean_cvm_descriptive": node["mean_cvm"],
                "node_minus_mmd_mean_cvm_descriptive": node["mean_cvm"] - mmd["mean_cvm"],
                "mmd_maximum_qq_absolute_deviation": mmd["maximum_qq_absolute_deviation"],
                "node_maximum_qq_absolute_deviation": node["maximum_qq_absolute_deviation"],
                "mmd_mean_absolute_u1_u2_rank_correlation": mmd["mean_absolute_rank_correlation"],
                "node_mean_absolute_u1_u2_rank_correlation": node["mean_absolute_rank_correlation"],
            })
    _write_csv(output / "rosenblatt_participant_summary.csv", aggregate_rows)
    global_rows = []
    for coordinate in ("u1", "u2"):
        subset = [row for row in aggregate_rows if row["coordinate"] == coordinate]
        global_rows.append({
            "coordinate": coordinate,
            "n_participants": len(subset),
            "participant_balanced_mmd_mean_qq_msd": float(np.mean([
                row["mmd_mean_qq_msd"] for row in subset
            ])),
            "participant_balanced_node_mean_qq_msd": float(np.mean([
                row["node_mean_qq_msd"] for row in subset
            ])),
            "participant_balanced_node_minus_mmd_mean_qq_msd": float(np.mean([
                row["node_minus_mmd_mean_qq_msd"] for row in subset
            ])),
            "fraction_participants_node_lower_qq_msd": float(np.mean([
                row["node_mean_qq_msd"] < row["mmd_mean_qq_msd"] for row in subset
            ])),
            "participant_balanced_mmd_mean_cvm_descriptive": float(np.mean([
                row["mmd_mean_cvm_descriptive"] for row in subset
            ])),
            "participant_balanced_node_mean_cvm_descriptive": float(np.mean([
                row["node_mean_cvm_descriptive"] for row in subset
            ])),
            "participant_balanced_node_minus_mmd_mean_cvm_descriptive": float(np.mean([
                row["node_minus_mmd_mean_cvm_descriptive"] for row in subset
            ])),
        })
    _write_csv(output / "rosenblatt_global_summary.csv", global_rows)
    payload = {
        "analysis": "descriptive bivariate Rosenblatt diagnostics",
        "run_directory": str(run_dir),
        "output_directory": str(output),
        "pedap_data_dir": str(args.pedap_data_dir),
        "dataset_fingerprint": expected_fingerprint,
        "n_participants": len(patient_curves),
        "n_blocks": len(rows) // 4,
        "weight_sources": ["discrete_mmd", "node_at_observed_time"],
        "coordinate_order": ["glucose", "velocity conditional on glucose"],
        "standardization": scaling,
        "variance_floor": args.variance_floor,
        "display_patient_ids": list(display_ids),
        "display_selection": (
            f"Display IDs {list(display_ids)} are the fixed choices in paper_config.py; "
            "they are retained for visual continuity and are not selected by goodness of fit."
        ),
        "display_time_selection": (
            f"{args.n_display_times} indices spaced deterministically from first to last "
            "available block; z-axis values are weeks since randomization."
        ),
        "inference_warning": (
            "These are in-sample descriptive diagnostics. Five-minute CGM observations "
            "are serially dependent, so ordinary i.i.d. KS p-values are intentionally omitted."
        ),
        "bandwidth_status": (
            "diagnostics use exactly the MMD weights and matching NODE saved in the selected run; "
            "fit/NODE provenance is recorded participant by participant"
        ),
        "historical_generator_recovery": {
            # Historical, pre-rebuild script; not part of this repository.
            "path": "Analysis/gof_bivariate_rosenblatt.py",
            "assessment": (
                "Its core transform formulas are correct, but its displayed curves use a "
                "representative held-out split whose weights are re-estimated by likelihood EM, "
                "not the saved MMD weights; it also reports i.i.d. KS p-values. Those choices "
                "are not carried into this canonical module."
            ),
        },
        "participants": provenance,
        "generated_files": {
            "block_csv": str((output / "rosenblatt_block_diagnostics.csv").resolve()),
            "participant_csv": str((output / "rosenblatt_participant_summary.csv").resolve()),
            "global_csv": str((output / "rosenblatt_global_summary.csv").resolve()),
            "six_display_figures": [
                str((output / f"patient_{patient_id}_qq_{coordinate}_3d.pdf").resolve())
                for patient_id in display_ids for coordinate in ("u1", "u2")
            ],
        },
    }
    with (output / "rosenblatt_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    print(f"[Rosenblatt] saved diagnostics to {output}")
    return payload


def main() -> None:
    run_diagnostics(_parse_args())


if __name__ == "__main__":
    main()
