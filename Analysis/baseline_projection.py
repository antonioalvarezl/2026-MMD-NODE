"""Project each participant's PRE-RANDOMIZATION distribution onto the run's
own MMD-GMM dictionary, giving the anchor for ``change_from_baseline``.

The old ``change`` estimand used each participant's FIRST fitted NODE state as
its baseline.  In PEDAP that state sits around week 1.5, by which time the CLC
arm has been on closed-loop for roughly ten days and the treatment effect is
already established, so that contrast measures post-effect drift rather than
the effect.  This module supplies the real anchor instead.

    change_from_baseline_i(t) = node_weights_i(t) - baseline_weights_i

Rules kept identical to the RCT analysis, deliberately:

* the SAME treatment-blind global dictionary (means, covariances) and the SAME
  global scaling that the run was fitted with -- never a refitted one;
* the SAME patient-specific local MMD bandwidth and ridge the run saved for
  that participant, so the baseline weights are estimated exactly as the
  weekly ones were;
* d=2 derivatives computed with the same rule, over the baseline series only.
  The baseline artefact contains nothing on or after the randomization date,
  so no derivative triple can straddle it; this module asserts that.

A participant without a >= 168 h baseline simply has no anchor: it is excluded
from ``change_from_baseline`` and stays available for ``absolute``.  Nothing is
imputed and no post-randomization data is ever used to build a baseline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PEDAP.baseline_rules import BASELINE_MIN_CGM_HOURS, is_baseline_usable
from Utils.MMD_utils import update_local_weights
from load import DERIVATIVE_TOLERANCE_SECONDS, compute_pedap_derivatives

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = PROJECT_ROOT / "data" / "processed" / "pedap_benchmark" / "pre_randomization_baseline"


def _manifest(directory: Path) -> dict:
    with open(Path(directory) / "manifest.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_baseline_frames(baseline_dir=BASELINE_DIR):
    """Per-participant baseline CGM plus the coverage table."""
    baseline_dir = Path(baseline_dir)
    manifest = _manifest(baseline_dir)
    cgm = pd.read_parquet(baseline_dir / manifest["files"]["cgm"])
    qc = pd.read_csv(baseline_dir / manifest["files"]["baseline_qc"])
    qc["baseline_end_exclusive"] = pd.to_datetime(qc["baseline_end_exclusive"])
    return cgm, qc, manifest


def baseline_points(patient_id, cgm, qc, representation="univariate",
                    derivative_tolerance_seconds=DERIVATIVE_TOLERANCE_SECONDS):
    """The participant's baseline distribution, in ORIGINAL clinical units.

    Returns ``(n, d)`` or ``None`` when the participant has no usable baseline.
    """
    patient_id = int(patient_id)
    row = qc[qc["PtID"] == patient_id]
    if row.empty or not bool(row["usable"].iloc[0]):
        return None
    if not is_baseline_usable(float(row["cgm_hours"].iloc[0])):
        return None

    frame = cgm[cgm["PtID"] == patient_id].sort_values("DeviceDtTm")
    if frame.empty:
        return None

    # HARD GUARANTEE: nothing on or after randomization may enter a baseline.
    cutoff = row["baseline_end_exclusive"].iloc[0]
    assert bool((frame["DeviceDtTm"] < cutoff).all()), (
        f"patient {patient_id}: a baseline reading falls on or after "
        "randomization"
    )

    values = frame["GlucoseMgDl"].to_numpy(dtype=float)
    if representation == "univariate":
        return values.reshape(-1, 1)
    if representation != "bivariate":
        raise ValueError("representation must be 'univariate' or 'bivariate'")

    derived = compute_pedap_derivatives(
        frame["DeviceDtTm"].to_numpy(dtype="datetime64[ns]"),
        values,
        frame["EventSubType"].to_numpy(),
        tolerance_seconds=derivative_tolerance_seconds,
    )
    if len(derived["derivative"]) == 0:
        return None
    # The derivative centres come from the baseline series alone, so they are
    # all strictly before randomization by construction; assert it anyway.
    assert bool(
        (derived["timestamps"] < np.datetime64(cutoff)).all()
    ), f"patient {patient_id}: a baseline derivative centre is not pre-randomization"
    return np.column_stack([derived["glucose"], derived["derivative"]])


def project_onto_dictionary(points, means, covs, scaling_mean, scaling_std,
                            bandwidth, lambda_ridge, w_init=None):
    """Local MMD weights of ``points`` on the FIXED dictionary.

    ``points`` are in original units; the dictionary lives in the run's
    standardized analysis space, so the same affine scaling is applied here.
    """
    points = np.asarray(points, dtype=float)
    means = np.asarray(means, dtype=float)
    covs = np.asarray(covs, dtype=float)
    d = means.shape[1]
    standardized = (points - np.asarray(scaling_mean)) / np.asarray(scaling_std)

    bandwidth = float(bandwidth)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("baseline projection needs a positive bandwidth")
    gamma = (bandwidth ** 2) * np.eye(d)

    if w_init is None:
        w_init = np.full(means.shape[0], 1.0 / means.shape[0])
    weights = update_local_weights(
        standardized, means, covs, gamma, lambda_ridge, np.asarray(w_init, float).copy()
    )
    weights = np.asarray(weights, dtype=float)
    if not np.all(np.isfinite(weights)) or abs(weights.sum() - 1.0) > 1e-5:
        raise RuntimeError("baseline projection produced invalid mixture weights")
    return weights


def baseline_weights_for_run(run_folder, patient_ids, baseline_dir=BASELINE_DIR,
                             verbose=True):
    """``{patient_id: baseline_weights}`` using the run's own dictionary.

    Participants without a usable baseline are simply absent from the mapping;
    the reasons are returned alongside.
    """
    run_folder = Path(run_folder)
    with open(run_folder / "global_dictionary.json", "r", encoding="utf-8") as handle:
        dictionary = json.load(handle)

    means = np.asarray(dictionary["means"], dtype=float)
    covs = np.asarray(dictionary["covariances"], dtype=float)
    scaling = dictionary["scaling"]
    representation = str(dictionary.get("representation", "univariate"))
    if int(dictionary["d"]) == 2:
        representation = "bivariate"
    elif int(dictionary["d"]) == 1:
        representation = "univariate"

    cgm, qc, manifest = load_baseline_frames(baseline_dir)

    weights, excluded = {}, []
    for patient_id in sorted(int(p) for p in patient_ids):
        patient_dir = run_folder / f"patient_{patient_id}"
        fit_path = patient_dir / "result_fit.json"
        if not fit_path.exists():
            excluded.append({"patient_id": patient_id, "reason": "no fit in this run"})
            continue
        with open(fit_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)["config"]

        points = baseline_points(patient_id, cgm, qc, representation)
        if points is None:
            excluded.append({
                "patient_id": patient_id,
                "reason": f"no baseline with >= {BASELINE_MIN_CGM_HOURS:g} h of CGM",
            })
            continue

        bandwidth = config.get("patient_bandwidth_reference")
        if bandwidth is None:
            excluded.append({"patient_id": patient_id,
                             "reason": "run saved no patient bandwidth"})
            continue

        weights[patient_id] = project_onto_dictionary(
            points, means, covs, scaling["mean"], scaling["std"],
            bandwidth, config.get("lambda_ridge", 0.0),
        )

    if verbose:
        print(f"[baseline] projected {len(weights)}/{len(list(patient_ids))} "
              f"participants onto the {run_folder.name} dictionary "
              f"(d={dictionary['d']}, K={dictionary['K']})")
        if excluded:
            reasons = {}
            for item in excluded:
                reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
            for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
                print(f"           excluded {count}: {reason}")
    return weights, excluded, {
        "baseline_dataset_fingerprint": manifest["dataset_fingerprint"],
        "run_dataset_fingerprint": dictionary.get("dataset_fingerprint"),
        "representation": representation,
        "n_projected": len(weights),
    }
