"""PEDAP RCT run validation, coverage and deterministic inference windows."""
from pathlib import Path
import json
from Analysis.paper_config import PEDAP_DATA_DIR, dataset_fingerprint

INFERENCE_ESTIMANDS = ("absolute", "change_from_baseline")
GROUP_CURVE_RULE = (
    "available-case: average only participants whose fitted support contains t; "
    "no extrapolation; save n_clc(t) and n_sc(t)"
)

def validate_rct_run(run_folder):
    """Validate saved fits against the primary dataset, without refitting."""
    path = Path(run_folder) / "fixed_params.json"
    params = json.loads(path.read_text())
    if params.get("dataset_version") != "pedap_v2":
        raise ValueError("Only pedap_v2 RCT runs are active")
    if params.get("dataset_fingerprint") != dataset_fingerprint():
        raise ValueError("Run fingerprint does not match the primary PEDAP dataset")
    return params


def verify_interval_data(data, verbose=True):
    """Verify available-case counts and absence of extrapolated values."""
    import numpy as np
    trajectories = np.asarray(data["trajectories"])
    grid = np.asarray(data["common_grid"])
    supports = np.asarray(data["fitted_node_supports_original_week_coordinates"])
    if data["common_grid_mode"] != "available_case":
        raise RuntimeError("Descriptive group curves require available-case coverage")
    finite = np.isfinite(trajectories).all(axis=2)
    outside = (grid[None, :] < supports[:, :1] - 1e-12) | (grid[None, :] > supports[:, 1:] + 1e-12)
    if np.any(finite & outside):
        raise RuntimeError("A fitted curve was extrapolated outside observed support")
    counts = np.asarray(data["n_available_per_grid_point"])
    if not np.array_equal(counts, finite.sum(axis=0)):
        raise RuntimeError("Coverage counts do not match finite trajectories")
    labels = np.asarray(data["labels"])
    for code, label in (("CLC", 1), ("SC", 0)):
        if not np.array_equal(data["n_available_per_grid_point_by_arm"][code], finite[labels == label].sum(axis=0)):
            raise RuntimeError("Arm coverage counts do not match trajectories")
    report = {"common_grid_mode": "available_case", "n_subjects": len(labels),
              "n_per_grid_point_min": int(counts.min()),
              "n_per_grid_point_max": int(counts.max()),
              "grid_support_weeks": list(data["common_grid_support_weeks"])}
    if verbose:
        print(f"[PEDAP RCT] available-case coverage: {counts.min()}--{counts.max()}")
    return report


def select_inference_window(supports, labels, interval=None,
                            min_per_arm=2, min_cohort_fraction=None,
                            verbose=True):
    """Pick the complete-case window for inference for PEDAP RCT inference.

    Inference needs ONE fixed cohort for the whole curve, so the window and the
    cohort are chosen together: among all windows, take the widest whose
    covering cohort still holds at least ``inference_min_cohort_fraction`` of
    EACH ARM separately.  A global fraction is not enough -- the arms are
    unbalanced (68 CLC vs 34 SC), so a window could satisfy a 90% overall floor
    while quietly dropping a third of the smaller arm, which is exactly the arm
    whose precision the contrast depends on.

    Selection looks ONLY at the participants' observed supports.  It never
    touches outcomes, MMD statistics or p-values, and the window is fixed
    before any test runs, so it cannot be tuned towards significance.

    Ties in width break towards the larger cohort, then the earlier start, so
    the choice is deterministic and reproducible.

    Returns ``(low, high, keep_mask, report)``.  ``keep_mask`` selects the
    participants who cover the whole window: exactly the complete-case cohort
    the wild bootstrap is defined on.
    """
    import numpy as np

    supports = np.asarray(supports, dtype=float)
    labels = np.asarray(labels, dtype=int)
    starts, ends = supports[:, 0], supports[:, 1]
    n_total = len(starts)
    if n_total == 0:
        raise RuntimeError("no participants to choose an inference window from")

    lo_bound = -np.inf if interval is None else float(interval[0])
    hi_bound = np.inf if interval is None else float(interval[1])

    fraction = float(
        0.90
        if min_cohort_fraction is None else min_cohort_fraction
    )
    n_clc_total = int((labels == 1).sum())
    n_sc_total = int((labels == 0).sum())
    required_clc = int(np.ceil(fraction * n_clc_total))
    required_sc = int(np.ceil(fraction * n_sc_total))

    lo_candidates = np.unique(np.clip(starts, lo_bound, hi_bound))
    hi_candidates = np.unique(np.clip(ends, lo_bound, hi_bound))

    best = None
    for low in lo_candidates:
        covers_low = starts <= low + 1e-12
        for high in hi_candidates:
            if high <= low:
                continue
            keep = covers_low & (ends >= high - 1e-12)
            n_clc = int((labels[keep] == 1).sum())
            n_sc = int((labels[keep] == 0).sum())
            # PER-ARM floor, not a pooled one.
            if n_clc < required_clc or n_sc < required_sc:
                continue
            if n_clc < min_per_arm or n_sc < min_per_arm:
                continue
            key = (high - low, n_clc + n_sc, -low)
            if best is None or key > best[0]:
                best = (key, float(low), float(high), keep)

    if best is None:
        raise RuntimeError(
            f"no contiguous window retains >= {fraction:.0%} of EACH arm "
            f"(needs CLC >= {required_clc}/{n_clc_total}, "
            f"SC >= {required_sc}/{n_sc_total}) with >= {min_per_arm} per arm"
        )

    _, low, high, keep = best
    full_low, full_high = float(starts.max()), float(ends.min())
    report = {
        "rule": (
            f"widest contiguous complete-case window retaining >= "
            f"{fraction:.0%} of EACH arm separately; chosen from observed "
            "supports only (never from outcomes, statistics or p-values) and "
            "fixed before any test is run"
        ),
        "min_cohort_fraction": fraction,
        "per_arm_floor": True,
        "required_clc": required_clc,
        "required_sc": required_sc,
        "window_weeks": [low, high],
        "window_span_weeks": high - low,
        "n_retained": int(keep.sum()),
        "n_total": n_total,
        "n_retained_clc": int((labels[keep] == 1).sum()),
        "n_total_clc": n_clc_total,
        "n_retained_sc": int((labels[keep] == 0).sum()),
        "n_total_sc": n_sc_total,
        "n_dropped": int(n_total - keep.sum()),
        "full_intersection_weeks": [full_low, full_high],
        "full_intersection_span_weeks": max(0.0, full_high - full_low),
        "span_gained_weeks": (high - low) - max(0.0, full_high - full_low),
    }
    if verbose:
        print(
            f"[inference window] PEDAP RCT @ {fraction:.0%} per arm: "
            f"[{low:.2f}, {high:.2f}] weeks (span {high - low:.2f}) | "
            f"CLC {report['n_retained_clc']}/{n_clc_total}, "
            f"SC {report['n_retained_sc']}/{n_sc_total}, "
            f"total {report['n_retained']}/{n_total}"
        )
    return low, high, keep, report


# Pre-specified sensitivity thresholds.  The primary analysis uses 90% per arm; these two are reported alongside it and are NEVER selected by
# looking at a p-value.
SENSITIVITY_COHORT_FRACTIONS = (0.85, 0.95)


def inference_window_sensitivity(supports, labels, interval=None,
                                 fractions=SENSITIVITY_COHORT_FRACTIONS,
                                 verbose=True):
    """The same selection at pre-specified alternative thresholds."""
    out = {}
    for fraction in fractions:
        try:
            _, _, _, report = select_inference_window(
                supports, labels, interval=interval,
                min_cohort_fraction=fraction, verbose=verbose,
            )
            out[f"{fraction:.2f}"] = report
        except RuntimeError as error:
            out[f"{fraction:.2f}"] = {"error": str(error),
                                      "min_cohort_fraction": fraction}
    return out


def coverage_table(data):
    """Per-grid-point coverage as a tidy frame: time, n_total, n_clc, n_sc."""
    import numpy as np
    import pandas as pd

    grid = np.asarray(data["common_grid"], dtype=float)
    counts = data.get("n_available_per_grid_point")
    by_arm = data.get("n_available_per_grid_point_by_arm")
    n_subjects = int(np.asarray(data["labels"]).size)
    if counts is None:
        counts = np.full(grid.shape, n_subjects, dtype=int)
    frame = pd.DataFrame({
        "time_weeks": grid,
        "n_total": np.asarray(counts, dtype=int),
    })
    if by_arm is not None:
        frame["n_clc"] = np.asarray(by_arm["CLC"], dtype=int)
        frame["n_sc"] = np.asarray(by_arm["SC"], dtype=int)
    return frame
