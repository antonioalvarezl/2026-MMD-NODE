"""PEDAP clinical data loading.

Reads the processed PEDAP dataset built by ``scripts/build_pedap_dataset.py``
and turns it into the weekly distributions consumed by the MMD-GMM / Neural
ODE pipeline.  This module never reads a raw PEDAP file, never rebuilds the
dataset and never re-derives a scientific convention: the SAP rules live in
``PEDAP.rules`` and were already applied by the builder.

"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from PEDAP.rules import (
    CGM_SAMPLING_MINUTES,
    EVENT_SUBTYPE_HIGH,
    EVENT_SUBTYPE_LOW,
    N_RCT_WEEKS,
    WEEKLY_MIN_CGM_HOURS,
)

PEDAP_DATA_DIR = "data/processed/pedap_benchmark/rct_followup"

PARTICIPANTS_FILE = "pedap_participants.csv"
CGM_FILE = "pedap_cgm_rct.parquet"
WEEKLY_QC_FILE = "pedap_weekly_qc.csv"
MANIFEST_FILE = "manifest.json"

# Only the manifest is required by name; the rest are resolved from its
# ``files`` mapping, so an artefact may name its CGM table differently
# Primary RCT readings: pedap_cgm_rct.parquet.
REQUIRED_FILES = (MANIFEST_FILE,)

MISSING_DATASET_MESSAGE = (
    "PEDAP processed dataset not found.\n"
    "Run:\n"
    "python scripts/build_pedap_dataset.py"
)

# A central difference needs genuinely adjacent five-minute samples.  The raw
# Dexcom timestamps drift by one to three seconds, so an exactly-300-second
# requirement would reject 22.6% of interior points for clock jitter alone,
# while only 0.21% of consecutive gaps are real data gaps.  5 seconds covers
# the whole observed jitter peak around 300s with margin (observed jitter is
# 1-3s) while still rejecting a genuine gap (e.g. 600s); set it to 0 for the
# literal reading.  This is a modelling choice, not a PEDAP/SAP rule: the
# derivative itself is not part of the SAP.
DERIVATIVE_TOLERANCE_SECONDS = 5.0


class PedapDatasetNotFound(FileNotFoundError):
    """Raised when the processed PEDAP dataset has not been built yet."""


# =============================================================================
# Dataset access (cached)
# =============================================================================

def _dataset_dir(data_dir: Optional[str] = None) -> Path:
    path = Path(data_dir or PEDAP_DATA_DIR)
    missing = [name for name in REQUIRED_FILES if not (path / name).exists()]
    if missing:
        raise PedapDatasetNotFound(MISSING_DATASET_MESSAGE)
    return path


@lru_cache(maxsize=4)
def load_pedap_manifest(data_dir: Optional[str] = None) -> Dict:
    """Load ``manifest.json``: how the dataset was built, and its fingerprint."""
    path = _dataset_dir(data_dir) / MANIFEST_FILE
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("dataset_version") != "pedap_v2":
        raise ValueError("Clinical fitting requires the PEDAP RCT dataset pedap_v2")
    return manifest


def _dataset_file(data_dir: Optional[str], key: str, default: str) -> Path:
    """Resolve one artefact file by its manifest ``files`` entry."""
    root = _dataset_dir(data_dir)
    manifest = load_pedap_manifest(data_dir)
    name = manifest.get("files", {}).get(key, default)
    path = root / name
    if not path.exists():
        raise PedapDatasetNotFound(MISSING_DATASET_MESSAGE)
    return path


@lru_cache(maxsize=4)
def load_pedap_participants(data_dir: Optional[str] = None) -> pd.DataFrame:
    """One row per randomized participant, with the analysis window."""
    path = _dataset_file(data_dir, "participants", PARTICIPANTS_FILE)
    frame = pd.read_csv(path)
    for column in (
        "RandDate", "EnrollDt", "Week13VisitDate", "week13_visit_date",
        "analysis_start_date", "analysis_end_date", "analysis_end_exclusive",
        "LastVisitDate", "RandVisitDate",
    ):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


@lru_cache(maxsize=4)
def load_pedap_cgm(data_dir: Optional[str] = None) -> pd.DataFrame:
    """Every CGM reading of the artefact, sorted by PtID/time."""
    return pd.read_parquet(_dataset_file(data_dir, "cgm", CGM_FILE))


@lru_cache(maxsize=4)
def load_pedap_weekly_qc(data_dir: Optional[str] = None) -> pd.DataFrame:
    """One row per participant-week retained by the artefact."""
    path = _dataset_file(data_dir, "weekly_qc", WEEKLY_QC_FILE)
    frame = pd.read_csv(path)
    for column in (
        "nominal_start_date", "nominal_end_date", "actual_start_datetime",
        "actual_end_datetime", "first_timestamp", "last_timestamp",
    ):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


@lru_cache(maxsize=4)
def _cgm_by_patient(data_dir: Optional[str] = None) -> Dict[int, pd.DataFrame]:
    """Split the CGM table once, so per-participant loading stays cheap."""
    cgm = load_pedap_cgm(data_dir)
    return {int(pt): frame for pt, frame in cgm.groupby("PtID", sort=True)}


def clear_pedap_cache() -> None:
    """Drop every cached dataset frame."""
    load_pedap_manifest.cache_clear()
    load_pedap_participants.cache_clear()
    load_pedap_cgm.cache_clear()
    load_pedap_weekly_qc.cache_clear()
    _cgm_by_patient.cache_clear()


def pedap_dataset_fingerprint(data_dir: Optional[str] = None) -> str:
    """The dataset fingerprint that every clinical result must be stamped with."""
    return str(load_pedap_manifest(data_dir)["dataset_fingerprint"])


def pedap_group_ids(data_dir: Optional[str] = None) -> Dict[str, Tuple[int, ...]]:
    """Trial arms straight from the processed participants table."""
    participants = load_pedap_participants(data_dir)
    return {
        arm: tuple(
            int(pt) for pt in
            participants.loc[participants["TrtGroup"] == arm, "PtID"].sort_values()
        )
        for arm in ("CLC", "SC")
    }


# =============================================================================
# Derivatives
# =============================================================================

def compute_pedap_derivatives(
    timestamps: np.ndarray,
    glucose: np.ndarray,
    high_low: Optional[np.ndarray] = None,
    tolerance_seconds: float = DERIVATIVE_TOLERANCE_SECONDS,
) -> Dict[str, np.ndarray]:
    """Central-difference dG/dt over one participant's whole follow-up series.

    The derivative at ``t_i`` exists only when ``t_{i-1}`` and ``t_{i+1}`` are
    the adjacent five-minute samples; a gap on either side removes the point.
    Nothing is interpolated and no artificial observation is created.

    HIGH/LOW values have already been imputed in ``GlucoseMgDl`` and are kept,
    as the SAP requires.  ``uses_imputed`` flags the centre points whose
    triple touches an imputed value, so an "exclude derivatives involving
    HIGH/LOW" sensitivity can be run later without rebuilding the dataset.

    Returns the centre timestamps, the glucose values there, dG/dt in
    mg/dL/min and the imputation flags.
    """
    times = np.asarray(timestamps, dtype="datetime64[ns]").reshape(-1)
    values = np.asarray(glucose, dtype=float).reshape(-1)
    if len(times) != len(values):
        raise ValueError("timestamps and glucose must have the same length")
    if len(times) < 3:
        empty_t = np.empty(0, dtype="datetime64[ns]")
        return {
            "timestamps": empty_t,
            "glucose": np.empty(0),
            "derivative": np.empty(0),
            "uses_imputed": np.empty(0, dtype=bool),
        }
    if np.any(np.diff(times) <= np.timedelta64(0, "ns")):
        raise ValueError("timestamps must be strictly increasing")

    step = np.timedelta64(int(round(CGM_SAMPLING_MINUTES * 60)), "s")
    tol = np.timedelta64(int(round(float(tolerance_seconds))), "s")
    gaps = np.diff(times)
    adjacent = np.abs(gaps - step) <= tol

    # Interior point i is usable when gap(i-1, i) and gap(i, i+1) are adjacent.
    centre = adjacent[:-1] & adjacent[1:]
    idx = np.flatnonzero(centre) + 1
    if idx.size == 0:
        empty_t = np.empty(0, dtype="datetime64[ns]")
        return {
            "timestamps": empty_t,
            "glucose": np.empty(0),
            "derivative": np.empty(0),
            "uses_imputed": np.empty(0, dtype=bool),
        }

    span_minutes = (
        (times[idx + 1] - times[idx - 1]) / np.timedelta64(1, "s")
    ) / 60.0
    derivative = (values[idx + 1] - values[idx - 1]) / span_minutes

    if high_low is None:
        uses_imputed = np.zeros(idx.size, dtype=bool)
    else:
        # EventSubType strings from PEDAPDexcomClarityCGM.txt ("HIGH"/"LOW"/
        # missing).  A plain object array with float NaN for missing (not
        # pandas' nullable "string" dtype) compares cleanly here.
        codes = np.asarray(high_low).reshape(-1)
        imputed = (codes == EVENT_SUBTYPE_HIGH) | (codes == EVENT_SUBTYPE_LOW)
        uses_imputed = imputed[idx - 1] | imputed[idx] | imputed[idx + 1]

    return {
        "timestamps": times[idx],
        "glucose": values[idx],
        "derivative": derivative,
        "uses_imputed": uses_imputed,
    }


# =============================================================================
# Per-participant weekly distributions
# =============================================================================

def load_patient_weeks(
    patient_id: int,
    data_dir: Optional[str] = None,
    representation: str = "univariate",
    add_deriv: Optional[bool] = None,
    weekly_min_cgm_hours: Optional[float] = None,
    min_node_observations: int = 0,
    derivative_tolerance_seconds: float = DERIVATIVE_TOLERANCE_SECONDS,
    return_metadata: bool = True,
):
    """Load one participant as the weekly CGM distributions of the PEDAP RCT.

    ``data`` holds only the blocks whose CGM coverage meets the SAP 72-hour
    rule; the blocks that were left out stay in ``metadata`` so missingness
    can be audited without losing the temporal structure.

    Missing weeks are allowed. Usable blocks retain their real observation
    times; min_node_observations is the participant inclusion gate.

    Returns ``({patient_id: [X_t0, X_t1, ...]}, metadata)``, each ``X_t`` of
    shape ``(n_t, 1)`` for the univariate representation and ``(n_t, 2)``
    (glucose, dG/dt) for the bivariate one.  ``n_t`` varies by week and by
    participant, which is expected.
    """
    if add_deriv is None:
        add_deriv = representation == "bivariate"
    if representation not in ("univariate", "bivariate"):
        raise ValueError("representation must be 'univariate' or 'bivariate'")
    threshold = (
        WEEKLY_MIN_CGM_HOURS if weekly_min_cgm_hours is None
        else float(weekly_min_cgm_hours)
    )

    patient_id = int(patient_id)
    participants = load_pedap_participants(data_dir)
    row = participants[participants["PtID"] == patient_id]
    if row.empty:
        raise KeyError(f"PtID {patient_id} is not a randomized PEDAP participant")
    row = row.iloc[0]

    weekly = load_pedap_weekly_qc(data_dir)
    blocks_qc = weekly[weekly["PtID"] == patient_id].sort_values("nominal_week_index")

    frame = _cgm_by_patient(data_dir).get(patient_id)
    if frame is None or frame.empty:
        times = np.empty(0, dtype="datetime64[ns]")
        glucose = np.empty(0)
        codes = np.empty(0, dtype=object)
    else:
        frame = frame.sort_values("DeviceDtTm")
        times = frame["DeviceDtTm"].to_numpy(dtype="datetime64[ns]")
        glucose = frame["GlucoseMgDl"].to_numpy(dtype=float)
        codes = frame["EventSubType"].to_numpy()

    if add_deriv:
        # The derivative is computed once over the whole follow-up series and
        # only then split into weeks, so no point is lost at a week boundary.
        deriv = compute_pedap_derivatives(
            times, glucose, codes, tolerance_seconds=derivative_tolerance_seconds
        )
        point_times = deriv["timestamps"]
        point_values = np.column_stack([deriv["glucose"], deriv["derivative"]])
        point_imputed = deriv["uses_imputed"]
    else:
        point_times = times
        point_values = glucose.reshape(-1, 1)
        point_imputed = (codes == EVENT_SUBTYPE_HIGH) | (codes == EVENT_SUBTYPE_LOW)

    weekly_data: List[np.ndarray] = []
    block_metadata: List[Dict] = []
    omitted: List[Dict] = []

    for block in blocks_qc.itertuples():
        usable = bool(block.usable) and float(block.cgm_hours) >= threshold
        has_window = bool(block.has_window)

        if has_window and len(point_times):
            mask = (
                (point_times >= np.datetime64(block.actual_start_datetime))
                & (point_times < np.datetime64(block.actual_end_datetime))
            )
        else:
            mask = np.zeros(len(point_times), dtype=bool)

        values = point_values[mask]
        record = {
            "week_index": int(block.nominal_week_index),
            "actual_start": (
                pd.Timestamp(block.actual_start_datetime).isoformat()
                if has_window else None
            ),
            "actual_end": (
                pd.Timestamp(block.actual_end_datetime).isoformat()
                if has_window else None
            ),
            "observation_time_weeks": float(block.observation_time_weeks),
            "n_observations": int(len(values)),
            "n_cgm_readings": int(block.n_cgm_readings),
            "cgm_hours": float(block.cgm_hours),
            "usable": usable,
            "n_high": int(block.n_high),
            "n_low": int(block.n_low),
            "n_points_touching_imputed": int(point_imputed[mask].sum()),
        }
        record["nominal_start_date"] = pd.Timestamp(
            block.nominal_start_date).date().isoformat()
        record["nominal_end_date"] = pd.Timestamp(
            block.nominal_end_date).date().isoformat()

        if usable and len(values) > 0:
            record["included"] = True
            weekly_data.append(np.ascontiguousarray(values, dtype=float))
            block_metadata.append(record)
        else:
            record["included"] = False
            record["omission_reason"] = (
                "no_window" if not has_window
                else "insufficient_cgm_hours" if not usable
                else "no_observations_after_derivative_filter"
            )
            omitted.append(record)

    observation_times = [b["observation_time_weeks"] for b in block_metadata]
    if len(observation_times) > 1 and np.any(np.diff(observation_times) <= 0):
        raise RuntimeError(
            f"PtID {patient_id}: block observation times are not strictly increasing"
        )

    metadata = {
        "patient_id": patient_id,
        "treatment_group": row["TrtGroup"],
        "rand_date": pd.Timestamp(row["RandDate"]).date().isoformat(),
        "rand_datetime_raw": row["RandDt"],
        "week13_visit_date": (
            pd.Timestamp(row["Week13VisitDate"]).date().isoformat()
            if pd.notna(row["Week13VisitDate"]) else None
        ),
        "week13_out_of_window": (
            int(row["Week13OutOfWin"]) if pd.notna(row["Week13OutOfWin"]) else None
        ),
        "analysis_start_date": pd.Timestamp(row["analysis_start_date"]).date().isoformat(),
        "analysis_end_date": (
            pd.Timestamp(row["analysis_end_date"]).date().isoformat()
            if "analysis_end_date" in row.index and pd.notna(row["analysis_end_date"])
            else None
        ),
        "analysis_end_source": (
            row["analysis_end_source"] if "analysis_end_source" in row.index else None
        ),
        "is_dropout_before_week13": (
            bool(row["is_dropout_before_week13"])
            if "is_dropout_before_week13" in row.index else None
        ),
        "nominal_week_indices": [b["week_index"] for b in block_metadata],
        "observation_times_weeks": observation_times,
        "blocks": block_metadata,
        "omitted_blocks": omitted,
        "n_usable_blocks": len(block_metadata),
        "n_nominal_weeks": int(N_RCT_WEEKS),
        "representation": representation,
        "weekly_min_cgm_hours": threshold,
        "derivative_tolerance_seconds": (
            float(derivative_tolerance_seconds) if add_deriv else None
        ),
        "dataset_fingerprint": pedap_dataset_fingerprint(data_dir),
    }

    if len(weekly_data) < int(min_node_observations):
        metadata["excluded"] = True
        metadata["exclusion_reason"] = "insufficient_model_observations"
        return ({}, metadata) if return_metadata else {}

    metadata["excluded"] = False
    result = {patient_id: weekly_data} if weekly_data else {}
    return (result, metadata) if return_metadata else result


# =============================================================================
# Global dictionary cohort (treatment blind)
# =============================================================================

def pedap_dictionary_cohort(
    data_dir: Optional[str] = None,
    weekly_min_cgm_hours: Optional[float] = None,
) -> List[int]:
    """Pre-specified, treatment-blind cohort backing the global GMM dictionary.

    The rule is fixed in advance and never looks at ``TrtGroup``: every
    randomized participant holding at least one usable patient-week, in
    ascending PtID order.
    """
    threshold = (
        WEEKLY_MIN_CGM_HOURS if weekly_min_cgm_hours is None
        else float(weekly_min_cgm_hours)
    )
    weekly = load_pedap_weekly_qc(data_dir)
    usable = weekly[weekly["cgm_hours"] >= threshold]
    return sorted(int(pt) for pt in usable["PtID"].unique())


# A patient with many usable weeks would otherwise contribute proportionally
# more points to the calibration pool than a patient with few, letting sheer
# CGM volume (missingness, follow-up length) shape the dictionary geometry
# rather than the participant population itself.  This caps each participant
# at roughly one week's worth of five-minute readings (7 days * 24 h * 12
# samples/h = 2016) regardless of how many usable weeks they have.  It is a
# numerical/statistical choice about the calibration pool, decided
# independently of arm: it never reads TrtGroup and the subsampling seed never
# depends on it either.
DICTIONARY_MAX_POINTS_PER_PATIENT = 2016


def aggregate_patients_for_init(
    data_dir: Optional[str] = None,
    patient_ids: Optional[Sequence[int]] = None,
    representation: str = "univariate",
    add_deriv: Optional[bool] = None,
    weekly_min_cgm_hours: Optional[float] = None,
    derivative_tolerance_seconds: float = DERIVATIVE_TOLERANCE_SECONDS,
    max_points_per_patient: Optional[int] = DICTIONARY_MAX_POINTS_PER_PATIENT,
    subsample_seed: int = 42,
    return_info: bool = False,
):
    """Pool usable weekly blocks into the calibration set for the dictionary.

    When ``max_points_per_patient`` is set, each participant's own pooled
    usable weeks are subsampled deterministically (without replacement, seeded
    by ``subsample_seed + patient_id``) down to that cap before joining the
    pool, so one patient with many usable weeks cannot dominate the
    calibration geometry over one with few.  Pass ``None`` to disable this and
    use every usable point from every selected participant, as before.
    """
    if patient_ids is None:
        patient_ids = pedap_dictionary_cohort(data_dir, weekly_min_cgm_hours)
    selected = sorted(int(pid) for pid in patient_ids)

    dim = 2 if (add_deriv if add_deriv is not None else representation == "bivariate") else 1
    print(f"\n[INIT] Aggregating {dim}D data from {len(selected)} participants "
          f"(treatment-blind dictionary cohort)")
    if max_points_per_patient is not None:
        print(f"[INIT] Balanced calibration: capped at "
              f"{max_points_per_patient:,} points/participant "
              f"(seed={subsample_seed})")

    measurements: List[np.ndarray] = []
    used: List[int] = []
    n_blocks = 0
    per_patient_points: Dict[int, Dict[str, int]] = {}
    for pid in selected:
        data, _ = load_patient_weeks(
            pid,
            data_dir=data_dir,
            representation=representation,
            add_deriv=add_deriv,
            weekly_min_cgm_hours=weekly_min_cgm_hours,
            derivative_tolerance_seconds=derivative_tolerance_seconds,
            return_metadata=True,
        )
        if pid not in data:
            continue
        pooled_patient = np.vstack(data[pid])
        n_available = int(len(pooled_patient))

        if max_points_per_patient is not None and n_available > int(max_points_per_patient):
            rng = np.random.default_rng(int(subsample_seed) + pid)
            keep = np.sort(
                rng.choice(n_available, size=int(max_points_per_patient), replace=False)
            )
            pooled_patient = pooled_patient[keep]

        measurements.append(pooled_patient)
        per_patient_points[pid] = {
            "n_available": n_available,
            "n_used": int(len(pooled_patient)),
        }
        n_blocks += len(data[pid])
        used.append(pid)

    if not measurements:
        raise RuntimeError("No usable PEDAP data found for dictionary initialization")

    aggregated = np.vstack(measurements)
    print(f"[INIT] {len(used)} participants, {n_blocks} usable weeks, "
          f"{len(aggregated):,} points, {aggregated.shape[1]}D")

    if not return_info:
        return aggregated
    info = {
        "init_patient_ids": used,
        "requested_dictionary_patient_ids": selected,
        "n_patients_used": len(used),
        "n_weeks_used": n_blocks,
        "n_points": int(len(aggregated)),
        "dimensionality": int(aggregated.shape[1]),
        "cohort_rule": (
            "every randomized participant with at least one usable patient-week; "
            "treatment blind"
        ),
        "balanced_subsampling": {
            "max_points_per_patient": (
                int(max_points_per_patient) if max_points_per_patient is not None else None
            ),
            "subsample_seed": int(subsample_seed),
            "per_patient": per_patient_points,
        },
        "dataset_fingerprint": pedap_dataset_fingerprint(data_dir),
    }
    return aggregated, info


# =============================================================================
# Experiment preparation
# =============================================================================

def prepare_cgm_experiments(
    data_dir: Optional[str] = None,
    patient_ids: Optional[Sequence[int]] = None,
    representation: str = "univariate",
    add_deriv: Optional[bool] = None,
    weekly_min_cgm_hours: Optional[float] = None,
    min_node_observations: int = 5,
    derivative_tolerance_seconds: float = DERIVATIVE_TOLERANCE_SECONDS,
    dictionary_patient_ids: Optional[Sequence[int]] = None,
    dictionary_max_points_per_patient: Optional[int] = DICTIONARY_MAX_POINTS_PER_PATIENT,
    dictionary_subsample_seed: int = 42,
    bandwidth_seed: int = 42,
    bandwidth_reference_sample_size: int = 1000,
) -> Tuple[Dict, np.ndarray, Dict]:
    """Prepare one experiment per participant from the processed PEDAP dataset.

    Missing weeks are allowed; each participant retains real observation times.
    The dictionary is calibrated on pooled usable weeks without treatment labels.
    """
    from data import compute_gammaopt

    manifest = load_pedap_manifest(data_dir)
    fingerprint = str(manifest["dataset_fingerprint"])
    participants = load_pedap_participants(data_dir)

    if patient_ids is None:
        candidates = [int(pt) for pt in participants["PtID"]]
    else:
        wanted = {int(pid) for pid in patient_ids}
        candidates = [int(pt) for pt in participants["PtID"] if int(pt) in wanted]

    init_data, init_info = aggregate_patients_for_init(
        data_dir=data_dir,
        patient_ids=dictionary_patient_ids,
        representation=representation,
        add_deriv=add_deriv,
        weekly_min_cgm_hours=weekly_min_cgm_hours,
        derivative_tolerance_seconds=derivative_tolerance_seconds,
        max_points_per_patient=dictionary_max_points_per_patient,
        subsample_seed=dictionary_subsample_seed,
        return_info=True,
    )

    # The rule itself never reads TrtGroup, but its RESULT can still be
    # single-arm when one arm has no usable CGM at all in this dataset (as it
    # currently does for SC).  That is a fact about the data, not the rule,
    # and must not be read as the dictionary having used treatment labels.
    dictionary_arms = set(
        participants.loc[
            participants["PtID"].isin(init_info["init_patient_ids"]), "TrtGroup"
        ]
    )
    init_info["dictionary_cohort_arms"] = sorted(dictionary_arms)
    if len(dictionary_arms) == 1:
        print(
            f"\n[INIT] NOTE: the treatment-blind dictionary cohort is, in this "
            f"dataset, 100% {next(iter(dictionary_arms))} ({len(init_info['init_patient_ids'])} "
            "participants) -- the rule did not select on arm, but the other arm "
            "currently has zero usable patient-weeks to select from. Re-derive "
            "the dictionary once that is fixed.\n"
        )

    # One treatment-blind standardization, learned on the dictionary cohort and
    # reused unchanged for every participant.
    if init_data.shape[1] > 1:
        scaling_mean = np.mean(init_data, axis=0)
        scaling_std = np.std(init_data, axis=0, ddof=0)
        if not np.all(np.isfinite(scaling_mean)) or not np.all(np.isfinite(scaling_std)):
            raise ValueError("Non-finite global standardization parameters")
        if np.any(scaling_std <= 0):
            raise ValueError("Global standard deviations must be positive")
        init_data = (init_data - scaling_mean) / scaling_std
        scaling_method = "global_zscore"
    else:
        scaling_mean = np.zeros(init_data.shape[1], dtype=float)
        scaling_std = np.ones(init_data.shape[1], dtype=float)
        scaling_method = "none"

    init_info["scaling"] = {
        "method": scaling_method,
        "mean": scaling_mean.tolist(),
        "std": scaling_std.tolist(),
        "calibration_patient_ids": init_info["init_patient_ids"],
    }
    init_info["dataset_fingerprint"] = fingerprint
    init_info["data_quality"] = {
        "n_cgm_readings": manifest.get("n_cgm_readings"),
        "n_high": manifest.get("n_high"),
        "n_low": manifest.get("n_low"),
        "n_normal": manifest.get("n_normal"),
        "n_usable_patient_weeks": manifest.get("n_usable_patient_weeks"),
        "n_unusable_patient_weeks": manifest.get("n_unusable_patient_weeks"),
        "scope_caveat": manifest.get("SCOPE_CAVEAT"),
    }

    experiments: Dict[str, Dict] = {}
    excluded: List[Dict] = []

    for pid in candidates:
        data, metadata = load_patient_weeks(
            pid,
            data_dir=data_dir,
            representation=representation,
            add_deriv=add_deriv,
            weekly_min_cgm_hours=weekly_min_cgm_hours,
            min_node_observations=min_node_observations,
            derivative_tolerance_seconds=derivative_tolerance_seconds,
            return_metadata=True,
        )
        if pid not in data:
            excluded.append(
                {
                    "patient_id": pid,
                    "treatment_group": metadata["treatment_group"],
                    "n_usable_blocks": metadata["n_usable_blocks"],
                    "exclusion_reason": metadata.get(
                        "exclusion_reason", "no_usable_blocks"
                    ),
                }
            )
            continue

        weekly_blocks = [(block - scaling_mean) / scaling_std for block in data[pid]]
        pooled = np.vstack(weekly_blocks)
        reference = compute_gammaopt(
            pooled,
            sample_size=bandwidth_reference_sample_size,
            seed=bandwidth_seed + 1000 * pid,
        )

        experiments[f"Patient_{pid}"] = {
            "combo": {
                "patient_id": pid,
                "treatment_group": metadata["treatment_group"],
                "n_weeks": len(weekly_blocks),
                "nominal_week_indices": metadata["nominal_week_indices"],
                "observation_times_weeks": metadata["observation_times_weeks"],
                "block_metadata": metadata["blocks"],
                "omitted_blocks": metadata["omitted_blocks"],
                "rand_date": metadata["rand_date"],
                "week13_visit_date": metadata["week13_visit_date"],
                "week13_out_of_window": metadata["week13_out_of_window"],
                "analysis_start_date": metadata["analysis_start_date"],
                "analysis_end_date": metadata["analysis_end_date"],
                "analysis_end_source": metadata["analysis_end_source"],
                "is_dropout_before_week13": metadata["is_dropout_before_week13"],
                "observed_support_weeks": [
                    metadata["observation_times_weeks"][0],
                    metadata["observation_times_weeks"][-1],
                ],
                "analysis_window_definition": (
                    manifest.get("followup_definition")
                    or manifest.get("selection_rule")
                ),
                "dataset_fingerprint": fingerprint,
                "dataset_version": manifest["dataset_version"],
            },
            "data": weekly_blocks,
            "gamma": [float(reference)] * len(weekly_blocks),
            "patient_bandwidth_reference": float(reference),
            "patient_bandwidth_reference_n_total": int(len(pooled)),
            "patient_bandwidth_reference_n_sample": int(
                min(len(pooled), bandwidth_reference_sample_size)
            ),
        }

    init_info["excluded_participants"] = excluded
    init_info["min_node_observations"] = int(min_node_observations)

    arms_kept = {}
    for exp in experiments.values():
        arm = exp["combo"]["treatment_group"]
        arms_kept[arm] = arms_kept.get(arm, 0) + 1
    print(f"\n[PEDAP RCT] Prepared {len(experiments)} participants "
          f"({', '.join(f'{arm}={n}' for arm, n in sorted(arms_kept.items())) or 'none'}); "
          f"{len(excluded)} excluded")
    if experiments:
        counts = [e["combo"]["n_weeks"] for e in experiments.values()]
        dim = next(iter(experiments.values()))["data"][0].shape[1]
        print(f"Usable blocks per participant: min {min(counts)}, "
              f"median {int(np.median(counts))}, max {max(counts)}; {dim}D")
    for record in excluded:
        print(f"  [EXCLUDED] Patient {record['patient_id']} "
              f"({record['treatment_group']}): {record['exclusion_reason']} "
              f"(usable blocks: {record['n_usable_blocks']})")

    return experiments, init_data, init_info
