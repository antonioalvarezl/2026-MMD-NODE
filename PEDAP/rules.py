"""Single source of truth for the PEDAP scientific conventions.

Every rule that the PEDAP Statistical Analysis Plan (SAP) fixes -- the
follow-up window, the HIGH/LOW imputation, the weekly blocks, the 72-hour
criterion -- is implemented exactly once in this module.  The dataset
builder, the audits and the experiment loader all import it, so a change of
convention is a change in one place and is reflected in the dataset
fingerprint.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Dataset identity
# -----------------------------------------------------------------------------

DATASET_NAME = "pedap_wadwa_rct"
DATASET_VERSION = "pedap_v2"
PREPROCESSING_SCRIPT_VERSION = "2.0.0"

# -----------------------------------------------------------------------------
# HIGH / LOW handling (SAP)
# -----------------------------------------------------------------------------

# SAP imputation for out-of-range sensor readings, shared by every raw
# encoding below.
HIGH_IMPUTED_VALUE = 401.0
LOW_IMPUTED_VALUE = 39.0

# -- Encoding A: numeric HighLowIndicator (PEDAPTandemCGMDATAGXB.txt).
# Auxiliary/historical: this raw file is not part of the v2 RCT dataset (see
# the CGM provenance decision below), but the encoding is documented and
# tested here in case Tandem is ever used again for an auxiliary purpose.
HL_NORMAL = 0
HL_HIGH = 1
HL_LOW = 2
KNOWN_HIGH_LOW_CODES = (HL_NORMAL, HL_HIGH, HL_LOW)

# -- Encoding B: string EventSubType (PEDAPDexcomClarityCGM.txt).  This is the
# ACTIVE encoding for the v2 dataset: PEDAPDexcomClarityCGM.txt is the sole
# CGM source for both arms (see CGM_SOURCE_DECISION below).
EVENT_SUBTYPE_HIGH = "HIGH"
EVENT_SUBTYPE_LOW = "LOW"
KNOWN_EVENT_SUBTYPES = (EVENT_SUBTYPE_HIGH, EVENT_SUBTYPE_LOW)

# -----------------------------------------------------------------------------
# CGM source (decided after auditing PEDAPTandemCGMDATAGXB.txt against
# PtRoster/PEDAPVisitInfo: it holds essentially no RCT-phase CGM for the SC
# arm, because SC only received the Tandem device at the 13-week visit).
# -----------------------------------------------------------------------------

CGM_SOURCE_DECISION = (
    "PEDAPDexcomClarityCGM.txt (the study Dexcom G6 / Dexcom Clarity export) "
    "is the SOLE CGM source for BOTH arms.  It is symmetric: both CLC and SC "
    "wore a study Dexcom G6 throughout the RCT phase, so using one source for "
    "both arms avoids introducing an artificial device/stream effect between "
    "arms.  PEDAPTandemCGMDATAGXB.txt (Tandem pump-relayed CGM) is NOT used: "
    "during the RCT phase only CLC used the Tandem pump, so it is "
    "structurally CLC-only in the SAP follow-up window and would confound "
    "treatment arm with data source if mixed with Dexcom for SC."
)

# -----------------------------------------------------------------------------
# Follow-up window (SAP)
# -----------------------------------------------------------------------------

# "CGM data from the fourth day after the randomization visit through the
# 13-week visit".  The randomization visit day is day 0; the first day after
# it is day 1, so the fourth day after it is day 4.  All boundaries are
# calendar days: the time-of-day carried by RandDt is audit metadata only.
FOLLOWUP_START_OFFSET_DAYS = 4

# The 13-week visit date is a date without a time.  The whole calendar day of
# the visit is included in follow-up, i.e. the window is the half-open
# interval [start 00:00, visit_date + 1 day 00:00).
FOLLOWUP_END_INCLUDES_VISIT_DAY = True

FOLLOWUP_DEFINITION = (
    "PEDAP SAP: CGM from the fourth calendar day after the randomization "
    "visit through the actual 13-week visit (or, for participants without a "
    "13-week visit, through their last recorded visit)."
)
FOLLOWUP_START_RULE = (
    f"analysis_start = calendar date of RandDt + {FOLLOWUP_START_OFFSET_DAYS} "
    "days, taken at 00:00 local device time (inclusive)."
)
FOLLOWUP_END_RULE = (
    "analysis_end = calendar date of the actual '13 Week' visit, inclusive of "
    "the whole visit day; the half-open upper bound is visit date + 1 day at "
    "00:00.  Participants with no '13 Week' visit fall back to their last "
    "recorded visit date (analysis_end_source = last_visit_before_dropout)."
)

# -----------------------------------------------------------------------------
# Weekly blocks
# -----------------------------------------------------------------------------

N_RCT_WEEKS = 13
WEEK_CONSTRUCTION_RULE = (
    "13 nominal weekly blocks anchored at the randomization calendar date: "
    "block w covers [RandDate + 7w, RandDate + 7(w+1)) in calendar days. "
    "Every block is intersected with the SAP follow-up window, so block 0 is "
    "truncated at its start; the final block (w=12) always ends at the actual "
    "follow-up end, extending or truncating the nominal boundary."
)

# -----------------------------------------------------------------------------
# CGM cadence and usability
# -----------------------------------------------------------------------------

CGM_SAMPLING_MINUTES = 5.0
WEEKLY_MIN_CGM_HOURS = 72.0
USABILITY_RULE = (
    f"A patient-week is usable when it holds at least {WEEKLY_MIN_CGM_HOURS:g} "
    f"hours of CGM, computed as n_readings * {CGM_SAMPLING_MINUTES:g} / 60 "
    "after de-duplication (SAP minimum of 72 hours)."
)

DUPLICATE_RULE = (
    "Exact duplicates (same PtID, timestamp and glucose encoding) collapse to "
    "a single reading.  Conflicting duplicates (same PtID and timestamp, "
    "different value or HIGH/LOW flag) are resolved deterministically by "
    "keeping the smallest RecID; their counts are recorded in manifest.json "
    "as duplicate_timestamp_groups and conflicting_duplicate_groups."
)

MISSING_DATA_RULE = (
    "CGM gaps are never interpolated and no artificial observation is "
    "created.  No 40/400 validity threshold and no winsorization is applied: "
    "HIGH and LOW readings enter the analysis at their SAP imputed values.  "
    "A week without enough CGM is flagged usable=False but the participant "
    "and all their other weeks are retained."
)

HIGH_LOW_RULE = (
    f"v2 ACTIVE encoding (PEDAPDexcomClarityCGM.txt, EventSubType): "
    f"EventSubType=='HIGH' -> {HIGH_IMPUTED_VALUE:g} mg/dL; "
    f"EventSubType=='LOW' -> {LOW_IMPUTED_VALUE:g} mg/dL; otherwise keep the "
    "raw CGM reading.  The imputed value comes from EventSubType, not from "
    "trusting the raw CGM number, though the raw file's CGM column is "
    "asserted to already agree (401/39) as a data-quality check.  No "
    "40/400 threshold, no winsorization.  (An auxiliary numeric "
    "HighLowIndicator encoding for the unused Tandem file is also "
    "implemented -- see apply_high_low -- but does not participate in v2.)"
)

# -----------------------------------------------------------------------------
# Raw date parsing
# -----------------------------------------------------------------------------

_DATE_FORMAT = "%m/%d/%Y"
_DATETIME_FORMATS = ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y")


def parse_pedap_date(values) -> pd.Series:
    """Parse a Jaeb date column (``M/D/YYYY``) into normalized timestamps."""
    parsed = pd.to_datetime(values, format=_DATE_FORMAT, errors="coerce")
    return pd.Series(parsed).dt.normalize()


def parse_pedap_datetime(values) -> pd.Series:
    """Parse a Jaeb datetime column, trying each documented raw layout."""
    series = pd.Series(values, dtype="object")
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    remaining = series.notna()
    for fmt in _DATETIME_FORMATS:
        if not remaining.any():
            break
        attempt = pd.to_datetime(series[remaining], format=fmt, errors="coerce")
        ok = attempt.notna()
        result.loc[attempt.index[ok]] = attempt[ok]
        remaining.loc[attempt.index[ok]] = False
    return result


# -----------------------------------------------------------------------------
# The centralized SAP rules
# -----------------------------------------------------------------------------

def pedap_followup_start(rand_date) -> pd.Timestamp:
    """Return the SAP follow-up start: the fourth day after randomization.

    The randomization *calendar date* is the anchor; the time of day recorded
    in ``RandDt`` is deliberately ignored so that weekly boundaries never fall
    at times such as 13:22.  The returned timestamp is inclusive.
    """
    anchor = pd.Timestamp(rand_date)
    if pd.isna(anchor):
        raise ValueError("randomization date is missing")
    return anchor.normalize() + pd.Timedelta(days=FOLLOWUP_START_OFFSET_DAYS)


def pedap_followup_end_exclusive(
    end_visit_date, include_visit_day: bool = FOLLOWUP_END_INCLUDES_VISIT_DAY
) -> pd.Timestamp:
    """Return the exclusive upper bound of the SAP follow-up window.

    ``end_visit_date`` is the actual 13-week visit date (or the last recorded
    visit for a participant who dropped out).  ``VisitDt`` carries no time of
    day, so the PRIMARY rule includes the entire visit calendar day
    (``include_visit_day=True``, the default and the module-level
    ``FOLLOWUP_END_INCLUDES_VISIT_DAY``).

    ``include_visit_day=False`` is a documented SECONDARY sensitivity: it ends
    follow-up at the visit date itself (00:00), excluding readings recorded
    on the visit day.  This exists because in the CLC-arm Tandem source the
    device handover to Control-IQ happens on this exact day for the SC arm
    (see the CGM provenance audit); once the correct SC-arm CGM source is
    available this lets the last block's sensitivity to that day be
    quantified without redefining the primary window.  It is never used to
    build the primary processed dataset.
    """
    anchor = pd.Timestamp(end_visit_date)
    if pd.isna(anchor):
        raise ValueError("follow-up end visit date is missing")
    extra = 1 if include_visit_day else 0
    return anchor.normalize() + pd.Timedelta(days=extra)


def apply_high_low(cgm_value, high_low_indicator):
    """Map raw (CGMValue, HighLowIndicator) pairs to SAP glucose values.

    ``HighLowIndicator == 0`` keeps the numeric reading; ``1`` is imputed to
    401 mg/dL and ``2`` to 39 mg/dL.  A raw ``CGMValue`` recorded alongside a
    non-zero indicator is never interpreted as a real glucose value.  Any
    other code raises, because its meaning is not documented.
    """
    values = np.asarray(cgm_value, dtype=float)
    codes = np.asarray(high_low_indicator)
    codes_numeric = pd.to_numeric(pd.Series(codes.ravel()), errors="coerce")
    if codes_numeric.isna().any():
        raise ValueError("HighLowIndicator contains non-numeric entries")
    codes_int = codes_numeric.to_numpy().astype(np.int64).reshape(codes.shape)

    unknown = sorted(set(np.unique(codes_int).tolist()) - set(KNOWN_HIGH_LOW_CODES))
    if unknown:
        raise ValueError(
            f"Unknown HighLowIndicator code(s) {unknown}; "
            f"only {list(KNOWN_HIGH_LOW_CODES)} are defined by the SAP"
        )

    glucose = np.where(
        codes_int == HL_HIGH,
        HIGH_IMPUTED_VALUE,
        np.where(codes_int == HL_LOW, LOW_IMPUTED_VALUE, values),
    )
    return glucose


def apply_high_low_event_subtype(cgm_value, event_subtype):
    """Map raw (CGM, EventSubType) pairs from PEDAPDexcomClarityCGM.txt to SAP
    glucose values.  This is the ACTIVE imputation rule for the v2 dataset.

    ``EventSubType == "HIGH"`` is imputed to 401 mg/dL; ``"LOW"`` to
    39 mg/dL; a blank/missing ``EventSubType`` keeps the raw ``CGM`` reading.
    The imputed value is used regardless of what the raw ``CGM`` column
    happens to hold for a HIGH/LOW row (the SAP value is the rule, not the
    raw number) -- callers should separately assert that the raw file's own
    ``CGM`` column already agrees with this rule, as a data-quality check
    that catches a change in the raw encoding rather than silently
    absorbing it.  Any ``EventSubType`` other than HIGH/LOW/blank raises,
    because its meaning is not documented.
    """
    values = np.asarray(cgm_value, dtype=float)
    codes = pd.Series(np.asarray(event_subtype).reshape(-1))

    unknown = sorted(
        set(codes.dropna().unique().tolist()) - set(KNOWN_EVENT_SUBTYPES)
    )
    if unknown:
        raise ValueError(
            f"Unknown EventSubType value(s) {unknown}; "
            f"only {list(KNOWN_EVENT_SUBTYPES)} (or blank) are defined by the SAP"
        )

    is_high = (codes == EVENT_SUBTYPE_HIGH).to_numpy().reshape(values.shape)
    is_low = (codes == EVENT_SUBTYPE_LOW).to_numpy().reshape(values.shape)
    glucose = np.where(is_high, HIGH_IMPUTED_VALUE, np.where(is_low, LOW_IMPUTED_VALUE, values))
    return glucose


def cgm_hours_from_readings(n_readings) -> float:
    """Observed CGM hours implied by de-duplicated readings at 5-minute cadence.

    This is deliberately not the span between the first and last timestamp:
    two readings a week apart carry ten minutes of coverage, not a week.
    """
    return float(np.asarray(n_readings, dtype=float) * CGM_SAMPLING_MINUTES / 60.0)


def is_week_usable(cgm_hours) -> bool:
    """SAP criterion: at least 72 hours of CGM in the weekly block."""
    return bool(float(cgm_hours) >= WEEKLY_MIN_CGM_HOURS)


def build_nominal_weeks(
    rand_date,
    analysis_start,
    analysis_end_exclusive,
    n_weeks: int = N_RCT_WEEKS,
) -> List[Dict]:
    """Build the participant's 13 nominal RCT weeks intersected with follow-up.

    Nominal boundaries are calendar weeks anchored at the randomization date.
    The intersection with the SAP window truncates the first block; the last
    block always ends at the real follow-up end, so a late 13-week visit adds
    its extra days to that block and an early visit truncates it.

    Returns one record per ``week_index`` in ``range(n_weeks)``.  A block whose
    intersection is empty is still returned, with ``has_window=False``.
    """
    anchor = pd.Timestamp(rand_date).normalize()
    start = pd.Timestamp(analysis_start)
    end = pd.Timestamp(analysis_end_exclusive)

    blocks: List[Dict] = []
    for week_index in range(n_weeks):
        nominal_start = anchor + pd.Timedelta(days=7 * week_index)
        nominal_end = anchor + pd.Timedelta(days=7 * (week_index + 1))

        actual_start = max(nominal_start, start)
        if week_index == n_weeks - 1:
            # SAP: "the 13 week visit date will be used as end date of the
            # Week 12-13 period", whether it falls early or late.
            actual_end = end
        else:
            actual_end = min(nominal_end, end)

        has_window = actual_end > actual_start
        blocks.append(
            {
                "nominal_week_index": week_index,
                "nominal_start_date": nominal_start,
                "nominal_end_date": nominal_end,
                "actual_start_datetime": actual_start,
                "actual_end_datetime": actual_end,
                "has_window": bool(has_window),
                "observation_time_weeks": (
                    observation_time_weeks(actual_start, actual_end, anchor)
                    if has_window
                    else float("nan")
                ),
            }
        )
    return blocks


def observation_time_weeks(actual_start, actual_end, rand_date) -> float:
    """Midpoint of the block's real window, in weeks elapsed since randomization.

    This is the time the Neural ODE should use: it differs from the nominal
    ``week_index + 0.5`` whenever the block is truncated (week 0) or defined by
    the real visit date (week 12).
    """
    anchor = pd.Timestamp(rand_date).normalize()
    start = pd.Timestamp(actual_start)
    end = pd.Timestamp(actual_end)
    midpoint = start + (end - start) / 2
    return float((midpoint - anchor) / pd.Timedelta(weeks=1))


# -----------------------------------------------------------------------------
# Fingerprint
# -----------------------------------------------------------------------------

def scientific_config() -> Dict[str, object]:
    """The scientific preprocessing configuration that defines the dataset."""
    return {
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "cgm_source_decision": CGM_SOURCE_DECISION,
        "followup_definition": FOLLOWUP_DEFINITION,
        "followup_start_rule": FOLLOWUP_START_RULE,
        "followup_end_rule": FOLLOWUP_END_RULE,
        "followup_start_offset_days": FOLLOWUP_START_OFFSET_DAYS,
        "followup_end_includes_visit_day": FOLLOWUP_END_INCLUDES_VISIT_DAY,
        "high_imputation": HIGH_IMPUTED_VALUE,
        "low_imputation": LOW_IMPUTED_VALUE,
        # The ACTIVE raw encoding for v2 (Dexcom Clarity's EventSubType
        # string flags).  The numeric HighLowIndicator encoding still exists
        # in PEDAP.rules for the auxiliary Tandem file, but it does not
        # participate in the v2 dataset or its fingerprint.
        "known_event_subtypes": list(KNOWN_EVENT_SUBTYPES),
        "n_rct_weeks": N_RCT_WEEKS,
        "week_construction": WEEK_CONSTRUCTION_RULE,
        "cgm_sampling_minutes": CGM_SAMPLING_MINUTES,
        "weekly_min_cgm_hours": WEEKLY_MIN_CGM_HOURS,
        "usability_rule": USABILITY_RULE,
        "duplicate_rule": DUPLICATE_RULE,
        "missing_data_rule": MISSING_DATA_RULE,
        "high_low_rule": HIGH_LOW_RULE,
    }


def sha256_file(path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a raw file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(
    raw_file_sha256: Dict[str, str],
    config: Optional[Dict[str, object]] = None,
    cgm_source_files: Optional[Iterable[str]] = None,
) -> str:
    """Deterministic fingerprint of raw inputs plus scientific configuration.

    The fingerprint changes whenever the CGM source, the follow-up window, the
    HIGH/LOW handling, the 72-hour rule, the duplicate rule or the week
    construction changes, so stale caches and stale results can be detected.
    """
    payload = {
        "raw_file_sha256": dict(sorted(raw_file_sha256.items())),
        "cgm_source_files": sorted(cgm_source_files or []),
        "scientific_config": config if config is not None else scientific_config(),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{DATASET_VERSION}_{hashlib.sha256(blob).hexdigest()[:16]}"
