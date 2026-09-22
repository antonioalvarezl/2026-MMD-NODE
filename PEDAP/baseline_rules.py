"""Pre-randomization baseline rules for the Wadwa-like endpoint analysis.

This module is DELIBERATELY separate from ``PEDAP.rules``.  The pedap_v2
fingerprint is computed over ``PEDAP.rules.scientific_config()``, so adding
anything to that function would silently invalidate the existing processed
dataset and every result fitted on it.  Nothing here feeds that fingerprint.

The PEDAP SAP compares a pre-randomization baseline against the cumulative
13-week follow-up period.  That is NOT the "week 0 versus week 12" contrast:
the follow-up endpoint is the WHOLE follow-up period pooled, not its last
weekly block, and the baseline is CGM recorded BEFORE randomization, which
the RCT dataset does not contain at all.

Baseline window
    The two calendar weeks immediately before the randomization date, as the
    half-open interval ``[RandDate.normalize() - 14d, RandDate.normalize())``.
    The randomization calendar day itself is excluded, exactly as
    ``pedap_followup_start`` ignores the time of day recorded in ``RandDt``:
    no reading on or after the randomization date can enter the baseline, so
    no baseline quantity -- including a derivative -- can straddle it.

Coverage
    At least 168 hours (7 of the 14 days) of de-duplicated CGM, at the same
    5-minute cadence accounting used throughout.  A participant below that
    has a MISSING baseline; it is never imputed and never widened.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from PEDAP.rules import (
    CGM_SAMPLING_MINUTES,
    HIGH_IMPUTED_VALUE,
    HIGH_LOW_RULE,
    LOW_IMPUTED_VALUE,
    DUPLICATE_RULE,
    MISSING_DATA_RULE,
    cgm_hours_from_readings,
)

BASELINE_DATASET_NAME = "pedap_wadwa_baseline"
BASELINE_DATASET_VERSION = "pedap_baseline_v1"
BASELINE_SCRIPT_VERSION = "1.0.0"

# Two calendar weeks immediately before randomization.
BASELINE_WINDOW_DAYS = 14

# SAP-style coverage floor for a period-level (not weekly) summary: 168 h.
BASELINE_MIN_CGM_HOURS = 168.0
FOLLOWUP_MIN_CGM_HOURS = 168.0

BASELINE_WINDOW_RULE = (
    f"The {BASELINE_WINDOW_DAYS} calendar days immediately before the "
    "randomization date, as the half-open interval "
    "[RandDate.normalize() - 14 days, RandDate.normalize()). The "
    "randomization calendar day itself is excluded, so no baseline reading "
    "is recorded on or after randomization and no baseline derivative can "
    "straddle it."
)

BASELINE_COVERAGE_RULE = (
    f"A participant has a baseline when the window holds at least "
    f"{BASELINE_MIN_CGM_HOURS:g} hours of de-duplicated CGM, computed as "
    f"n_readings * {CGM_SAMPLING_MINUTES:g} / 60. Below that the baseline is "
    "MISSING: it is never imputed, never widened, and the participant simply "
    "does not contribute to baseline-anchored quantities."
)

FOLLOWUP_COVERAGE_RULE = (
    f"The follow-up endpoint pools ALL CGM in the pedap_v2 analysis window "
    f"(fourth day after randomization through the actual 13-week visit, or "
    f"the last valid visit for a dropout) and requires at least "
    f"{FOLLOWUP_MIN_CGM_HOURS:g} hours in total. This is the cumulative "
    "13-week follow-up period of the SAP, NOT the final weekly block."
)

ENDPOINT_MODE = "wadwa_baseline_followup"
ENDPOINT_DEFINITION = (
    "PEDAP SAP-style contrast: pre-randomization baseline (last two weeks "
    "before RandDate) versus the cumulative 13-week follow-up period. Both "
    "sides require >= 168 h of CGM. This is a DESCRIPTIVE within-participant "
    "change and a between-arm comparison of those changes; it is not the "
    "'week 0 vs week 12' block contrast, which is a different estimand on a "
    "small subset."
)


def baseline_window(rand_date) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Half-open ``[start, end)`` of the pre-randomization baseline window."""
    anchor = pd.Timestamp(rand_date)
    if pd.isna(anchor):
        raise ValueError("randomization date is missing")
    end_exclusive = anchor.normalize()
    return end_exclusive - pd.Timedelta(days=BASELINE_WINDOW_DAYS), end_exclusive


def is_baseline_usable(cgm_hours) -> bool:
    """At least 168 hours of CGM in the two-week baseline window."""
    return bool(float(cgm_hours) >= BASELINE_MIN_CGM_HOURS)


def is_followup_usable(cgm_hours) -> bool:
    """At least 168 hours of CGM pooled over the whole follow-up period."""
    return bool(float(cgm_hours) >= FOLLOWUP_MIN_CGM_HOURS)


def baseline_scientific_config() -> Dict[str, object]:
    """Configuration defining the baseline artefact (never pedap_v2's)."""
    return {
        "dataset_name": BASELINE_DATASET_NAME,
        "dataset_version": BASELINE_DATASET_VERSION,
        "script_version": BASELINE_SCRIPT_VERSION,
        "baseline_window_days": BASELINE_WINDOW_DAYS,
        "baseline_window_rule": BASELINE_WINDOW_RULE,
        "baseline_min_cgm_hours": BASELINE_MIN_CGM_HOURS,
        "baseline_coverage_rule": BASELINE_COVERAGE_RULE,
        "followup_min_cgm_hours": FOLLOWUP_MIN_CGM_HOURS,
        "followup_coverage_rule": FOLLOWUP_COVERAGE_RULE,
        "endpoint_mode": ENDPOINT_MODE,
        "endpoint_definition": ENDPOINT_DEFINITION,
        "cgm_sampling_minutes": CGM_SAMPLING_MINUTES,
        "high_imputation": HIGH_IMPUTED_VALUE,
        "low_imputation": LOW_IMPUTED_VALUE,
        "high_low_rule": HIGH_LOW_RULE,
        "duplicate_rule": DUPLICATE_RULE,
        "missing_data_rule": MISSING_DATA_RULE,
    }


def baseline_dataset_fingerprint(
    raw_file_sha256: Dict[str, str],
    config: Optional[Dict[str, object]] = None,
    cgm_source_files: Optional[Iterable[str]] = None,
) -> str:
    """Fingerprint of the baseline artefact: raw inputs plus its own config."""
    payload = {
        "raw_file_sha256": dict(sorted(raw_file_sha256.items())),
        "cgm_source_files": sorted(cgm_source_files or []),
        "scientific_config": (
            config if config is not None else baseline_scientific_config()
        ),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return (
        f"{BASELINE_DATASET_VERSION}_"
        f"{hashlib.sha256(blob).hexdigest()[:16]}"
    )
