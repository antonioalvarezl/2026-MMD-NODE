"""PEDAP (Wadwa et al.) clinical data layer.

This package holds the single authoritative implementation of every
scientific convention used to turn the raw PEDAP files into the processed
analysis dataset.  Nothing in here depends on the modelling code, so the
dataset can be built and audited without importing torch.
"""

from PEDAP.rules import (  # noqa: F401
    CGM_SAMPLING_MINUTES,
    FOLLOWUP_START_OFFSET_DAYS,
    HIGH_IMPUTED_VALUE,
    LOW_IMPUTED_VALUE,
    N_RCT_WEEKS,
    WEEKLY_MIN_CGM_HOURS,
    apply_high_low,
    apply_high_low_event_subtype,
    build_nominal_weeks,
    cgm_hours_from_readings,
    dataset_fingerprint,
    pedap_followup_end_exclusive,
    pedap_followup_start,
    scientific_config,
)
