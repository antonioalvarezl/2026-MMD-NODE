#!/usr/bin/env python3
"""Build the pre-randomization BASELINE artefact from Dexcom Clarity.

pedap_v2 contains only the RCT follow-up window, so it holds no
pre-randomization CGM at all.  The PEDAP SAP's primary descriptive contrast,
however, is baseline versus the cumulative 13-week follow-up.  This script
materialises the missing baseline side as a SEPARATE derived artefact with its
own manifest and fingerprint.

It never reads, writes or invalidates pedap_v2: the only
input is the raw Dexcom Clarity file plus PtRoster/VisitInfo, and the only
output is data/processed/pedap_benchmark/pre_randomization_baseline/.

Cleaning rules are identical to pedap_v2 (HIGH=401, LOW=39 from EventSubType;
smallest-RecID de-duplication; no 40/400 clipping; no interpolation); only the
WINDOW and the COVERAGE FLOOR differ, and both live in PEDAP.baseline_rules.

    python scripts/build_pedap_baseline_dataset.py [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import PEDAP.raw as pedap_raw  # noqa: E402
from PEDAP.baseline_rules import (  # noqa: E402
    BASELINE_DATASET_VERSION,
    BASELINE_MIN_CGM_HOURS,
    BASELINE_SCRIPT_VERSION,
    BASELINE_WINDOW_DAYS,
    baseline_dataset_fingerprint,
    baseline_scientific_config,
    baseline_window,
    is_baseline_usable,
)
from PEDAP.rules import (  # noqa: E402
    EVENT_SUBTYPE_HIGH,
    EVENT_SUBTYPE_LOW,
    HIGH_IMPUTED_VALUE,
    KNOWN_EVENT_SUBTYPES,
    LOW_IMPUTED_VALUE,
    apply_high_low_event_subtype,
    cgm_hours_from_readings,
)

DEFAULT_OUT_DIR = Path("data/processed/pedap_benchmark/pre_randomization_baseline")
CGM_FILE = "pedap_baseline_cgm.parquet"
QC_FILE = "pedap_baseline_qc.csv"
MANIFEST_FILE = "manifest.json"

EXPECTED_RANDOMIZED, EXPECTED_CLC, EXPECTED_SC = 102, 68, 34


def build_baseline_cgm(raw_dir: Path, participants: pd.DataFrame):
    """Dexcom Clarity readings inside each participant's baseline window."""
    cgm = pedap_raw.read_cgm(raw_dir)
    stats = {
        "raw_cgm_rows": int(len(cgm)),
        "raw_unparseable_timestamps": int(cgm["DeviceDtTmParsed"].isna().sum()),
    }
    assert stats["raw_unparseable_timestamps"] == 0, "unparseable DeviceDtTm values"

    observed = set(cgm["EventSubType"].dropna().unique().tolist())
    unexpected = observed - set(KNOWN_EVENT_SUBTYPES)
    assert not unexpected, f"unexpected EventSubType value(s): {sorted(unexpected)}"

    high_rows = cgm["EventSubType"] == EVENT_SUBTYPE_HIGH
    low_rows = cgm["EventSubType"] == EVENT_SUBTYPE_LOW
    assert bool((cgm.loc[high_rows, "CGM"] == HIGH_IMPUTED_VALUE).all()), \
        "raw CGM value is not 401 for every HIGH row"
    assert bool((cgm.loc[low_rows, "CGM"] == LOW_IMPUTED_VALUE).all()), \
        "raw CGM value is not 39 for every LOW row"

    cgm = cgm.rename(columns={"CGM": "CGMValueRaw"})
    cgm["GlucoseMgDl"] = apply_high_low_event_subtype(
        cgm["CGMValueRaw"].to_numpy(), cgm["EventSubType"].to_numpy()
    )

    window = participants.set_index("PtID")[
        ["baseline_start", "baseline_end_exclusive"]
    ]
    cgm = cgm.merge(window, left_on="PtID", right_index=True, how="inner")
    inside = (
        (cgm["DeviceDtTmParsed"] >= cgm["baseline_start"])
        & (cgm["DeviceDtTmParsed"] < cgm["baseline_end_exclusive"])
    )
    cgm = cgm[inside].copy()
    stats["cgm_rows_in_baseline_window"] = int(len(cgm))

    # CRITICAL: nothing at or after randomization may survive this filter.
    assert bool((cgm["DeviceDtTmParsed"] < cgm["baseline_end_exclusive"]).all()), \
        "a baseline reading falls on or after the randomization date"

    # De-duplication, identical to pedap_v2: smallest RecID wins.
    cgm = cgm.sort_values(["PtID", "DeviceDtTmParsed", "RecID"], kind="mergesort")
    dup_mask = cgm.duplicated(subset=["PtID", "DeviceDtTmParsed"], keep=False)
    duplicates = cgm[dup_mask]
    n_conflicting = 0
    if len(duplicates):
        n_values = duplicates.groupby(
            ["PtID", "DeviceDtTmParsed"], sort=False)["CGMValueRaw"].nunique()
        n_subtypes = duplicates.groupby(
            ["PtID", "DeviceDtTmParsed"], sort=False
        )["EventSubType"].nunique(dropna=False)
        n_conflicting = int(((n_values > 1) | (n_subtypes > 1)).sum())
    stats["duplicate_timestamp_rows"] = int(len(duplicates))
    stats["conflicting_duplicate_groups"] = n_conflicting

    cgm = cgm.drop_duplicates(subset=["PtID", "DeviceDtTmParsed"], keep="first")
    stats["cgm_rows_after_deduplication"] = int(len(cgm))

    out = cgm.drop(columns=["DeviceDtTm"]).rename(
        columns={"DeviceDtTmParsed": "DeviceDtTm"}
    )[["PtID", "DeviceDtTm", "CGMValueRaw", "EventSubType", "GlucoseMgDl",
       "SourceDeviceID"]].copy()
    out["PtID"] = out["PtID"].astype("int32")
    out["CGMValueRaw"] = out["CGMValueRaw"].astype("float32")
    out["GlucoseMgDl"] = out["GlucoseMgDl"].astype("float32")
    out = out.sort_values(["PtID", "DeviceDtTm"], kind="mergesort").reset_index(drop=True)

    stats["n_high"] = int((out["EventSubType"] == EVENT_SUBTYPE_HIGH).sum())
    stats["n_low"] = int((out["EventSubType"] == EVENT_SUBTYPE_LOW).sum())
    stats["n_normal"] = int(out["EventSubType"].isna().sum())
    return out, stats


def build_baseline_qc(participants: pd.DataFrame, cgm: pd.DataFrame) -> pd.DataFrame:
    counts = cgm.groupby("PtID").agg(
        n_cgm_readings=("GlucoseMgDl", "size"),
        first_timestamp=("DeviceDtTm", "min"),
        last_timestamp=("DeviceDtTm", "max"),
        n_high=("EventSubType", lambda s: int((s == EVENT_SUBTYPE_HIGH).sum())),
        n_low=("EventSubType", lambda s: int((s == EVENT_SUBTYPE_LOW).sum())),
        mean_glucose=("GlucoseMgDl", "mean"),
    )
    qc = participants[[
        "PtID", "TrtGroup", "RandDate", "baseline_start", "baseline_end_exclusive",
    ]].merge(counts, on="PtID", how="left")
    qc["n_cgm_readings"] = qc["n_cgm_readings"].fillna(0).astype(int)
    qc["n_high"] = qc["n_high"].fillna(0).astype(int)
    qc["n_low"] = qc["n_low"].fillna(0).astype(int)
    qc["n_normal"] = qc["n_cgm_readings"] - qc["n_high"] - qc["n_low"]
    # cgm_hours_from_readings returns a Python float, so it is scalar-only by
    # design; apply it elementwise rather than reimplementing the rule here.
    qc["cgm_hours"] = [
        cgm_hours_from_readings(n) for n in qc["n_cgm_readings"]
    ]
    qc["usable"] = [is_baseline_usable(h) for h in qc["cgm_hours"]]
    return qc.sort_values("PtID").reset_index(drop=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(pedap_raw.RAW_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    existing = [out_dir / n for n in (CGM_FILE, QC_FILE, MANIFEST_FILE)
                if (out_dir / n).exists()]
    if existing and not args.force:
        print("Baseline artefact already exists; re-run with --force.")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    roster = pedap_raw.read_pt_roster(raw_dir)
    participants = pedap_raw.randomized_cohort(roster, strict=True).copy()
    assert len(participants) == EXPECTED_RANDOMIZED
    assert int((participants["TrtGroup"] == "CLC").sum()) == EXPECTED_CLC
    assert int((participants["TrtGroup"] == "SC").sum()) == EXPECTED_SC

    windows = [baseline_window(d) for d in participants["RandDate"]]
    participants["baseline_start"] = [w[0] for w in windows]
    participants["baseline_end_exclusive"] = [w[1] for w in windows]

    print(f"CGM source: {pedap_raw.CGM_FILES[0]}")
    print(f"Baseline window: {BASELINE_WINDOW_DAYS} days before RandDate, "
          f">= {BASELINE_MIN_CGM_HOURS:g} h required")

    cgm, stats = build_baseline_cgm(raw_dir, participants)
    qc = build_baseline_qc(participants, cgm)

    n_usable = int(qc["usable"].sum())
    by_arm = qc.groupby("TrtGroup")["usable"].agg(["sum", "count"])
    print(f"\nBaseline available: {n_usable}/{len(qc)}")
    print(by_arm.rename(columns={"sum": "with_baseline", "count": "randomized"}))
    print("\nMean glucose among participants WITH a baseline:")
    print(qc[qc["usable"]].groupby("TrtGroup")["mean_glucose"].agg(["mean", "count"]))

    # VALIDATION: both arms must retain a usable baseline cohort, or the
    # contrast is not estimable and the artefact should not be written.
    for arm in ("CLC", "SC"):
        got = int(qc.loc[qc["TrtGroup"] == arm, "usable"].sum())
        assert got >= 2, f"arm {arm} has only {got} participants with a baseline"

    cgm.to_parquet(out_dir / CGM_FILE, index=False)
    qc.to_csv(out_dir / QC_FILE, index=False)

    raw_hashes = pedap_raw.raw_file_hashes(raw_dir)
    config = baseline_scientific_config()
    fingerprint = baseline_dataset_fingerprint(
        raw_hashes, config, cgm_source_files=list(pedap_raw.CGM_FILES)
    )
    manifest = {
        "dataset_version": BASELINE_DATASET_VERSION,
        "script_version": BASELINE_SCRIPT_VERSION,
        "dataset_fingerprint": fingerprint,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {"cgm": CGM_FILE, "baseline_qc": QC_FILE},
        "scientific_config": config,
        "raw_file_sha256": raw_hashes,
        "cgm_source_files": list(pedap_raw.CGM_FILES),
        "cgm_stats": stats,
        "n_randomized": int(len(qc)),
        "n_with_baseline": n_usable,
        "n_with_baseline_by_arm": {
            arm: int(qc.loc[qc["TrtGroup"] == arm, "usable"].sum())
            for arm in ("CLC", "SC")
        },
        "scope_caveat": (
            "PRE-RANDOMIZATION CGM ONLY. This artefact exists solely to "
            "supply the baseline side of the SAP's baseline-vs-follow-up "
            "contrast and the baseline anchor for change_from_baseline. It "
            "is never mixed into the RCT follow-up dataset, and it does not "
            "affect the pedap_v2 fingerprints."
        ),
    }
    (out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote {out_dir}/  fingerprint={fingerprint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
