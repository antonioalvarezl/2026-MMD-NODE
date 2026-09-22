#!/usr/bin/env python3
"""Build the processed PEDAP analysis dataset from the raw Jaeb files.

This script is run MANUALLY.  ``main.py`` never invokes it: the experiments
assume the processed dataset already exists.

    python scripts/build_pedap_dataset.py            # first build
    python scripts/build_pedap_dataset.py --force    # rebuild in place

Outputs, under ``data/processed/pedap_benchmark/rct_followup/`` (nothing else):

    pedap_participants.csv   one row per randomized participant
    pedap_cgm_rct.parquet    every CGM reading inside the SAP follow-up window
    pedap_weekly_qc.csv      one row per participant x nominal RCT week
    manifest.json            how the dataset was built, plus its fingerprint

CGM SOURCE DECISION (v2): PEDAPDexcomClarityCGM.txt -- the study Dexcom G6 /
Dexcom Clarity export -- is the SOLE CGM source for BOTH arms.  It covers the
102 randomized participants symmetrically (both CLC and SC wore a study
Dexcom G6 throughout the RCT).  PEDAPTandemCGMDATAGXB.txt (Tandem
pump-relayed CGM) is NOT used to build this dataset: during the RCT phase
only CLC used the Tandem pump, so it is structurally CLC-only in the SAP
follow-up window; mixing it with Dexcom for SC would confound treatment arm
with data source.  See PEDAP.rules.CGM_SOURCE_DECISION.

Every scientific invariant this depends on is checked with a hard assertion
below rather than written out to a separate audit report: see "VALIDATIONS".
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PEDAP import raw as pedap_raw
from PEDAP.raw import EXPECTED_CLC, EXPECTED_RANDOMIZED, EXPECTED_SC
from PEDAP.rules import (
    DATASET_NAME,
    DATASET_VERSION,
    EVENT_SUBTYPE_HIGH,
    EVENT_SUBTYPE_LOW,
    HIGH_IMPUTED_VALUE,
    KNOWN_EVENT_SUBTYPES,
    LOW_IMPUTED_VALUE,
    N_RCT_WEEKS,
    PREPROCESSING_SCRIPT_VERSION,
    WEEKLY_MIN_CGM_HOURS,
    apply_high_low_event_subtype,
    build_nominal_weeks,
    cgm_hours_from_readings,
    dataset_fingerprint,
    is_week_usable,
    pedap_followup_end_exclusive,
    pedap_followup_start,
    scientific_config,
)

DEFAULT_OUT_DIR = Path("data/processed/pedap_benchmark/rct_followup")

PARTICIPANTS_FILE = "pedap_participants.csv"
CGM_FILE = "pedap_cgm_rct.parquet"
WEEKLY_QC_FILE = "pedap_weekly_qc.csv"
MANIFEST_FILE = "manifest.json"

# A conflicting-duplicate rate above this is treated as a sign that something
# is structurally wrong (e.g. a schema mismatch), not ordinary sensor noise:
# the CGM provenance audit found ~0.01% of windowed rows in conflicting
# groups, so 1% leaves ~100x headroom while still catching a real problem.
MAX_CONFLICTING_DUPLICATE_RATE = 0.01

PARTICIPANT_COLUMNS = [
    "PtID",
    "SiteID",
    "EnrollDt",
    "RandDt",
    "RandDate",
    "PtStatus",
    "TrtGroup",
    "AgeAsofEnrollDt",
    "Week13VisitDate",
    "Week13OutOfWin",
    "Week13OutOfWinReason",
    "analysis_start_date",
    "analysis_end_date",
    "analysis_end_source",
]


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Participants
# -----------------------------------------------------------------------------

def build_participants(raw_dir: Path) -> tuple[pd.DataFrame, int]:
    """Build the participant table.  TrtGroup comes only from PtRoster."""
    roster = pedap_raw.read_pt_roster(raw_dir)
    cohort = pedap_raw.randomized_cohort(roster, strict=True)

    visits = pedap_raw.read_visit_info(raw_dir)
    wk13 = pedap_raw.week13_visit(visits)
    last_visit = pedap_raw.last_visit_before(visits)
    rand_visit = pedap_raw.randomization_visit_dates(visits)

    participants = (
        cohort.merge(wk13, on="PtID", how="left")
        .merge(last_visit, on="PtID", how="left")
        .merge(rand_visit, on="PtID", how="left")
    )

    # RandDt from PtRoster is the primary source because it carries a time.
    # A discrepancy against the Randomization visit is recorded, never fixed
    # silently.
    participants["rand_date_matches_visit"] = (
        participants["RandVisitDate"] == participants["RandDate"]
    )
    n_discrepancies = int(
        (
            participants["RandVisitDate"].notna()
            & ~participants["rand_date_matches_visit"]
        ).sum()
    )

    starts, ends, sources = [], [], []
    for record in participants.itertuples():
        starts.append(pedap_followup_start(record.RandDate))
        if pd.notna(record.Week13VisitDate):
            ends.append(pd.Timestamp(record.Week13VisitDate))
            sources.append("week13_visit")
        elif pd.notna(record.LastVisitDate):
            ends.append(pd.Timestamp(record.LastVisitDate))
            sources.append("last_visit_before_dropout")
        else:
            ends.append(pd.NaT)
            sources.append("none")

    participants["analysis_start_date"] = starts
    participants["analysis_end_date"] = ends
    participants["analysis_end_source"] = sources
    # Primary rule: the whole calendar day of the actual 13-week visit (or
    # last visit) is included; the half-open upper bound is date + 1 day.
    participants["analysis_end_exclusive"] = [
        pedap_followup_end_exclusive(end) if pd.notna(end) else pd.NaT
        for end in ends
    ]
    participants["followup_days"] = (
        participants["analysis_end_exclusive"] - participants["analysis_start_date"]
    ).dt.days
    participants["is_dropout_before_week13"] = (
        participants["analysis_end_source"] == "last_visit_before_dropout"
    )

    empty = participants["analysis_end_exclusive"] <= participants["analysis_start_date"]
    if empty.any():
        print(
            "  ! participants whose follow-up window is empty: "
            f"{participants.loc[empty, 'PtID'].tolist()}"
        )

    return participants.sort_values("PtID").reset_index(drop=True), n_discrepancies


# -----------------------------------------------------------------------------
# CGM (PEDAPDexcomClarityCGM.txt only)
# -----------------------------------------------------------------------------

def build_cgm(raw_dir: Path, participants: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Build the RCT-window Dexcom Clarity CGM table."""
    cgm = pedap_raw.read_cgm(raw_dir)  # CGM_FILES default = Dexcom Clarity only
    stats = {
        "raw_cgm_rows": int(len(cgm)),
        "raw_cgm_participants": int(cgm["PtID"].nunique()),
        "raw_unparseable_timestamps": int(cgm["DeviceDtTmParsed"].isna().sum()),
    }
    assert stats["raw_unparseable_timestamps"] == 0, (
        f"{stats['raw_unparseable_timestamps']} DeviceDtTm values in "
        f"{pedap_raw.CGM_FILES} could not be parsed"
    )

    # VALIDATION: only the documented EventSubType vocabulary appears.
    observed_subtypes = set(cgm["EventSubType"].dropna().unique().tolist())
    unexpected_subtypes = observed_subtypes - set(KNOWN_EVENT_SUBTYPES)
    assert not unexpected_subtypes, (
        f"unexpected EventSubType value(s) in {pedap_raw.CGM_FILES}: "
        f"{sorted(unexpected_subtypes)}; only {list(KNOWN_EVENT_SUBTYPES)} "
        "(or blank) are defined by the SAP"
    )

    # VALIDATION: even though the raw CGM column already carries 401/39 for
    # HIGH/LOW rows, don't trust that alone -- assert it, so a change in the
    # raw encoding is caught rather than silently propagated.
    high_rows = cgm["EventSubType"] == EVENT_SUBTYPE_HIGH
    low_rows = cgm["EventSubType"] == EVENT_SUBTYPE_LOW
    assert bool((cgm.loc[high_rows, "CGM"] == HIGH_IMPUTED_VALUE).all()), (
        "raw CGM value is not 401 for every EventSubType=='HIGH' row"
    )
    assert bool((cgm.loc[low_rows, "CGM"] == LOW_IMPUTED_VALUE).all()), (
        "raw CGM value is not 39 for every EventSubType=='LOW' row"
    )
    stats["n_high_raw"] = int(high_rows.sum())
    stats["n_low_raw"] = int(low_rows.sum())

    # SAP HIGH/LOW imputation, applied before anything else looks at values.
    # The imputed value comes from the RULE (EventSubType), not from trusting
    # the raw CGM number -- the asserts above are the separate consistency
    # check against the raw encoding.
    cgm = cgm.rename(columns={"CGM": "CGMValueRaw"})
    cgm["GlucoseMgDl"] = apply_high_low_event_subtype(
        cgm["CGMValueRaw"].to_numpy(), cgm["EventSubType"].to_numpy()
    )

    window = participants.set_index("PtID")[
        ["analysis_start_date", "analysis_end_exclusive"]
    ]
    cgm = cgm.merge(window, left_on="PtID", right_index=True, how="inner")
    stats["cgm_rows_for_randomized"] = int(len(cgm))

    inside = (
        (cgm["DeviceDtTmParsed"] >= cgm["analysis_start_date"])
        & (cgm["DeviceDtTmParsed"] < cgm["analysis_end_exclusive"])
    )
    cgm = cgm[inside].copy()
    stats["cgm_rows_in_followup_window"] = int(len(cgm))

    # ------------------------------------------------------- de-duplication
    cgm = cgm.sort_values(["PtID", "DeviceDtTmParsed", "RecID"], kind="mergesort")
    dup_mask = cgm.duplicated(subset=["PtID", "DeviceDtTmParsed"], keep=False)
    duplicates = cgm[dup_mask]

    n_conflicting_groups = 0
    if len(duplicates):
        # dropna=False: a NaN (normal) EventSubType vs. a "HIGH"/"LOW" one at
        # the same timestamp IS a real conflict; nunique()'s default drops
        # NaN and would silently miss exactly that case.
        n_distinct_values = duplicates.groupby(
            ["PtID", "DeviceDtTmParsed"], sort=False
        )["CGMValueRaw"].nunique()
        n_distinct_subtypes = duplicates.groupby(
            ["PtID", "DeviceDtTmParsed"], sort=False
        )["EventSubType"].nunique(dropna=False)
        n_conflicting_groups = int(
            ((n_distinct_values > 1) | (n_distinct_subtypes > 1)).sum()
        )

    stats["duplicate_timestamp_rows"] = int(len(duplicates))
    stats["duplicate_timestamp_groups"] = (
        int(duplicates.groupby(["PtID", "DeviceDtTmParsed"]).ngroups) if len(duplicates) else 0
    )
    stats["conflicting_duplicate_groups"] = n_conflicting_groups

    # VALIDATION: a conflicting-duplicate rate far above what was observed in
    # the provenance audit signals a structural problem, not sensor noise.
    conflict_rate = n_conflicting_groups / max(stats["cgm_rows_in_followup_window"], 1)
    assert conflict_rate <= MAX_CONFLICTING_DUPLICATE_RATE, (
        f"{n_conflicting_groups} conflicting duplicate groups "
        f"({100 * conflict_rate:.2f}% of windowed rows) exceeds the "
        f"{100 * MAX_CONFLICTING_DUPLICATE_RATE:.1f}% sanity threshold"
    )

    # Smallest RecID wins; identical repeats collapse to the same single row.
    cgm = cgm.drop_duplicates(subset=["PtID", "DeviceDtTmParsed"], keep="first")
    stats["cgm_rows_after_deduplication"] = int(len(cgm))

    out = cgm.drop(columns=["DeviceDtTm"]).rename(
        columns={"DeviceDtTmParsed": "DeviceDtTm"}
    )[["PtID", "DeviceDtTm", "CGMValueRaw", "EventSubType", "GlucoseMgDl", "SourceDeviceID"]].copy()
    out["PtID"] = out["PtID"].astype("int32")
    out["CGMValueRaw"] = out["CGMValueRaw"].astype("float32")
    out["GlucoseMgDl"] = out["GlucoseMgDl"].astype("float32")
    out = out.sort_values(["PtID", "DeviceDtTm"], kind="mergesort").reset_index(drop=True)

    stats["n_high"] = int((out["EventSubType"] == EVENT_SUBTYPE_HIGH).sum())
    stats["n_low"] = int((out["EventSubType"] == EVENT_SUBTYPE_LOW).sum())
    stats["n_normal"] = int(out["EventSubType"].isna().sum())
    assert stats["n_high"] + stats["n_low"] + stats["n_normal"] == len(out)

    return out, stats


# -----------------------------------------------------------------------------
# Weekly QC
# -----------------------------------------------------------------------------

def build_weekly_qc(participants: pd.DataFrame, cgm: pd.DataFrame) -> pd.DataFrame:
    """One row per participant x nominal RCT week, present even when empty."""
    by_patient = {pt: frame for pt, frame in cgm.groupby("PtID", sort=True)}

    rows = []
    for record in participants.itertuples():
        if pd.isna(record.analysis_end_exclusive):
            continue
        blocks = build_nominal_weeks(
            record.RandDate,
            record.analysis_start_date,
            record.analysis_end_exclusive,
            n_weeks=N_RCT_WEEKS,
        )
        frame = by_patient.get(record.PtID)
        times = (
            frame["DeviceDtTm"].to_numpy() if frame is not None else np.array([], dtype="datetime64[ns]")
        )
        subtypes = (
            frame["EventSubType"].to_numpy() if frame is not None else np.array([], dtype=object)
        )

        for block in blocks:
            if block["has_window"] and len(times):
                mask = (
                    (times >= np.datetime64(block["actual_start_datetime"]))
                    & (times < np.datetime64(block["actual_end_datetime"]))
                )
            else:
                mask = np.zeros(len(times), dtype=bool)

            n_readings = int(mask.sum())
            hours = cgm_hours_from_readings(n_readings)
            block_subtypes = subtypes[mask]
            n_high = int((block_subtypes == EVENT_SUBTYPE_HIGH).sum())
            n_low = int((block_subtypes == EVENT_SUBTYPE_LOW).sum())
            rows.append(
                {
                    "PtID": int(record.PtID),
                    "nominal_week_index": block["nominal_week_index"],
                    "nominal_start_date": block["nominal_start_date"],
                    "nominal_end_date": block["nominal_end_date"],
                    "actual_start_datetime": (
                        block["actual_start_datetime"] if block["has_window"] else pd.NaT
                    ),
                    "actual_end_datetime": (
                        block["actual_end_datetime"] if block["has_window"] else pd.NaT
                    ),
                    "has_window": block["has_window"],
                    "observation_time_weeks": block["observation_time_weeks"],
                    "n_cgm_readings": n_readings,
                    "cgm_hours": hours,
                    "usable": is_week_usable(hours),
                    "n_high": n_high,
                    "n_low": n_low,
                    "n_normal": n_readings - n_high - n_low,
                    "first_timestamp": pd.Timestamp(times[mask][0]) if n_readings else pd.NaT,
                    "last_timestamp": pd.Timestamp(times[mask][-1]) if n_readings else pd.NaT,
                    "TrtGroup": record.TrtGroup,
                    "Week13OutOfWin": record.Week13OutOfWin,
                }
            )

    weekly = pd.DataFrame(rows)
    return weekly.sort_values(["PtID", "nominal_week_index"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Arm coverage
# -----------------------------------------------------------------------------

def arm_coverage_table(participants: pd.DataFrame, weekly: pd.DataFrame) -> pd.DataFrame:
    """CLC/SC coverage summary; also the basis for the builder's hard asserts."""
    per_patient = weekly.groupby("PtID").agg(
        readings=("n_cgm_readings", "sum"),
        usable_weeks=("usable", "sum"),
    )
    joined = participants.set_index("PtID")[["TrtGroup"]].join(per_patient).fillna(0)
    rows = []
    for arm in pedap_raw.TREATMENT_ARMS:
        sub = joined[joined["TrtGroup"] == arm]
        with_cgm = int((sub["readings"] > 0).sum())
        rows.append(
            {
                "arm": arm,
                "randomized": int(len(sub)),
                "patients_with_rct_cgm": with_cgm,
                "patients_with_usable_week": int((sub["usable_weeks"] > 0).sum()),
                "usable_patient_weeks": int(sub["usable_weeks"].sum()),
                "cgm_readings": int(sub["readings"].sum()),
                "median_usable_weeks_per_patient": float(sub["usable_weeks"].median()),
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(pedap_raw.RAW_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--force", action="store_true",
        help="rebuild over an existing processed dataset",
    )
    args = parser.parse_args(argv)

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    existing = [
        out_dir / name
        for name in (PARTICIPANTS_FILE, CGM_FILE, WEEKLY_QC_FILE, MANIFEST_FILE)
        if (out_dir / name).exists()
    ]
    if existing and not args.force:
        print("PEDAP processed dataset already exists:")
        for path in existing:
            print(f"  {path}")
        print("\nRefusing to overwrite it.  Re-run with --force to rebuild.")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"CGM source: {pedap_raw.CGM_FILES[0]} (sole source, both arms)")

    # ---------------------------------------------------------- participants
    participants, n_discrepancies = build_participants(raw_dir)

    # VALIDATIONS: cohort.
    randomized_n = len(participants)
    clc_n = int((participants["TrtGroup"] == "CLC").sum())
    sc_n = int((participants["TrtGroup"] == "SC").sum())
    assert randomized_n == EXPECTED_RANDOMIZED, (
        f"expected {EXPECTED_RANDOMIZED} randomized, got {randomized_n}"
    )
    assert clc_n == EXPECTED_CLC, f"expected {EXPECTED_CLC} CLC, got {clc_n}"
    assert sc_n == EXPECTED_SC, f"expected {EXPECTED_SC} SC, got {sc_n}"

    n_no_wk13 = int(participants["Week13VisitDate"].isna().sum())
    out_of_win = participants["Week13OutOfWin"].fillna(0).astype(int)
    n_out_of_window = int((out_of_win == 1).sum())

    # ------------------------------------------------------------------ CGM
    cgm, cgm_stats = build_cgm(raw_dir, participants)

    # -------------------------------------------------------------- weekly
    weekly = build_weekly_qc(participants, cgm)
    n_usable = int(weekly["usable"].sum())
    n_unusable = int((~weekly["usable"]).sum())

    per_patient_readings = weekly.groupby("PtID")["n_cgm_readings"].sum()
    n_with_cgm = int((per_patient_readings > 0).sum())
    n_without_cgm = int(randomized_n - n_with_cgm)
    participants = participants.merge(
        weekly.groupby("PtID").agg(
            n_cgm_readings=("n_cgm_readings", "sum"),
            n_usable_weeks=("usable", "sum"),
        ),
        on="PtID", how="left",
    )
    participants["n_cgm_readings"] = participants["n_cgm_readings"].fillna(0).astype(int)
    participants["n_usable_weeks"] = participants["n_usable_weeks"].fillna(0).astype(int)
    participants["has_rct_cgm"] = participants["n_cgm_readings"] > 0

    # ----------------------------------------------------------- arm coverage
    coverage = arm_coverage_table(participants, weekly)
    clc_row = coverage[coverage["arm"] == "CLC"].iloc[0]
    sc_row = coverage[coverage["arm"] == "SC"].iloc[0]

    # VALIDATIONS: symmetric arm coverage (the whole point of switching CGM
    # source).  Hard assertions, not a soft/bypassable gate.
    assert int(clc_row["patients_with_rct_cgm"]) == EXPECTED_CLC, (
        f"expected all {EXPECTED_CLC} CLC participants to have RCT CGM, "
        f"got {int(clc_row['patients_with_rct_cgm'])}"
    )
    assert int(sc_row["patients_with_rct_cgm"]) == EXPECTED_SC, (
        f"expected all {EXPECTED_SC} SC participants to have RCT CGM, "
        f"got {int(sc_row['patients_with_rct_cgm'])}"
    )
    assert n_with_cgm == randomized_n, (
        f"{randomized_n - n_with_cgm} randomized participant(s) have no RCT CGM at all"
    )

    clc_rate = clc_row["usable_patient_weeks"] / max(clc_row["randomized"] * N_RCT_WEEKS, 1)
    sc_rate = sc_row["usable_patient_weeks"] / max(sc_row["randomized"] * N_RCT_WEEKS, 1)
    coverage_ratio = sc_rate / clc_rate if clc_rate > 0 else float("nan")
    assert np.isfinite(coverage_ratio) and 0.5 <= coverage_ratio <= 2.0, (
        f"SC/CLC usable-week rate ratio {coverage_ratio:.4f} is outside the "
        "reasonable [0.5, 2.0] range: weekly CGM coverage is not comparable "
        "between arms"
    )

    median_by_arm = {
        "CLC": float(clc_row["median_usable_weeks_per_patient"]),
        "SC": float(sc_row["median_usable_weeks_per_patient"]),
    }

    # -------------------------------------------------------------- writing
    participant_out = participants[
        PARTICIPANT_COLUMNS
        + [
            "RandVisitDate",
            "rand_date_matches_visit",
            "LastVisitDate",
            "analysis_end_exclusive",
            "followup_days",
            "is_dropout_before_week13",
            "n_cgm_readings",
            "n_usable_weeks",
            "has_rct_cgm",
        ]
    ]
    participant_out.to_csv(out_dir / PARTICIPANTS_FILE, index=False)
    cgm.to_parquet(out_dir / CGM_FILE, index=False, compression="zstd")
    weekly.to_csv(out_dir / WEEKLY_QC_FILE, index=False)

    raw_hashes = pedap_raw.raw_file_hashes(raw_dir)  # PtRoster, VisitInfo, Dexcom Clarity
    fingerprint = dataset_fingerprint(raw_hashes, scientific_config(), pedap_raw.CGM_FILES)

    manifest = {
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "dataset_fingerprint": fingerprint,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "preprocessing_script_version": PREPROCESSING_SCRIPT_VERSION,
        "git_commit_if_available": _git_commit(),
        "raw_files": sorted(raw_hashes),
        "raw_file_sha256": raw_hashes,
        "randomized_n": randomized_n,
        "clc_n": clc_n,
        "sc_n": sc_n,
        "cgm_source": pedap_raw.CGM_FILES[0],
        "participants_with_rct_cgm_by_arm": {
            "CLC": int(clc_row["patients_with_rct_cgm"]),
            "SC": int(sc_row["patients_with_rct_cgm"]),
        },
        "usable_patient_weeks_by_arm": {
            "CLC": int(clc_row["usable_patient_weeks"]),
            "SC": int(sc_row["usable_patient_weeks"]),
        },
        "median_usable_weeks_by_arm": median_by_arm,
        "n_cgm_readings": int(len(cgm)),
        "n_high": cgm_stats["n_high"],
        "n_low": cgm_stats["n_low"],
        "n_normal": cgm_stats["n_normal"],
        "duplicate_timestamp_groups": cgm_stats["duplicate_timestamp_groups"],
        "conflicting_duplicate_groups": cgm_stats["conflicting_duplicate_groups"],
        "n_without_week13_visit": n_no_wk13,
        "n_week13_out_of_window": n_out_of_window,
        "n_dropout_endpoints": int(participants["is_dropout_before_week13"].sum()),
        "n_randomization_date_discrepancies": n_discrepancies,
        "n_participants_with_cgm": n_with_cgm,
        "n_participants_without_cgm": n_without_cgm,
        "n_usable_patient_weeks": n_usable,
        "n_unusable_patient_weeks": n_unusable,
        "sc_over_clc_usable_week_ratio": float(coverage_ratio),
        "cgm_build_stats": cgm_stats,
        "arm_coverage": coverage.to_dict(orient="records"),
        "files": {
            "participants": PARTICIPANTS_FILE,
            "cgm": CGM_FILE,
            "weekly_qc": WEEKLY_QC_FILE,
        },
        **scientific_config(),
    }
    # followup_start_rule / followup_end_rule / weekly_min_cgm_hours /
    # missing_data_rule / high_low_rule (via scientific_config's rule text)
    # are already included through **scientific_config() above.
    with open(out_dir / MANIFEST_FILE, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=False)

    # ------------------------------------------------------------ summary
    print("\nPEDAP v2 built successfully\n")
    print(f"Randomized: {randomized_n}")
    print(f"CLC: {clc_n}")
    print(f"SC: {sc_n}\n")
    print("Participants with RCT CGM:")
    print(f"CLC: {int(clc_row['patients_with_rct_cgm'])}/{clc_n}")
    print(f"SC: {int(sc_row['patients_with_rct_cgm'])}/{sc_n}\n")
    print("Usable patient-weeks:")
    print(f"CLC: {int(clc_row['usable_patient_weeks'])}")
    print(f"SC: {int(sc_row['usable_patient_weeks'])}\n")
    print("Median usable weeks:")
    print(f"CLC: {median_by_arm['CLC']:g}")
    print(f"SC: {median_by_arm['SC']:g}\n")
    print(f"CGM readings: {len(cgm):,}")
    print(f"HIGH: {cgm_stats['n_high']:,}")
    print(f"LOW: {cgm_stats['n_low']:,}\n")
    print(f"Week13 missing: {n_no_wk13}")
    print(f"Week13 out-of-window: {n_out_of_window}")
    print(f"Dropout endpoints: {int(participants['is_dropout_before_week13'].sum())}\n")
    print(f"Duplicate groups: {cgm_stats['duplicate_timestamp_groups']}")
    print(f"Conflicting duplicate groups: {cgm_stats['conflicting_duplicate_groups']}\n")
    print(f"Dataset fingerprint: {fingerprint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
