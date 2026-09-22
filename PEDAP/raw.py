"""Readers for the raw PEDAP (Jaeb) pipe-delimited files.

The raw files are never modified.  These readers only parse them and apply
the cohort definition from ``PtRoster``; no scientific filtering happens here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from PEDAP.rules import parse_pedap_date, parse_pedap_datetime

# Expected PEDAP randomized cohort (Wadwa et al.).
EXPECTED_RANDOMIZED = 102
EXPECTED_CLC = 68
EXPECTED_SC = 34

TREATMENT_ARMS = ("CLC", "SC")

RAW_DIR = Path("data/raw/pedap")
PT_ROSTER_FILE = "PtRoster.txt"
VISIT_INFO_FILE = "PEDAPVisitInfo.txt"

# The sole CGM source for the v2 RCT dataset: the study Dexcom G6 / Dexcom
# Clarity export, symmetric for both arms (see
# PEDAP.rules.CGM_SOURCE_DECISION for why).
CGM_FILES = ("PEDAPDexcomClarityCGM.txt",)

# AUXILIARY raw file, kept locally but NOT part of the RCT dataset pipeline.
# During the RCT phase only the CLC arm used the Tandem pump, so this file is
# structurally CLC-only in the SAP follow-up window; mixing it with Dexcom
# for SC would confound treatment arm with data source.  Not read by
# read_cgm() unless a caller explicitly passes files=(TANDEM_CGM_FILE,).
TANDEM_CGM_FILE = "PEDAPTandemCGMDATAGXB.txt"

RANDOMIZATION_VISIT = "Randomization"
WEEK13_VISIT = "13 Week"


def read_pt_roster(raw_dir=RAW_DIR) -> pd.DataFrame:
    """Read PtRoster and add the normalized randomization calendar date."""
    path = Path(raw_dir) / PT_ROSTER_FILE
    roster = pd.read_csv(path, sep="|", dtype=str, keep_default_na=False)
    roster = roster.replace("", np.nan)
    roster["PtID"] = roster["PtID"].astype(int)
    roster["SiteID"] = pd.to_numeric(roster["SiteID"], errors="coerce").astype("Int64")
    roster["AgeAsofEnrollDt"] = pd.to_numeric(
        roster["AgeAsofEnrollDt"], errors="coerce"
    ).astype("Int64")
    roster["EnrollDate"] = parse_pedap_date(roster["EnrollDt"]).to_numpy()
    # RandDt carries a time of day; it is kept verbatim for audit and reduced
    # to a calendar date for every window computation.
    roster["RandDtParsed"] = parse_pedap_datetime(roster["RandDt"]).to_numpy()
    roster["RandDate"] = pd.Series(roster["RandDtParsed"]).dt.normalize().to_numpy()
    return roster.sort_values("PtID").reset_index(drop=True)


def randomized_cohort(roster: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """Select randomized participants: the cohort is ``RandDt`` not missing."""
    randomized = roster[roster["RandDate"].notna()].copy()
    randomized = randomized.sort_values("PtID").reset_index(drop=True)
    if strict:
        n = len(randomized)
        n_clc = int((randomized["TrtGroup"] == "CLC").sum())
        n_sc = int((randomized["TrtGroup"] == "SC").sum())
        assert n == EXPECTED_RANDOMIZED, (
            f"expected {EXPECTED_RANDOMIZED} randomized participants, got {n}"
        )
        assert n_clc == EXPECTED_CLC, f"expected {EXPECTED_CLC} CLC, got {n_clc}"
        assert n_sc == EXPECTED_SC, f"expected {EXPECTED_SC} SC, got {n_sc}"
        unknown = sorted(set(randomized["TrtGroup"].dropna()) - set(TREATMENT_ARMS))
        assert not unknown, f"unexpected treatment arm label(s): {unknown}"
    return randomized


def read_visit_info(raw_dir=RAW_DIR) -> pd.DataFrame:
    """Read PEDAPVisitInfo with parsed visit dates."""
    path = Path(raw_dir) / VISIT_INFO_FILE
    visits = pd.read_csv(path, sep="|", dtype=str, keep_default_na=False)
    visits = visits.replace("", np.nan)
    visits["PtID"] = visits["PtID"].astype(int)
    visits["VisitDate"] = parse_pedap_date(visits["VisitDt"]).to_numpy()
    visits["OutOfWin"] = pd.to_numeric(visits["OutOfWin"], errors="coerce").astype("Int64")
    visits["VisitMiss"] = pd.to_numeric(visits["VisitMiss"], errors="coerce").astype("Int64")
    return visits


def read_cgm(
    raw_dir=RAW_DIR,
    files=CGM_FILES,
    usecols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Read every raw CGM source file and stack them with provenance.

    Returns the untouched raw columns plus ``DeviceDtTmParsed`` and
    ``source_file``.  No filtering, no HIGH/LOW mapping, no de-duplication.

    The dtype map below is the union of both documented raw CGM schemas --
    Dexcom Clarity's ``EventSubType``/``SourceDeviceID``/``CGM`` (the ACTIVE
    schema, ``CGM_FILES`` default) and Tandem's ``CGMValue``/
    ``HighLowIndicator`` (auxiliary, ``TANDEM_CGM_FILE``) -- so this one
    reader works for either without a schema branch; pandas only applies a
    dtype to a column that is actually present in the file being read.
    """
    frames = []
    for name in files:
        path = Path(raw_dir) / name
        frame = pd.read_csv(
            path,
            sep="|",
            usecols=usecols,
            dtype={
                "PtID": "int32",
                "RecID": "int64",
                "DeviceDtTm": "string",
                # Dexcom Clarity schema.  EventSubType is deliberately plain
                # "object" (not pandas' nullable "string" dtype): a missing
                # entry then reads back as a plain float NaN, so
                # ``column == "HIGH"`` compares cleanly to False instead of
                # raising on pandas.NA's ambiguous boolean value once the
                # column round-trips through ``.to_numpy()``.
                "EventSubType": "object",
                "SourceDeviceID": "string",
                "CGM": "float32",
                "TransmitterTime": "int64",
                # Tandem schema (auxiliary; not part of CGM_FILES).
                "CGMValue": "float32",
                "HighLowIndicator": "int16",
            },
        )
        frame["source_file"] = name
        frames.append(frame)
    cgm = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    cgm["DeviceDtTmParsed"] = parse_pedap_datetime(cgm["DeviceDtTm"]).to_numpy()
    return cgm


def randomization_visit_dates(visits: pd.DataFrame) -> pd.DataFrame:
    """One row per participant with a ``Randomization`` visit date."""
    rand = visits[visits["Visit"] == RANDOMIZATION_VISIT]
    return (
        rand.groupby("PtID", as_index=False)["VisitDate"]
        .min()
        .rename(columns={"VisitDate": "RandVisitDate"})
    )


def week13_visit(visits: pd.DataFrame) -> pd.DataFrame:
    """One row per participant with the actual ``13 Week`` visit and its QC flags."""
    wk13 = visits[visits["Visit"] == WEEK13_VISIT].copy()
    wk13 = wk13.sort_values(["PtID", "VisitDate"])
    wk13 = wk13.groupby("PtID", as_index=False).first()
    return wk13[["PtID", "VisitDate", "OutOfWin", "OutOfWinReason"]].rename(
        columns={
            "VisitDate": "Week13VisitDate",
            "OutOfWin": "Week13OutOfWin",
            "OutOfWinReason": "Week13OutOfWinReason",
        }
    )


def last_visit_before(visits: pd.DataFrame, not_after=None) -> pd.DataFrame:
    """Last recorded visit date per participant (dropout fallback endpoint)."""
    dated = visits[visits["VisitDate"].notna()]
    return (
        dated.groupby("PtID", as_index=False)["VisitDate"]
        .max()
        .rename(columns={"VisitDate": "LastVisitDate"})
    )


def raw_file_hashes(raw_dir=RAW_DIR, files=None) -> Dict[str, str]:
    """SHA-256 of each raw file that feeds the processed dataset."""
    from PEDAP.rules import sha256_file

    names = files or (PT_ROSTER_FILE, VISIT_INFO_FILE, *CGM_FILES)
    return {name: sha256_file(Path(raw_dir) / name) for name in names}
