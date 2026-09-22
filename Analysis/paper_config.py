"""Shared configuration for the PEDAP CGM paper post-processing pipeline.

The notebook selects ONE dimension and ONE run folder at a time.  This module
contains only scientific/display defaults shared by the analysis modules.

Cohort and arm membership are DERIVED from the processed PEDAP dataset
(``pedap_participants.csv``, i.e. ultimately ``PtRoster``).  There are no
hard-coded participant lists here: the trial's own randomization is the only
source of allocation, and it is intention-to-treat, so a dropout keeps the arm
it was randomized to.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

# Analysis/paper_config.py lives under <PROJECT_ROOT>/Analysis/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "Results_paper" / "CGM"
FIGURES_ROOT = PROJECT_ROOT / "figures_paper"

# -----------------------------------------------------------------------------
# Canonical clinical files: the processed PEDAP dataset.
# -----------------------------------------------------------------------------
PEDAP_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "pedap_benchmark" / "rct_followup"
PARTICIPANTS_FILE = PEDAP_DATA_DIR / "pedap_participants.csv"
CGM_FILE = PEDAP_DATA_DIR / "pedap_cgm_rct.parquet"
WEEKLY_QC_FILE = PEDAP_DATA_DIR / "pedap_weekly_qc.csv"
MANIFEST_FILE = PEDAP_DATA_DIR / "manifest.json"

RAW_PEDAP_DIR = PROJECT_ROOT / "data" / "raw" / "pedap"


@lru_cache(maxsize=None)
def _manifest_at(data_dir: Path) -> Dict:
    manifest_file = Path(data_dir) / "manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(
            "PEDAP processed dataset not found.\n"
            "Run:\n"
            "python scripts/build_pedap_dataset.py"
        )
    with open(manifest_file, "r", encoding="utf-8") as handle:
        return json.load(handle)


def pedap_manifest(data_dir=None) -> Dict:
    """The manifest of the primary longitudinal PEDAP RCT dataset."""
    return _manifest_at(Path(data_dir) if data_dir is not None else PEDAP_DATA_DIR)


def dataset_fingerprint(data_dir=None) -> str:
    """Fingerprint that every clinical result must be stamped with."""
    return str(pedap_manifest(data_dir)["dataset_fingerprint"])


def treatment_comparison_supported() -> bool:
    """Whether the dataset's arm coverage supports a CLC-vs-SC comparison.

    v2 has no such manifest flag: PEDAPDexcomClarityCGM.txt is the sole CGM
    source for both arms and the builder hard-asserts symmetric coverage
    (every randomized participant in both arms has RCT CGM, and the SC/CLC
    usable-week rate ratio is within [0.5, 2.0]) before it will write the
    dataset at all.  So for any dataset that exists, this is true by
    construction; kept as a function for backward-compatible call sites.
    """
    pedap_manifest()  # still requires the dataset to exist
    return True


@lru_cache(maxsize=None)
def arm_ids(data_dir=None) -> Dict[str, Tuple[int, ...]]:
    """Participant ids per trial arm, straight from the randomization."""
    import pandas as pd

    base = Path(data_dir) if data_dir is not None else PEDAP_DATA_DIR
    participants_file = base / "pedap_participants.csv"
    if not participants_file.exists():
        raise FileNotFoundError(
            "PEDAP processed dataset not found.\n"
            "Run:\n"
            "python scripts/build_pedap_dataset.py"
        )
    participants = pd.read_csv(participants_file, usecols=["PtID", "TrtGroup"])
    return {
        arm: tuple(
            int(pt) for pt in
            participants.loc[participants["TrtGroup"] == arm, "PtID"].sort_values()
        )
        for arm in ARM_CODES
    }


# -----------------------------------------------------------------------------
# Group coding.  The trial's own codes are canonical; the display names are a
# presentation layer only, and the code always travels with the result.
# -----------------------------------------------------------------------------
CONTROL_ARM = "SC"      # standard care
TREATMENT_ARM = "CLC"   # closed-loop control
ARM_CODES = (TREATMENT_ARM, CONTROL_ARM)

CONTROL_LABEL = 0
TREATMENT_LABEL = 1
ARM_NAMES = {CONTROL_LABEL: "Control", TREATMENT_LABEL: "Treatment"}
ARM_CODE_BY_LABEL = {CONTROL_LABEL: CONTROL_ARM, TREATMENT_LABEL: TREATMENT_ARM}
ARM_LABEL_BY_CODE = {CONTROL_ARM: CONTROL_LABEL, TREATMENT_ARM: TREATMENT_LABEL}
ARM_DISPLAY_BY_CODE = {CONTROL_ARM: "Control (SC)", TREATMENT_ARM: "Treatment (CLC)"}


def __getattr__(name):
    """Derive CONTROL_IDS / TREATMENT_IDS on first use.

    They are module attributes for backwards compatibility with the analysis
    modules, but they are computed from the dataset rather than written down.
    """
    if name == "CONTROL_IDS":
        return arm_ids()[CONTROL_ARM]
    if name == "TREATMENT_IDS":
        return arm_ids()[TREATMENT_ARM]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# -----------------------------------------------------------------------------
# Scientific analysis window.
#
# The PEDAP RCT has 13 nominal weekly periods anchored at each participant's
# RANDOMIZATION date, intersected with the SAP follow-up window (fourth day
# after randomization through the actual 13-week visit).  Participants
# contribute the usable weekly distributions they actually have: this is NOT a
# requirement that every participant have 13 of them, and the observation times
# are the real block midpoints, not 0.5, 1.5, ..., 12.5.
# -----------------------------------------------------------------------------
N_NOMINAL_WEEKS = 13
PAPER_INTERVAL = (0.0, float(N_NOMINAL_WEEKS))
ANALYSIS_WINDOW_DEFINITION = (
    "PEDAP SAP follow-up: from the fourth calendar day after the randomization "
    "visit through the actual 13-week visit (or the last visit for a dropout); "
    "13 nominal weekly periods anchored at the randomization date"
)

PAPER_N_GRID = 101
PAPER_BAND_PCT = 90.0

# Fixed display patients used for Rosenblatt and individual panels.
# Keep the current study choices here; change deliberately if the manuscript set changes.
DISPLAY_PATIENT_IDS = (16, 26, 39)

# Treatment green is lighter than the original #588157: at the low alpha used
# for the faint per-subject background curves (paper_figures._plot_trajectory_
# figures), the old green faded to a grayish tone too close to faded Control
# red to tell apart; #55a868 keeps ~30% more perceptual separation at that
# alpha while still reading as the same "green" family at full opacity.
ARM_COLORS = {CONTROL_LABEL: "#ad2831", TREATMENT_LABEL: "#55a868"}

# Fixed component palette used throughout the manuscript.
COMPONENT_COLORS = (
    "#17becf",  # component 1: cyan
    "#ff7f0e",  # component 2: orange
    "#2ca02c",  # component 3: green
    "#d62728",  # component 4: red
    "#9467bd",  # component 5: purple
)

# Reproducibility defaults shared by orchestration code.
INFERENCE_SEED = 2025
WASSERSTEIN_SEED = 42
