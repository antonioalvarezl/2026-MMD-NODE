"""SAP-style baseline versus cumulative 13-week follow-up endpoint.

``endpoint_mode = "wadwa_baseline_followup"``.

This is the contrast the PEDAP SAP actually describes, and it is NOT the
``fixed_weeks`` week-0-versus-week-12 block contrast:

* the FOLLOW-UP side pools the WHOLE 13-week follow-up period, not its last
  weekly block;
* the BASELINE side is CGM recorded BEFORE randomization, which the RCT
  dataset does not contain at all.

Both sides require >= 168 h of CGM.  A participant missing either side is
excluded from this endpoint and reported as such; nothing is imputed.

Reads two artefacts, and writes neither:
    data/processed/pedap_benchmark/pre_randomization_baseline/   pre-randomization CGM
    data/processed/pedap_benchmark/rct_followup/            RCT follow-up CGM
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats

from PEDAP.baseline_rules import (
    BASELINE_MIN_CGM_HOURS,
    ENDPOINT_DEFINITION,
    ENDPOINT_MODE,
    FOLLOWUP_MIN_CGM_HOURS,
    is_baseline_usable,
    is_followup_usable,
)
from PEDAP.rules import cgm_hours_from_readings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = PROJECT_ROOT / "data" / "processed" / "pedap_benchmark" / "pre_randomization_baseline"
RCT_DIR = PROJECT_ROOT / "data" / "processed" / "pedap_benchmark" / "rct_followup"

ARM_ORDER = ("CLC", "SC")


def _manifest(directory: Path) -> dict:
    with open(Path(directory) / "manifest.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def baseline_followup_table(baseline_dir=BASELINE_DIR, rct_dir=RCT_DIR) -> pd.DataFrame:
    """One row per randomized participant, with both endpoint sides."""
    baseline_dir, rct_dir = Path(baseline_dir), Path(rct_dir)
    b_manifest, r_manifest = _manifest(baseline_dir), _manifest(rct_dir)

    qc = pd.read_csv(baseline_dir / b_manifest["files"]["baseline_qc"])
    baseline = qc[["PtID", "TrtGroup", "cgm_hours", "mean_glucose", "usable"]].rename(
        columns={
            "cgm_hours": "baseline_cgm_hours",
            "mean_glucose": "baseline_mean_glucose",
            "usable": "baseline_available",
        }
    )

    # Follow-up: the WHOLE analysis window pooled, not a weekly block.
    cgm = pd.read_parquet(rct_dir / r_manifest["files"]["cgm"])
    pooled = cgm.groupby("PtID").agg(
        followup_n_readings=("GlucoseMgDl", "size"),
        followup_mean_glucose=("GlucoseMgDl", "mean"),
        followup_first=("DeviceDtTm", "min"),
        followup_last=("DeviceDtTm", "max"),
    ).reset_index()
    pooled["followup_cgm_hours"] = [
        cgm_hours_from_readings(n) for n in pooled["followup_n_readings"]
    ]
    pooled["followup_available"] = [
        is_followup_usable(h) for h in pooled["followup_cgm_hours"]
    ]

    table = baseline.merge(pooled, on="PtID", how="left")
    table["followup_n_readings"] = table["followup_n_readings"].fillna(0).astype(int)
    table["followup_cgm_hours"] = table["followup_cgm_hours"].fillna(0.0)
    table["followup_available"] = table["followup_available"].fillna(False).astype(bool)
    table["baseline_available"] = table["baseline_available"].astype(bool)

    table["both_available"] = table["baseline_available"] & table["followup_available"]
    table["change"] = np.where(
        table["both_available"],
        table["followup_mean_glucose"] - table["baseline_mean_glucose"],
        np.nan,
    )
    return table.sort_values("PtID").reset_index(drop=True)


def _mean_ci(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 2:
        return {"n": n, "mean": float(values.mean()) if n else np.nan,
                "sd": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
    mean = float(values.mean())
    radius = float(stats.t.ppf(0.975, n - 1) * stats.sem(values))
    return {"n": n, "mean": mean, "sd": float(values.std(ddof=1)),
            "ci95_low": mean - radius, "ci95_high": mean + radius}


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """Wadwa-style summary: baseline, 13-week follow-up, and the change."""
    rows = []
    for arm in ARM_ORDER:
        arm_rows = table[table["TrtGroup"] == arm]
        paired = arm_rows[arm_rows["both_available"]]
        for label, values in (
            ("baseline", paired["baseline_mean_glucose"]),
            ("followup_13wk", paired["followup_mean_glucose"]),
            ("change_paired", paired["change"]),
        ):
            rows.append({"arm": arm, "quantity": label,
                         "n_randomized": int(len(arm_rows)), **_mean_ci(values)})
    return pd.DataFrame(rows)


def between_arm_change(table: pd.DataFrame) -> dict:
    """Descriptive between-arm difference in the baseline->follow-up change."""
    paired = table[table["both_available"]]
    clc = paired.loc[paired["TrtGroup"] == "CLC", "change"].to_numpy(dtype=float)
    sc = paired.loc[paired["TrtGroup"] == "SC", "change"].to_numpy(dtype=float)
    test = stats.ttest_ind(clc, sc, equal_var=False)
    ci = test.confidence_interval(confidence_level=0.95)
    return {
        "contrast": "change_CLC_minus_change_SC",
        "test": "two-sided Welch t-test (descriptive)",
        "estimate": float(clc.mean() - sc.mean()),
        "ci95_low": float(ci.low), "ci95_high": float(ci.high),
        "pvalue": float(test.pvalue), "df": float(test.df),
        "n_clc": int(len(clc)), "n_sc": int(len(sc)),
    }


def generate(output_dir, baseline_dir=BASELINE_DIR, rct_dir=RCT_DIR,
             verbose=True) -> dict:
    """Write the endpoint tables and return the summary payload."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = baseline_followup_table(baseline_dir, rct_dir)
    summary = summarize(table)
    contrast = between_arm_change(table)

    table.to_csv(output_dir / "wadwa_endpoint_participants.csv", index=False)
    summary.to_csv(output_dir / "wadwa_endpoint_summary.csv", index=False)

    availability = {
        arm: {
            "randomized": int((table["TrtGroup"] == arm).sum()),
            "with_baseline": int(
                ((table["TrtGroup"] == arm) & table["baseline_available"]).sum()),
            "with_followup": int(
                ((table["TrtGroup"] == arm) & table["followup_available"]).sum()),
            "with_both": int(
                ((table["TrtGroup"] == arm) & table["both_available"]).sum()),
        }
        for arm in ARM_ORDER
    }
    payload = {
        "endpoint_mode": ENDPOINT_MODE,
        "endpoint_definition": ENDPOINT_DEFINITION,
        "baseline_min_cgm_hours": BASELINE_MIN_CGM_HOURS,
        "followup_min_cgm_hours": FOLLOWUP_MIN_CGM_HOURS,
        "baseline_dataset_fingerprint": _manifest(baseline_dir)["dataset_fingerprint"],
        "rct_dataset_fingerprint": _manifest(rct_dir)["dataset_fingerprint"],
        "availability": availability,
        "summary": summary.to_dict(orient="records"),
        "between_arm_change": contrast,
        "caveat": (
            "DESCRIPTIVE. Participants missing either side are excluded, not "
            "imputed, so this is a complete-case contrast on the participants "
            "who have both. It is not the fixed_weeks block contrast and must "
            "not be labelled as such."
        ),
    }
    (output_dir / "wadwa_endpoint_metadata.json").write_text(
        json.dumps(payload, indent=2) + "\n")

    if verbose:
        print(f"[{ENDPOINT_MODE}] availability:")
        for arm, counts in availability.items():
            print(f"   {arm}: baseline {counts['with_baseline']}/{counts['randomized']}, "
                  f"follow-up {counts['with_followup']}/{counts['randomized']}, "
                  f"both {counts['with_both']}/{counts['randomized']}")
        print()
        print(summary.to_string(index=False))
        print()
        print(f"Between-arm change (CLC - SC): {contrast['estimate']:.2f} mg/dL, "
              f"95% CI [{contrast['ci95_low']:.2f}, {contrast['ci95_high']:.2f}], "
              f"p={contrast['pvalue']:.4g}, n={contrast['n_clc']}+{contrast['n_sc']}")
    return payload


if __name__ == "__main__":
    import sys
    generate(sys.argv[1] if len(sys.argv) > 1
             else PROJECT_ROOT / "figures_paper" / "baseline_endpoint")
