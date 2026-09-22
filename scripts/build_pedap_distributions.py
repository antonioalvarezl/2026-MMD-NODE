#!/usr/bin/env python3
"""Materialize the analysis-ready PEDAP distributions as benchmark artefacts.

The two CGM artefacts shipped in ``pedap_benchmark`` hold ONE ROW PER SENSOR
READING.  They are the auditable record, but they are not what the MMD-GMM /
Neural ODE pipeline actually consumes: the model sees, per participant, the
sequence of WEEKLY DISTRIBUTIONS ``X_t`` placed at the real observation times.
That aggregation lived only in memory, inside ``load.load_patient_weeks``, so a
third party receiving the benchmark had to re-implement it to reproduce a fit.

This script writes it out, under
``data/processed/pedap_benchmark/weekly_distributions/``:

    pedap_weekly_univariate.parquet    d=1 weekly blocks (glucose only)
    pedap_weekly_bivariate.parquet     d=2 weekly blocks (glucose, dG/dt)
    pedap_baseline_univariate.parquet  d=1 pre-randomization block
    pedap_baseline_bivariate.parquet   d=2 pre-randomization block
    manifest.json                      provenance, rules and fingerprints

Four tables and nothing else.  Everything a per-block index would have held is
already recoverable: the block structure by grouping on
``(PtID, nominal_week_index)``, and the clinical coverage columns from
``rct_followup/pedap_weekly_qc.csv`` on the same key.  The one quantity that is
NOT recoverable is whether a d=2 point's central difference touched an imputed
HIGH/LOW reading, so the bivariate tables carry it per point as
``UsesImputedNeighbour`` -- which is strictly more useful than the per-block
count it replaces, because the "exclude HIGH/LOW derivatives" sensitivity can
then be run by filtering rows.

Univariate and bivariate are SEPARATE artefacts on purpose.  They are not the
same points with a column dropped: d=1 keeps every de-duplicated reading in the
block, while d=2 keeps only the interior centres whose neighbours are genuinely
adjacent five-minute samples, so a d=2 block is a strict subset placed on
different points.  Shipping one table and asking the reader to filter it
correctly would invite exactly that mistake.

NOTHING here is a new scientific convention.  The weekly blocks come from
``load.load_patient_weeks`` verbatim and the baseline from
``Analysis.baseline_projection.baseline_points`` verbatim; this script only
reshapes their output to long form.  Every emitted block is checked against the
in-memory arrays before it is written, so the artefact cannot drift from the
code path the fits actually use.

Run:

    python scripts/build_pedap_distributions.py
    python scripts/build_pedap_distributions.py --force   # overwrite
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Analysis.baseline_projection import BASELINE_DIR, baseline_points, load_baseline_frames
from PEDAP.baseline_rules import BASELINE_MIN_CGM_HOURS, baseline_scientific_config
from PEDAP.rules import WEEKLY_MIN_CGM_HOURS
from load import (
    DERIVATIVE_TOLERANCE_SECONDS,
    _cgm_by_patient,
    compute_pedap_derivatives,
    load_patient_weeks,
    load_pedap_manifest,
    load_pedap_participants,
    pedap_dataset_fingerprint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processed" / "pedap_benchmark" / "weekly_distributions"

DISTRIBUTIONS_NAME = "pedap_wadwa_distributions"
DISTRIBUTIONS_VERSION = "pedap_distributions_v1"
SCRIPT_VERSION = "1.0.0"

REPRESENTATIONS = ("univariate", "bivariate")

FILES = {
    "weekly_univariate": "pedap_weekly_univariate.parquet",
    "weekly_bivariate": "pedap_weekly_bivariate.parquet",
    "baseline_univariate": "pedap_baseline_univariate.parquet",
    "baseline_bivariate": "pedap_baseline_bivariate.parquet",
}

GLUCOSE_COL = "GlucoseMgDl"
DERIVATIVE_COL = "dGlucoseDtMgDlPerMin"
IMPUTED_COL = "UsesImputedNeighbour"


def _imputed_flags(patient_id, data_dir, blocks_meta):
    """Per-point ``uses_imputed`` for one participant's d=2 weekly blocks.

    ``load_patient_weeks`` returns the block VALUES but only a per-block count
    of the points whose central difference touches an imputed HIGH/LOW reading.
    The flag itself is wanted per point, so it is recomputed here with the same
    primitive and the same block windows, then checked against that count --
    which is an exact test, so this cannot drift from the loader.
    """
    frame = _cgm_by_patient(data_dir).get(patient_id)
    if frame is None or frame.empty:
        return []
    frame = frame.sort_values("DeviceDtTm")
    derived = compute_pedap_derivatives(
        frame["DeviceDtTm"].to_numpy(dtype="datetime64[ns]"),
        frame["GlucoseMgDl"].to_numpy(dtype=float),
        frame["EventSubType"].to_numpy(),
        tolerance_seconds=DERIVATIVE_TOLERANCE_SECONDS,
    )
    times = derived["timestamps"]
    flags = derived["uses_imputed"]

    out = []
    for record in blocks_meta:
        mask = (
            (times >= np.datetime64(record["actual_start"]))
            & (times < np.datetime64(record["actual_end"]))
        )
        block_flags = flags[mask]
        if len(block_flags) != int(record["n_observations"]):
            raise RuntimeError(
                f"PtID {patient_id} week {record['week_index']}: "
                f"{len(block_flags)} flags for {record['n_observations']} points"
            )
        if int(block_flags.sum()) != int(record["n_points_touching_imputed"]):
            raise RuntimeError(
                f"PtID {patient_id} week {record['week_index']}: "
                f"{int(block_flags.sum())} imputed-touching points but the "
                f"loader reports {record['n_points_touching_imputed']}"
            )
        out.append(block_flags)
    return out


def _weekly_long(representation: str, data_dir=None):
    """Long-form weekly blocks for every randomized participant.

    One row per observation, keyed by participant and nominal week.  The block
    structure is recovered by grouping on ``(PtID, nominal_week_index)``, and
    the clinical coverage of each block lives in ``pedap_weekly_qc.csv`` under
    the same key, so no separate index table is written.
    """
    participants = load_pedap_participants(data_dir)
    arms = dict(zip(participants["PtID"].astype(int), participants["TrtGroup"]))
    bivariate = representation == "bivariate"

    pt_chunks = []
    week_chunks = []
    time_chunks = []
    value_chunks = []
    flag_chunks = []
    n_blocks = 0

    for patient_id in sorted(arms):
        data, metadata = load_patient_weeks(
            patient_id,
            data_dir=data_dir,
            representation=representation,
            min_node_observations=0,
            return_metadata=True,
        )
        blocks = data.get(patient_id, [])
        # load_patient_weeks appends one metadata record per emitted block, in
        # the same order; anything else means the loader changed underneath us.
        if len(blocks) != len(metadata["blocks"]):
            raise RuntimeError(
                f"PtID {patient_id}: {len(blocks)} blocks but "
                f"{len(metadata['blocks'])} block records"
            )
        flags = (
            _imputed_flags(patient_id, data_dir, metadata["blocks"])
            if bivariate else [None] * len(blocks)
        )

        for values, record, block_flags in zip(blocks, metadata["blocks"], flags):
            values = np.asarray(values, dtype=float)
            n = values.shape[0]
            if n != int(record["n_observations"]):
                raise RuntimeError(
                    f"PtID {patient_id} week {record['week_index']}: "
                    f"{n} rows but n_observations={record['n_observations']}"
                )
            pt_chunks.append(np.full(n, patient_id, dtype=np.int32))
            week_chunks.append(
                np.full(n, int(record["week_index"]), dtype=np.int16)
            )
            time_chunks.append(
                np.full(n, float(record["observation_time_weeks"]), dtype=np.float64)
            )
            value_chunks.append(values)
            if bivariate:
                flag_chunks.append(np.asarray(block_flags, dtype=bool))
            n_blocks += 1

    if not value_chunks:
        raise RuntimeError(f"no usable {representation} weekly blocks were produced")

    stacked = np.concatenate(value_chunks, axis=0)
    frame = pd.DataFrame(
        {
            "PtID": np.concatenate(pt_chunks),
            "nominal_week_index": np.concatenate(week_chunks),
            "observation_time_weeks": np.concatenate(time_chunks),
            GLUCOSE_COL: stacked[:, 0].astype(np.float32),
        }
    )
    if bivariate:
        # float64, unlike the glucose column.  Glucose survives float32 exactly
        # because it comes back from a float32 source, but dG/dt is a computed
        # central difference: narrowing it would make the artefact differ from
        # the arrays the fits use, which _verify_weekly duly refuses.
        frame[DERIVATIVE_COL] = stacked[:, 1].astype(np.float64)
        frame[IMPUTED_COL] = np.concatenate(flag_chunks)
    frame["TrtGroup"] = frame["PtID"].map(arms).astype("string")
    return frame, n_blocks


def _baseline_long(representation: str, baseline_dir=BASELINE_DIR, data_dir=None):
    """Long-form pre-randomization block, one per participant with a baseline.

    No week columns: the baseline is a single block anchored BEFORE
    randomization, it has no position on the follow-up timeline, and giving it
    one would invite treating it as week 0.
    """
    cgm, qc, _ = load_baseline_frames(baseline_dir)
    participants = load_pedap_participants(data_dir)
    arms = dict(zip(participants["PtID"].astype(int), participants["TrtGroup"]))
    bivariate = representation == "bivariate"

    pt_chunks = []
    value_chunks = []
    flag_chunks = []

    for patient_id in sorted(arms):
        points = baseline_points(patient_id, cgm, qc, representation=representation)
        if points is None:
            continue
        points = np.asarray(points, dtype=float)
        n = points.shape[0]
        pt_chunks.append(np.full(n, patient_id, dtype=np.int32))
        value_chunks.append(points)

        if bivariate:
            frame = cgm[cgm["PtID"] == patient_id].sort_values("DeviceDtTm")
            derived = compute_pedap_derivatives(
                frame["DeviceDtTm"].to_numpy(dtype="datetime64[ns]"),
                frame["GlucoseMgDl"].to_numpy(dtype=float),
                frame["EventSubType"].to_numpy(),
                tolerance_seconds=DERIVATIVE_TOLERANCE_SECONDS,
            )
            # baseline_points ran the same computation over the same series, so
            # the flags must align point for point with what it returned.
            if not np.array_equal(
                np.column_stack([derived["glucose"], derived["derivative"]]), points
            ):
                raise RuntimeError(
                    f"PtID {patient_id}: baseline derivative flags do not align "
                    "with baseline_points"
                )
            flag_chunks.append(np.asarray(derived["uses_imputed"], dtype=bool))

    if not value_chunks:
        raise RuntimeError(f"no usable {representation} baseline blocks were produced")

    stacked = np.concatenate(value_chunks, axis=0)
    frame = pd.DataFrame(
        {
            "PtID": np.concatenate(pt_chunks),
            GLUCOSE_COL: stacked[:, 0].astype(np.float32),
        }
    )
    if bivariate:
        frame[DERIVATIVE_COL] = stacked[:, 1].astype(np.float64)
        frame[IMPUTED_COL] = np.concatenate(flag_chunks)
    frame["TrtGroup"] = frame["PtID"].map(arms).astype("string")
    return frame


def _verify_weekly(frame: pd.DataFrame, representation: str, data_dir=None) -> int:
    """Re-read the written long form and compare it to the in-memory arrays.

    The artefact only earns its place if it is the SAME data the fits use, so
    every block is checked value by value rather than by row count alone.
    """
    cols = [GLUCOSE_COL] + ([DERIVATIVE_COL] if representation == "bivariate" else [])
    grouped = {
        key: group[cols].to_numpy(dtype=float)
        for key, group in frame.groupby(["PtID", "nominal_week_index"], sort=False)
    }
    participants = load_pedap_participants(data_dir)
    checked = 0
    for patient_id in sorted(participants["PtID"].astype(int)):
        data, metadata = load_patient_weeks(
            patient_id,
            data_dir=data_dir,
            representation=representation,
            min_node_observations=0,
            return_metadata=True,
        )
        for values, record in zip(data.get(patient_id, []), metadata["blocks"]):
            key = (patient_id, int(record["week_index"]))
            if key not in grouped:
                raise RuntimeError(f"block {key} missing from the {representation} artefact")
            if not np.array_equal(grouped[key], np.asarray(values, dtype=float)):
                raise RuntimeError(f"block {key} differs from load_patient_weeks")
            checked += 1
    if checked != len(grouped):
        raise RuntimeError(
            f"{representation}: artefact holds {len(grouped)} blocks, loader gives {checked}"
        )
    return checked


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(out_dir: Path, force: bool = False, data_dir=None,
          baseline_dir=BASELINE_DIR) -> Path:
    out_dir = Path(out_dir)
    existing = [out_dir / name for name in FILES.values()]
    if not force and any(path.exists() for path in existing):
        raise SystemExit(
            f"{out_dir} already holds distribution artefacts; pass --force to rebuild"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    written = {}

    for representation in REPRESENTATIONS:
        print(f"[weekly:{representation}] aggregating ...", flush=True)
        frame, emitted = _weekly_long(representation, data_dir=data_dir)
        n_blocks = _verify_weekly(frame, representation, data_dir=data_dir)
        if n_blocks != emitted:
            raise RuntimeError(
                f"{representation}: emitted {emitted} blocks, verified {n_blocks}"
            )
        path = out_dir / FILES[f"weekly_{representation}"]
        frame.to_parquet(path, index=False)
        written[f"weekly_{representation}"] = path
        print(
            f"[weekly:{representation}] {n_blocks} blocks, {len(frame):,} rows "
            f"-> {path.name}",
            flush=True,
        )

        print(f"[baseline:{representation}] aggregating ...", flush=True)
        frame = _baseline_long(
            representation, baseline_dir=baseline_dir, data_dir=data_dir
        )
        path = out_dir / FILES[f"baseline_{representation}"]
        frame.to_parquet(path, index=False)
        written[f"baseline_{representation}"] = path
        print(
            f"[baseline:{representation}] {frame['PtID'].nunique()} participants, "
            f"{len(frame):,} rows -> {path.name}",
            flush=True,
        )

    manifest = {
        "dataset_name": DISTRIBUTIONS_NAME,
        "dataset_version": DISTRIBUTIONS_VERSION,
        "script_version": SCRIPT_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "built_by": "scripts/build_pedap_distributions.py",
        "derived_from": {
            # These two fingerprints are what makes the artefact safe to reuse:
            # a rebuilt CGM dataset under different conventions gets a different
            # fingerprint, and these distributions are then visibly stale.
            "rct_followup_fingerprint": pedap_dataset_fingerprint(data_dir),
            "rct_followup_manifest": load_pedap_manifest(data_dir).get("dataset_version"),
            "baseline_dir": str(Path(baseline_dir).name),
        },
        "aggregation": {
            "source_of_truth": "load.load_patient_weeks / Analysis.baseline_projection.baseline_points",
            "weekly_min_cgm_hours": float(WEEKLY_MIN_CGM_HOURS),
            "baseline_min_cgm_hours": float(BASELINE_MIN_CGM_HOURS),
            "derivative_tolerance_seconds": float(DERIVATIVE_TOLERANCE_SECONDS),
            "missing_weeks": "left missing; never imputed or interpolated",
            "observation_times": "real clinical timeline, in weeks since randomization",
            "univariate_vs_bivariate": (
                "separate artefacts: d=1 keeps every reading in the block, d=2 keeps "
                "only interior centres with adjacent five-minute neighbours, so d=2 "
                "is a strict subset on different points, not a dropped column"
            ),
            "baseline_block": (
                "single pre-randomization block per participant; it has no position "
                "on the follow-up timeline and is not week 0"
            ),
        },
        "baseline_scientific_config": baseline_scientific_config(),
        "files": {key: path.name for key, path in written.items()},
        "file_sha256": {key: _file_digest(path) for key, path in written.items()},
        "columns": {
            "weekly": ["PtID", "TrtGroup", "nominal_week_index",
                       "observation_time_weeks", GLUCOSE_COL],
            "bivariate_extra": [DERIVATIVE_COL, IMPUTED_COL],
            "baseline": ["PtID", "TrtGroup", GLUCOSE_COL],
            "note": (
                f"{DERIVATIVE_COL} (mg/dL/min) and {IMPUTED_COL} appear only in "
                "the bivariate artefacts; the baseline artefacts carry no week "
                "columns. Block structure is recovered by grouping on "
                "(PtID, nominal_week_index); each block's clinical coverage is "
                "in rct_followup/pedap_weekly_qc.csv under the same key."
            ),
        },
        "uses_imputed_neighbour": (
            f"{IMPUTED_COL} marks a d=2 point whose central difference used a "
            "HIGH/LOW reading imputed to 401/39 mg/dL, at the centre or at "
            "either neighbour. Filtering those rows out runs the "
            "'exclude HIGH/LOW derivatives' sensitivity without rebuilding."
        ),
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"\nmanifest -> {manifest_path}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--data-dir", default=None,
                        help="rct_followup directory (default: the configured one)")
    parser.add_argument("--baseline-dir", default=str(BASELINE_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build(Path(args.out_dir), force=args.force, data_dir=args.data_dir,
          baseline_dir=Path(args.baseline_dir))


if __name__ == "__main__":
    main()
