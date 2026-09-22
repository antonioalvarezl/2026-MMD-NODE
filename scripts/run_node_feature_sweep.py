"""Run the NODE time-feature sweep (Fourier m=0..2 at p=1; polynomial p=1..4 at m=0)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Analysis.paper_config import DISPLAY_PATIENT_IDS, FIGURES_ROOT, RESULTS_ROOT
from Analysis.patient_sensitivity import node_feature_sweep
from Visualization.plots import _get_patient_dir, _load_patient_results


def load_bundle(patient_id, run_dir):
    patient_dir = Path(_get_patient_dir(run_dir, patient_id))
    result = _load_patient_results(patient_dir, include_covs=True)
    fit = json.loads((patient_dir / 'result_fit.json').read_text())
    return {
        'patient_dir': patient_dir,
        'fit': fit,
        'fit_weights': np.asarray(fit['results']['weights'], dtype=float),
        'observation_times': np.asarray(fit['config']['observation_times_weeks'], dtype=float),
        'trajectory_times': np.asarray(result['trajectory_time_grid_weeks'], dtype=float),
        'node_config': result.get('node_config', {}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dimension', choices=('d1', 'd2'), default='d2')
    parser.add_argument('--run', default=None)
    parser.add_argument('--patients', type=int, nargs='+', default=list(DISPLAY_PATIENT_IDS))
    args = parser.parse_args()
    run = args.run or ('run14' if args.dimension == 'd1' else 'run15')
    run_dir = RESULTS_ROOT / run
    for patient_id in args.patients:
        out = FIGURES_ROOT / args.dimension / run / 'sensitivity' / f'patient_{patient_id}' / 'node_features'
        print(node_feature_sweep(load_bundle(patient_id, run_dir), out))


if __name__ == '__main__':
    main()
