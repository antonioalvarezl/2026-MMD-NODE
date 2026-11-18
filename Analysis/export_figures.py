"""Inventory and package the two active PEDAP RCT figure collections."""
from pathlib import Path
import re
import zipfile
from Analysis.paper_config import FIGURES_ROOT, RESULTS_ROOT, PROJECT_ROOT
from Analysis.clinical_support import validate_rct_run

ACTIVE_RUNS = {'d1': 'run16', 'd2': 'run17'}
# Hand-curated sensitivity collection mixing historical runs; provenance is in its
# figure_sources.csv. Documented in INVENTORY.txt but never packaged in the export.
EXTRA_FOLDERS = ('sensitivity',)
# Per-participant diagnostics: one file per participant is useful to keep on
# disk but would swamp both the inventory and the submitted figure set.
EXCLUDED_SUBPATHS = ('rosenblatt/participants',)

# (pattern on the file name, description). First match wins; {name} fields are
# filled from the pattern's named groups.
DESCRIPTIONS = [
    (r'global_(?:mean|variance|covariance)_(?:glucose|rate|glucose_rate)_by_arm',
     'Single mixture-moment coordinate: individual trajectories and available-case arm averages.'),
    # Group trajectories and Fréchet means
    (r'Mixed_comp(?P<k>\d)_bands',
     'Mixture weight of component {k} over weeks since randomization: arm means '
     '(Treatment green, Control red) with 90% across-participant bands, available-case.'),
    (r'density_initial_final_difference',
     'Summary panel: W2 Fréchet mean (Wasserstein barycenter) density per arm at the '
     'initial and final state, and the signed final - initial difference per arm.'),
    (r'density_(?P<when>initial|final)_comparison_frechet',
     'd=1: W2 Fréchet mean glucose density at each participant\'s own {when} state, '
     'Treatment vs Control overlaid.'),
    (r'density_(?P<when>initial|final)_(?P<arm>treatment|control)_frechet',
     'd=2: W2 Fréchet mean (free-support barycenter, KDE for display) of the joint '
     'glucose / rate-of-change density, {arm} arm, {when} state.'),
    (r'density_frechet_difference_measure_(?P<arm>treatment|control)',
     'Signed difference of Fréchet mean densities, final - initial, {arm} arm.'),
    (r'inference_window_sensitivity',
     'Inference window chosen at 85/90/95% minimum per-arm coverage, with retained '
     'CLC/SC counts; chosen without looking at outcomes.'),
    # Global moments
    (r'global_(?P<m>mean|covariance)_by_arm',
     'Global mixture {m} over time: one thin curve per participant, thick arm average.'),
    (r'global_(?P<m>mean|covariance)_patient_(?P<p>\d+)',
     'Global mixture {m} over time for patient {p}.'),
    (r'global_(?P<m>mean|covariance)_treatment_minus_control',
     'Treatment - Control difference of arm-average global {m} over time.'),
    (r'component_cross_covariance_within_by_arm',
     'd=2: per-component contribution w_k Sigma_k to the glucose / rate cross-covariance, '
     'arm averages.'),
    (r'component_cross_covariance_within_patient_(?P<p>\d+)',
     'd=2: per-component within-component cross-covariance contribution, patient {p}.'),
    (r'component_cross_covariance_total_by_arm',
     'd=2: per-component total contribution (within + between-mean term) to the '
     'cross-covariance, arm averages; components sum to the global C_12.'),
    (r'component_cross_covariance_total_patient_(?P<p>\d+)',
     'd=2: per-component total cross-covariance contribution, patient {p}.'),
    # Wild bootstrap inference
    (r'wild_bootstrap_(?P<e>absolute|change_from_baseline)_null',
     'Global functional MMD test ({e} estimand): wild-bootstrap null distribution and '
     'permutation validation, observed statistic marked.'),
    (r'wild_bootstrap_mmd_(?P<e>absolute|change_from_baseline)_weekly_pointwise_pvalues',
     'Weekly pointwise wild-bootstrap MMD p-values per component ({e} estimand).'),
    (r'wild_bootstrap_mmd_(?P<e>absolute|change_from_baseline)_weekly_cumulative_acat_pvalues',
     'Cumulative ACAT combination of the weekly p-values per component ({e} estimand).'),
    (r'cumulative_acat_change_from_baseline_(?P<g>weekly|fine)(?:_grid)?_tau_sweep',
     'Cumulative ACAT p-values (change from baseline) on the {g} grid for '
     'tau = 1, 3, 5, 7, 10, 14 days, one row per tau.'),
    (r'cumulative_acat_change_from_baseline_(?P<g>weekly|fine)(?:_grid)?_tau(?P<t>\d+)',
     'Cumulative ACAT p-values (change from baseline) on the {g} grid, tau = {t} days.'),
    # Individual participants
    (r'individual/patient_(?P<p>\d+)/composite',
     'Patient {p}: raw daily CGM, MMD weights + NODE trajectory, and initial/final '
     'fitted density over the empirical cloud.'),
    (r'individual/patient_(?P<p>\d+)/trajectories',
     'Patient {p}: weekly MMD weights (points) and fitted NODE trajectory (lines).'),
    (r'individual/patient_(?P<p>\d+)/(?P<when>initial|final)',
     'Patient {p}: fitted GMM density (Treatment green, Control red) over the empirical CGM cloud, {when} week.'),
    (r'timesignal_(?P<arm>\w+?)_patient_?(?P<p>\d+)',
     'Patient {p} ({arm}): daily CGM curves overlaid by time of day.'),
    (r'rosenblatt/patient_(?P<p>\d+)_qq_(?P<u>u\d)_3d',
     'Patient {p}: Rosenblatt-transform QQ diagnostic for coordinate {u} across weeks, '
     'discrete MMD weights vs NODE at the same time.'),
    # Per-run sensitivity
    (r'bandwidth/bandwidth_endpoint_weights|bandwidth_endpoint_weights_(?P<ctx>historical|randomized_phase)',
     'Bandwidth sensitivity: endpoint local MMD weights refitted for gamma = c * d_ref '
     '(fixed dictionary and ridge), initial and final.'),
    (r'bandwidth_endpoints_coordinate_(?P<j>\d)',
     'Bandwidth sensitivity: endpoint marginal density of coordinate {j} for each c, '
     'and difference from c = 1.'),
    (r'bandwidth_endpoints_joint',
     'Bandwidth sensitivity: endpoint joint density for each c over the empirical cloud.'),
    (r'node_fourier_features',
     'NODE time-feature sensitivity: Fourier features of order m = 0, 1, 2 with polynomial '
     'degree 1; weekly MMD weights (points) and NODE fit (lines). Training fits only.'),
    (r'node_polynomial_features',
     'NODE time-feature sensitivity: cumulative monomials t, ..., t^p for p = 1..4 with no '
     'Fourier terms; weekly MMD weights (points) and NODE fit (lines). Training fits only.'),
    (r'node_fourier_feature_maps_patient_(?P<p>\d+)',
     'Historical (d=1, run4) NODE Fourier feature-map comparison, patient {p}; see TXT.'),
    (r'bandwidth_sensitivity_random_patient_(?P<p>\d+)',
     'Historical (run4) MMD bandwidth sensitivity, patient {p}; see TXT.'),
    (r'ridge_sensitivity_random_patient_(?P<p>\d+)',
     'Historical (run4) ridge penalty sensitivity, patient {p}; see TXT.'),
]


def describe(relative_path):
    text = str(relative_path)
    for pattern, template in DESCRIPTIONS:
        match = re.search(pattern, text)
        if match:
            fields = {k: (v or '').replace('_', ' ') for k, v in match.groupdict().items()}
            description = template.format(**fields)
            if 'historical' in text and 'Historical' not in description:
                description = 'Historical run13 fit. ' + description
            elif 'randomized_phase' in text:
                description = 'run15 fit. ' + description
            return description
    raise KeyError(f'No inventory description for {text}; add one to DESCRIPTIONS')


def write_inventory(figures_root=FIGURES_ROOT, results_root=RESULTS_ROOT):
    root = Path(figures_root)
    folders = []
    for dimension, run in ACTIVE_RUNS.items():
        params = validate_rct_run(Path(results_root) / run)
        if params['d'] != int(dimension[1:]):
            raise ValueError('Run dimension does not match figure collection')
        folders.append(root / dimension / run)
    paths = []
    lines = ['PEDAP paper figures: one line per PDF, "path: description".',
             'Arm colours: Treatment (CLC) green, Control (SC) red.', '']
    for folder in folders:
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        found = [p for p in sorted(folder.rglob('*.pdf'))
                 if not any(x in p.as_posix() for x in EXCLUDED_SUBPATHS)]
        paths.extend(found)
        lines.append(f'## {folder.relative_to(root)}')
        lines += [f'{p.relative_to(root)}: {describe(p.relative_to(root))}' for p in found]
        lines.append('')
    for name in EXTRA_FOLDERS:
        lines.append(f'## {name} (local only; not included in the export zip)')
        lines += [f'{p.relative_to(root)}: {describe(p.relative_to(root))}'
                  for p in sorted((root / name).rglob('*.pdf'))]
        lines.append('')
    root.mkdir(parents=True, exist_ok=True)
    (root / 'INVENTORY.txt').write_text('\n'.join(lines))
    return paths


def export_figures(destination=None):
    paths = write_inventory()
    target = Path(destination) if destination else PROJECT_ROOT / 'exports/PEDAP_paper_figures.zip'
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(FIGURES_ROOT))
        archive.write(FIGURES_ROOT / 'INVENTORY.txt', 'INVENTORY.txt')
    return target


if __name__ == '__main__':
    print(export_figures())
