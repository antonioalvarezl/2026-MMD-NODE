import ast
import inspect
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import pytest
from Analysis.clinical_support import select_inference_window, inference_window_sensitivity, verify_interval_data
from Analysis.export_figures import write_inventory, export_figures, ACTIVE_RUNS
from Analysis.paper_config import PROJECT_ROOT, TREATMENT_IDS, CONTROL_IDS
from Analysis.paper_figures import _load_interval_data


def test_removed_entrypoints():
    import main
    import load
    from Analysis.wild_bootstrap_mmd import ESTIMANDS, run_global_functional_mmd_analyses
    with pytest.raises(SystemExit) as error:
        main._parse_args(['--observation-mode', 'shared_support'])
    assert error.value.code == 2
    for function in (load.load_patient_weeks, load.prepare_cgm_experiments, _load_interval_data, run_global_functional_mmd_analyses):
        assert 'observation_mode' not in inspect.signature(function).parameters
        with pytest.raises(TypeError):
            function(observation_mode='shared_support')
    assert set(ESTIMANDS) == {'absolute', 'change_from_baseline'}


def test_active_sources_and_notebook():
    sources = [*PROJECT_ROOT.glob('*.py'), *PROJECT_ROOT.glob('Analysis/*.py'), *PROJECT_ROOT.glob('scripts/*.py')]
    for p in sources:
        text = p.read_text()
        ast.parse(text)
        assert 'observation_mode' not in text, p
        assert 'shared_support' not in text, p
    notebook = json.loads((PROJECT_ROOT / 'Analysis/Medical_DataAnalysis.ipynb').read_text())
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        assert 'observation_mode' not in source
        assert 'shared_support' not in source
        assert 'POLICY' not in source
        if cell['cell_type'] == 'code':
            ast.parse(source)


def test_export_only_primary_runs(tmp_path):
    paths = write_inventory()
    assert paths
    for p in paths:
        rel = p.relative_to(PROJECT_ROOT / 'figures_paper')
        assert rel.parts[1] == ACTIVE_RUNS[rel.parts[0]]
        assert 'shared_support' not in str(rel)
        assert 'change_from_first' not in str(rel)
    target = export_figures(tmp_path / 'figures.zip')
    with zipfile.ZipFile(target) as archive:
        assert all('shared_support' not in name for name in archive.namelist())
        assert 'shared_support' not in archive.read('INVENTORY.txt').decode()


@pytest.mark.parametrize('run', ['run14', 'run15'])
def test_saved_supports_and_coverage(run):
    data = _load_interval_data(TREATMENT_IDS, CONTROL_IDS, PROJECT_ROOT / 'Results_paper/CGM' / run)
    assert verify_interval_data(data, verbose=False)['n_subjects'] == 99
    supports, labels = data['fitted_node_supports_original_week_coordinates'], data['labels']
    lo, hi, keep, report = select_inference_window(supports, labels, verbose=False)
    assert (lo, hi) == (1.5, 11.5)
    assert (report['n_retained_clc'], report['n_retained_sc']) == (62, 31)
    perm = np.random.default_rng(42).permutation(len(labels))
    reordered = select_inference_window(np.asarray(supports)[perm], np.asarray(labels)[perm], verbose=False)
    assert reordered[:2] == (lo, hi)
    np.testing.assert_array_equal(reordered[2], keep[perm])
    sens = inference_window_sensitivity(supports, labels, verbose=False)
    assert sens['0.85']['window_weeks'] == [1.5, 11.5]
    assert sens['0.95']['window_weeks'] == [2.5, 11.5]
    assert sens['0.95']['n_retained'] == 96


def test_per_arm_floor_prevents_pooled_selection():
    supports = [[0, 10]] * 90 + [[0, 10]] * 5 + [[1, 9]] * 5
    labels = [1] * 90 + [0] * 10
    lo, hi, _, report = select_inference_window(supports, labels, verbose=False)
    assert (lo, hi) == (1, 9)
    assert report['n_retained_sc'] == 10


def test_endpoint_summary_uses_same_complete_cases():
    from Analysis.wadwa_endpoints import summarize
    frame = pd.DataFrame({'TrtGroup':['CLC','CLC','SC'], 'both_available':[True,False,True],
        'baseline_available':[True,False,True], 'followup_available':[True,True,True],
        'baseline_mean_glucose':[170,np.nan,180], 'followup_mean_glucose':[150,300,175],
        'change':[-20,np.nan,-5]})
    result = summarize(frame)
    assert (result['n'] == 1).all()
    assert result.query("arm == 'CLC' and quantity == 'followup_13wk'")['mean'].iloc[0] == 150


@pytest.mark.parametrize('mode,run', [('cgm_univariate','run14'),('cgm_bivariate','run15')])
def test_current_configuration_reuses_fits(mode, run, monkeypatch):
    import input as config
    from argparse import Namespace
    import Utils.IO_utils as io
    def reuse_only(parent, params):
        saved = json.loads((PROJECT_ROOT / 'Results_paper/CGM' / run / 'fixed_params.json').read_text())
        assert json.loads(json.dumps(params)) == saved
        return str(PROJECT_ROOT / 'Results_paper/CGM' / run)
    monkeypatch.setattr(io, '_find_or_create_run_folder', reuse_only)
    options, *_ = config.get_params(Namespace(analysis_mode=mode, id_start=None, id_end=None))
    assert Path(options['base_folder']).name == run


@pytest.mark.parametrize('dimension,run', [('d1','run14'),('d2','run15')])
@pytest.mark.parametrize('estimand', ['absolute','change_from_baseline'])
def test_functional_statistics_match_saved_results(dimension, run, estimand):
    from Analysis.wild_bootstrap_mmd import functional_gaussian_kernel, scaled_mmd_v_from_kernels
    saved = json.loads((PROJECT_ROOT / 'figures_paper' / dimension / run / 'wild_bootstrap' / f'wild_bootstrap_mmd_{estimand}.json').read_text())
    ids = saved['population']['included_ids']
    data = _load_interval_data([p for p in TREATMENT_IDS if p in ids], [p for p in CONTROL_IDS if p in ids],
        PROJECT_ROOT / 'Results_paper/CGM' / run, interval=(1.5,11.5), n_grid=len(saved['common_time_grid_weeks']), common_grid_mode='intersection')
    curves, labels = data['trajectories'], np.asarray(data['labels'])
    if estimand == 'change_from_baseline':
        from Analysis.baseline_projection import baseline_weights_for_run
        anchors, _, _ = baseline_weights_for_run(PROJECT_ROOT / 'Results_paper/CGM' / run, list(data['included_ids']), verbose=False)
        assert len(anchors) == 80
        curves = curves - np.array([anchors[p] for p in data['included_ids']])[:,None,:]
    kernel, bandwidth, _ = functional_gaussian_kernel(curves, data['common_grid'])
    treated, control = np.flatnonzero(labels==1), np.flatnonzero(labels==0)
    statistic = scaled_mmd_v_from_kernels(kernel[np.ix_(treated,treated)], kernel[np.ix_(control,control)], kernel[np.ix_(treated,control)])
    assert bandwidth == pytest.approx(saved['global']['kernel_bandwidth_h'], rel=1e-12)
    assert statistic == pytest.approx(saved['global']['scaled_mmd_v_statistic'], rel=1e-12)
