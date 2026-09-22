"""Reproducible patient sensitivity studies for the selected notebook run."""
from pathlib import Path
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal, norm

from Analysis.paper_config import COMPONENT_COLORS
from Analysis.paper_figures import _save_figure


def node_feature_sweep(bundle, output_dir, fourier_orders=(0, 1, 2),
                       polynomial_degrees=(1, 2, 3, 4), max_epochs=None):
    """Vary Fourier at polynomial degree 1, and polynomial at Fourier order 0.

    Training uses the production trainer and saved hyperparameters. A content
    hash includes the fit, configuration and training source for cache identity.
    These are in-sample fits, not held-out model-selection scores.
    """
    import torch
    from Run.run_n import _train_node

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = dict(bundle['node_config'])
    base.update(K=bundle['fit_weights'].shape[1], print_freq=100)
    if max_epochs is not None:
        base['max_epochs'] = int(max_epochs)
    families = {
        'fourier': [(int(f), 1) for f in fourier_orders],
        'polynomial': [(0, int(p)) for p in polynomial_degrees],
    }
    times = np.asarray(bundle['observation_times'])
    if np.any(np.diff(times) <= 0):
        raise ValueError('Observation times must increase strictly')
    grid = np.asarray(bundle['trajectory_times'])
    if not np.allclose([grid[0], grid[-1]], [times[0], times[-1]]):
        raise ValueError('NODE and observation supports must agree')
    normalized = (times - times[0]) / (times[-1] - times[0]) * base['T']
    normalized_grid = (grid - times[0]) / (times[-1] - times[0]) * base['T']
    source_root = Path(__file__).resolve().parents[1]
    source_hash = hashlib.sha256(b''.join(
        (source_root / p).read_bytes() for p in
        ('Core/node.py', 'Core/train_node.py', 'Run/run_n.py', 'Analysis/patient_sensitivity.py')
    )).hexdigest()
    results = {}
    for f, p in dict.fromkeys(pair for pairs in families.values() for pair in pairs):
        config = dict(base, time_fourier_order=f, time_polynomial_degree=p)
        identity = {'config': config, 'fit': bundle['fit'], 'grid': grid.tolist(),
                    'source_hash': source_hash, 'torch_version': torch.__version__, 'device': 'cpu'}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        cache = output_dir / f'node_f{f}_p{p}_{digest[:16]}.json'
        if cache.exists():
            result = json.loads(cache.read_text())
        else:
            print(f'[features] {output_dir.name}: Fourier={f}, polynomial={p}', flush=True)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(config['seed_node'])
                prediction, history, diagnostics = _train_node(
                    config, torch.tensor(bundle['fit_weights'], dtype=torch.float32).unsqueeze(1),
                    torch.tensor(normalized, dtype=torch.float32),
                    torch.tensor(normalized_grid, dtype=torch.float32), torch.device('cpu'),
                )
            prediction = np.asarray(prediction).reshape(len(grid), -1)
            at_observed = np.column_stack([
                np.interp(times, grid, prediction[:, k]) for k in range(prediction.shape[1])
            ])
            result = {'identity': identity, 'trajectory': prediction.tolist(),
                      'loss_history': history, 'diagnostics': diagnostics,
                      'training_weight_mse': float(np.mean((at_observed - bundle['fit_weights']) ** 2))}
            cache.write_text(json.dumps(result, indent=2) + '\n')
        results[(f, p)] = result

    files = []
    for family, pairs in families.items():
        fig, axes = plt.subplots(len(pairs), 1, figsize=(10, 2.5 * len(pairs)),
                                 sharex=True, sharey=True, squeeze=False, layout='constrained')
        for ax, pair in zip(axes[:, 0], pairs):
            result = results[pair]
            trajectory = np.asarray(result['trajectory'])
            for k in range(trajectory.shape[1]):
                color = COMPONENT_COLORS[k % len(COMPONENT_COLORS)]
                ax.scatter(times, bundle['fit_weights'][:, k], color=color, s=13, alpha=0.6)
                ax.plot(grid, trajectory[:, k], color=color, lw=1.6, label=f'Component {k+1}')
            ax.set_ylabel('Weight')
            ax.set_title(f'Fourier order {pair[0]}, polynomial degree {pair[1]} '
                         f"| training MSE={result['training_weight_mse']:.3g}", fontsize=10)
            ax.grid(alpha=0.18)
        axes[0, 0].legend(ncol=bundle['fit_weights'].shape[1], fontsize=8, frameon=False)
        axes[-1, 0].set_xlabel('Study week')
        files += _save_figure(fig, output_dir, f'node_{family}_features')
    return files


def bandwidth_endpoint_sweep(bundle, raw_blocks, raw_metadata, output_dir,
                             multipliers=(0.1, 0.5, 1.0, 2.0, 5.0)):
    """Refit only endpoint local MMD weights with gamma=c*d_ref; fix geometry/ridge."""
    from Utils.MMD_utils import update_local_weights

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = bundle['fit']['config']
    dictionary = config['dictionary']
    means, covs = bundle['means'], bundle['covs']
    d = means.shape[1]
    cs = np.asarray(multipliers, dtype=float)
    if len(cs) == 0 or np.any(~np.isfinite(cs)) or np.any(cs <= 0) or len(np.unique(cs)) != len(cs):
        raise ValueError('Bandwidth multipliers must be distinct, positive and finite')
    if not np.any(cs == 1.0):
        raise ValueError('Include c=1 as the reference for density differences')
    if len(raw_blocks) != len(bundle['observation_times']):
        raise ValueError('Raw blocks do not match the fitted window')
    # PEDAP v2 anchors every patient at randomisation (RandDt), not at the
    # first CGM reading, so the calendar origin is ``rand_date``.  Fits saved
    # before that key was carried into the run config simply lack it; the
    # per-block check below compares absolute ISO start/end datetimes, which
    # subsumes an origin comparison, so its absence is not a loss of rigour.
    saved_origin = config.get('rand_date', config.get('calendar_origin_date'))
    if saved_origin is not None and raw_metadata['rand_date'] != saved_origin:
        raise ValueError('Raw and fitted calendar origins differ')
    saved_blocks = [b for b in config['block_metadata'] if b.get('included', True)]
    raw_included = [b for b in raw_metadata['blocks'] if b.get('included', True)]
    if len(saved_blocks) != len(raw_included):
        raise ValueError('Raw/fitted block count mismatch')
    for saved, raw in zip(saved_blocks, raw_included):
        for key in ('week_index', 'n_observations', 'observation_time_weeks',
                    'actual_start', 'actual_end'):
            if saved[key] != raw[key]:
                raise ValueError(f'Raw/fitted block mismatch: {key}')
    if not np.allclose(means, dictionary['means']) or not np.allclose(covs, dictionary['covariances']):
        raise ValueError('Local fit must use the selected global dictionary')
    scaling = dictionary['scaling']
    center, scale = np.asarray(scaling['mean']), np.asarray(scaling['std'])
    reference = float(config['patient_bandwidth_reference'])
    if not np.isfinite(reference) or reference <= 0 or np.any(scale <= 0):
        raise ValueError('Invalid saved bandwidth reference or scaling')
    originals = [np.asarray(raw_blocks[i]).reshape(-1, d) for i in (0, -1)]
    standardized = [(x - center) / scale for x in originals]
    weights = np.asarray([
        [update_local_weights(x, means, covs, (c * reference)**2 * np.eye(d),
                              config['lambda_ridge'], bundle['fit_weights'][i].copy())
         for x, i in zip(standardized, (0, -1))] for c in cs
    ])
    mu, sigma = bundle['means_original'], bundle['covs_original']
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(cs)))
    styles = ['-', '--', '-.', ':', (0, (5, 2))]
    files = []
    grids = []
    for j in range(d):
        pooled = np.concatenate([x[:, j] for x in originals])
        lo = min(pooled.min(), np.min(mu[:, j] - 4*np.sqrt(sigma[:, j, j])))
        hi = max(pooled.max(), np.max(mu[:, j] + 4*np.sqrt(sigma[:, j, j])))
        grids.append(np.linspace(lo, hi, 400))
        basis = norm.pdf(grids[j][:, None], mu[:, j], np.sqrt(sigma[:, j, j]))
        densities = np.einsum('cek,xk->cex', weights, basis)
        fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True, sharey='row', layout='constrained')
        for e, title in enumerate(('Initial', 'Final')):
            axes[0, e].hist(originals[e][:, j], bins=45, density=True, color='0.85')
            for ci, c in enumerate(cs):
                style = dict(color=colors[ci], linestyle=styles[ci % len(styles)], lw=1.7)
                axes[0, e].plot(grids[j], densities[ci, e], label=f'c={c:g}', **style)
                axes[1, e].plot(grids[j], densities[ci, e] - densities[cs == 1][0, e], **style)
            axes[0, e].set_title(f'{title}: week {bundle["observation_times"][[0, -1][e]]:g}')
            axes[1, e].axhline(0, color='0.4', lw=0.6)
            axes[1, e].set_xlabel('Glucose (mg/dL)' if j == 0 else 'Glucose rate of change (mg/dL/time)')
        axes[0, 0].set_ylabel('Marginal density' if d == 2 else 'Density')
        axes[1, 0].set_ylabel('Density difference from c=1')
        axes[0, 1].legend(frameon=False)
        files += _save_figure(fig, output_dir, f'bandwidth_endpoints_coordinate_{j+1}')

    # In d=2 the joint panels also reveal changes hidden by the marginals.
    if d == 2:
        xx, yy = np.meshgrid(grids[0][::2], grids[1][::2])
        positions = np.stack((xx, yy), axis=-1)
        basis = np.stack([multivariate_normal(m, s).pdf(positions) for m, s in zip(mu, sigma)])
        densities = np.einsum('cek,kyx->ceyx', weights, basis)
        fig, axes = plt.subplots(len(cs), 2, figsize=(10, 2.7*len(cs)), sharex=True, sharey=True,
                                 squeeze=False, layout='constrained')
        vmax = float(densities.max())
        for ci, c in enumerate(cs):
            for e, title in enumerate(('Initial', 'Final')):
                ax = axes[ci, e]
                mesh = ax.pcolormesh(xx, yy, densities[ci, e], shading='auto', cmap='Greens', vmin=0, vmax=vmax)
                ax.scatter(*originals[e].T, s=1, color='black', alpha=0.1, rasterized=True)
                ax.set_title(f'{title}, c={c:g}', fontsize=10)
                if e == 0:
                    ax.set_ylabel('Rate (mg/dL/time)')
        for ax in axes[-1]:
            ax.set_xlabel('Glucose (mg/dL)')
        fig.colorbar(mesh, ax=axes.ravel().tolist(), label='Joint density', shrink=0.65)
        files += _save_figure(fig, output_dir, 'bandwidth_endpoints_joint')

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True, layout='constrained')
    for e, ax in enumerate(axes):
        for k in range(means.shape[0]):
            ax.plot(cs, weights[:, e, k], 'o-', color=COMPONENT_COLORS[k % len(COMPONENT_COLORS)], label=f'Component {k+1}')
        ax.set(xscale='log', xlabel='Bandwidth multiplier c', ylabel='Local MMD weight',
               title=('Initial', 'Final')[e])
        ax.grid(alpha=0.18)
    axes[1].legend(frameon=False, fontsize=8)
    files += _save_figure(fig, output_dir, 'bandwidth_endpoint_weights')
    metadata = {'fit_path': str(bundle['patient_dir'] / 'result_fit.json'),
                'dictionary': dictionary, 'multipliers': cs.tolist(),
                'bandwidth_reference': reference, 'bandwidths': (cs*reference).tolist(),
                'lambda_ridge': config['lambda_ridge'], 'weights': weights.tolist(),
                'endpoint_weeks': bundle['observation_times'][[0, -1]].tolist(),
                'raw_metadata': raw_metadata,
                'interpretation': 'Fixed global MMD dictionary; refitted local weights, no NODE. '
                                  'Raw MMD values across c are not model-selection scores.'}
    (output_dir / 'bandwidth_sensitivity.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return files
