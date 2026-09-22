import os
import json
import numpy as np
from itertools import product
from Run.run_f import run_fitting
from Run.run_n import run_node
from Run.run_t import run_test
from Core.train_mmd import MMD_FIT_ALGORITHM_VERSION
from data import generate_data, compute_gammaopt
from Utils.IO_utils import create_folder, save_data_csv, save_json, load_experiment
BANDWIDTH_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)


def _load_global_bandwidth_selection(
    path, representation, expected_lambda_ridge=None
):
    """Load and validate a treatment-blind complete-day bandwidth selection."""
    if path is None:
        raise ValueError("clinical fitting requires global_bandwidth_selection_file")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    summary = payload.get("summary", payload)
    if summary.get("representation") != representation:
        raise ValueError("bandwidth selection representation does not match the run")
    selected = float(summary["selected_global_multiplier"])
    candidates = tuple(float(value) for value in summary["candidate_multipliers"])
    if selected not in candidates or selected <= 0 or not np.isfinite(selected):
        raise ValueError("invalid selected global bandwidth multiplier")
    if not summary.get("selection_is_treatment_blind", False):
        raise ValueError("bandwidth selection is not documented as treatment-blind")
    if expected_lambda_ridge is not None:
        selected_ridge = payload.get("configuration", {}).get("lambda_ridge")
        if selected_ridge is None or not np.isclose(
            float(selected_ridge), float(expected_lambda_ridge)
        ):
            raise ValueError(
                "bandwidth selection used a different lambda_ridge; "
                "regenerate it before fitting"
            )
    return selected, {
        "path": os.path.abspath(path),
        "selection_analysis": payload.get("analysis"),
        "selected_global_multiplier": selected,
        "candidate_multipliers": list(candidates),
        "lambda_ridge": payload.get("configuration", {}).get("lambda_ridge"),
        "training_bandwidth": summary.get("training_bandwidth"),
        "evaluation_bandwidth": summary.get("evaluation_bandwidth"),
    }


def _median_positive_pairwise_distance(X):
    """Median of finite strictly positive pairwise distances."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] < 2:
        return 1.0
    from scipy.spatial.distance import pdist
    dists = pdist(X)
    positive = dists[np.isfinite(dists) & (dists > 0)]
    return float(np.median(positive)) if positive.size else 1.0


def _gaussian_kernel_matrix(X, Y, gamma):
    """Gaussian kernel matrix K(X, Y) with bandwidth gamma."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.size == 0 or Y.size == 0:
        return np.empty((len(X), len(Y)), dtype=float)
    sq_x = np.sum(X ** 2, axis=1, keepdims=True)
    sq_y = np.sum(Y ** 2, axis=1, keepdims=True).T
    dist2 = sq_x + sq_y - 2.0 * (X @ Y.T)
    dist2 = np.maximum(dist2, 0.0)
    gamma2 = max(float(gamma) ** 2, 1e-12)
    return np.exp(-0.5 * dist2 / gamma2)


def _mmd_validation_score(train, valid, gamma):
    """Treatment-blind MMD score for selecting the discrete bandwidth multiplier."""
    train = np.asarray(train, dtype=float)
    valid = np.asarray(valid, dtype=float)
    if train.shape[0] == 0 or valid.shape[0] == 0:
        return 0.0
    k_tt = _gaussian_kernel_matrix(train, train, gamma)
    k_vv = _gaussian_kernel_matrix(valid, valid, gamma)
    k_tv = _gaussian_kernel_matrix(train, valid, gamma)
    return float(np.mean(k_tt) + np.mean(k_vv) - 2.0 * np.mean(k_tv))


def _select_window_bandwidth(block, seed=42, multipliers=BANDWIDTH_MULTIPLIERS):
    """Select the discrete bandwidth multiplier c using the article's validation rule."""
    ref = _median_positive_pairwise_distance(block)
    if ref <= 0 or not np.isfinite(ref):
        ref = 1.0

    if len(block) < 6:
        return float(ref), float(1.0)

    rng = np.random.default_rng(seed)
    n_valid = min(max(3, len(block) // 2), len(block) - 1)
    if n_valid < 3:
        return float(ref), float(1.0)

    valid_idx = rng.choice(len(block), size=n_valid, replace=False)
    valid = np.asarray(block[valid_idx], dtype=float)
    train = np.delete(np.asarray(block, dtype=float), valid_idx, axis=0)
    if len(train) == 0:
        return float(ref), float(1.0)

    best_gamma = float(ref)
    best_c = float(1.0)
    best_score = np.inf
    for c in multipliers:
        gamma = ref * float(c)
        score = _mmd_validation_score(train, valid, gamma)
        if score < best_score:
            best_score = score
            best_gamma = float(gamma)
            best_c = float(c)
    return best_gamma, best_c

# ----------------------------------------------------------------------------- 
# Experiment : one hyper-parameter configuration  ➜  one complete run
# -----------------------------------------------------------------------------

class Experiment:
    """
    Hold everything related to ONE experiment:
      – fixed hyper-parameter combo
      – the time-series datasets generated or loaded
      – the γ* found for every time-step
      – the results dict that comes back from fitting / NODE
    """
    def __init__(self, exp_id, combo, data, gamma):
        self.exp_id = exp_id
        self.combo  = combo          # fixed params
        self.data   = data           # list of arrays (len = t_steps)
        self.gamma  = gamma          # list of γ* (len = t_steps)
        self.results = None
        self.folder  = None

    # ------------------------------------------------------------------ #
    # run : launch fitting   →   launch NODE   →   collect every result
    # ------------------------------------------------------------------ #
    def run(self, options):
        """
        Run complete time-series experiment:
        1) Fit weights/means/covariances via MMD-GMM
        2) Train Neural ODE for weight trajectories
        3) Evaluate and test the model
        """
        
        # 0) ----------------------------------------------------------------
        print(f"[Experiment]  {self.exp_id}")
        self.combo.setdefault("covariance_floor", options.get("covariance_floor", 1e-2))

        # For medical data: one folder per REAL patient ID
        if options.get("load_data", False):
            # real patient id comes from combo (patient_ids from the RDS)
            patient_id = self.combo.get("ids", self.combo.get("patient_id", "unknown"))
            base_folder = options["base_folder"]
            self.folder = os.path.join(base_folder, f"patient_{patient_id}")
        else:
            # synthetic experiments keep the original behavior
            self.folder = create_folder(options, self.combo)

        os.makedirs(self.folder, exist_ok=True)

        
        # 1) MMD-GMM fitting
        fit_res = run_fitting(self.combo, options, self.folder, self.data, self.gamma)
        all_trajs = []
        all_errs = []

        m  = fit_res["avg_means"]
        c  = fit_res["avg_covariances"]
        w  = fit_res["avg_weights"]
            
        # 2) Train Neural ODE and evaluate
        traj, f = run_node(self.combo, self.data, m, c, w, self.folder, options)
        errs = (run_test(self.combo, m, c, w, traj, f, self.data, options)
                if options.get("test", True) or options.get("load_data", False) else {})
        all_trajs.append(traj)
        all_errs.append(errs)

        # 3) Store results
        fit_res["traj_weights"] = all_trajs
        fit_res["test_errors"] = all_errs
        
        self.results = fit_res
        print(f"Experiment {self.exp_id} finished. Results saved in {self.folder}")
        return self.results

def prepare_experiments(params_data,
                        params_optim,
                        params_uq,
                        params_node,
                        options):
    """
    Build the dict {exp_id: {"combo", "data", "gamma"}} consumed later by
    `Experiment`.  Works for
        • purely synthetic runs          (params_load is None)
        • CSV time-series runs   (params_load is a dict from get_params)
    """

        # ------------------------------------------------------------------ #
    # 1. LOADED CSV / CGM BRANCH
    # ------------------------------------------------------------------ #
    folder = options["base_folder"]
    
    params_load  = options.get("load_data", False)
    if params_load:
        from load import prepare_cgm_experiments
        
        experiments_dict, init_data, init_info = prepare_cgm_experiments(
            data_dir=params_data["pedap_data_dir"],
            patient_ids=params_data.get("ids"),
            representation=params_data["representation"],
            add_deriv=params_data.get("add_deriv", False),
            weekly_min_cgm_hours=params_data.get("weekly_min_cgm_hours"),
            min_node_observations=params_data.get("min_node_observations", 5),
            derivative_tolerance_seconds=params_data.get(
                "derivative_tolerance_seconds", 5.0
            ),
            dictionary_patient_ids=params_data.get("global_dictionary_ids"),
            dictionary_max_points_per_patient=params_data.get(
                "dictionary_max_points_per_patient", 2016
            ),
            dictionary_subsample_seed=params_data.get("dictionary_subsample_seed", 42),
            bandwidth_seed=params_data.get("bandwidth_seed", 42),
            bandwidth_reference_sample_size=params_data.get(
                "bandwidth_reference_sample_size", 1000
            ),
        )
        dataset_fingerprint = init_info["dataset_fingerprint"]
        params_data["dataset_fingerprint"] = dataset_fingerprint

        clinical_ridge = np.asarray(
            params_optim.get("lambda_ridge", 0.01), dtype=float
        ).reshape(-1)
        if clinical_ridge.size != 1:
            raise ValueError(
                "clinical bandwidth selection requires one scalar lambda_ridge"
            )
        configured_multiplier = params_data.get("global_bandwidth_multiplier")
        if configured_multiplier is not None:
            global_multiplier = float(configured_multiplier)
            if global_multiplier <= 0 or not np.isfinite(global_multiplier):
                raise ValueError("global_bandwidth_multiplier must be finite and positive")
            bandwidth_selection = {
                "path": None,
                "selection_analysis": "fixed production multiplier",
                "selection_mode": "fixed",
                "selected_global_multiplier": global_multiplier,
                "candidate_multipliers": [],
                "lambda_ridge": float(clinical_ridge[0]),
                "training_bandwidth": (
                    "fixed c times one patient-pooled median distance"
                ),
                "evaluation_bandwidth": (
                    "same absolute patient-specific bandwidth at every block"
                ),
            }
        else:
            global_multiplier, bandwidth_selection = _load_global_bandwidth_selection(
                params_data.get("global_bandwidth_selection_file"),
                params_data.get("representation"),
                expected_lambda_ridge=float(clinical_ridge[0]),
            )
        for exp_id, exp_data in experiments_dict.items():
            reference = float(exp_data["patient_bandwidth_reference"])
            gamma = float(global_multiplier * reference)
            gamma_list = [gamma] * len(exp_data["data"])
            c_list = [float(global_multiplier)] * len(exp_data["data"])
            exp_data["gamma"] = gamma_list
            exp_data["gamma_multipliers"] = c_list
        params_data["global_bandwidth_multiplier"] = global_multiplier
        params_data["bandwidth_selection"] = bandwidth_selection
        
        # Fix means/covs if requested - do a FULL MMD-GMM fit on representative patients
        if options.get("fix_mv", False):
            from Core.train_mmd import mmd_gmm_fit
            from Utils.MMD_utils import backtransform_gmm_parameters
            
            # Helper to extract value (handles both list and scalar params)
            def get_val(d, key, default):
                v = d.get(key, default)
                return v[0] if isinstance(v, list) else v
            
            # Subsample init_data if too large to avoid OOM
            MAX_INIT_POINTS = params_data.get("max_dictionary_points", 10000)
            if len(init_data) > MAX_INIT_POINTS:
                rng = np.random.default_rng(params_data.get("dictionary_subsample_seed", 42))
                idx = rng.choice(len(init_data), MAX_INIT_POINTS, replace=False)
                init_data_fit = init_data[idx]
                print(f"[FIX_MV] Subsampled {len(init_data)} -> {MAX_INIT_POINTS} points for fitting")
            else:
                init_data_fit = init_data
            
            print("\n" + "="*60)
            print("FITTING GMM ON REPRESENTATIVE PATIENTS")
            print("="*60)
            print(f"Patient IDs: {init_info['init_patient_ids']}")
            print(f"Original data points: {init_info['n_points']}")
            print(f"Fitting with: {len(init_data_fit)} points")
            print(f"Dimensionality: {init_info['dimensionality']}D")
            print("="*60 + "\n")
            
            dictionary_file = "global_dictionary.json"
            stored_dictionary = load_experiment(folder, dictionary_file)
            K_dictionary = get_val(params_optim, "K", 3)
            expected_ids = init_info["init_patient_ids"]
            scaling = init_info["scaling"]
            dictionary_optimizer = {
                "max_iter": int(get_val(params_optim, "max_iter", 50)),
                "tol_rel_param": float(get_val(params_optim, "tol_rel_param", 1e-4)),
                "tol_rel_err": float(get_val(params_optim, "tol_rel_err", 1e-6)),
                "tol_abs": float(get_val(params_optim, "tol_abs", 1e-8)),
                "lr": float(get_val(params_optim, "lr", 0.001)),
                "grad_steps": int(get_val(params_optim, "grad_steps", 10)),
                "lambda_ridge": np.asarray(
                    get_val(params_optim, "lambda_ridge", 0.01), dtype=float
                ).reshape(-1).tolist(),
            }

            if stored_dictionary is not None:
                compatible = (
                    stored_dictionary.get("representation") == params_data.get("representation") and
                    stored_dictionary.get("dataset_fingerprint") == dataset_fingerprint and
                    stored_dictionary.get("K") == K_dictionary and
                    stored_dictionary.get("d") == init_data.shape[1] and
                    stored_dictionary.get("calibration_patient_ids") == expected_ids and
                    stored_dictionary.get("dictionary_seed") == params_data.get("dictionary_seed", 42) and
                    stored_dictionary.get("fit_algorithm_version") == MMD_FIT_ALGORITHM_VERSION and
                    stored_dictionary.get("covariance_floor") == options.get("covariance_floor", 1e-2) and
                    stored_dictionary.get("subsample_seed") == params_data.get("dictionary_subsample_seed", 42) and
                    # Compare only the two scalars that determine the balanced
                    # subsample, not the full per-patient breakdown: that dict
                    # has integer patient-id keys, which JSON round-trips to
                    # strings, so a whole-dict compare would always miss after
                    # the first cache write even with nothing changed.
                    (stored_dictionary.get("balanced_subsampling") or {}).get("max_points_per_patient")
                    == (init_info.get("balanced_subsampling") or {}).get("max_points_per_patient") and
                    (stored_dictionary.get("balanced_subsampling") or {}).get("subsample_seed")
                    == (init_info.get("balanced_subsampling") or {}).get("subsample_seed") and
                    stored_dictionary.get("n_fitted_points") == len(init_data_fit) and
                    stored_dictionary.get("bandwidth_reference_sample_size") == params_data.get("bandwidth_reference_sample_size", 1000) and
                    stored_dictionary.get("optimizer_config") == dictionary_optimizer and
                    stored_dictionary.get("global_bandwidth_multiplier") == global_multiplier and
                    stored_dictionary.get("bandwidth_selection", {}).get("path") == bandwidth_selection["path"] and
                    np.allclose(stored_dictionary.get("scaling", {}).get("mean", []), scaling["mean"]) and
                    np.allclose(stored_dictionary.get("scaling", {}).get("std", []), scaling["std"])
                )
                if stored_dictionary.get("dataset_fingerprint") != dataset_fingerprint:
                    raise RuntimeError(
                        "Cached global dictionary was built on PEDAP dataset "
                        f"{stored_dictionary.get('dataset_fingerprint')!r}, but this run "
                        f"uses {dataset_fingerprint!r}. Clinical caches are never reused "
                        "across datasets: delete the run folder and refit."
                    )
                if not compatible:
                    raise RuntimeError("Stored global dictionary does not match this clinical configuration")
                params_data["means"] = np.asarray(stored_dictionary["means"], dtype=float)
                params_data["covs"] = np.asarray(stored_dictionary["covariances"], dtype=float)
                init_gamma = float(stored_dictionary["bandwidth"])
                print(f"[FIX_MV] Reusing {os.path.join(folder, dictionary_file)}")
            else:
                init_gamma = global_multiplier * compute_gammaopt(
                    init_data_fit,
                    sample_size=params_data.get("bandwidth_reference_sample_size", 1000),
                    seed=params_data.get("bandwidth_seed", 42)
                )
                fit_result = mmd_gmm_fit(
                    X=[init_data_fit],
                    K=K_dictionary,
                    gamma=[init_gamma],
                    max_iter=dictionary_optimizer["max_iter"],
                    tol_rel_param=dictionary_optimizer["tol_rel_param"],
                    tol_rel_err=dictionary_optimizer["tol_rel_err"],
                    tol_abs=dictionary_optimizer["tol_abs"],
                    lr=dictionary_optimizer["lr"],
                    grad_steps=dictionary_optimizer["grad_steps"],
                    verbose=True,
                    lambda_ridge=dictionary_optimizer["lambda_ridge"],
                    seed=params_data.get("dictionary_seed", 42),
                    covariance_floor=options.get("covariance_floor", 1e-2),
                )
                params_data["means"] = fit_result.means
                params_data["covs"] = fit_result.covariances
                raw_means, raw_covs = backtransform_gmm_parameters(
                    fit_result.means, fit_result.covariances,
                    np.asarray(scaling["mean"]), np.asarray(scaling["std"])
                )
                stored_dictionary = {
                    "representation": params_data.get("representation"),
                    "dataset_fingerprint": dataset_fingerprint,
                    "dataset_version": params_data.get("dataset_version"),
                    "analysis_window_definition": params_data.get(
                        "clinical_window", "pedap_rct"
                    ),
                    "cohort_rule": init_info.get("cohort_rule"),
                    "K": K_dictionary,
                    "d": int(init_data.shape[1]),
                    "means": fit_result.means.tolist(),
                    "covariances": fit_result.covariances.tolist(),
                    "means_original_units": raw_means.tolist(),
                    "covariances_original_units": raw_covs.tolist(),
                    "bandwidth": float(init_gamma),
                    "global_bandwidth_multiplier": float(global_multiplier),
                    "bandwidth_selection": bandwidth_selection,
                    "scaling": scaling,
                    "calibration_patient_ids": expected_ids,
                    "dictionary_seed": params_data.get("dictionary_seed", 42),
                    "fit_algorithm_version": MMD_FIT_ALGORITHM_VERSION,
                    "covariance_floor": float(options.get("covariance_floor", 1e-2)),
                    "optimizer_config": dictionary_optimizer,
                    "subsample_seed": params_data.get("dictionary_subsample_seed", 42),
                    "balanced_subsampling": init_info.get("balanced_subsampling"),
                    "dictionary_cohort_arms": init_info.get("dictionary_cohort_arms"),
                    "bandwidth_seed": params_data.get("bandwidth_seed", 42),
                    "bandwidth_reference_sample_size": params_data.get(
                        "bandwidth_reference_sample_size", 1000
                    ),
                    "n_calibration_points": int(len(init_data)),
                    "n_fitted_points": int(len(init_data_fit))
                }
                save_json(stored_dictionary, folder, dictionary_file)
                init_info["fix_mv_fit"] = {
                    "training_error": float(fit_result.mmd_history[fit_result.best_index]),
                    "training_objective": float(
                        fit_result.objective_history[fit_result.best_index]
                    ),
                    "convergence_iter": fit_result.convergence_iter,
                    "fit_time": fit_result.fit_time,
                    "best_index": fit_result.best_index
                }

                print("\n" + "="*60)
                print("FIX_MV FIT COMPLETE")
                print(f"  Best MMD²: {fit_result.mmd_history[fit_result.best_index]:.6f}")
                print(f"  Converged at iter: {fit_result.convergence_iter}")
                print(f"  Time: {fit_result.fit_time:.2f}s")
                print("="*60 + "\n")

            params_data["dictionary"] = stored_dictionary
        
        # Merge params and build experiments dict
        experiments = {}
        for exp_id, exp_data in experiments_dict.items():
            merged = {**params_data, **params_optim, **params_uq, **params_node}
            # The dictionary cohort is one scientific list, not a grid value.
            list_valued_params = {"global_dictionary_ids"}
            combo = {
                k: (v if k in list_valued_params
                    else v[0] if isinstance(v, list) else v)
                for k, v in merged.items()
            }
            
            # Determine dimensionality from data
            data_dim = exp_data["data"][0].shape[1]
            
            combo.update({
                "ids": exp_data["combo"]["patient_id"],   # REAL patient ID
                # t_steps is the number of USABLE observation distributions this
                # participant contributes, which need not be 13.
                "t_steps": exp_data["combo"]["n_weeks"],
                "nominal_week_indices": exp_data["combo"]["nominal_week_indices"],
                "analysis_window_definition": exp_data["combo"][
                    "analysis_window_definition"
                ],
                "observed_support_weeks": exp_data["combo"]["observed_support_weeks"],
                "treatment_group": exp_data["combo"]["treatment_group"],
                "rand_date": exp_data["combo"]["rand_date"],
                "week13_visit_date": exp_data["combo"]["week13_visit_date"],
                "week13_out_of_window": exp_data["combo"]["week13_out_of_window"],
                "analysis_start_date": exp_data["combo"]["analysis_start_date"],
                "analysis_end_date": exp_data["combo"]["analysis_end_date"],
                "analysis_end_source": exp_data["combo"]["analysis_end_source"],
                "is_dropout_before_week13": exp_data["combo"][
                    "is_dropout_before_week13"
                ],
                "dataset_fingerprint": exp_data["combo"]["dataset_fingerprint"],
                "dataset_version": exp_data["combo"]["dataset_version"],
                "n_points": exp_data["data"][0].shape[0],
                "d": data_dim,  # 1D or 2D based on the representation
                "init_info": init_info,
                "observation_times_weeks": exp_data["combo"]["observation_times_weeks"],
                "block_metadata": exp_data["combo"]["block_metadata"],
                "omitted_blocks": exp_data["combo"]["omitted_blocks"],
                "window_bandwidths": exp_data["gamma"],
                "window_multipliers": exp_data.get("gamma_multipliers", []),
                "bandwidth_scope": "patient_fixed_over_time",
                "patient_bandwidth_reference": exp_data["patient_bandwidth_reference"],
                "patient_bandwidth_reference_n_total": exp_data[
                    "patient_bandwidth_reference_n_total"
                ],
                "patient_bandwidth_reference_n_sample": exp_data[
                    "patient_bandwidth_reference_n_sample"
                ],
                "global_bandwidth_multiplier": global_multiplier,
                "bandwidth_selection": bandwidth_selection,
                "dictionary": params_data.get("dictionary", {})
            })
            
            experiments[exp_id] = {
                "combo": combo,
                "data": exp_data["data"],
                "gamma": exp_data["gamma"],
                "multipliers": exp_data.get("gamma_multipliers", [])
            }
        
        return experiments

    # ------------------------------------------------------------------ #
    # 2. SYNTHETIC BRANCH  (identical logic to previous version)
    # ------------------------------------------------------------------ #
    merged   = {**params_data, **params_optim, **params_uq, **params_node}
    ts_keys  = {"means", "vars"}

    fixed_keys   = [k for k in merged if k not in ts_keys]
    fixed_params = {k: (merged[k] if isinstance(merged[k], list) else [merged[k]]) for k in fixed_keys}
    fixed_combos = [dict(zip(fixed_keys, vals)) for vals in product(*fixed_params.values())]

    experiments = {}
    T       = params_node.get("T", 0)
    t_steps = params_data.get("t_steps", 1)
    d       = params_data["d"]

    for i, combo in enumerate(fixed_combos, start=1):
        exp_id = f"exp_{i}"
        seed   = combo.get("seed")

        if T == 0:
            ts_params = {"means": params_data["means"][0], "vars": params_data["vars"][0]}
            X     = generate_data("gaussian", d, {**combo, **ts_params}, seed=seed)
            gamma = compute_gammaopt(X)
            experiments[exp_id] = {"combo": combo, "data": [X], "gamma": [gamma]}
        else:
            data_list, gamma_list, c_list = [], [], []
            for t in range(t_steps):
                ts_params = {"means": params_data["means"][t], "vars": params_data["vars"][t]}
                X_t   = generate_data("gaussian", d, {**combo, **ts_params}, seed=seed)
                data_list.append(X_t)
                gamma_val, c_val = _select_window_bandwidth(
                    X_t,
                    seed=int(seed) + 1000 * (t + 1),
                    multipliers=tuple(options.get("bandwidth_candidates", BANDWIDTH_MULTIPLIERS))
                )
                gamma_list.append(gamma_val)
                c_list.append(c_val)
            experiments[exp_id] = {"combo": combo, "data": data_list, "gamma": gamma_list, "multipliers": c_list}

    if folder is not None:
        save_data_csv(experiments, output_dir=folder, filename="data.csv")

    return experiments
