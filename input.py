from data import get_distrparams
from Utils.IO_utils import validate_params, set_base_folder


# Select a clinical representation or a synthetic experiment.
ANALYSIS_MODE = "cgm_bivariate"  # Options: "synthetic_d1", "synthetic_d10", "cgm_univariate", "cgm_bivariate"

# The PEDAP RCT has 13 nominal weekly periods.  This is the calendar structure
# of the trial, NOT a requirement that every participant contribute 13 usable
# distributions: participants keep whatever usable weeks they have.
N_NOMINAL_WEEKS = 13
VALID_ANALYSIS_MODES = {
    "synthetic_d1",
    "synthetic_d10",
    "cgm_univariate",
    "cgm_bivariate",
}

def get_params(args=None):
    analysis_mode = getattr(args, "analysis_mode", None) or ANALYSIS_MODE
    if analysis_mode not in VALID_ANALYSIS_MODES:
        raise ValueError(
            f"analysis mode must be one of {sorted(VALID_ANALYSIS_MODES)}"
        )
    load_clinical_data = analysis_mode.startswith("cgm_")
    options = {
        "gif": False,                        # Generate GIF of trajectories
        "gif_error": False,                  # Generate GIF of error trajectories
        "load_cached_fit": True,            # Load cached data in fitting global parameters if available
        "load_cached_node": True,           # Load cached data in fitting neural ode if available
        "load_cached_test": True,           # Load cached data in loss test if available
        "verbose": True,                    # Verbose output during training
        "plots": True,                     # Per-patient NODE diagnostic plots
        "covariance_floor": 1e-2,           # Registered floor used for covariance stabilization
        "bandwidth_candidates": (
            [0.05, 0.1, 0.5, 1.0, 2.0]
            if load_clinical_data else [0.25, 0.5, 1.0, 2.0, 4.0]
        ),
        "test": True,                       # Run evaluation for both synthetic and clinical experiments
        "load_data": load_clinical_data,
        "fix_mv": load_clinical_data,       # One frozen global dictionary for CGM
        "results_root": "Results_paper"
    }
    
    if options["load_data"]:
        options["threshold"] = False

        # The cohort comes from the processed PEDAP dataset (PtRoster), never
        # from a hard-coded id range.  --id-start/--id-end only narrow it.
        if args is not None and args.id_start is not None and args.id_end is not None:
            ids = list(range(args.id_start, args.id_end + 1))
        else:
            ids = None

        representation = analysis_mode.removeprefix("cgm_")
        params_data = {
            "analysis_mode": analysis_mode,
            "pedap_data_dir": "data/processed/pedap_benchmark/rct_followup",
            "dataset_version": "pedap_v2",
            "clinical_window": "pedap_rct",
            "ids": ids,
            "representation": representation,
            "add_deriv": representation == "bivariate",
            # PEDAP SAP minimum CGM per weekly period.
            "weekly_min_cgm_hours": 72.0,
            # Tolerance around the exact five-minute spacing a central
            # difference requires; 0 is the literal rule.  The derivative
            # itself is a modelling choice, not part of the SAP, so this is
            # decided on the observed data: raw Dexcom timestamps jitter by
            # 1-3s around the nominal 300s spacing (99.5% within +/-5s), while
            # a genuine gap is far larger (e.g. 600s).  5s covers the full
            # jitter peak with margin while still rejecting real gaps.
            "derivative_tolerance_seconds": 5.0,
            # Purely numerical requirement for fitting a Neural ODE to a
            # participant's trajectory.  Not a PEDAP exclusion criterion.
            # 2-3 usable weeks under-identify a longitudinal trajectory; 5 is
            # a reasonable floor.  Report sensitivity to 3 and 8 once the
            # SC-arm CGM source is fixed and the usable-weeks distribution can
            # actually be inspected across both arms.
            "min_node_observations": 5,
            "n_nominal_weeks": N_NOMINAL_WEEKS,
            # None = the pre-specified treatment-blind dictionary cohort.
            # No participant is required to have a usable week 0 (or any other
            # specific nominal week): each contributes whichever usable weekly
            # distributions it has, at their real observation times, and the
            # NODE simply starts at that participant's own first observed t_i.
            "global_dictionary_ids": None,
            "dictionary_seed": 42,
            "dictionary_subsample_seed": 42,
            # Caps each dictionary-cohort participant's contribution to the
            # calibration pool at ~1 week of 5-min readings (7*24*12=2016),
            # subsampled deterministically from ALL of that participant's
            # usable weeks.  Prevents a participant with many usable weeks
            # from dominating the calibration geometry over one with few;
            # never reads TrtGroup.  Set to None to use every usable point
            # from every dictionary-cohort participant (pre-balancing
            # behavior).  Both arms now contribute usable weeks under v2
            # (Dexcom Clarity, symmetric CGM source for CLC and SC).
            "dictionary_max_points_per_patient": 2016,
            "bandwidth_seed": 42,
            "bandwidth_reference_sample_size": 1000,
            "global_bandwidth_multiplier": 1.0,
            "max_dictionary_points": 10000
        }

        # Fail fast, and stamp the dataset fingerprint into the run identity so
        # that results and caches can never be shared across datasets.
        from load import load_pedap_manifest
        manifest = load_pedap_manifest(params_data["pedap_data_dir"])
        params_data["dataset_fingerprint"] = manifest["dataset_fingerprint"]

    else:
        options["threshold"] = False
        params_data = {
            "analysis_mode": analysis_mode,
            "distribution": 'gaussian',
            "d": 1 if analysis_mode == "synthetic_d1" else 10,
            "n_points": [20, 50, 100, 200, 300],
            "n_clusters": 3,
            "seed": list(range(1, 101)),
            "t_steps": 11,
            "dictionary_seed": 0
        }


            
    params_optim = {
        "seed_split": 42,                   # Seed for splitting data into train/test sets
        "max_iter": 50,                     # Maximum iterations for MMD-GMM fitting
        "tol_rel_param": 1e-6,              # Practical relative tolerance for FP32 parameters
        "tol_rel_err": 1e-5,                # Practical relative tolerance for an FP32 objective
        "tol_abs": 1e-6,                    # Absolute tolerance for convergence
        "K": [
            10 if not options["load_data"]
            else (5 if params_data.get("representation") == "bivariate" else 5)
        ],
        "update_method": ['gradient'],      # Options: 'gradient' or 'heuristic'
        "lr": [0.01],                       # Learning rate for the MMD-GMM fitting
        "grad_steps": [50],                 # Number of gradient substeps for each step of the MMD-GMM fitting
        "lambda_ridge": [[0.01]],            # Regularization parameter. Each value of the list can be a number/list of size 1 
                                            # (same for all K) or a list of size K (different for each K).
    }
    
    # Uncertainty quantification parameters
    params_uq = {
        "n_bags": 1                         # Number of bagging iterations (>1 enables bagging)
    }   
    
    # Neural ODE parameters
    params_node = {
        "T": 1.0,
        "dt": 0.01,
        "seed_node": 42,
        "hidden_dim": 64,
        "non_linearity": "relu",
        # Final cost/performance choice: the reduced ablation found no material
        # benefit from retaining the slower 64x2 model in the main pipeline.
        "N_layers": 1,
        "method": "dopri5",
        "lr": 1e-2,
        "loss_func": "CE",         # Options: "MSE", "L1", "CE"
        "print_freq": 50,
        "max_epochs": 1000,        # plateau typically around 900-1000; cap earlier to keep runs practical
        "solver_rtol": 1e-7,
        "solver_atol": 1e-9,
        "early_stopping_min_delta": 1e-5,
        "early_stopping_patience": 100,
        "n_MC": 10000,
        "normalization": "softmax",
        "trainable_initial": True,
        "time_fourier_order": 0,
        "time_polynomial_degree": 4,
        "softmax_epsilon": 1e-8
    }
    # Validate input parameters
    validate_params(params_data, params_uq, options)      
    
    # Set up result directories
    options, params_data = set_base_folder(
        options, params_data, params_optim, params_uq, params_node, 
        ignore_keys=["load_cached_fit", "load_cached_node", "load_cached_test", 
                    "load_data", "gif", "verbose", "test", 
                    "ids"]
    )

    # Generate synthetic distribution parameters if not loading external data
    if not options["load_data"]:
        params_data["means"], params_data["vars"] = get_distrparams(
            params_data, params_node["T"], params_data["t_steps"]
        )
            
    return options, params_data, params_optim, params_uq, params_node
