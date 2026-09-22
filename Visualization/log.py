


def log_experiment(*args, phase=1, errors=None, best_index=None, conv_it=None, test_error=None, options = None):
    """
    Logs configuration and optional results.

    For:
      - Phase 1: expects (current_iteration, total_iterations).
      - Phases 2 and 3: expects (exp_params,).

    Args:
      *args: Variable arguments depending on the phase.
      errors (list, optional): List of squared MMD errors (phase 3).
      best_index (int, optional): Index of the best result (phase 3).
      conv_it (int, optional): Number of iterations until convergence.
      test_error (float or iterable, optional): Testing error.
      phase (int, optional): Logging phase (default=1).
      options (dict, optional): Options dictionary for additional parameters.
    """
    if phase == 1:
        # Phase 1: Log iteration progress.
        if len(args) != 2:
            raise ValueError("Phase 1 requires two arguments: current iteration and total iterations.")
        iteration, total_iterations = args
        print("===================================")
        print("===================================")
        print(f"Iteration {iteration}/{total_iterations}")
        print("===================================")
        
    elif phase == 2:
        if len(args) != 1:
            raise ValueError("Phase 2 requires one argument: exp_params.")
        exp_params = args[0]

        parts = [
            f"d={exp_params['d']}",
            f"K={exp_params['K']}",
            f"lr={exp_params['lr']}",
            f"grad_steps={exp_params['grad_steps']}",
        ]
        if not options.get("load_data"):
            parts.append(f"n_points={exp_params['n_points']}, seed={exp_params['seed']}")

        parts.extend([
            f"lambda_ridge={exp_params['lambda_ridge']}",
            f"t_steps={exp_params['t_steps']}"
        ])

        base_msg = ", ".join(parts) + ", "
        print(base_msg)
        print("===================================")

        
    elif phase == 3:
        # Phase 3: Log performance metrics.
        if len(args) != 1:
            raise ValueError("Phase 3 requires one argument: exp_params.")
        exp_params = args[0]
        if errors is not None and best_index is not None:
            print("Minimum (global) MMD^2 error:", errors[best_index])
            print(f"Best index: {best_index} ({conv_it} iterations)")
            if hasattr(test_error, '__iter__'):
                print("Testing (global) MMD^2 error:", sum(test_error))
            else:
                print("Testing (global) MMD^2 error:", test_error)
            print("===================================")

def log_error(type, avgL2_ts, stdL2_ts, avgL2, stdL2, avgAlt_ts, stdAlt_ts, avgAlt, stdAlt):

    if type.startswith("synthetic"):
        # ---------------------------------------------------------------------
        #  Console summary 
        # ---------------------------------------------------------------------
        print("===================================")
        print("MMD FITTING (coarse grid):")
        print(
            f"Avg L2: {avgL2_ts:.12f} ± {stdL2_ts:.12f}"
        )
        print("-----------------------------------")
        print("NODE TRAJECTORY (fine grid):")
        print(
            f"Avg L2: {avgL2:.12f} ± {stdL2:.12f}"
        )
        print("===================================")
    elif type == "load":
        print("===================================")
        print("MMD FITTING (coarse grid):")
        print(
            f"Avg L2: {avgL2_ts:.12f} ± {stdL2_ts:.12f}\nAvg MMD: {avgAlt_ts:.12f} ± {stdAlt_ts:.12f}"
        )
        print("-----------------------------------")
        print("NODE TRAJECTORY (fine grid):")
        print(
            f"Avg L2: {avgL2:.12f} ± {stdL2:.12f}\nAvg MMD: {avgAlt:.12f} ± {stdAlt:.12f}"
        )
        print("===================================")
