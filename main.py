# -*- coding: utf-8 -*-

import argparse

from input import get_params       
from Experiment import Experiment, prepare_experiments
from Visualization.log import log_experiment
from load import PedapDatasetNotFound


def _parse_args(argv=None):
   parser = argparse.ArgumentParser(description="Run AAMM experiments")
   parser.add_argument(
      "--analysis-mode",
      choices=("synthetic_d1", "synthetic_d10", "cgm_univariate", "cgm_bivariate"),
      default=None,
   )
   parser.add_argument("--id-start", type=int, default=None)
   parser.add_argument("--id-end", type=int, default=None)
   args = parser.parse_args(argv)
   if (args.id_start is None) != (args.id_end is None):
      parser.error("--id-start and --id-end must be supplied together")
   if args.id_start is not None and args.id_start > args.id_end:
      parser.error("--id-start must not exceed --id-end")
   return args


def main(argv=None):
   """Main execution flow."""

   # Get input parameters.  Clinical modes require the processed PEDAP dataset
   # to exist already: main.py never builds it.
   try:
      options, params_data, params_optim, params_uq, params_node = get_params(
         _parse_args(argv)
      )
   except PedapDatasetNotFound as error:
      print(error)
      return 1
   
   # Generate experiments (hyperparameters, datasets)
   experiments_dict = prepare_experiments(params_data, params_optim, params_uq, params_node, options)
   experiments = []
   for exp_id, exp_data in experiments_dict.items():
      experiments.append(Experiment(exp_id, exp_data["combo"], exp_data["data"], exp_data["gamma"]))

   # Process each experiment
   for j, exp in enumerate(experiments):
      log_experiment(j+1, len(experiments), phase=1)
      exp.run(options)

   return 0


if __name__ == "__main__":
    raise SystemExit(main())


 
"""
Global-local GMM Fitting and Neural ODE Weight Evolution for Time-Series Data

1. Input:
   - A set of time points and data pairs: { (t_i, X_i) } for i = 1,...,n
   - Number of Gaussian components: K
a
2. Construct aggregated data:
   - Merge all X_i into a single set X = ∪ X_i

3. Joint global GMM fitting and local weight estimation:
   - Define the following objective to optimize both the global means/variances (μ_k, σ_k^2)
     and local weights α_k(t_i) simultaneously:
       min_{α_k(t_i), μ_k, σ_k^2} ∑_{i=1}^{n} MMD^2(P_{X_i}, ∑_{k=1}^{K} α_k(t_i) · N(μ_k, σ_k^2)).
   - This process yields:
       (a) Global means μ_k* and variances σ_k*^2.
       (b) Local weights α_k*(t_i) for each time t_i.

4. Neural ODE modeling:
   - Define the continuous-time dynamics of α(t) as dα/dt = f(α), i.e., α'(t) = f(α(t)),
     where α(t) = (α_1(t), ..., α_K(t)).
   - Initialize α(t_i) with the locally fitted weights α_k*(t_i) from Step 3.

5. Final optimization:
   - Solve the Neural ODE to obtain α(t_i) for each time point t_i. 
   - Fit the Neural ODE parameters θ* that describe how α(t) evolves over time.
"""
