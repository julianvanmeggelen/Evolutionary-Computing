from datetime import datetime
from hyper_parameter_optimization.config.revolve_neat_config import RevolveNeatConfig
from hyper_parameter_optimization.optimizer.optimizer import OptunaHyperOptimizer, SpotHyperOptimizer
from hyper_parameter_optimization.result.optimization_run import OptimizationRun
from hyper_parameter_optimization.result.optimization_result import OptimizationResult
from hyper_parameter_optimization.optimizer.tunable_param import TunableFloat
import main
import os
import sys
import pandas as pd 
import argparse
from revolve2.ci_group.logging import setup_logging
import logging
setup_logging()

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

N_RUNS = 10
def objective(config: RevolveNeatConfig) -> OptimizationRun:
    """Define the objective.
    :param config: the config that should be evaluated
    """
    fitness = 0.0
    stats = [] #store the statistics for each iteration
    for i in range(N_RUNS):
        iter_fitness, iter_stats = main.main(plot=False, config = config, verbose=0)
        fitness += iter_fitness
        stats.append(iter_stats)
       
    mbf = fitness/N_RUNS

    return OptimizationRun(
        utility = mbf,
        statistics = stats
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--spot', default=0, type=int)  #wether to use spot
    parser.add_argument('--name', default=f"tuneresult_{datetime.now().strftime('%Y-%m-%d-%H:%M')}", type=str)  #path to save result
    parser.add_argument('--timeout', default=600, type=int)  #path to save result
    args = parser.parse_args()

    logging.info(f"Running with args: {args.spot=}, {args.name=}, {args.timeout=}")

    if not bool(args.spot):
        #Optuna tuner
        logging.info("Using Optuna tuner")
        tuner = OptunaHyperOptimizer(
            objective=objective,
            bias_mutate_power = TunableFloat(0.1,2),
            bias_replace_rate = TunableFloat(0.0,1),
            conn_add_prob = TunableFloat(0.0,1),
            conn_delete_prob = TunableFloat(0.0,1),
        )
    else:
        #Spot
        logging.info("Using Spot tuner")
        tuner = SpotHyperOptimizer(
            objective=objective,
            bias_mutate_power = TunableFloat(0.1,2),
            bias_replace_rate = TunableFloat(0.0,1),
            conn_add_prob = TunableFloat(0.0,1),
            conn_delete_prob = TunableFloat(0.0,1),
        )

    result = tuner.run(timeout=args.timeout)
    result.save(args.name)

    loaded = OptimizationResult.load(args.name)
    print(loaded.summary())

