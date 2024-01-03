from datetime import datetime
from sklearn import base
from hyper_parameter_optimization.config.revolve_neat_config import RevolveNeatConfig
from hyper_parameter_optimization.optimizer.optimizer import (
    OptunaHyperOptimizer,
    SpotHyperOptimizer
)
from hyper_parameter_optimization.result.optimization_run import OptimizationRun
from hyper_parameter_optimization.result.optimization_result import OptimizationResult
from hyper_parameter_optimization.optimizer.tunable_param import TunableFloat, TunableCategory
import main
import os
import sys
import pandas as pd
import argparse
from revolve2.experimentation.logging import setup_logging
import logging

setup_logging()

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

N_RUNS = int(os.getenv("NRUNS", 10))
N_GENERATIONS = int(os.getenv('NGEN', 100))


def objective(config: RevolveNeatConfig) -> OptimizationRun:
    """Define the objective.
    :param config: the config that should be evaluated
    """
    fitness = 0.0
    stats = []  # store the statistics for each iteration
    for i in range(N_RUNS):
        iter_fitness, iter_stats = main.main(plot=False, config=config, verbose=0)
        fitness += iter_fitness
        stats.append(iter_stats)

    mbf = fitness / N_RUNS

    return OptimizationRun(utility=mbf, statistics=stats)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--spot", default=0, type=int)  # wether to use spot
    parser.add_argument(
        "--name",
        default=f"tuneresult_{datetime.now().strftime('%Y-%m-%d-%H:%M')}",
        type=str,
    )  # path to save result
    parser.add_argument("--timeout", default=600, type=int)  # path to save result
    args = parser.parse_args()

    logging.info(f"Running with args: {args.spot=}, {args.name=}, {args.timeout=}")
    logging.info(f"Using fitness function {os.getenv('FIT_FUN')}")

    base_config = RevolveNeatConfig(
        body_num_inputs=5, 
        body_num_outputs=5, 
        brain_num_inputs=7, 
        brain_num_outputs=1,
        NUM_GENERATIONS=N_GENERATIONS
    )


    tuner_type = OptunaHyperOptimizer if not bool(args.spot) else SpotHyperOptimizer
    tuner = tuner_type(
            objective=objective,
            config_template=base_config,
            node_delete_prob = TunableFloat(0.0, 0.2),
            node_add_prob = TunableFloat(0.0, 0.2),
            conn_add_prob=TunableFloat(0.0, 0.2),
            conn_delete_prob=TunableFloat(0.0, 0.2),
            bias_mutate_rate=TunableFloat(0.0, 0.2, init=0),
            weight_mutate_rate=TunableFloat(0.0, 1.0, init=0.9),
            activation_default=TunableCategory(["tanh","gauss","sin","identity"], init='tanh'),
            activation_mutate_rate=TunableFloat(0.0,1.0)
        )
    logging.info(f'Using tuner type {tuner_type.__name__}')

    result = tuner.run(timeout=args.timeout, n_jobs=1)
    if not os.path.isdir('./results'):
        os.mkdir('./results')
    save_path =  os.path.join('./results/', args.name)
    result.save(
       save_path
    )

    loaded = OptimizationResult.load(save_path)
    print(loaded.summary())
