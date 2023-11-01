from hyper_parameter_optimization.config.revolve_neat_config import RevolveNeatConfig
from hyper_parameter_optimization.optimizer.optimizer import OptunaHyperOptimizer, SpotHyperOptimizer
from hyper_parameter_optimization.result.optimization_run import OptimizationRun
from hyper_parameter_optimization.result.optimization_result import OptimizationResult
from hyper_parameter_optimization.optimizer.tunable_param import TunableFloat

import main

import os

import sys
#sys.path.append('../')

N_RUNS = 10
def objective(config: RevolveNeatConfig):
    fitness = 0.0
    mins, maxs, means, sizes_nodes, sizes_connections, all_fitnesses = [],[],[],[],[],[]

    for i in range(N_RUNS):
        iter_fitness, maxs_curr, means_curr, mins_curr, net_size_nodes, net_size_connections, all_fitness = main.main(plot=False, config_in = config, verbose=0)
        fitness += iter_fitness
        mins.append(mins_curr)
        maxs.append(maxs_curr)
        means.append(means_curr)
        sizes_connections.append(net_size_connections)
        sizes_nodes.append(net_size_nodes)
        all_fitnesses.append(all_fitness)

    return OptimizationRun(
        utility = fitness/N_RUNS,
        statistics = dict(
            mins=mins,
            maxs=maxs,
            means=means,
            sizes_connections = sizes_connections,
            sizes_nodes=sizes_nodes,
            all_fitnesses = all_fitnesses
        )
    )

if __name__ == "__main__":

    optuna=False
    if optuna:
        #Optuna tuner
        tuner = OptunaHyperOptimizer(
            objective=objective,
            bias_mutate_power = TunableFloat(0.1,2),
            bias_replace_rate = TunableFloat(0.0,1),
            conn_add_prob = TunableFloat(0.0,1),
            conn_delete_prob = TunableFloat(0.0,1),
        )
    else:
        #Spot
        tuner = SpotHyperOptimizer(
            objective=objective,
            bias_mutate_power = TunableFloat(0.1,2),
            bias_replace_rate = TunableFloat(0.0,1),
            conn_add_prob = TunableFloat(0.0,1),
            conn_delete_prob = TunableFloat(0.0,1),
        )

    result = tuner.run(timeout=20)
    result.save('tuneresult')

    loaded = OptimizationResult.load('tuneresult')
    print(loaded.best_run().config)

