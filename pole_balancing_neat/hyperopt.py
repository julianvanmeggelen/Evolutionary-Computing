from dataclasses import dataclass
from argparse import Namespace
import optuna
from typing import TypedDict
from argparse import Namespace
from datetime import datetime
import uuid 
from itertools import count
import main

from neat.activations import ActivationFunctionSet
from neat.aggregations import AggregationFunctionSet


from neat.genes import DefaultConnectionGene, DefaultNodeGene







class DummyGenomeConfig:
    def __init__(self):
        self.node_indexer = count(1)

    def get_new_node_key(self, node_dict):
        return next(self.node_indexer)
    
    def check_structural_mutation_surer(self):
        if self.structural_mutation_surer == 'true':
            return True
        elif self.structural_mutation_surer == 'false':
            return False
        elif self.structural_mutation_surer == 'default':
            return self.single_structural_mutation
        else:
            error_string = f"Invalid structural_mutation_surer {self.structural_mutation_surer!r}"
            raise RuntimeError(error_string)


def callback(trial, fitness, evals):
    trial.report(fitness, evals)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    
def objective(trial, study_name):
    config = Namespace()
    config.genome_config = DummyGenomeConfig()


    #These are revole parameters
    config.POPULATION_SIZE = 50#trial.suggest_int("POP_SIZE", 10,50)
    config.OFFSPRING_SIZE = 10 #int(config.POPULATION_SIZE  * trial.suggest_float("OFFSPRING_FRAC", 0.1,1))
    config.NUM_EVALS = 200
    config.NUM_GENERATIONS = 100 #(config.NUM_EVALS-config.POPULATION_SIZE) / config.OFFSPRING_SIZE


    #These are normally provided in neat's config file
    config.genome_config.activation_default      = "sigmoid"
    config.genome_config.activation_mutate_rate  = 0.0
    config.genome_config.activation_options      = "sigmoid"
    # node aggregation options
    config.genome_config.aggregation_default     = "sum"
    config.genome_config.aggregation_mutate_rate = 0.0
    config.genome_config.aggregation_options     = "sum"
    # node bias options
    config.genome_config.bias_init_mean          = 0.0
    config.genome_config.bias_init_stdev         = 1.0
    config.genome_config.bias_max_value          = 30.0
    config.genome_config.bias_min_value          = -30.0
    config.genome_config.bias_mutate_power       = trial.suggest_float("bias_mutate_power", 0.1,2)
    config.genome_config.bias_mutate_rate        = 0.7
    config.genome_config.bias_replace_rate       = trial.suggest_float("bias_replace_rate", 0.0,1)
    # genome compatibility options
    config.genome_config.compatibility_disjoint_coefficient = 1.0
    config.genome_config.compatibility_weight_coefficient   = 0.5
    # connection add/remove rates
    config.genome_config.conn_add_prob           = trial.suggest_float("conn_add_prob", 0.0,1)
    config.genome_config.conn_delete_prob        = trial.suggest_float("conn_delete_prob", 0.0,1)
    # connection enable options
    config.genome_config.enabled_default         = True
    config.genome_config.enabled_mutate_rate     = 0.01
    config.genome_config.feed_forward            = True
    config.genome_config.initial_connection      = "full_direct"
    # node add/remove rates
    config.genome_config.node_add_prob           = trial.suggest_float("node_add_prob", 0.1,1)
    config.genome_config.node_delete_prob        = trial.suggest_float("node_delete_prob", 0.1,1)
    # network parameters
    config.genome_config.num_hidden              = 1
    config.genome_config.num_inputs              = 4
    config.genome_config.num_outputs             = 1
    # node response options
    config.genome_config.response_init_mean      = 1.0
    config.genome_config.response_init_stdev     = 0.0
    config.genome_config.response_max_value      = 30.0
    config.genome_config.response_min_value      = -30.0
    config.genome_config.response_mutate_power   = 0.0
    config.genome_config.response_mutate_rate    = 0.0
    config.genome_config.response_replace_rate   = 0.0
    # connection weight options
    config.genome_config.weight_init_mean        = 0.0
    config.genome_config.weight_init_stdev       = 1.0
    config.genome_config.weight_max_value        = 30
    config.genome_config.weight_min_value        = -30
    config.genome_config.weight_mutate_power     = 0.5
    config.genome_config.weight_mutate_rate      = 0.8
    config.genome_config.weight_replace_rate     = 0.1

    #these are needed besides the params that come from the file
    config.genome_config.output_keys = [i for i in range(config.genome_config.num_outputs)]
    config.genome_config.input_keys = [-i - 1 for i in range(config.genome_config.num_inputs)]
    config.genome_config.node_gene_type = DefaultNodeGene
    config.genome_config.connection_gene_type   = DefaultConnectionGene
    config.genome_config.bias_init_type = "gaussian"
    config.genome_config.response_init_type = "gaussian"
    config.genome_config.weight_init_type = "gaussian"
    config.genome_config.activation_defs = ActivationFunctionSet()
    config.genome_config.aggregation_function_defs = AggregationFunctionSet()
    config.genome_config.aggregation_defs =  config.genome_config.aggregation_function_defs
    config.genome_config.single_structural_mutation= False
    config.genome_config.enabled_rate_to_false_add= 0.0
    config.genome_config.enabled_rate_to_true_add=0.0
    config.genome_config.structural_mutation_surer='default'




    epoch_callback = None #lambda fitness, evals: callback(trial, fitness,evals)
    fitness = 0.0
    n_runs= 20
    mins, maxs, means, sizes_nodes, sizes_connections, all_fitnesses = [],[],[],[],[],[]
    for i in range(n_runs):
        iter_fitness, maxs_curr, means_curr, mins_curr, net_size_nodes, net_size_connections, all_fitness = main.main(callback=epoch_callback, plot=False, config_in = config, verbose=0)
        fitness += iter_fitness
        mins.append(mins_curr)
        maxs.append(maxs_curr)
        means.append(means_curr)
        sizes_connections.append(net_size_connections)
        sizes_nodes.append(net_size_nodes)
        all_fitnesses.append(all_fitness)

    fitness_avg = fitness/n_runs

    save_trial(trial._trial_id, config, maxs, mins, means, sizes_nodes, sizes_connections, all_fitnesses, study_name)
    return fitness_avg

def hyperOpt(timeout=1200, name=None):
    pruner = None; #optuna.pruners.PercentilePruner(0.5, n_startup_trials=5, n_warmup_steps=20)
    study_name = name or f'test1{uuid.uuid4()}'
    study = optuna.create_study(pruner=pruner,direction="maximize", study_name=study_name, load_if_exists=True, storage="sqlite:///hyperopt_results.sqlite")
    study.optimize(lambda trial: objective(trial, study_name), n_trials=50, timeout=timeout, n_jobs=1)
    return study

def hyperOptParallel():
    from joblib import Parallel, delayed
    name = f'test1{uuid.uuid4()}'
    Parallel(n_jobs=-1)(delayed(hyperOpt)(1200) for i in range(4))

def get_dir(trial_id, study_name):
    import os
    save_dir  = './trial_results'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    study_dir = f'{save_dir}/{study_name}'
    if not os.path.exists(study_dir):
        os.mkdir(study_dir)
    trial_dir = f'{study_dir}/{trial_id}'
    if not os.path.exists(trial_dir):
        os.mkdir(trial_dir)
    return trial_dir


def save_trial(trial_id, config, maxs, mins, means, sizes_nodes, sizes_connections, all_fitnesses, study_name):
    import os 
    import pickle
    import numpy as np
    import logging

    print([len(_) for _ in mins])

    trial_dir = get_dir(trial_id, study_name)
    with open(f'{trial_dir}/config.pkl', 'wb') as f:
        pickle.dump(config, f)
    np.save(f'{trial_dir}/mins.npy', np.stack(mins))
    np.save(f'{trial_dir}/means.npy', np.stack(means))
    np.save(f'{trial_dir}/maxs.npy', np.stack(maxs))
    np.save(f'{trial_dir}/net_size_nodes.npy', sizes_nodes)
    np.save(f'{trial_dir}/net_size_connections.npy', sizes_connections)
    np.save(f'{trial_dir}/all_fitness.npy', all_fitnesses)
    logging.info(f"Saved trial results to {trial_dir}")

    

if __name__ == "__main__":
    hyperOpt()
