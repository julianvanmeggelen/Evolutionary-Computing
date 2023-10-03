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
    
def objective(trial):
    config = Namespace()
    config.genome_config = DummyGenomeConfig()

    config.POPULATION_SIZE = trial.suggest_int("POP_SIZE", 50,200)
    config.OFFSPRING_SIZE = int(config.POPULATION_SIZE  *trial.suggest_float("OFFSPRING_FRAC", 0.1,1))
    config.NUM_EVALS = 5000
    config.NUM_GENERATIONS = config.NUM_EVALS / (config.POPULATION_SIZE + config.OFFSPRING_SIZE)


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
    config.genome_config.bias_mutate_power       = 0.5
    config.genome_config.bias_mutate_rate        = 0.7
    config.genome_config.bias_replace_rate       = 0.1
    # genome compatibility options
    config.genome_config.compatibility_disjoint_coefficient = 1.0
    config.genome_config.compatibility_weight_coefficient   = 0.5
    # connection add/remove rates
    config.genome_config.conn_add_prob           = 0.5
    config.genome_config.conn_delete_prob        = 0.5
    # connection enable options
    config.genome_config.enabled_default         = True
    config.genome_config.enabled_mutate_rate     = 0.01
    config.genome_config.feed_forward            = True
    config.genome_config.initial_connection      = "full"
    # node add/remove rates
    config.genome_config.node_add_prob           = 0.2
    config.genome_config.node_delete_prob        = 0.2
    # network parameters
    config.genome_config.num_hidden              = 1
    config.genome_config.num_inputs              = 2
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
    for i in range(10):
        fitness += main.main(callback=epoch_callback, plot=False, config_in = config)
    return fitness/10
    

def hyperOpt():
    pruner = None; #optuna.pruners.PercentilePruner(0.5, n_startup_trials=5, n_warmup_steps=20)
    study = optuna.create_study(pruner=pruner,direction="maximize", study_name=f'test{uuid.uuid4()}')#, storage="sqlite:///hyperopt_results.sqlite")
    study.optimize(objective, n_trials=50, timeout=600)

    return study


if __name__ == "__main__":
    hyperOpt()
