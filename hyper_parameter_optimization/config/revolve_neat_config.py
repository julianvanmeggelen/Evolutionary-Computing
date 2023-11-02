from dataclasses import dataclass, replace, asdict
from functools import cache
from hyper_parameter_optimization.config.neat_genome_config import (
    NeatGenomeConfigGenericMixin
)

@dataclass
class RevolveNeatConfig(NeatGenomeConfigGenericMixin):
    # General config
    POPULATION_SIZE: int = 50
    OFFSPRING_SIZE: int = 10
    NUM_EVALS: int = 200
    NUM_GENERATIONS: int = 100

    #Neat config 
    activation_default = "sigmoid"
    activation_mutate_rate = 0.0
    activation_options: str = "sigmoid"
    aggregation_default: str = "sum"
    aggregation_mutate_rate: float = 0.0
    aggregation_options: str = "sum"
    bias_init_mean: float = 0.0
    bias_init_stdev: float = 1.0
    bias_max_value: float = 30.0
    bias_min_value: float = -30.0
    bias_mutate_power: float = 0.5
    bias_mutate_rate: float = 0.7
    bias_replace_rate: float = 0.1
    compatibility_disjoint_coefficient = 1.0
    compatibility_weight_coefficient = 0.5
    conn_add_prob: float = 0.2
    conn_delete_prob: float = 0.2
    enabled_default: bool = True
    enabled_mutate_rate: float = 0.01
    feed_forward: bool = True
    initial_connection: str = "full_direct"
    node_add_prob: float = 0.2
    node_delete_prob: float = 0.2
    num_hidden: int = 1
    num_inputs: int = 4
    num_outputs: int = 1
    response_init_mean: float = 1.0
    response_init_stdev: float = 0.0
    response_max_value: float = 30.0
    response_min_value: float = -30.0
    response_mutate_power: float = 0.0
    response_mutate_rate: float = 0.0
    response_replace_rate: float = 0.0
    weight_init_mean: float = 0.0
    weight_init_stdev: float = 1.0
    weight_max_value: float = 30
    weight_min_value: float = -30
    weight_mutate_power: float = 0.5
    weight_mutate_rate: float = 0.8
    weight_replace_rate: float = 0.1
    single_structural_mutation = False
    enabled_rate_to_false_add: float = 0.0
    enabled_rate_to_true_add: float = 0.0
    structural_mutation_surer: str = "default"
    bias_init_type: str = "gaussian"
    response_init_type: str = "gaussian"
    weight_init_type: str = "gaussian"

    # Neat genome config (neat accesses the config via config.genome_config)
    @property
    def genome_config(self):
        return self

    def copy(self):
        return replace(self)

    def dict(self):
        return asdict(self)

    @cache
    def _slots(self):
        return [k for k in self.__dict__ if not k.startswith('_') or k == 'genome_config']

    def __contains__(self, key: str) -> bool:

        return key in self.__dict__


