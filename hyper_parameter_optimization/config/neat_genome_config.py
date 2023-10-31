from dataclasses import dataclass
from itertools import count

from neat.activations import ActivationFunctionSet
from neat.aggregations import AggregationFunctionSet
from neat.genes import DefaultConnectionGene, DefaultNodeGene


@dataclass
class NeatGenomeConfig:
    """
    The configuration options that are needed to pass to neat DefaultGenome objects
    """
    # activation
    activation_default: str
    activation_mutate_rate: float
    activation_options: str

    # node aggregation options
    aggregation_default: str
    aggregation_mutate_rate: float
    aggregation_options: str

    # node bias options
    bias_init_mean: float
    bias_init_stdev: float
    bias_max_value: float
    bias_min_value: float
    bias_mutate_power: float
    bias_mutate_rate: float
    bias_replace_rate: float

    # genome compatibility options
    compatibility_disjoint_coefficient: float
    compatibility_weight_coefficient: float

    # connection add/remove rates
    conn_add_prob: float
    conn_delete_prob: float

    # connection enable options
    enabled_default: bool
    enabled_mutate_rate: float
    feed_forward: bool
    initial_connection: str

    # node add/remove rates
    node_add_prob: float
    node_delete_prob: float

    # network parameters
    num_hidden: int
    num_inputs: int
    num_outputs: int

    # node response options
    response_init_mean: float
    response_init_stdev: float
    response_max_value: float
    response_min_value: float
    response_mutate_power: float
    response_mutate_rate: float
    response_replace_rate: float

    # connection weight options
    weight_init_mean: float
    weight_init_stdev: float
    weight_max_value: float
    weight_min_value: float
    weight_mutate_power: float
    weight_mutate_rate: float
    weight_replace_rate: float

    # these are properties that are not considered configuration by neat-python
    node_gene_type = DefaultNodeGene
    connection_gene_type = DefaultConnectionGene
    bias_init_type: str = "gaussian"
    response_init_type: str = "gaussian"
    weight_init_type: str = "gaussian"
    activation_defs: ActivationFunctionSet = ActivationFunctionSet()
    aggregation_function_defs: AggregationFunctionSet = AggregationFunctionSet()
    aggregation_defs = aggregation_function_defs
    single_structural_mutation = False
    enabled_rate_to_false_add: float = 0.0
    enabled_rate_to_true_add: float = 0.0
    structural_mutation_surer: str = "default"
    node_indexer: count = count(1)

    @property
    def output_keys(self):
        return [i for i in range(self.num_outputs)]

    @property
    def input_keys(self):
        return [-i - 1 for i in range(self.num_inputs)]

    def get_new_node_key(self, node_dict):
        return next(self.node_indexer)

    def check_structural_mutation_surer(self):
        if self.structural_mutation_surer == "true":
            return True
        elif self.structural_mutation_surer == "false":
            return False
        elif self.structural_mutation_surer == "default":
            return self.single_structural_mutation
        else:
            error_string = (
                f"Invalid structural_mutation_surer {self.structural_mutation_surer!r}"
            )
            raise RuntimeError(error_string)

class DefaultNeatGenomeConfig(NeatGenomeConfig):
    """
    Default settings (as provided by neat-python)
    """
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
