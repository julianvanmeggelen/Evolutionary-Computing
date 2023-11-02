from dataclasses import dataclass
from itertools import count

from neat.activations import ActivationFunctionSet
from neat.aggregations import AggregationFunctionSet
from neat.genes import DefaultConnectionGene, DefaultNodeGene


class NeatGenomeConfigGenericMixin:
    # these are properties that are not considered configuration by neat-python
    node_gene_type = DefaultNodeGene
    connection_gene_type = DefaultConnectionGene
    activation_defs: ActivationFunctionSet = ActivationFunctionSet()
    aggregation_function_defs: AggregationFunctionSet = AggregationFunctionSet()
    aggregation_defs = aggregation_function_defs
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
