from dataclasses import dataclass
from hyper_parameter_optimization.config.neat_genome_config import (
    NeatGenomeConfig,
    DefaultNeatGenomeConfig,
)

NON_NEAT_ARGS = ["POPULATION_SIZE", "OFFSPRING_SIZE", "NUM_EVALS", "NUM_GENERATIONS"]


@dataclass
class RevolveNeatConfig:
    # General config
    POPULATION_SIZE: int
    OFFSPRING_SIZE: int
    NUM_EVALS: int
    NUM_GENERATIONS: int

    # Neat genome config
    genome_config: NeatGenomeConfig


@dataclass
class DefaultRevolveNeatConfig:
    """
    Create a default configuration
    """

    POPULATION_SIZE: int = 50
    OFFSPRING_SIZE: int = 10
    NUM_EVALS: int = 200
    NUM_GENERATIONS: int = 100

    # Neat genome config
    genome_config: NeatGenomeConfig = DefaultNeatGenomeConfig()
