from dataclasses import dataclass


from hyper_parameter_optimization.config.neat_genome_config import NeatGenomeConfig, DefaultNeatGenomeConfig


@dataclass
class RevolveNeatConfig:
    # General config
    POPULATION_SIZE:int 
    OFFSPRING_SIZE:int 
    NUM_EVALS:int 
    NUM_GENERATIONS:int 

    #Neat genome config
    genome_config: NeatGenomeConfig


class DefaultRevolveNeatConfig(RevolveNeatConfig):
    """
    Create a default configuration
    """
    POPULATION_SIZE:int = 50 
    OFFSPRING_SIZE:int = 10
    NUM_EVALS:int = 200
    NUM_GENERATIONS:int = 100 

    #Neat genome config
    genome_config: NeatGenomeConfig = DefaultNeatGenomeConfig()


