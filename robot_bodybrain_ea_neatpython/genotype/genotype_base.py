from __future__ import annotations

import copy
from itertools import count

from hyper_parameter_optimization.config.revolve_neat_config import RevolveNeatConfig
import neat
import numpy as np
from neat.genome import DefaultGenome, DefaultGenomeConfig
from typing_extensions import Self

indexer = count(1)


class BaseNeatGenotype:
    """Most basic neat-python based genotype, meant for use as a baseclass"""

    indexer = count(1)

    def __init__(self, neatGenome: DefaultGenome, config: RevolveNeatConfig):
        self.config: DefaultGenomeConfig = config.genome_config
        self.neatGenome = neatGenome

    @classmethod
    def random(
        cls, rng: np.random.Generator, config: RevolveNeatConfig
    ) -> BaseNeatGenotype:
        """
        Create a random genotype.

        :param rng: Random number generator. WARNING: not used
        :returns: The created genotype.
        """
        ret = DefaultGenome(next(indexer))
        ret.configure_new(config)
        return cls(ret, config)

    def mutate(self, rng: np.random.Generator) -> Self:
        """
        Mutate this genotype.

        This genotype will not be changed; a mutated copy will be returned.

        :param rng: Random number generator.
        :returns: A mutated copy of the provided genotype.
        """
        neatGenomeCopy = copy.deepcopy(self.neatGenome)
        neatGenomeCopy.key = next(self.indexer)
        neatGenomeCopy.mutate(self.config)
        return type(self)(neatGenomeCopy, self.config)

    @classmethod
    def crossover(
        cls,
        parent1: BaseNeatGenotype,
        parent2: BaseNeatGenotype,
        rng: np.random.Generator,
        config: RevolveNeatConfig,
    ) -> BaseNeatGenotype:
        """
        Perform uniform crossover between two genotypes.

        :param parent1: The first genotype.
        :param parent2: The second genotype.
        :param rng: Random number generator.
        :returns: A newly created genotype.
        """
        child = DefaultGenome(next(cls.indexer))
        child.configure_crossover(parent1.neatGenome, parent2.neatGenome, config)
        return cls(child, config)
