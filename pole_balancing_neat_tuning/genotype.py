"""Genotype class. This borrows the gene from"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from itertools import count
import copy
from neat.genome import DefaultGenome, DefaultGenomeConfig
from hyper_parameter_optimization.config.revolve_neat_config import RevolveNeatConfig


class Genotype:
    indexer = count(1)
    def __init__(self, neatGenome: DefaultGenome, config: RevolveNeatConfig):
        self.config: DefaultGenomeConfig = config.genome_config
        self.neatGenome = neatGenome

    @classmethod
    def random(
        cls,
        rng: np.random.Generator,
        config: RevolveNeatConfig
    ) -> Genotype:
        """
        Create a random genotype.

        :param rng: Random number generator. WARNING: not used
        :returns: The created genotype.
        """
        ret = DefaultGenome(next(cls.indexer))
        ret.configure_new(config)
        return Genotype(ret, config)

    def mutate(
        self,
        rng: np.random.Generator
    ) -> Genotype:
        """
        Mutate this genotype.

        This genotype will not be changed; a mutated copy will be returned.

        :param rng: Random number generator.
        :returns: A mutated copy of the provided genotype.
        """
        neatGenomeCopy = copy.deepcopy(self.neatGenome)
        neatGenomeCopy.key = next(self.indexer)
        neatGenomeCopy.mutate(self.config)
        return Genotype(
            neatGenomeCopy,
            self.config
        )

    @classmethod
    def crossover(
        cls,
        parent1: Genotype,
        parent2: Genotype,
        rng: np.random.Generator,
        config: RevolveNeatConfig
    ) -> Genotype:
        """
        Perform uniform crossover between two genotypes.

        :param parent1: The first genotype.
        :param parent2: The second genotype.
        :param rng: Random number generator.
        :returns: A newly created genotype.
        """
        child = DefaultGenome(next(cls.indexer))
        child.configure_crossover(parent1.neatGenome, parent2.neatGenome, config)
        return Genotype(child, config)
    


