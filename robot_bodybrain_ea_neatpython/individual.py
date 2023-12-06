"""Individual class."""

from dataclasses import dataclass

from genotype.genotype import Genotype


class Individual:
    """An individual in a population."""

    def __init__(self, genotype: Genotype, fitness: float) -> None:
        self.genotype: Genotype = genotype
        self.fitness: float = fitness
        self.genotype.body.neatGenome.fitness = fitness
        self.genotype.brain.neatGenome.fitness = fitness
