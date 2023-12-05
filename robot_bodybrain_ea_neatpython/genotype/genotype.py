from __future__ import annotations
from abc import ABC
from itertools import count
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkNeighbor
from revolve2.modular_robot.body.base import ActiveHinge, Body
from hyper_parameter_optimization.config.revolve_neat_config import RevolveNeatConfig
import neat
import numpy as np
from neat.genome import DefaultGenome, DefaultGenomeConfig

from genotype.body import BodyGenotype
from genotype.brain import BrainGenotype

class Genotype():
    def __init__(self, body: BodyGenotype, brain: BrainGenotype) -> None:
        self.body = body
        self.brain = brain

    @classmethod
    def random(cls, rng, config: RevolveNeatConfig) -> Genotype:
        """Initialize the body and brain"""
        return Genotype(
            body=BodyGenotype.random(rng, config.body_config()),
            brain=BrainGenotype.random(rng, config.brain_config())
        )
    
    def mutate(self, rng) -> Genotype:
        """Mutate the body and brain"""
        return Genotype(
            body=self.body.mutate(rng),
            brain=self.brain.mutate(rng)
        )
    
    @classmethod
    def crossover(self, parent1, parent2, rng, config: RevolveNeatConfig ) -> Genotype:
        """Do crossover on the body and brain"""
        return Genotype(
            body=BodyGenotype.crossover(parent1.body, parent2.body, rng, config.body_config()),
            brain=BrainGenotype.crossover(parent1.brain, parent2.brain, rng, config.brain_config())
        )

    def develop(self) -> ModularRobot: 
        body = self.body.develop()
        return ModularRobot(
            body = body,
            brain = self.brain.develop(body)
        )
        


