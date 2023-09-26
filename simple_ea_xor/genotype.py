import numpy.typing as npt
import numpy as np
from typing import Optional
import config

class Genotype:
    rng: np.random.Generator 

    def __init__(self, params: npt.NDArray[np.float_]):
        self.params = params

    def mutate(self):
        return Genotype(
            self.rng.normal(scale=config.MUTATE_STD, size=config.NUM_PARAMETERS)
            + self.params
        )
    
    @classmethod
    def random(cls):
        return Genotype(cls.rng.random(size=config.NUM_PARAMETERS) * 2 - 1)
    
    @classmethod
    def crossOver(cls, a, b):
        mask = cls.rng.random(config.NUM_PARAMETERS)
        return Genotype(np.where(mask, a.params, b.params))


