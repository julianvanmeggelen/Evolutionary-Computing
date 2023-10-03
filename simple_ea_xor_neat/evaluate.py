"""Evaluation functions. Uses neat-python nn.feedforward"""

import numpy as np
import numpy.typing as npt

import neat 
from neat.config import Config
from neat.genome import DefaultGenome


config: Config = None 
def evaluate(genotype, config) -> float:
    #print(genotype.neatGenome)
    """
    Measure one set of parameters.

    :param parameters: The parameters to measure.
    :returns: Negative sum of squared errors and each individual error. 5x1 floats.
    """
    # Define all possible inputs for xor and the expected outputs
    inputs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    expected_outputs = np.array([0, 1, 1, 0])

    # Evaluate the provided network parameters
    net = neat.nn.FeedForwardNetwork.create(genotype.neatGenome, config)
    outputs = np.array([net.activate(input) for input in inputs])
    #print(outputs)

    # Calculate the difference between the network outputs and the expect outputs
    errors = outputs - expected_outputs
    print(np.mean(outputs.round() == expected_outputs))

    # Return the sum of squared errors.
    # 0 would be an optimizal result.
    # We invert so we can maximize the fitness instead of minimize.
    # Finally we convert from a numpy float_ type to the python float type. This is not really important.

    fitness =  - float(np.sum(errors**2))
    genotype.neatGenome.fitness = fitness
    return fitness
