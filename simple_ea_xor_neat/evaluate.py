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
    inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    expected_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]

    # Evaluate the provided network parameters
    net = neat.nn.FeedForwardNetwork.create(genotype.neatGenome, config)
    mse = []
    for xi, xo in zip(inputs, expected_outputs):
        output = net.activate(xi)
        mse.append((output[0] - xo[0]) ** 2)
    
    fitness = -np.mean(mse)

    #print(len(genotype.neatGenome.connections))

    # Calculate the difference between the network outputs and the expect outputs
    #print(np.mean(outputs.round() == expected_outputs))

    # Return the sum of squared errors.
    # 0 would be an optimizal result.
    # We invert so we can maximize the fitness instead of minimize.
    # Finally we convert from a numpy float_ type to the python float type. This is not really important.

    genotype.neatGenome.fitness = fitness
    return fitness
