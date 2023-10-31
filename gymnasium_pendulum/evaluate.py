"""Evaluation functions. Uses neat-python nn.feedforward"""

import numpy as np
import numpy.typing as npt

import neat 
from neat.config import Config
from neat.genome import DefaultGenome

import gymnasium as gym
from tqdm import tqdm

from joblib import Parallel, delayed


env = gym.make('Pendulum-v1', g=9.81)

config: Config = None 


"""
Single-pole balancing experiment using a feed-forward neural network.
"""
import neat

runs_per_net = 1
max_steps = 500


def evaluate(genotype, config) -> float:
    #print(genotype.neatGenome)
    """
    Measure one set of parameters.

    :param parameters: The parameters to measure.
    :returns: Negative sum of squared errors and each individual error. 5x1 floats.
    """
    
    net = neat.nn.FeedForwardNetwork.create(genotype.neatGenome, config)
    fitnesses = []
    for runs in range(runs_per_net):
        observation, observation_init_info = env.reset()
        # Run the given simulation for up to num_steps time steps.
        fitness = -float('inf')
        step = 0
        while True:
            step+=1
            #print(step)
            output = np.array(net.activate(observation)) * 2-1 #because we need (-2,2)
            #print(output)
          
            observation, reward, terminated, done, info = env.step(output)
            #fitness = reward if reward > fitness else fitness
            if terminated or step > max_steps:
                break

        fitnesses.append(reward)

    fitness = min(fitnesses)

    genotype.neatGenome.fitness = fitness
    return fitness


def evaluate_genotypes(genotypes, config):
    # ret = []
    # for geno in tqdm(genotypes):
    #     ret.append(evaluate(geno, config))
    ret = Parallel(n_jobs=-1, backend='loky', verbose=0)(delayed(evaluate)(gen, config) for gen in genotypes)
    for gen, fitness in zip(genotypes, ret):
        gen.neatGenome.fitness = fitness
    return ret