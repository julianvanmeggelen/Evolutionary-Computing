"""Evaluation functions. Uses neat-python nn.feedforward"""

import numpy as np
import numpy.typing as npt

import neat 
from neat.config import Config
from neat.genome import DefaultGenome

import gymnasium as gym
from tqdm import tqdm

from joblib import Parallel, delayed
import flappy_bird_gymnasium
import gymnasium
env = gymnasium.make("FlappyBird-v0")


config: Config = None 


"""
Single-pole balancing experiment using a feed-forward neural network.
"""
import neat

runs_per_net = 1
max_steps = float('inf')#5000


def process_state(state):
    #print(state)
    dy_bottom = state[[9]] - state[[5]] #difference between player y and pipe y
    dy_top= state[[9]] - state[[4]] #difference between player y and pipe y

    #ret =  np.concatenate([state, dy_bottom, dy_top])
    ret =  np.concatenate([dy_top, dy_bottom, state[[3]]])
    #print(ret)
    return ret

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
        observation, observation_init_info = env.reset(seed=69)
        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        step = 0
        while True:
            step+=1
            #print(step)
            output = np.round(net.activate(process_state(observation)))
            #print(output)
          
            observation, reward, terminated, done, info = env.step(output)
            fitness += reward
            if terminated or step > max_steps:
                break

        fitnesses.append(fitness)

    fitness = min(fitnesses)**2

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