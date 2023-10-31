"""
Optimize a neural network to calculate XOR, using an evolutionary algorithm.

To better understand, first look at the 'experiment_setup' example.

You learn:
- How to create a simple evolutionary loop.
"""

import logging
import math

import numpy as np
import numpy.typing as npt
from evaluate import evaluate
from genotype import Genotype
from individual import Individual
from revolve2.ci_group.logging import setup_logging
from revolve2.ci_group.rng import make_rng_time_seed
from revolve2.experimentation.optimization.ea import population_management, selection
import pickle

import neat


def select_parents(
    rng: np.random.Generator,
    population: list[Individual],
    offspring_size: int,
) -> npt.NDArray[np.float_]:
    """
    Select pairs of parents using a tournament.

    :param rng: Random number generator.
    :param population: The population to select from.
    :param offspring_size: The number of parent pairs to select.
    :returns: Pairs of indices of selected parents. offspring_size x 2 ints.
    """
    return np.array(
        [
            selection.multiple_unique(
                2,
                [individual.genotype for individual in population],
                [individual.fitness for individual in population],
                lambda _, fitnesses: selection.tournament(rng, fitnesses, k=1),
            )
            for _ in range(offspring_size)
        ],
    )


def select_survivors(
    rng: np.random.Generator,
    original_population: list[Individual],
    offspring_population: list[Individual],
) -> list[Individual]:
    """
    Select survivors using a tournament.

    :param rng: Random number generator.
    :param original_population: The population the parents come from.
    :param offspring_population: The offspring.
    :returns: A newly created population.
    """
    original_survivors, offspring_survivors = population_management.steady_state(
        [i.genotype for i in original_population],
        [i.fitness for i in original_population],
        [i.genotype for i in offspring_population],
        [i.fitness for i in offspring_population],
        lambda n, genotypes, fitnesses: selection.multiple_unique(
            n,
            genotypes,
            fitnesses,
            lambda _, fitnesses: selection.tournament(rng, fitnesses, k=2),
        ),
    )

    return [
        Individual(
            original_population[i].genotype,
            original_population[i].fitness,
        )
        for i in original_survivors
    ] + [
        Individual(
            offspring_population[i].genotype,
            offspring_population[i].fitness,
        )
        for i in offspring_survivors
    ]


def init_neat(config=None):
    """This is needed to set the neat config
    """
    import os
    if config is None:
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config-feedforward')
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation,
            config_path)
    evaluate.config  = config
    Genotype.config = config.genome_config
    return config

def main(plot=True, config_in=None, callback=None, save_winner = False, verbose=1) -> None:
    """Run the program."""
    # Set up standard logging.
    setup_logging()

    if config_in is None:
        global config; import config as config # use config.py
        neat_config = init_neat()
        logging.info("Using config.py")
    else:
        global config; config = config_in #use provided config
        neat_config = init_neat(config_in)
        logging.info("Using config_in")



    #print('\n'.join([f'{k}: {v}' for k,v in neat_config.genome_config.__dict__.items() if "__" not in k]))

    # Set up the random number generater.
    rng = make_rng_time_seed()

    # Create an initial population.
    logging.info("Generating initial population.")
    initial_genotypes = [
        Genotype.random(
            rng=rng,
        )
        for _ in range(config.POPULATION_SIZE)
    ]

    # Evaluate the initial population.
    if verbose > 0: logging.info("Evaluating initial population.")
    initial_fitnesses = [
        evaluate(genotype, neat_config) for genotype in initial_genotypes
    ]

    # Create a population of individuals, combining genotype with fitness.
    population = [
        Individual(genotype, fitness)
        for genotype, fitness in zip(initial_genotypes, initial_fitnesses)
    ]

    # Set the current generation to 0.
    generation_index = 0

    # Start the actual optimization process.
    if verbose > 0: logging.info("Start optimization process.")
    mins, maxs, means = [], [], []
    while generation_index < config.NUM_GENERATIONS:
        if verbose > 0: logging.info(f"Generation {generation_index + 1} / {config.NUM_GENERATIONS}.")
        maxs.append(np.max([_.fitness for _ in population]))
        mins.append(np.min([_.fitness for _ in population]))
        means.append(np.mean([_.fitness for _ in population]))

        # Create offspring.
        parents = select_parents(rng, population, config.OFFSPRING_SIZE)
        offspring_genotypes = [
            Genotype.crossover(
                population[parent1_i].genotype,
                population[parent2_i].genotype,
                rng,
            ).mutate(rng)
            for parent1_i, parent2_i in parents
        ]

        # Evaluate the offspring.
        offspring_fitnesses = [
            evaluate(genotype, neat_config) for genotype in offspring_genotypes
        ]

        if verbose > 0: logging.info(f"Max fitness: {maxs[-1]}")

        # Make an intermediate offspring population.
        offspring_population = [
            Individual(genotype, fitness)
            for genotype, fitness in zip(offspring_genotypes, offspring_fitnesses)
        ]

        # Create the next generation by selecting survivors between original population and offspring.
        population = select_survivors(
            rng,
            population,
            offspring_population,
        )

        # Increase the generation index counter.
        generation_index += 1
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(maxs, label = 'max'); plt.plot(mins, label = 'min'); plt.plot(means, label = 'avg');
        plt.xlabel("Generation"); plt.ylabel("fitness"); plt.legend()
        plt.show()

    if save_winner:
        winner = max(population, key=lambda individual: individual.fitness)
        with open('winner-feedforward', 'wb') as f:
            pickle.dump(winner, f)

    net_size_nodes = [len(ind.genotype.neatGenome.nodes) for ind in population] 
    net_size_connections = [len(ind.genotype.neatGenome.connections) for ind in population]
    all_fitness = [ind.fitness for ind in population]
    return maxs[-1], maxs, means, mins, net_size_nodes, net_size_connections, all_fitness



if __name__ == "__main__":
    main(save_winner=True)
