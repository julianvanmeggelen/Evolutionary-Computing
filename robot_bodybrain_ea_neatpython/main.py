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
from genotype.genotype import Genotype
from individual import Individual
from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.rng import make_rng_time_seed
from revolve2.experimentation.optimization.ea import population_management, selection

from hyper_parameter_optimization.config.revolve_neat_config import RevolveNeatConfig


import pickle

import neat

from evaluator import Evaluator

# Set up standard logging.
setup_logging()


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


def main(
    config: RevolveNeatConfig, plot=True, callback=None, save_winner=False, verbose=1
) -> None:
    """Does a single optimization run.

    :returns:
        fitness: best fitness in the last generation
        stats: A dictionary with keys:
            mins: The minimum population fitness over the generations
            maxs: The maximum population fitness over the generations
            means: The mean population fitness over the generations
            net_size_nodes: The sizes (measured in #nodes) of the nets in the last generation
            net_size_connections = The sizes (measured in #connections) of the nets in the last generation
            all_fitness: A list with all the fitness values of the individuals in the last generation
    """

    evaluator = Evaluator(headless=True, num_simulators=config.NUM_SIMULATORS)

    # Set up the random number generater.
    rng = make_rng_time_seed()

    # Create an initial population.
    logging.info("Generating initial population.")
    initial_genotypes = [
        Genotype.random(rng=rng, config=config) for _ in range(config.POPULATION_SIZE)
    ]

    # Evaluate the initial population.
    if verbose > 0:
        logging.info("Evaluating initial population.")
    initial_fitnesses = evaluator.evaluate(
        [genotype.develop() for genotype in initial_genotypes]
    )

    # Create a population of individuals, combining genotype with fitness.
    population = [
        Individual(genotype, fitness)
        for genotype, fitness in zip(initial_genotypes, initial_fitnesses)
    ]

    # Set the current generation to 0.
    generation_index = 0

    # Start the actual optimization process.
    if verbose > 0:
        logging.info("Start optimization process.")
    mins, maxs, means = [], [], []
    while generation_index < config.NUM_GENERATIONS:
        if verbose > 0:
            logging.info(
                f"Generation {generation_index + 1} / {config.NUM_GENERATIONS}."
            )
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
                config,
            ).mutate(rng)
            for parent1_i, parent2_i in parents
        ]

        # Evaluate the offspring.
        offspring_fitnesses = evaluator.evaluate(
            [genotype.develop() for genotype in offspring_genotypes]
        )

        if verbose > 0:
            logging.info(f"Max fitness: {maxs[-1]}")

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

        plt.plot(maxs, label="max")
        plt.plot(mins, label="min")
        plt.plot(means, label="avg")
        plt.xlabel("Generation")
        plt.ylabel("fitness")
        plt.legend()
        plt.show()

    if save_winner:
        winner = max(population, key=lambda individual: individual.fitness)
        with open("winner-feedforward", "wb") as f:
            pickle.dump(winner, f)

    net_size_nodes = [len(ind.genotype.brain.neatGenome.nodes) for ind in population]
    net_size_connections = [
        len(ind.genotype.brain.neatGenome.connections) for ind in population
    ]
    all_fitness = [ind.fitness for ind in population]

    stats = dict(
        mins=mins,
        maxs=maxs,
        means=means,
        sizes_connections=net_size_connections,
        sizes_nodes=net_size_nodes,
        all_fitnesses=all_fitness,
    )
    fitness = maxs[-1]

    plt.plot(maxs)
    plt.plot(mins)
    plt.plot(means)

    return fitness, stats


if __name__ == "__main__":
    config = RevolveNeatConfig(
        body_num_inputs=5,
        body_num_outputs=5,
        brain_num_inputs=7,
        brain_num_outputs=1,
        POPULATION_SIZE=100,
        OFFSPRING_SIZE=50,
    )  # default config
    main(config, plot=True)
