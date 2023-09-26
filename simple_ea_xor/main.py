from revolve2.ci_group.rng import make_rng_time_seed
from revolve2.experimentation.optimization.ea import population_management, selection
from revolve2.ci_group.logging import setup_logging

setup_logging()
from genotype import Genotype
from individual import Individual
from evaluate import evaluate
import logging
import config
rng = make_rng_time_seed()
Genotype.rng = rng
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

def init_population()  -> list[Genotype]:
    return [Individual(Genotype.random()) for i in range(config.POPULATION_SIZE)]

def eval_population(pop: list[Individual]) -> list[float]:
    ret = []
    for ind in pop:
        if ind.fitness is None:
            fitness =  evaluate(ind)
        else:
            fitness = ind.fitness
        ret.append(fitness)
    return ret

def select_parents(pop: list[Individual]) ->list[tuple[Individual]]:
    index_pairs = [selection.multiple_unique(
            2,
            [individual.genotype for individual in pop],
            [individual.fitness for individual in pop],
            lambda _, fitnesses: selection.tournament(rng, fitnesses, k=1),
        )
        for _ in range(config.OFFSPRING_SIZE)]
    return [(pop[i], pop[j]) for i, j in index_pairs]

def select_survivors(pop: list[Individual], offspring_pop: list[Individual]) ->list[Individual]:
    original_survivors, offspring_survivors = population_management.steady_state(
        [i.genotype for i in pop],
        [i.fitness for i in pop],
        [i.genotype for i in offspring_pop],
        [i.fitness for i in offspring_pop],
        lambda n, genotypes, fitnesses: selection.multiple_unique(
            n,
            genotypes,
            fitnesses,
            lambda _, fitnesses: selection.tournament(rng, fitnesses, k=2),
        ),
    )

    return [
        Individual(
            pop[i].genotype,
            pop[i].fitness,
        )
        for i in original_survivors
    ] + [
        Individual(
            offspring_pop[i].genotype,
            offspring_pop[i].fitness,
        )
        for i in offspring_survivors
    ]

def make_next_pop(pop: list[Individual]) ->list[Individual]:
    parents = select_parents(pop)
    offspring_pop = [Individual(Genotype.crossOver(a.genotype,b.genotype).mutate()) for a,b in parents]
    eval_population(offspring_pop) #OFFSPRING_SIZE evaluations
    pop = select_survivors(pop, offspring_pop)
    return pop

def main(config_overwrite=None, callback: callable = None, plot = False):
    global config
    if config_overwrite:
        config = config_overwrite

    print(f"Config: gens: {config.POPULATION_SIZE}, off: {config.OFFSPRING_SIZE}, mutate_std:{config.MUTATE_STD}")
    maxs, mins, avgs = [], [], []
    logging.info("Making initial population")
    pop: list[Individual] = init_population()
    logging.info("Starting optimization")

    num_generations = config.NUM_EVALS / (config.POPULATION_SIZE + config.OFFSPRING_SIZE)
    logging.info(f"Using {num_generations} generations")
    for i in range(int(num_generations)):
        pop_fitness = eval_population(pop) #GENERATION_SIZE evaluations
        maxs.append(max(pop_fitness));  mins.append(min(pop_fitness)); avgs.append(sum(pop_fitness)/len(pop_fitness));
        logging.info(f"Generation {i} | max(fitness): {maxs[-1]}")
        pop = make_next_pop(pop)
        if callback: callback(maxs[-1], i * (config.OFFSPRING_SIZE + config.POPULATION_SIZE))

    if plot:
        plt.plot(maxs, label = 'max'); plt.plot(mins, label = 'min'); plt.plot(avgs, label = 'avg');
        plt.xlabel("Generation"); plt.ylabel("fitness"); plt.legend()
        plt.show()

    return maxs[-1]
        

if __name__ == "__main__":
    main(plot=True)


