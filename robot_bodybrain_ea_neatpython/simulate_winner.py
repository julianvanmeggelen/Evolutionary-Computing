"""Main script for the example."""
from revolve2.ci_group import terrains
from revolve2.ci_group.simulation import make_standard_batch_parameters
from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.rng import make_rng_time_seed
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body import RightAngles
from revolve2.modular_robot.body.v1 import ActiveHingeV1, BodyV1, BrickV1
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkNeighborRandom
from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.simulators.mujoco_simulator import LocalSimulator
from individual import Individual
import pickle


def find_best_robot(
    current_best: Individual | None, population: list[Individual]
) -> Individual:
    """
    Return the best robot between the population and the current best individual.

    :param current_best: The current best individual.
    :param population: The population.
    :returns: The best individual.
    """
    return max(
        population + [] if current_best is None else [current_best],
        key=lambda x: x.fitness,
    )

def find_nth_best_robot(population: list[Individual], current_best: Individual | None = None, n: int = 1) -> Individual:
    """
    Return the nth best robot between the population and the current best individual.

    :param population: The population.
    :param current_best: The current best individual.
    :param n: The rank of the best individual to find.
    :returns: The nth best individual.
    """
    # Combine the population and the current best individual into one list
    individuals = population + ([current_best] if current_best is not None else [])

    # Sort the individuals by their fitness in descending order
    sorted_individuals = sorted(individuals, key=lambda x: x.fitness, reverse=True)

    # Return the nth best individual
    return sorted_individuals[n-1]



def main() -> None:
    """Run the simulation."""
    # Set up logging.
    setup_logging()


    # TODO: Load the winner
    print("Loading winner is hardcoded in simulate_winner.py, you need to modify it to load the winner.")
    return

    # Load the winner
    with open("winner-feedforward7", "rb") as f:
        individual = pickle.load(f)

    # # load last population
    # try:
    #     with open("last_pop", "rb") as f:
    #         population = pickle.load(f)
    # except (FileNotFoundError, EOFError):
    #     print("Error: 'last_pop' file is missing or empty.")
    #     return

    # if not population:
    #     print("Error: 'last_pop' file is empty.")
    #     return

    # # find the best individual in the population
    # individual = find_nth_best_robot(population, None, 1)
    # # print the fitness of the winner
    # print("FITNESS", individual.fitness)

    body = individual.genotype.body.develop()
    brain = individual.genotype.brain.develop(body)
    robot = ModularRobot(body, brain)

    # Create a modular robot scene.
    # This is a combination of one or more modular robots positioned in a given terrain.
    scene = ModularRobotScene(terrain=terrains.flat())
    scene.add_robot(robot)

    # Create a simulator that will perform the simulation.
    # This tutorial chooses to use Mujoco, but your version of revolve might contain other simulators as well.
    simulator = LocalSimulator()

    # `batch_parameters` are important parameters for simulation.
    # Here, we use the parameters that are standard in CI Group.
    batch_parameters = make_standard_batch_parameters()

    # Simulate the scene.
    # A simulator can run multiple sets of scenes sequentially; it can be reused.
    # However, in this tutorial we only use it once.
    simulate_scenes(
        simulator=simulator,
        batch_parameters=batch_parameters,
        scenes=scene,
    )


if __name__ == "__main__":
    main()
