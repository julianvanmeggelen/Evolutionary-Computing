
import pickle
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



def simulate(individuals):

    # Create a modular robot scene.
    # This is a combination of one or more modular robots positioned in a given terrain.
    scene = ModularRobotScene(terrain=terrains.flat())


    for individual in individuals:
        body = individual.genotype.body.develop()
        brain = individual.genotype.brain.develop(body)
        robot = ModularRobot(body, brain)
        scene.add_robot(robot)

    # Create a simulator that will perform the simulation.
    # This tutorial chooses to use Mujoco, but your version of revolve might contain other simulators as well.
    simulator = LocalSimulator(num_simulators=1)

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
    all_stats = []
    for i in range(24):
        with open(f"results_4733654/stats_{i}", "rb") as f:
            all_stats.append(
                pickle.load(
                    f
                )
            )

    all_best = sorted([stats['winner_individual'] for stats in all_stats], key=lambda el: el.fitness)[-10:]
    simulate(all_best)
