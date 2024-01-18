"""Evaluator class."""

import os
from typing import Callable
from revolve2.ci_group import fitness_functions, terrains
from revolve2.ci_group.simulation import make_standard_batch_parameters
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot_simulation import (
    ModularRobotScene,
    Terrain,
    simulate_scenes,
)
from revolve2.simulators.mujoco_simulator import LocalSimulator

from revolve2.modular_robot_simulation import ModularRobotSimulationState

from hyper_parameter_optimization.fitness_functions import rotation, targeted_locomotion
class FitnessFunctions:
    XY_DISPLACEMENT = 'XY_DISPLACEMENT'
    ROTATION = 'ROTATION'
    TARGETED_LOCOMOTION = 'TARGETED_LOCOMOTION'
    DEFAULT = 'XY_DISPLACEMENT'

def load_fitness_function() -> tuple[str, Callable[[ModularRobotSimulationState,ModularRobotSimulationState], float]]:
    """ Load fitness function from environment or load default (XY_DISPLACEMENT)
    """
    env_var = os.getenv("FITNESS_FUN", FitnessFunctions.DEFAULT).upper()
    if env_var == FitnessFunctions.XY_DISPLACEMENT:
        return  FitnessFunctions.XY_DISPLACEMENT, fitness_functions.xy_displacement
    if env_var == FitnessFunctions.ROTATION:
        return FitnessFunctions.ROTATION, rotation
    if env_var == FitnessFunctions.TARGETED_LOCOMOTION:
        return FitnessFunctions.TARGETED_LOCOMOTION, targeted_locomotion
    
    raise NotImplementedError(f'Fitness function {env_var} not recognized as a fitness function')


def evaluate_fitness_function(robots, scene_states) -> list[float]:
    fit_fun_name, fit_fun = load_fitness_function()
    if fit_fun_name == FitnessFunctions.ROTATION:
        fit_vals = [
            fit_fun(
                [state.get_modular_robot_simulation_state(robot) for state in states]
            )
            for robot, states in zip(robots, scene_states)
        ]
    else:
        fit_vals = [
            fit_fun(
                states[0].get_modular_robot_simulation_state(robot),
                states[-1].get_modular_robot_simulation_state(robot),
            )
            for robot, states in zip(robots, scene_states)
        ]

    return fit_vals



class Evaluator:
    """Provides evaluation of robots."""

    _simulator: LocalSimulator
    _terrain: Terrain

    def __init__(
        self,
        headless: bool,
        num_simulators: int = os.getenv("NSIM", None),
    ) -> None:
        """
        Initialize this object.

        :param headless: `headless` parameter for the physics simulator.
        :param num_simulators: `num_simulators` parameter for the physics simulator.
        """
        self._simulator = LocalSimulator(
            headless=headless, num_simulators=num_simulators
        )
        self._terrain = terrains.flat()

    def evaluate(
        self,
        robots: list[ModularRobot],
    ) -> list[float]:
        """
        Evaluate multiple robots.

        Fitness is the distance traveled on the xy plane.

        :param robots: The robots to simulate.
        :returns: Fitnesses of the robots.
        """
        # Create the scenes.
        scenes = []
        for robot in robots:
            scene = ModularRobotScene(terrain=self._terrain)
            scene.add_robot(robot)
            scenes.append(scene)

        # Simulate all scenes.
        scene_states = simulate_scenes(
            simulator=self._simulator,
            batch_parameters=make_standard_batch_parameters(),
            scenes=scenes,
        )

        # Calculate the fitness.
        return evaluate_fitness_function(robots, scene_states)