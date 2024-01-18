import numpy as np
from revolve2.modular_robot_simulation import ModularRobotSimulationState

def rotation(
    states: list[ModularRobotSimulationState]
) -> float:
    """
    Calculate the rotation by a single modular robot.

    :param begin_state: Begin state of the robot.
    :param end_state: End state of the robot.
    :returns: The calculated fitness.
    """
    angles = [state.get_pose().orientation.angle for state in states]
    angles = np.array(angles)
    rotation = np.diff(angles)
    rotation = (rotation + np.pi) % (2 * np.pi) - np.pi
    total_rotation = np.abs(np.sum(rotation))
    return total_rotation.item()

def targeted_locomotion(
    begin_state: ModularRobotSimulationState, end_state: ModularRobotSimulationState
) -> float:
    """
    Calculate the progress towards travelling to the corner of the terrain

    :param begin_state: Begin state of the robot.
    :param end_state: End state of the robot.
    :returns: The calculated fitness.
    """
    begin_position = begin_state.get_pose().position
    end_position = end_state.get_pose().position
    return end_position.x + end_position.y