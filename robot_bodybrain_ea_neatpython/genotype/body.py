import math
from dataclasses import dataclass
from queue import Queue
from typing import Any
from revolve2.ci_group.genotypes.cppnwin.modular_robot.v1._body_develop import __Module, __add, __timesscalar, __cross, __dot, __rotate
import neat

from revolve2.modular_robot.body import Module
from revolve2.modular_robot.body.v1 import ActiveHingeV1, BodyV1, BrickV1, CoreV1

from hyper_parameter_optimization.config.revolve_neat_config import RevolveNeatConfig
from robot_bodybrain_ea_neatpython.genotype.genotype_base import BaseNeatGenotype


# @dataclass
# class __Module:
#     position: tuple[int, int, int]
#     forward: tuple[int, int, int]
#     up: tuple[int, int, int]
#     chain_length: int
#     module_reference: Module

class BodyGenotype(BaseNeatGenotype):
    def develop(self):
        return develop(self.neatGenome, self.config)

def develop(
    genotype: neat.DefaultGenome,
    config: RevolveNeatConfig
) -> BodyV1:
    """
    Develop a CPPNWIN genotype into a modular robot body.

    It is important that the genotype was created using a compatible function.

    :param genotype: The genotype to create the body from.
    :returns: The created body.
    :raises RuntimeError: In case a module is encountered that is not supported.
    """
    max_parts = 10

    body_net = neat.nn.FeedForwardNetwork.create(genotype, config)

    to_explore: Queue[__Module] = Queue()
    grid: set[tuple[int, int, int]] = set()

    body = BodyV1()

    to_explore.put(__Module((0, 0, 0), (0, -1, 0), (0, 0, 1), 0, body.core))
    grid.add((0, 0, 0))
    part_count = 1

    while not to_explore.empty():
        module = to_explore.get()

        children: list[tuple[int, int]] = []  # child index, rotation

        if isinstance(module.module_reference, CoreV1):
            children.append((CoreV1.FRONT, 0))
            children.append((CoreV1.LEFT, 1))
            children.append((CoreV1.BACK, 2))
            children.append((CoreV1.RIGHT, 3))
        elif isinstance(module.module_reference, BrickV1):
            children.append((BrickV1.FRONT, 0))
            children.append((BrickV1.LEFT, 1))
            children.append((BrickV1.RIGHT, 3))
        elif isinstance(module.module_reference, ActiveHingeV1):
            children.append((ActiveHingeV1.ATTACHMENT, 0))
        else:  # Should actually never arrive here but just checking module type to be sure
            raise RuntimeError()

        for index, rotation in children:
            if part_count < max_parts:
                child = __add_child(body_net, module, index, rotation, grid)
                if child is not None:
                    to_explore.put(child)
                    part_count += 1

    return body


def __evaluate_cppn(
    body_net: neat.nn.FeedForwardNetwork,
    position: tuple[int, int, int],
    chain_length: int,
) -> tuple[Any, int]:
    """
    Get module type and orientation from a multineat CPPN network.

    :param body_net: The CPPN network.
    :param position: Position of the module.
    :param chain_length: Tree distance of the module from the core.
    :returns: (module type, orientation)
    """
   
    outputs =  body_net.activate(
        [1.0, position[0], position[1], position[2], chain_length]
    )

    # get module type from output probabilities
    type_probs = [outputs[0], outputs[1], outputs[2]]
    types = [None, BrickV1, ActiveHingeV1]
    module_type = types[type_probs.index(min(type_probs))]

    # get rotation from output probabilities
    rotation_probs = [outputs[3], outputs[4]]
    rotation = rotation_probs.index(min(rotation_probs))

    return (module_type, rotation)


def __add_child(
    body_net: neat.nn.FeedForwardNetwork,
    module: __Module,
    child_index: int,
    rotation: int,
    grid: set[tuple[int, int, int]],
) -> __Module | None:
    forward = __rotate(module.forward, module.up, rotation)
    position = __add(module.position, forward)
    chain_length = module.chain_length + 1

    # if grid cell is occupied, don't make a child
    # else, set cell as occupied
    if position in grid:
        return None
    else:
        grid.add(position)

    child_type, orientation = __evaluate_cppn(body_net, position, chain_length)
    if child_type is None:
        return None
    up = __rotate(module.up, forward, orientation)

    child = child_type(orientation * (math.pi / 2.0))
    module.module_reference.set_child(child, child_index)

    return __Module(
        position,
        forward,
        up,
        chain_length,
        child,
    )

