"""Evaluation functions. Uses neat-python nn.feedforward"""

import neat 




"""
Single-pole balancing experiment using a feed-forward neural network.
"""
import cart_pole
import neat

runs_per_net = 5
simulation_seconds = 60.0


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
        sim = cart_pole.CartPole()
        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        while sim.t < simulation_seconds:
            inputs = sim.get_scaled_state()
            action = net.activate(inputs)
            #print(action)

            # Apply action to the simulated cart-pole
            force = cart_pole.discrete_actuator_force(action)
            #print(sim.x,force)
            sim.step(force)

            # Stop if the network fails to keep the cart within the position or angle limits.
            # The per-run fitness is the number of time steps the network can balance the pole
            # without exceeding these limits.
            if abs(sim.x) >= sim.position_limit or abs(sim.theta) >= sim.angle_limit_radians:
                #print('stopped')
                break

        fitness = sim.t

        fitnesses.append(fitness)
    fitness = min(fitnesses)

    genotype.neatGenome.fitness = fitness
    return fitness
