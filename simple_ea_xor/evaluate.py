import numpy as np
from individual import Individual

def model_forward(params, inputs):
    # First layer
    n0 = np.maximum(0, np.dot(params[:2], inputs) + params[2])
    n1 = np.maximum(0, np.dot(params[3:5], inputs) + params[5])

    # Second layer
    output: np.float_ = np.maximum(0, n0 * params[6] + n1 * params[7] + params[8])

    return output

def evaluate(individual: Individual) -> float:
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([model_forward(individual.genotype.params, input) for input in X])
    fitness = float(-np.sum((y_true - y_pred)**2))
    individual.fitness =  fitness
    return fitness
