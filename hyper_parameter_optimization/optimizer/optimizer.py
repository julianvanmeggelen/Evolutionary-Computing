from hyper_parameter_optimization.optimization_objective import OptimizationObjective
from hyper_parameter_optimization.result.optimization_result import OptimizationResult

class HyperParameterOptimizer:
    """
    Base class for several optimization framework backends
    """

    def __init__(self, objective: OptimizationObjective):
        self.objective = objective
 
    def run(self) -> OptimizationResult:
        raise NotImplementedError