from typing import Callable
from typing import Protocol

from hyper_parameter_optimization.config.revolve_neat_config import RevolveNeatConfig
from hyper_paramer_optimization.result.optimization_run import OptimizationRun



class OptimizationObjective(Protocol):
    """
    An objective must receive a RevolveNeatConfig and return a OptimizationRun
    """
    def __call__(self, config: RevolveNeatConfig, *args, **kwargs) -> OptimizationRun:
        ...
