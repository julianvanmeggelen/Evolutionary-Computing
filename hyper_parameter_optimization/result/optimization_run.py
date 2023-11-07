from dataclasses import dataclass, field
from itertools import count
from typing import Any
from hyper_parameter_optimization.config.revolve_neat_config import RevolveNeatConfig


@dataclass
class OptimizationRun:
    """
    Stores the config, utility and other statistics of a run
    """
    utility: float
    statistics: list[dict[str, Any]]  #this stores arbitrary stats (one dict for each iteration)
    config: RevolveNeatConfig = None
    id: int = field(default_factory=count().__next__)

