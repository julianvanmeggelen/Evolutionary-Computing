from dataclasses import dataclass, field
from datetime import datetime
from itertools import count
from typing import Any
import numpy as np
from hyper_parameter_optimization.config.revolve_neat_config import RevolveNeatConfig

@dataclass
class OptimizationRun:
    """
    Stores the config, utility and other statistics of a run
    """
    utility: float
    statistics: list[dict[str, Any]]  #this stores arbitrary stats (one dict for each iteration)
    config: RevolveNeatConfig = None
    timestamp: datetime = datetime.now()
    id: int = field(default_factory=count().__next__)

    def n_iterations(self):
        return len(self.statistics)

    def get_stats(self, key:str) -> np.ndarray:
        """Get all run statistics for this key

        returns np.ndarray where the first axis has size n_iterations
        """
        all_stats = [stats[key] for stats in self.statistics]
        return np.array(all_stats)


