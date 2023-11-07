import pickle
from typing import Any
from hyper_parameter_optimization.result.optimization_run import OptimizationRun
from hyper_parameter_optimization.optimizer.tunable_param import (
    TunableParameter
)
from dataclasses import dataclass, asdict
import pandas as pd

class OptimizationResult:
    """
    Stores all the results of an optimization result
    """

    def __init__(self, tune_params: dict[str, TunableParameter], runs: list[OptimizationRun] = []):
        self.tune_params = tune_params
        self.runs: list[OptimizationRun] = []        
        self._tuner = None #the tuner object used for tuning

    @staticmethod
    def load(file_name):
        with open(file_name, "rb") as file:
            tune_params, runs, _tuner = pickle.load(file)
        ret = OptimizationResult(tune_params = tune_params)
        ret._tuner = _tuner
        ret.runs = runs
        return ret

    def add(self, run: OptimizationRun):
        self.runs.append(run)

    def best_run(self) -> OptimizationRun:
        return max(self.runs, key=lambda run: run.utility)

    def summary(self) -> pd.DataFrame:
        tune_param_names = list(self.tune_params)
        dict_items = []
        for run in self.runs:
            dict_item = {}
            dict_item['id'] = run.id 
            dict_item['score'] = run.utility
            run_relevant_params = {k:v for k,v in run.config.dict().items() if k in tune_param_names}
            dict_item.update(run_relevant_params)
            dict_items.append(dict_item)
        return pd.DataFrame(dict_items).sort_values(by='score', ascending=False)

    def get_stats(self, key:str) -> list[Any]:
        """
        Return a list of all the statistics with key for all the runs
        """
        ret = []
        for run in self.runs:
            ret.append(
                run.statistics[key]
            )
        return ret

    def save(self, file_name: str):
        with open(file_name, "wb") as file:
            pickle.dump((self.tune_params, self.runs, self._tuner), file)  # we only care about storing the runs
