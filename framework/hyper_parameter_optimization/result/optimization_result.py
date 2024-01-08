import pickle
from typing import Any
from hyper_parameter_optimization.result.optimization_run import OptimizationRun
from hyper_parameter_optimization.optimizer.tunable_param import (
    TunableParameter
)
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
import pandas as pd

class OptimizationResult:
    """
    Stores all the results of an optimization result
    """

    def __init__(self, tune_params: dict[str, TunableParameter], fitness_function:str, runs: list[OptimizationRun] = []):
        self.tune_params = tune_params
        self.runs: list[OptimizationRun] = []        
        self._tuner = None #the original tuner object used for tuning
        self.importance: dict[str,float] = None
        self.fitness_function = fitness_function

    @staticmethod
    def load(file_name):
        with open(file_name, "rb") as file:
            loaded = pickle.load(file)
            if len(loaded) == 5:
                tune_params, runs, _tuner, importance, fitness_function = loaded
            else:
                tune_params, runs, _tuner, importance = loaded
                fitness_function= None
        ret = OptimizationResult(tune_params = tune_params, fitness_function=fitness_function)
        ret._tuner = _tuner
        ret.runs = runs
        ret.importance = importance
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
        df =  pd.DataFrame(dict_items).sort_values(by='score', ascending=False)
        df = df.rename_axis('Parameter', axis=1)
        df = df.rename_axis('Run', axis=0)
        return df

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

    def plot_utility(self, use_timestamp=True):
        """
        Plot the utility over time
        """
        plt.figure()
        utility_vals = [run.utility for run in self.runs]
        x= [run.timestamp for run in self.runs] if use_timestamp else range(len(utility_vals))
        plt.scatter(x, utility_vals)
        plt.plot(utility_vals)
        plt.xlabel('Tuning run')
        plt.ylabel('Utility')

    def save(self, file_name: str):
        with open(file_name, "wb") as file:
            pickle.dump((self.tune_params, self.runs, self._tuner, self.importance, self.fitness_function), file) 
