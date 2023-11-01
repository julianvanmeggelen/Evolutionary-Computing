import pickle
from hyper_parameter_optimization.result.optimization_run import OptimizationRun


class OptimizationResult:
    """
    Stores all the results of an optimization result
    """

    def __init__(self, runs: list[OptimizationRun] = []):
        self.runs = runs

    @staticmethod
    def load(file_name):
        with open(file_name, "rb") as file:
            runs = pickle.load(file)
        return OptimizationResult(runs=runs)

    def add(self, run: OptimizationRun):
        self.runs.append(run)

    def best_run(self) -> OptimizationRun:
        return max(self.runs, key=lambda run: run.utility)

    def save(self, file_name: str):
        with open(file_name, "wb") as file:
            pickle.dump(self.runs, file)  # we only care about storing the runs
