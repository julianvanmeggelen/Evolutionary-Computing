from hyper_paramer_optimization.result.optimization_run import OptimizationRun

class OptimizationResult:
    """
    Stores all the results of an optimization result
    """

    def __init__(self, runs: list[OptimizationRun] = []):
        self.runs = runs

    def best_run(self) -> OptimizationRun:
        return max(self.runs, key = lambda run: run.utility)
    
    

