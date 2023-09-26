import optuna


from typing import TypedDict
from argparse import Namespace
from datetime import datetime
import uuid 

import config
import main

#Type definitions
class HyperOptConfig(TypedDict):
    POPULATION_SIZE: int
    OFFSPRING_SIZE: int
    NUM_EVALS: int
    MUTATE_STD: float


def callback(trial, fitness, evals):
    trial.report(fitness, evals)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    
def objective(trial):
    config.POPULATION_SIZE = trial.suggest_int("POP_SIZE", 50,200)
    config.OFFSPRING_SIZE = int(config.POPULATION_SIZE  *trial.suggest_float("OFFSPRING_FRAC", 0.1,1))
    config.MUTATE_STD = trial.suggest_float("MUTATE_STD", 0.01,0.9)
    config.NUM_EVALS = 5000
    epoch_callback = None #lambda fitness, evals: callback(trial, fitness,evals)
    fitness = 0.0
    for i in range(10):
        fitness += main.main(callback=epoch_callback)
    return fitness/10
    

def hyperOpt():
    pruner = None; #optuna.pruners.PercentilePruner(0.5, n_startup_trials=5, n_warmup_steps=20)
    study = optuna.create_study(pruner=pruner,direction="maximize", study_name=f'test{uuid.uuid4()}', storage="sqlite:///hyperopt_results.sqlite")
    study.optimize(objective, n_trials=50, timeout=600)

    return study


if __name__ == "__main__":
    hyperOpt()