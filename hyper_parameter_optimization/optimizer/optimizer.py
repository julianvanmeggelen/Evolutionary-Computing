# from hyper_parameter_optimization.optimization_objective import OptimizationObjective
import optuna
import numpy as np
from spotPython.spot import spot
from scipy.optimize import differential_evolution
from hyper_parameter_optimization.config.revolve_neat_config import (
    DefaultRevolveNeatConfig,
    RevolveNeatConfig,
    NON_NEAT_ARGS,
)
from hyper_parameter_optimization.optimizer.optimization_objective import (
    OptimizationObjective,
)
from hyper_parameter_optimization.optimizer.tunable_param import (
    TunableFloat,
    TunableParameter,
)
from hyper_parameter_optimization.result.optimization_result import OptimizationResult


class HyperOptimizer:
    """
    Base class for several optimization framework backends
    """

    def __init__(self, objective: OptimizationObjective, **tune_params):
        self.objective: OptimizationObjective = objective
        self.tune_params: dict[str, TunableParameter] = tune_params
        self.result: OptimizationResult = OptimizationResult()

    def _generate_config(self, *args, **kwargs) -> RevolveNeatConfig:
        """
        Generate the config object, must be implemented by backend specific optimizer
        """
        raise NotImplementedError

    def _eval_config(self, config: RevolveNeatConfig) -> float:
        """
        Evaluate the configuration and return the utility
        """
        optimization_run = self.objective(config)
        optimization_run.config = config
        self.result.add(optimization_run)
        return optimization_run.utility

    def _internal_objective(self, *args, **kwargs) -> float:
        """
        This is the function that must be called by the backend. Args & kwargs are passed to _generate_config
        """
        config = self._generate_config(*args, **kwargs)
        utility = self._eval_config(config)
        return utility

    def run(self, *args, **kwargs) -> OptimizationResult:
        """
        Calling this method starts the optimization process, several backends can accept different arguments
        """
        raise NotImplementedError


class OptunaHyperOptimizer(HyperOptimizer):
    def _generate_config(self, trial) -> RevolveNeatConfig:

        # start with default
        config = DefaultRevolveNeatConfig()

        for param_name, param in self.tune_params.items():

            # determine wether to set neat config or not
            relevant_config = (
                config if param_name in NON_NEAT_ARGS else config.genome_config
            )

            if param_name not in relevant_config.__dict__:
                raise ValueError(f"Unrecognized parameter {param_name}")

            if type(param) is TunableFloat:
                setattr(
                    relevant_config,
                    param_name,
                    trial.suggest_float(param_name, param.min, param.max),
                )
            else:
                raise NotImplementedError(
                    "Only TunableFloats are supported by this backend"
                )

        return config

    def run(self, n_trials=50, timeout=1200, n_jobs=1) -> OptimizationResult:
        study = optuna.create_study(direction="maximize")
        study.optimize(
            self._internal_objective, n_trials=n_trials, n_jobs=n_jobs, timeout=timeout
        )
        return self.result


class SpotHyperOptimizer(HyperOptimizer):
    def _generate_config(self, row) -> RevolveNeatConfig:

        # start with default
        config = DefaultRevolveNeatConfig()

        for (param_name, param), param_val in zip(self.tune_params.items(), row):

            # determine wether to set neat config or not
            relevant_config = (
                config if param_name in NON_NEAT_ARGS else config.genome_config
            )

            if param_name not in relevant_config.__dict__:
                raise ValueError(f"Unrecognized parameter {param_name}")

            if type(param) is TunableFloat:
                setattr(relevant_config, param_name, param_val)
            else:
                raise NotImplementedError(
                    "Only TunableFloats are supported by this backend"
                )

        return config

    def run(self, n_trials=50, timeout=1200, n_jobs=1) -> OptimizationResult:
        def spot_objective(X: np.ndarray, fun_control):
            # Layer between spot and _internal_objective
            y = np.empty((0, 1))
            for row in X:
                utility = self._internal_objective(row=row)
                y = np.append(y, -utility)
            return y

        X_start = np.array([param.init for param in self.tune_params.values()])
        lower = np.array([param.min for param in self.tune_params.values()])
        upper = np.array([param.max for param in self.tune_params.values()])

        spot_model = spot.Spot(
            fun=spot_objective,  # objective function
            lower=lower,  # lower bound of the search space
            upper=upper,  # upper bound of the search space
            fun_evals=15,  # default value
            max_time=timeout,  # 10 mins
            var_name=list(self.tune_params),
            show_progress=True,
            surrogate_control={
                "n_theta": 6,
                "model_optimizer": differential_evolution,
            },
            # fun_control= fun_control,
        )
        spot_model.run(
            X_start=X_start,
        )  # initial design points

        return self.result
