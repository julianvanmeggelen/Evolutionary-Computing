import optuna
import numpy as np
import dataclasses
from spotPython.spot import spot
from scipy.optimize import differential_evolution
from hyper_parameter_optimization.config.revolve_neat_config import (
    RevolveNeatConfig
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

    def __init__(self, objective: OptimizationObjective, config_template: RevolveNeatConfig = None, **tune_params: dict[str,TunableParameter]):
        """
        args:
            objective: A Callable that accepts a RevolveNeatConfig object and returns an OptimizationRun
            config_template: the values for non-tuned parameters. If not provided the defaults as provided in RevolveNeatConfig are used.
            tune_params: A dict of param_name: TunableParameter. param_name must exist in RevolveNeatConfig
        """
        self.objective: OptimizationObjective = objective
        self.tune_params = tune_params
        self._config_template = config_template or RevolveNeatConfig()
        self.result: OptimizationResult = OptimizationResult(tune_params = tune_params)

    def _base_config(self):
        """
        Return a copy of the config_template
        """
        return self._config_template.copy()

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

        # start with the base
        config = self._base_config()

        # set the generic params (e.g OFFSPRING_SIZE)
        for param_name, param in self.tune_params.items():
            if param_name not in config:
                raise ValueError(f"Unrecognized parameter {param_name}")

            if type(param) is TunableFloat:
                setattr(
                    config,
                    param_name,
                    trial.suggest_float(param_name, param.min, param.max),
                )
            else:
                raise NotImplementedError(
                    "Only TunableFloats are supported by this backend"
                )          

        return config

    def run(self, n_trials=None, timeout=600, n_jobs=1) -> OptimizationResult:
        study = optuna.create_study(direction="maximize")
        self._tuner = study

        study.optimize(
            self._internal_objective, n_jobs=n_jobs, timeout=timeout, n_trials=n_trials
        )
        return self.result


class SpotHyperOptimizer(HyperOptimizer):
    def _generate_config(self, row) -> RevolveNeatConfig:

        # start with the base
        config = self._base_config()

        for (param_name, param), param_val in zip(self.tune_params.items(), row):

            if param_name not in config.__dict__:
                raise ValueError(f"Unrecognized parameter {param_name}")

            if type(param) is TunableFloat:
                setattr(config, param_name, param_val)
            else:
                raise NotImplementedError(
                    "Only TunableFloats are supported by this backend"
                )

        return config

    def run(self, n_trials=None, timeout=600, n_jobs=1) -> OptimizationResult:
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
            fun_evals=n_trials,  # default value
            max_time=timeout/60,  # timeout in mins
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

        self._tuner = spot_model
        return self.result
