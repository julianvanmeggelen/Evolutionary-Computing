from datetime import datetime, timedelta
import optuna
import optuna.importance  
import numpy as np
import dataclasses
from spotPython.spot import spot
from scipy.optimize import differential_evolution
from functools import partial
import logging
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

from spotPython.hyperparameters.values import (
    add_core_model_to_fun_control,
    assign_values,
    generate_one_config_from_var_dict,
    get_bound_values,
    get_default_hyperparameters_as_array,
    get_var_name,
    get_var_type,
)
from hyper_parameter_optimization.optimizer.spot_specific import DummyModel, SpotHyperDict
import json
from typing import Dict
from hyper_parameter_optimization.optimizer.tunable_param import TunableParameter, TunableFloat, TunableCategory, TunableDataType



class HyperOptimizer:
    """
    Base class for several optimization framework backends
    """

    def __init__(self, objective: OptimizationObjective, fitness_function:str, config_template: RevolveNeatConfig = None, checkpoint_dir: str = None, **tune_params: dict[str,TunableParameter]):
        """
        args:
            objective: A Callable that accepts a RevolveNeatConfig object and returns an OptimizationRun
            config_template: the values for non-tuned parameters. If not provided the defaults as provided in RevolveNeatConfig are used.
            tune_params: A dict of param_name: TunableParameter. param_name must exist in RevolveNeatConfig
        """
        self.objective: OptimizationObjective = objective
        self.tune_params = tune_params
        self._config_template = config_template or RevolveNeatConfig()
        self.result: OptimizationResult = OptimizationResult(tune_params = tune_params, fitness_function = fitness_function) 
        self.checkpoint_dir = checkpoint_dir
      
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
        eval_bt = datetime.now()
        optimization_run = self.objective(config)
        eval_et = datetime.now()
        optimization_run.config = config
        self.result.add(optimization_run)
        return optimization_run.utility

    def _internal_objective(self, *args, **kwargs) -> float:
        """
        This is the function that must be called by the backend. Args & kwargs are passed to _generate_config
        """
        config = self._generate_config(*args, **kwargs)
        utility = self._eval_config(config)
        self._checkpoint()
        return utility

    def _checkpoint(self):
        if self.checkpoint_dir:
            self.result.save(self.checkpoint_dir)

    def run(self, *args, **kwargs) -> OptimizationResult:
        """
        Calling this method starts the optimization process, several backends can accept different arguments
        """
        raise NotImplementedError
    
    def _pre_run(self):
        """Hook that needs to be called at start of run
        """
        logging.level = logging.exception
        self.bt = datetime.now()
       
    def _post_run(self):
        """Hook that needs to be called at end of run
        """
        self.et = datetime.now()
        self.total_time = self.et - self.bt
        print("="*30)
        print("Finished tuning run")
        print(f"\t Total time: {self.total_time}")
        n_runs = len(self.result.runs)
        print(f"\t Total runs: {n_runs}")
        total_time_minutes = self.total_time.total_seconds()/60
        print(f"\t \t per minute: {n_runs/total_time_minutes}")
        print("="*30)


class BaselineDummyTuner(HyperOptimizer):
    """Do not tune but run the baseline. Only gives one trial.
    """

    def __init__(self, objective: OptimizationObjective, config_template: RevolveNeatConfig, fitness_function:str, checkpoint_dir: str = None):
        super().__init__(objective=objective, config_template=config_template, checkpoint_dir=checkpoint_dir, fitness_function=fitness_function)

    def _generate_config(self, *args, **kwargs) -> RevolveNeatConfig:
        return self._base_config()
    
    def run(self, timeout, n_jobs) -> OptimizationResult:
        self._pre_run()
        utility = self._internal_objective()
        self._post_run()
        return self.result

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

            elif type(param) is TunableCategory:
                setattr(
                    config,
                    param_name,
                    trial.suggest_categorical(param_name, param.categories),
                )
                
            else:
                raise NotImplementedError(
                    "Only TunableFloats are supported by this backend"
                )          

        return config

    def run(self, n_trials=None, timeout=600, n_jobs=-1) -> OptimizationResult:
        self._pre_run()
        study = optuna.create_study(direction="maximize")
        self.result._tuner = study

        study.optimize(
            self._internal_objective, n_jobs=n_jobs, timeout=timeout, n_trials=n_trials
        )

        self.result.importance = optuna.importance.get_param_importances(study)
        self._post_run()
        return self.result


class SpotHyperOptimizer(HyperOptimizer):
    def _generate_config(self, row) -> RevolveNeatConfig:
        # TODO: spot has some methods to create a dict from the "row"

        # start with the base
        config = self._base_config()

        for (param_name, param), param_val in zip(self.tune_params.items(), row):

            if param_name not in config.__dict__:
                raise ValueError(f"Unrecognized parameter {param_name}")

            if type(param) is TunableFloat:
                setattr(config, param_name, param_val)

            elif type(param) is TunableCategory:
                setattr(config, param_name, param.categories[int(param_val)])

            else:
                raise NotImplementedError(
                    "Only TunableFloats are supported by this backend"
                )

        return config

    def _spot_objective(self, X: np.ndarray, fun_control):
            # Layer between spot and _internal_objective
            y = np.empty((0, 1))
            for row in X:
                utility = self._internal_objective(row=row)
                print(f"Row: {row}, Utility: {utility}")
                y = np.append(y, -utility)
            return y
    
    def run(self, n_trials=15, timeout=600, n_jobs=-1) -> OptimizationResult:
        self._pre_run()

        fun_control = {}
        self.generate_hyper_spot_json(self.tune_params)
        add_core_model_to_fun_control(core_model=DummyModel,
                              fun_control=fun_control,
                              hyper_dict=SpotHyperDict,
                              filename=None)
        var_type = get_var_type(fun_control)
        var_name = get_var_name(fun_control)
        lower = get_bound_values(fun_control, "lower")
        upper = get_bound_values(fun_control, "upper")
        # TODO: !!! default value (param.init) should not be None
        # currently, tunable float ones are none and they're set as the mean of min and max
        # but when some are none and some are not then it triggers an error from spot
        X_start = get_default_hyperparameters_as_array(fun_control) 

        spot_model = spot.Spot(
            fun=self._spot_objective,  # objective function
            lower=lower,  # lower bound of the search space
            upper=upper,  # upper bound of the search space
            fun_evals=n_trials,  # default value
            max_time=timeout/60,  # timeout in mins
            var_name=var_name,
            var_type=var_type,
            show_progress=True,
            surrogate_control={
                "n_theta": len(var_name),
            },
            # fun_control= fun_control,
        )
        spot_model.run(
            X_start=X_start,
        )  # initial design points

        spot_model.fun = None #remove the fun before saving, it is not needed
        self.result._tuner = spot_model
        self.result.importance = dict(spot_model.print_importance())
        self._post_run()
        return self.result

    def generate_hyper_spot_json(self, params: Dict[str, TunableParameter]) -> None:
        hyper_spot_data = {"DummyModel": {}}
        for param_name, param in params.items():
            if isinstance(param, TunableFloat):
                # TODO: change the logic for default value
                default_value = param.init if param.init is not None else (param.min + param.max) / 2
                hyper_spot_data["DummyModel"][param_name] = {
                    "type": "float",
                    "default": default_value,
                    "transform": "None",
                    "lower": param.min,
                    "upper": param.max
                }
            elif isinstance(param, TunableCategory):
                hyper_spot_data["DummyModel"][param_name] = {
                    "type": "factor",
                    "default": param.init,
                    "transform": "None",
                    "core_model_parameter_type": "str",
                    "lower": 0,
                    "upper": len(param.categories) - 1,
                    "levels": param.categories
                }
        
        with open('hyper_spot_generated.json', 'w') as file:
            json.dump(hyper_spot_data, file, indent=4)