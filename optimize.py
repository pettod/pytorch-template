import json
import os
import optuna
from src.optunaobjective import Objective
from time import strftime


OPTUNA_BEST_PARAMS_FILE_NAME = f"optuna_runs/{strftime('%Y-%m-%d_%H%M%S')}_optuna_best_params.txt"


def printCallback(study, trial):
    params = {
        "loss" : trial.value,
    }
    params.update(trial.params)
    print(f"\rTrial {trial.number} finished with value: {trial.value}, and params: {params}")


def saveBestparams(study, trial):
    if trial.value > 0.225:
        return
    params = {
        "flops": trial.user_attrs["flops"],
        "loss" : trial.value,
    }
    os.makedirs(os.path.dirname(OPTUNA_BEST_PARAMS_FILE_NAME), exist_ok=True)
    params.update(trial.params)
    file = open(OPTUNA_BEST_PARAMS_FILE_NAME, "a")
    file.write(json.dumps(params) + "\n")
    file.close()


def optimize(
        sampler_seed=73123,
        n_trials=10000,
        **predefined_params
    ):
    sampler = optuna.samplers.TPESampler(
        seed=sampler_seed,
        multivariate=True,
        constant_liar=True,
    )
    sampler = optuna.samplers.PartialFixedSampler(predefined_params, sampler)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    # study.sampler = sampler
    obj = Objective()
    study.optimize(obj, n_trials=n_trials, callbacks=[printCallback, saveBestparams])
    return study.best_params


def main():
    hyperparameters = optimize()


main()
