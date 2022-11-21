import datetime
import json
from typing import Optional, Sequence

import click
import mlflow
import numpy as np
import optuna
import torch
from arsene.config.parameter import create_parameters
from optuna import exceptions, pruners
from optuna.study import Study
from optuna.trial import FrozenTrial

from util import data
from util.trainer import get_loss, get_trainer


def _objective(
    trial: optuna.trial.FrozenTrial,
    dataset: Sequence[Optional[data.Dataset]],
    debug: bool,
    device: str,
    method: str,
) -> float:
    assert len(dataset) == 3
    dataset_train, dataset_val, dataset_test = dataset

    D = dataset_train.X.shape[1]
    K = int(torch.cat((dataset_train.y, dataset_val.y, dataset_test.y)).max()) + 1
    n_epochs = 2 if debug else 200
    min_epoch = 2 if debug else 10
    max_epoch = 2 if debug else 500

    config = {
        "fixed": {
            "model_args": {
                "D": D,
                "K": K,
                # "d_hidden": 1000,
                # "d_latent": 1000,
                # "n_layers": 1,
            },
            "train_args": {
                "device": device,
                "batch_size": 1024,
                # "optimizer_type": "adam",
                # "lr": 1e-5,
                # "n_epochs": n_epochs,
            },
        },
        "searching": {
            "model_args": {
                "d_hidden": ["int", 500, 1500, 50],
                "d_latent": ["int", 500, 1500, 50],
                "n_layers": ["int", 0, 2],
            },
            "train_args": {
                "optimizer_type": [
                    "categorical",
                    # "adadelta",
                    "adam",
                    "adam_w",
                    "adamax",
                    # "asgd",
                    "rms_prop",
                    # "r_prop",
                    "sgd",
                ],
                "lr": ["loguniform", 1e-6, 1e-4],
                "n_epochs": ["int", min_epoch, max_epoch, 10],
            },
        },
    }

    if method in ("dann_source_to_target", "dann_target_to_source"):
        config["searching"]["model_args"]["discriminator_loss_weight"] = [
            "loguniform",
            0.1,
            10,
        ]
        config["searching"]["model_args"]["scale"] = ["loguniform", 0.01, 1]
    elif method in ("source", "target"):
        config["fixed"]["model_args"]["discriminator_loss_weight"] = 0.0
    else:
        raise ValueError(f"Invalid method={method}")

    if method in ("dann_target_to_source", "source"):
        config["fixed"]["train_args"]["objective_domain"] = "source"
    elif method in ("dann_source_to_target", "target"):
        config["fixed"]["train_args"]["objective_domain"] = "target"

    params = create_parameters(config)
    params.set_trial(trial)
    # model_args = params.get("model_args")
    model_args = dict(
        [(key, params["model_args"][key].get()) for key in params["model_args"].keys()]
    )
    # train_args = params.get("train_args")
    train_args = dict(
        [(key, params["train_args"][key].get()) for key in params["train_args"].keys()]
    )
    print(params.dump_chosen_values())

    trainer = get_trainer(model_args, train_args)
    for step, loss in enumerate(trainer.train(dataset)):
        trial.report(loss, step)
        if trial.should_prune():
            raise exceptions.TrialPruned()

    assert trainer.result is not None
    trial.set_user_attr("result", trainer.result)

    assert trainer.pred is not None
    trial.set_user_attr("pred", trainer.pred)

    assert trainer.y is not None
    trial.set_user_attr("y", trainer.y)

    final_loss = get_loss(trainer.result, train_args["objective_domain"])
    return final_loss


def _callback(study: Study, trial: FrozenTrial, experiment_id: str) -> None:
    with mlflow.start_run(experiment_id=experiment_id, run_name=trial.number):
        mlflow.log_param("params.json", json.dumps(trial.params))

        result = getattr(trial.user_attrs, "result", [])
        mlflow.log_param("result.json", json.dumps(result))

        trial_value = getattr(trial, "value", float("nan"))
        trial_value = trial_value or float("nan")
        mlflow.log_metric("value", trial_value)


@click.command()
@click.option(
    "--method",
    "-m",
    default="dann_source_to_target",
    type=click.Choice(
        [
            "dann_source_to_target",
            "dann_target_to_source",
            "source",
            "target"
        ]
    )
)
@click.option("--debug", is_flag=True)
@click.option("--n-trials", "-t", default=100, type=int)
@click.option("--device", "-d", default="cuda:0", type=str)
@click.option("--seed", "-s", default=0, type=int)
def main(method: str, debug: bool, n_trials: int, device: str, seed: int) -> None:
    _datetime = datetime.datetime.now().isoformat()
    experiment_id = mlflow.create_experiment(_datetime)
    print(f"Experiment ID={experiment_id}")
    args = locals()

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    dataset = data.load(method)

    def objective(trial: FrozenTrial) -> float:
        return _objective(trial, dataset, debug, device, method)

    def callback(study: Study, trial: FrozenTrial) -> None:
        return _callback(study, trial, experiment_id)

    study = optuna.create_study(pruner=pruners.MedianPruner())
    n_trials = 2 if debug else n_trials
    study.optimize(objective, n_trials=n_trials, callbacks=[callback])

    with mlflow.start_run(run_name="all", experiment_id=experiment_id):
        mlflow.log_param("experiment_id", experiment_id)
        best_trial_params = {
            "number": study.best_trial.number,
            "params": study.best_trial.params,
            "value": study.best_trial.value,
        }
        mlflow.log_param("best_trial.json", json.dumps(best_trial_params))

        result = study.best_trial.user_attrs["result"]
        mlflow.log_param("result.json", json.dumps(result))
        mlflow.log_param("args.json", json.dumps(args))

        pred = study.best_trial.user_attrs["pred"]
        mlflow.log_param("pred.json", json.dumps(pred))

        y = study.best_trial.user_attrs["y"]
        mlflow.log_param("y.json", json.dumps(y))

if __name__ == "__main__":
    main()
