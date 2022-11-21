import json
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from . import dann
from .data import Dataset

OptionType = Dict[str, Any]


class Result(dict):
    pass


def _get_score(result: Result, dataset_type: str, domain_type: str) -> float:
    assert dataset_type in ("train", "val", "test"), f"Invalid: {dataset_type}"
    assert domain_type in ("source", "target"), f"Invalid: {domain_type}"
    return -result[dataset_type]["acc"][domain_type]


def get_loss(results: Sequence[Result], domain_type: str) -> float:
    total_loss_train = [_get_score(r, "train", domain_type) for r in results]
    idx = np.argmin(total_loss_train)
    return _get_score(results[idx], "test", domain_type)


def _get_optimizer(optimizer_type: str) -> torch.optim.Optimizer:
    if optimizer_type == "adadelta":
        return torch.optim.Adadelta
    elif optimizer_type == "adam":
        return torch.optim.Adam
    elif optimizer_type == "adam_w":
        return torch.optim.AdamW
    elif optimizer_type == "sparse_adam":
        return torch.optim.SparseAdam
    elif optimizer_type == "adamax":
        return torch.optim.Adamax
    elif optimizer_type == "asgd":
        return torch.optim.ASGD
    elif optimizer_type == "lbfgs":
        return torch.optim.LBFGS
    elif optimizer_type == "rms_prop":
        return torch.optim.RMSprop
    elif optimizer_type == "r_prop":
        return torch.optim.Rprop
    elif optimizer_type == "sgd":
        return torch.optim.SGD
    else:
        raise ValueError(f"Invalid optimizer: {optimizer_type}")


class Trainer(nn.Module):
    def __init__(self, model: nn.Module, **kwargs: OptionType) -> None:
        super().__init__()
        self.model = model
        self.args = kwargs

    def __call__(
        self, X: torch.Tensor, z: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        device = X.device
        X = X.to(self.args["device"])
        z = z.to(self.args["device"])
        y = y.to(self.args["device"])
        loss = self.model(X, z, y)
        return loss.to(device)

    def _predict(self, X: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        device = X.device
        X = X.to(self.args["device"])
        z = z.to(self.args["device"])
        y = self.model.predict(X, z)
        return y.to(device)

    def _evaluate_one_minibatch(
        self, X: torch.Tensor, z: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        device = X.device
        X = X.to(self.args["device"])
        z = z.to(self.args["device"])
        y = y.to(self.args["device"])
        count_correct = self.model.count_correct(X, z, y)
        return count_correct.to(device)

    def _train_one_epoch(
        self, dataset: Dataset, optimizer: torch.optim.Optimizer, args: OptionType
    ) -> Any:
        loader = torch.utils.data.DataLoader(dataset, args["batch_size"], shuffle=True)
        loss_all = 0.0
        loss_supervised_all = 0.0
        loss_unsupervised_all = 0.0
        y_true_all = []
        y_pred_all = []
        z_all = []
        for batch in loader:
            X, z, y = batch
            loss = self(X, z, y)
            loss.backward()
            optimizer.step()

            loss_all += float(loss)
            loss_supervised_all += float(self.model.loss_supervised)
            loss_unsupervised_all += float(self.model.loss_unsupervised)

            y_pred = self._predict(X, z)
            y_true_all.append(y)
            y_pred_all.append(y_pred)
            z_all.append(z)

        y_true_all = torch.cat(y_true_all)
        y_pred_all = torch.cat(y_pred_all)
        z_all = torch.cat(z_all)

        y_true_source = y_true_all[z_all == 1]
        y_true_target = y_true_all[z_all == 0]
        y_pred_source = y_pred_all[z_all == 1]
        y_pred_target = y_pred_all[z_all == 0]
        acc_source = (y_true_source == y_pred_source).sum() / len(y_true_source)
        acc_target = (y_true_target == y_pred_target).sum() / len(y_true_target)
        # print(confusion_matrix(y_true_source, y_pred_source))
        # print(confusion_matrix(y_true_target, y_pred_target))
        return {
            "loss": {
                "total": float(loss_all),
                "supervised": float(loss_supervised_all),
                "unsupervised": float(loss_unsupervised_all),
            },
            "acc": {
                "source": float(acc_source),
                "target": float(acc_target),
            },
        }

    def train(
        self,
        dataset: Sequence[Optional[Dataset]],
        **kwargs: OptionType,
    ) -> None:
        dataset_train, dataset_val, dataset_test = dataset
        assert dataset_train is not None

        args = dict(self.args, **kwargs)
        optimizer = _get_optimizer(args["optimizer_type"])(
            self.model.parameters(), lr=args["lr"]
        )
        self.to(args["device"])
        self.model.train()
        result: List[Result] = []
        pred = []
        for epoch in range(args["n_epochs"]):
            _result_train = self._train_one_epoch(dataset_train, optimizer, args)
            _result = Result({"train": _result_train})
            _pred = {}
            if dataset_val is not None:
                _result_val, _pred_val = self._evaluate(dataset_val, args)
                _result["val"] = _result_val
                _pred["val"] = _pred_val
            if dataset_test is not None:
                _result_test, _pred_test = self._evaluate(dataset_test, args)
                _result["test"] = _result_test
                _pred["test"] = _pred_test
            print(f"[Epoch{epoch}]\t", json.dumps(_result))
            result.append(_result)
            pred.append(_pred)

            _dataset_type = "train" if dataset_val is None else "val"
            score = _get_score(_result, _dataset_type, args["objective_domain"])
            yield score
        self._result = result
        self._pred = pred
        self._y = {
            "train": dataset_train.y.tolist(),
            "val": dataset_val.y.tolist(),
            "test": dataset_test.y.tolist()
        }

    @property
    def result(self) -> Any:
        return getattr(self, "_result", None)

    @property
    def pred(self) -> Any:
        return getattr(self, "_pred", None)

    @property
    def y(self) -> Any:
        return getattr(self, "_y", None)

    def _evaluate(self, dataset: Dataset, args: OptionType) -> Any:
        loader = torch.utils.data.DataLoader(dataset, args["batch_size"], shuffle=False)

        self.model.eval()
        loss_all = 0.0
        y_true_all = []
        y_pred_all = []
        z_all = []
        for batch in loader:
            X, z, y = batch
            loss = self(X, z, y)
            loss_all += float(loss)
            y_pred = self._predict(X, z)
            y_true_all.append(y)
            y_pred_all.append(y_pred)
            z_all.append(z)

        y_true_all = torch.cat(y_true_all)
        y_pred_all = torch.cat(y_pred_all)
        z_all = torch.cat(z_all)

        y_true_source = y_true_all[z_all == 1]
        y_true_target = y_true_all[z_all == 0]
        y_pred_source = y_pred_all[z_all == 1]
        y_pred_target = y_pred_all[z_all == 0]
        acc_source = (y_true_source == y_pred_source).sum() / len(y_true_source)
        acc_target = (y_true_target == y_pred_target).sum() / len(y_true_target)
        # print(confusion_matrix(y_true_source, y_pred_source))
        # print(confusion_matrix(y_true_target, y_pred_target))
        metric = {
            "loss": {
                "total": float(loss_all),
            },
            "acc": {"source": float(acc_source), "target": float(acc_target)},
        }
        pred = {
            "source": {
                "pred": y_pred_source.tolist(),
                "true": y_true_source.tolist(),
            },
            "target": {
                "pred": y_pred_target.tolist(),
                "true": y_true_target.tolist(),
            }
        }
        return metric, pred

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self(X), dim=1)

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self(X), dim=1)


def get_trainer(model_args: OptionType, train_args: OptionType):
    model = dann.get_dann(**model_args)
    trainer = Trainer(model, **train_args)
    return trainer
