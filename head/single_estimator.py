import argparse
from pathlib import Path
import os
import random

import chainer
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

import logger
from estimator_util import get_estimator
import project


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--result_dir', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument(
        '--estimator',
        type=str,
        default='random_forest',
        choices=(
            'random_forest',
            'logistic_regression',
            'logistic_regression_sag',
            'logistic_regression_saga',
            'extra_tree',
            'linear_svc',
            'gbdt',
            'mlp-3',
            'mlp-4',
            'knn-2',
            'knn-4',
            'knn-8',
            'knn-16',
            'knn-32',
            'knn-64',
            'knn-128',
            'knn-256',
        )
    )
    parser.add_argument('--n_samples', type=int, default=-1)
    parser.add_argument('--n_features', type=int, default=-1)
    args = parser.parse_args()
    return args


def set_random_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def main():
    args = _parse_args()

    logger.setup_logger()
    set_random_seed(args.seed)

    n_samples = args.n_samples
    n_features = args.n_features
    input_dir = Path(args.input_dir)
    result_dir = Path(args.result_dir)

    X_train = pd.read_csv(input_dir / "train" / "feature_vectors.csv", header=None).values.astype(np.float32)
    X_test = pd.read_csv(input_dir / "test"/ "feature_vectors.csv", header=None).values.astype(np.float32)
    y_train = pd.read_csv(input_dir / "train" / "labels.txt", header=None).values[:, 0].astype(np.int32)
    y_test = pd.read_csv(input_dir / "test" / "labels.txt", header=None).values[:, 0].astype(np.int32)

    if n_samples >= 0:
        X_train = X_train[:n_samples]
        y_train = y_train[:n_samples]

    if n_features >= 0:
        X_train = X_train[:, :n_features]
        X_test = X_test[:, :n_features]

    label_names = pd.read_csv(input_dir / "label_names.txt", header=None)
    estimator = get_estimator(args.estimator, gpu=-1, n_out=len(label_names), seed=args.seed)
    estimator.fit(X_train, y_train)

    pred_test = estimator.predict(X_test)
    np.savetxt(result_dir / "submission.txt", pred_test, fmt="%i")

    report = classification_report(y_test, pred_test)
    with open(result_dir / "classification_report.txt", "w") as f:
        f.write(report)

    if hasattr(estimator, "predict_proba"):
        prob_test = estimator.predict_proba(X_test)
    elif hasattr(estimator, "decision_function"):
        prob_test = estimator.decision_function(X_test)
    else:
        raise ValueError(f"No score function. {estimator}")
    np.save(result_dir / "prob.npy", prob_test)


if __name__ == '__main__':
    main()