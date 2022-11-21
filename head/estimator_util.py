import re

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from models.chainer import ChainerClassifier, MLP3, MLP4


def get_estimator(
    estimator_name: str,
    gpu: int,
    n_out: int,
    seed: int,
):
    if estimator_name == 'random_forest':
        return RandomForestClassifier()
    elif estimator_name == 'logistic_regression':
        return LogisticRegression()
    elif estimator_name == 'logistic_regression_sag':
        return LogisticRegression(solver='sag')
    elif estimator_name == 'logistic_regression_saga':
        return LogisticRegression(solver='saga')
    elif estimator_name == 'extra_tree':
        return ExtraTreesClassifier()
    elif estimator_name == 'linear_svc':
        return LinearSVC(
            penalty='l1', loss='squared_hinge', dual=False, C=50,
        )
    elif estimator_name == 'gbdt':
        return XGBClassifier(
            objective='multi:softmax',
            learning_rate=0.05,
            max_depth=5,
            n_estimators=1000,
            nthread=10,
            subsample=0.5,
            colsample_bytree=1.0,
            random_state=seed,
        )
    elif estimator_name == 'mlp-3':
        return ChainerClassifier(
            MLP3, gpu=gpu, n_epoch=100, n_out=n_out,
        )
    elif estimator_name == 'mlp-4':
        return ChainerClassifier(
            MLP4, gpu=gpu, n_epoch=200, n_out=n_out,
        )

    m = re.fullmatch('knn-([0-9]+)', estimator_name)
    if m:
        k = int(m.group(1))
        return KNeighborsClassifier(n_neighbors=k)

    raise ValueError(f'Invalid estimator name: {estimator_name}')
