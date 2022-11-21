import os

import pandas as pd

from lib.dataset import dataset as D
from lib.dataset import metadata as M
from lib.dataset import sample_set as S


_here = os.path.abspath(os.path.dirname(__file__))
_importance_score_file = os.path.join(
    _here, 'data/importance_score_xgboost.csv')
_importance_score = pd.read_csv(_importance_score_file, index_col=0)


def create_importance_score_filter(
        size, score=None):
    if size <= 0:
        raise ValueError('size must be positive, size={}'.format(size))
    if score is None:
        score = _importance_score
    score = score.sort_values(by=['score'], ascending=False)
    white_list = score[:size]['feature_name']

    def _filter(dataset):
        return _filter_by_white_list(dataset, white_list)
    return _filter


def _filter_by_white_list(dataset, white_list):
    metadata = dataset.metadata
    feature_names = pd.Series(metadata.feature_names)
    is_used = feature_names.isin(white_list)

    metadata = _filter_metadata(metadata, is_used)
    train = _filter_sample_set(dataset.train, is_used)
    test = _filter_sample_set(dataset.test, is_used)
    return D.Dataset(metadata, train, test)


def _filter_metadata(metadata, is_used):
    names = metadata.feature_names[is_used]
    label_names = metadata.label_names
    return M.Metadata(names, label_names)


def _filter_sample_set(sample_set, is_used):
    fv = sample_set.feature_vectors
    fv = fv[:, is_used]
    names = sample_set.instance_names
    labels = sample_set.labels
    return S.SampleSet(fv, names, labels)
