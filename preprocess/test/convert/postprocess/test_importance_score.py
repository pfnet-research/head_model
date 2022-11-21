import unittest

import numpy as np
import pandas as pd

from lib.convert import postprocess
from lib.dataset import metadata as M
from lib.dataset import sample_set as S
from lib.dataset import dataset as D


def _create_sample_set(row=4, col=5, n_label=3):
    fvs = np.random.randn(row, col)
    instance_names = np.array(['instance_%d' % i for i in range(row)])
    labels = np.random.randint(n_label, size=(row,))
    return S.SampleSet(fvs, instance_names, labels)


def _create_metadata(n_feature=5, n_label=3):
    fns = np.array(['feature_name_%d' % i for i in range(n_feature)])
    lns = np.array(['label_name_%d' % j for j in range(n_label)])
    return M.Metadata(fns, lns)


class TestImportanceScoreFilter(unittest.TestCase):

    def setUp(self):
        metadata = _create_metadata()
        train = _create_sample_set()
        test = _create_sample_set()
        self.dataset = D.Dataset(metadata, train, test)

    def test_non_positive_size_1(self):
        with self.assertRaises(ValueError):
            postprocess.create_importance_score_filter(0)

    def test_non_positive_size_2(self):
        with self.assertRaises(ValueError):
            postprocess.create_importance_score_filter(-1)

    def test_normal(self):
        size = 2
        score = pd.DataFrame(data={
            'score': [3., 1., 2., 0., -1.],
            'feature_name': ['feature_name_%d' % i for i in range(5)]})
        f = postprocess.create_importance_score_filter(size, score)
        dataset_actual = f(self.dataset)

        fns = ['feature_name_0', 'feature_name_2']
        lns = ['label_name_%d' % j for j in range(3)]
        metadata = M.Metadata(fns, lns)

        train = self.dataset.train
        fvs = train.feature_vectors[:, [0, 2]]
        train = S.SampleSet(fvs, train.instance_names, train.labels)

        test = self.dataset.test
        fvs = test.feature_vectors[:, [0, 2]]
        test = S.SampleSet(fvs, test.instance_names, test.labels)

        dataset_expect = D.Dataset(metadata, train, test)
        assert dataset_expect == dataset_actual
