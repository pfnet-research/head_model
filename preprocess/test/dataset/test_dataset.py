import copy
import unittest

import numpy as np

from lib.dataset import dataset as D
from lib.dataset import metadata as M
from lib.dataset import sample_set as S


def _create_sample_set(row=4, col=5, n_label=3):
    fvs = np.random.randn(row, col)
    instance_names = np.array(['instance_%d' % i for i in range(row)])
    labels = np.random.randint(n_label, size=(row,))
    return S.SampleSet(fvs, instance_names, labels)


def _create_metadata(n_feature=5, n_label=3):
    fns = np.array(['feature_name_%d' % i for i in range(n_feature)])
    lns = np.array(['label_name_%d' % j for j in range(n_label)])
    return M.Metadata(fns, lns)


class TestDataset(unittest.TestCase):

    def setUp(self):
        metadata = _create_metadata()
        train = _create_sample_set()
        test = _create_sample_set()
        self.dataset = D.Dataset(metadata, train, test)

    def test_equal(self):
        other = copy.deepcopy(self.dataset)
        self.assertEqual(self.dataset, other)


class TestInvalidDataset(unittest.TestCase):

    def test_train_test_inconsistence(self):
        metadata = _create_metadata()
        train = _create_sample_set(col=1)
        test = _create_sample_set(col=2)
        with self.assertRaises(AssertionError):
            D.Dataset(metadata, train, test)

    def test_feature_dimension_inconsistence(self):
        metadata = _create_metadata()
        train = _create_sample_set(col=10)
        test = _create_sample_set(col=10)
        with self.assertRaises(AssertionError):
            D.Dataset(metadata, train, test)

    def test_train_label_too_large(self):
        metadata = _create_metadata()
        train = _create_sample_set(n_label=100)
        test = _create_sample_set()
        with self.assertRaises(AssertionError):
            D.Dataset(metadata, train, test)

    def test_test_label_too_large(self):
        metadata = _create_metadata()
        train = _create_sample_set()
        test = _create_sample_set(n_label=100)
        with self.assertRaises(AssertionError):
            D.Dataset(metadata, train, test)
