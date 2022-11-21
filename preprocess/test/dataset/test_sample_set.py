import copy
import unittest

import numpy as np

from lib.dataset import sample_set as S


def _create_sample_set(row=4, col=5):
    fvs = np.random.randn(row, col)
    instance_names = np.random.randint(10, size=(row,))
    labels = np.random.randint(10, size=(row,))
    return S.SampleSet(fvs, instance_names, labels)


class TestSampleSetInvalid(unittest.TestCase):

    def setUp(self):
        self.fvs = np.random.randn(4, 5)
        self.instance_names = np.random.randint(10, size=(4,))
        self.labels = np.random.randint(10, size=(4,))

    def test_invalid_1(self):
        fvs = np.random.randn(40, 5)
        with self.assertRaises(AssertionError):
            S.SampleSet(fvs, self.instance_names, self.labels)

    def test_invalid_2(self):
        instance_names = np.random.randint(10, size=(40,))
        with self.assertRaises(AssertionError):
            S.SampleSet(self.fvs, instance_names, self.labels)

    def test_invalid_(self):
        labels = np.random.randint(10, size=(40,))
        with self.assertRaises(AssertionError):
            S.SampleSet(self.fvs, self.instance_names, labels)


class TestSampleSetEq(unittest.TestCase):

    def setUp(self):
        self.sample_set = _create_sample_set()

    def test_eq(self):
        other = copy.deepcopy(self.sample_set)
        self.assertEqual(self.sample_set, other)

    def test_ineq_1(self):
        other = copy.deepcopy(self.sample_set)
        other.feature_vectors = np.zeros_like(self.sample_set.feature_vectors)
        self.assertNotEqual(self.sample_set, other)

    def test_ineq_2(self):
        other = copy.deepcopy(self.sample_set)
        other.instance_names = np.zeros_like(self.sample_set.instance_names)
        self.assertNotEqual(self.sample_set, other)

    def test_ineq_3(self):
        other = copy.deepcopy(self.sample_set)
        other.labels = np.zeros_like(self.sample_set.labels)
        self.assertNotEqual(self.sample_set, other)


class TestSampleSetLen(unittest.TestCase):

    def setUp(self):
        self.sample_set = _create_sample_set()

    def test_len(self):
        self.assertEqual(len(self.sample_set), 4)


class TestSampleSetAdd(unittest.TestCase):

    def setUp(self):
        self.sample_set = _create_sample_set()
        self.other = copy.deepcopy(self.sample_set)

    def test_add(self):
        expect = S.SampleSet(np.tile(self.sample_set.feature_vectors, (2, 1)),
                             np.tile(self.sample_set.instance_names, 2),
                             np.tile(self.sample_set.labels, 2))
        actual = self.sample_set + self.other
        self.assertEqual(expect, actual)


class TestSampleSetSplit(unittest.TestCase):

    def setUp(self):
        self.sample_set = _create_sample_set()

    def test_normal(self):
        train_idx = [0, 2]
        test_idx = [1, 3]
        train, test = self.sample_set.split(train_idx, test_idx)
        train_expected = S.SampleSet(
            self.sample_set.feature_vectors[train_idx],
            self.sample_set.instance_names[train_idx],
            self.sample_set.labels[train_idx])
        test_expected = S.SampleSet(
            self.sample_set.feature_vectors[test_idx],
            self.sample_set.instance_names[test_idx],
            self.sample_set.labels[test_idx])
        self.assertEqual(train, train_expected)
        self.assertEqual(test, test_expected)

    def test_spit_invalid_1(self):
        with self.assertRaises(ValueError):
            self.sample_set.split([0], [0, 1, 2])

    def test_split_invalid_2(self):
        with self.assertRaises(ValueError):
            self.sample_set.split([0], [1, 2])

    def test_split_invalid_3(self):
        with self.assertRaises(ValueError):
            self.sample_set.split([0], [1, 2, 4])


class TestSampleSort(unittest.TestCase):

    def setUp(self):
        self.sample_set = _create_sample_set()
        self.sample_set.instance_names = np.array([3, 2, 0, 1])

    def test_sort(self):
        idx = [2, 3, 1, 0]
        expect = S.SampleSet(self.sample_set.feature_vectors[idx],
                             self.sample_set.instance_names[idx],
                             self.sample_set.labels[idx])
        self.sample_set.sort()
        self.assertEqual(self.sample_set, expect)
