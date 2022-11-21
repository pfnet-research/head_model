import math
import unittest

import numpy as np
import pandas as pd

from lib.convert import postprocess
from lib.dataset import dataset as D_
from lib.dataset import metadata as M
from lib.dataset import sample_set as S


class TestNormalize(unittest.TestCase):

    def _make_sample_set(self, N, D, K):
        x = np.random.uniform(size=(N, D)).astype(np.float32)
        names = np.arange(N).astype(str)
        y = np.random.randint(0, K, (N,), dtype=np.int32)
        return S.SampleSet(x, names, y)

    def _make_metadata(self, D, K):
        neg_cons = ['Negative Control 2',
                    'Negative Control 2_1',
                    'Negative Control 2_2']
        control_mirnas = postprocess.normalize_.Normalizer.control_mirnas
        other_mirnas = ['hsa-miR-9500p']
        minimum_columns = np.array(neg_cons + control_mirnas + other_mirnas,
                                   dtype=str)
        D_min = len(minimum_columns)
        assert D >= D_min
        feature_names = np.hstack((minimum_columns,
                                   np.arange(D - D_min).astype(str)))
        label_names = np.arange(K).astype(str)
        return M.Metadata(feature_names, label_names)

    def setUp(self):
        N_train = 10
        N_test = 5
        D = 8
        K = 3
        train = self._make_sample_set(N_train, D, K)
        test = self._make_sample_set(N_test, D, K)
        metadata = self._make_metadata(D, K)
        self.dataset = D_.Dataset(metadata, train, test)

    def test(self):
        postprocess.normalize(self.dataset)


class TestNormalizer(unittest.TestCase):

    def setUp(self):
        self.normalizer = postprocess.normalize_.Normalizer()
        self.neg_cons = ['Negative Control 2',
                         'Negative Control 2_1',
                         'Negative Control 2_2']
        self.control_mirnas = postprocess.normalize_.Normalizer.control_mirnas
        self.other_mirnas = ['hsa-miR-9500p']

    def test_shape(self):
        columns = self.neg_cons + self.control_mirnas + self.other_mirnas
        x = np.random.uniform(size=(2, len(columns))).astype(np.float32)
        df = pd.DataFrame(x, columns=columns)
        actual = self.normalizer.fit_transform(df)

        columns = self.control_mirnas + self.other_mirnas
        x = np.random.uniform(size=(2, len(columns))).astype(np.float32)
        expect = pd.DataFrame(x, columns=columns)
        pd.testing.assert_index_equal(actual.index, expect.index)
        pd.testing.assert_index_equal(actual.columns, expect.columns)

    def test_mean_of_control_mirnas(self):
        columns = self.neg_cons + self.control_mirnas + self.other_mirnas
        x = np.random.uniform(size=(2, len(columns))).astype(np.float32)
        df = pd.DataFrame(x, columns=columns)
        df_after = self.normalizer.fit_transform(df)
        actual = df_after[self.control_mirnas].mean()
        expect = postprocess.normalize_.Normalizer.preset_value
        np.testing.assert_array_equal(actual, expect)

    def check_value_errors(self, df):
        with self.assertRaises(ValueError):
            self.normalizer.fit_transform(df)

        with self.assertRaises(ValueError):
            self.normalizer.transform(df)

    def test_no_control_mirnas(self):
        columns = self.neg_cons + self.other_mirnas
        x = np.random.uniform(size=(2, len(columns))).astype(np.float32)
        df = pd.DataFrame(x, columns=columns)
        self.check_value_errors(df)

    def test_one_control_mirnas(self):
        columns = self.neg_cons + [self.control_mirnas[0]] + self.other_mirnas
        x = np.random.uniform(size=(2, len(columns))).astype(np.float32)
        df = pd.DataFrame(x, columns=columns)
        self.check_value_errors(df)


class TestSamplewisePreprocess(unittest.TestCase):

    def setUp(self):
        self.neg_cons = ['Negative Control 2',
                         'Negative Control 2_1',
                         'Negative Control 2_2']
        self.control_mirnas = list(
            postprocess.normalize_.Normalizer.control_mirnas)
        self.other_mirnas = ['hsa-miR-9400p']

    def test_shape(self):
        index = self.neg_cons + self.control_mirnas + self.other_mirnas
        x = np.random.uniform(size=len(index)).astype(np.float32)
        col = pd.Series(x, index=index)

        actual = postprocess.normalize_.samplewise_preprocess(col)
        index_expect = self.control_mirnas + self.other_mirnas
        expect = pd.Series(x[:len(index_expect)], index_expect)
        pd.testing.assert_index_equal(actual.index, expect.index)

    def test_no_negative_controls(self):
        index = self.control_mirnas + self.other_mirnas
        x = np.random.uniform(size=len(index)).astype(np.float32)
        col = pd.Series(x, index=index)
        with self.assertRaises(ValueError):
            postprocess.normalize_.samplewise_preprocess(col)

    def test_absent_mirna_value(self):
        index = self.neg_cons + self.other_mirnas
        x = np.array([90, 110, 1000, -100], dtype=np.float32)
        col = pd.Series(x, index=index)
        actual = postprocess.normalize_.samplewise_preprocess(col)
        index_expect = self.other_mirnas
        expect = pd.Series([0.1], index_expect, dtype=np.float32)
        pd.testing.assert_series_equal(actual, expect)

    def test_present_mirna_value(self):
        index = self.neg_cons + self.other_mirnas
        x = np.array([-1000, 100, 1000, 110], dtype=np.float32)
        col = pd.Series(x, index=index)
        actual = postprocess.normalize_.samplewise_preprocess(col)
        index_expect = self.other_mirnas
        expect_value = math.log2(110 - 100)
        expect = pd.Series([expect_value], index_expect, dtype=np.float32)
        pd.testing.assert_series_equal(actual, expect)


class TestComuteControlStatistics(unittest.TestCase):

    def setUp(self):
        self.neg_cons = ['Negative Control 2',
                         'Negative Control 2_1',
                         'Negative Control 2_2',
                         'Negative Control 2_3']
        self.other_mirnas = ['hsa-miR-9400p']

    def test_no_negative_controls(self):
        x = np.random.uniform(size=len(self.other_mirnas)).astype(np.float32)
        col = pd.Series(x, index=self.other_mirnas)
        with self.assertRaises(ValueError):
            postprocess.normalize_.compute_control_statistics(col)

    def test_few_negative_controls(self):
        index = self.neg_cons[:2]
        x = np.random.uniform(size=len(index)).astype(np.float32)
        col = pd.Series(x, index=index)
        with self.assertRaises(ValueError):
            postprocess.normalize_.compute_control_statistics(col)

    def test_mean_std(self):
        index = self.neg_cons + self.other_mirnas
        x = np.array([90, 100, -1000, 1000, -100], dtype=np.float32)
        col = pd.Series(x, index=index)
        actual = postprocess.normalize_.compute_control_statistics(col)
        expect = pd.np.array([95., 5.], dtype=np.float32)
        np.testing.assert_almost_equal(actual, expect)

    def test_mean_std_with_three_mirnas(self):
        index = self.neg_cons[:3]
        x = np.array([-1000, 100, 1000], dtype=np.float32)
        col = pd.Series(x, index=index)
        actual = postprocess.normalize_.compute_control_statistics(col)
        expect = pd.np.array([100., 0.], dtype=np.float32)
        np.testing.assert_almost_equal(actual, expect)
