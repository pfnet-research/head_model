import unittest

import numpy as np

from lib.util.feature_converter import convert_feature_vectors


class TestConvertFeatureVector(unittest.TestCase):

    def setUp(self):
        self.fvs = [{0: 1., 1: 2., 2: 3.},
                    {0: 4., 1: 5., 2: 6.}]

    def test_normal(self):
        actual = convert_feature_vectors(self.fvs, 3)
        self.assertIsInstance(actual, np.ndarray)

        expected = np.arange(1, 7).reshape(2, 3).astype(np.float32)
        np.testing.assert_array_equal(actual, expected)

    def test_large_col_num(self):
        actual = convert_feature_vectors(self.fvs, 4, False)
        self.assertIsInstance(actual, np.ndarray)

        expected = np.array([[1., 2., 3., np.nan],
                             [4., 5., 6., np.nan]],
                            dtype=np.float32)
        np.testing.assert_array_equal(actual, expected)

    def test_large_col_num_forbid_empty(self):
        with self.assertRaises(ValueError):
            convert_feature_vectors(self.fvs, 4, True)

    def test_missing_value(self):
        fvs = [{0: 1., 2: 3.},
               {0: 4., 1: 5.}]

        actual = convert_feature_vectors(fvs, 4, False)
        self.assertIsInstance(actual, np.ndarray)

        expected = np.array([[1., np.nan, 3., np.nan],
                             [4., 5., np.nan, np.nan]],
                            dtype=np.float32)
        np.testing.assert_array_equal(actual, expected)

    def test_missing_value_forbid_empty(self):
        fvs = [{0: 1., 2: 3.},
               {0: 4., 1: 5.}]

        with self.assertRaises(ValueError):
            convert_feature_vectors(fvs, 4, True)


class TestConvertFeatureVectorErrorCase(unittest.TestCase):

    def test_invalid_key_type(self):
        with self.assertRaises(ValueError):
            convert_feature_vectors([{'foo': 1.}], 3)

    def test_invalid_key_value(self):
        with self.assertRaises(ValueError):
            convert_feature_vectors([{3: 1.}], 3)
