import unittest

import numpy as np
import pandas as pd

from lib.convert import preprocess


class TestWhiteningFilter(unittest.TestCase):

    def test_filter(self):
        filter_ = preprocess.whiten

        val_actual = np.array([1.0, 2.0, 9.0], dtype=np.float32)
        data_actual = {'G_Name': ['hsa-miR-28-3p', 'foo', 'hsa-miR-9500'],
                       '635nm': val_actual}
        df = pd.DataFrame(data=data_actual)
        actual = filter_(df)

        mean = val_actual.mean()
        std = val_actual.std()
        val_expect = (val_actual - mean) / std
        data_expect = {'G_Name': ['hsa-miR-28-3p', 'foo', 'hsa-miR-9500'],
                       '635nm': val_expect}
        expect = pd.DataFrame(data=data_expect)

        pd.testing.assert_frame_equal(expect, actual)


class TestWhiten(unittest.TestCase):

    def setUp(self):
        self.x = np.array([1, 2, 3], dtype=np.float32)
        mean = 2.
        std = np.sqrt(2 / 3)
        self.expect = (self.x - mean) / std

    def test_ndarray(self):
        actual = preprocess.whiten_._whiten(self.x)
        np.testing.assert_array_equal(actual, self.expect)

    def test_data_frame(self):
        x = pd.DataFrame(self.x)
        actual = preprocess.whiten_._whiten(x)
        pd.testing.assert_frame_equal(actual, pd.DataFrame(self.expect))
