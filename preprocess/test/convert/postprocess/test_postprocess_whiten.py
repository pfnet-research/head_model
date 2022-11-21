import unittest

import numpy as np
import pandas as pd

from lib.convert.postprocess import whiten_


class TestWhiten(unittest.TestCase):

    def setUp(self):
        self.x = np.array([1, 2, 3], dtype=np.float32)
        mean = 2.
        std = np.sqrt(2 / 3)
        self.expect = (self.x - mean) / std

    def test_ndarray(self):
        actual = whiten_._whiten_array(self.x)
        np.testing.assert_array_equal(actual, self.expect)

    def test_data_frame(self):
        x = pd.DataFrame(self.x)
        actual = whiten_._whiten_array(x)
        pd.testing.assert_frame_equal(actual, pd.DataFrame(self.expect))


class TestWhitenWithTestSet1D(unittest.TestCase):

    def setUp(self):
        self.train = np.array([1, 2, 3], dtype=np.float32)
        self.test = np.array([4, 5, 6], dtype=np.float32)
        mean = 2.
        std = np.sqrt(2 / 3)
        self.expect_train = (self.train - mean) / std
        self.expect_test = (self.test - mean) / std

    def test_ndarray(self):
        actual_train, actual_test = whiten_._whiten_array(
            self.train, self.test)
        np.testing.assert_array_equal(actual_train, self.expect_train)
        np.testing.assert_array_equal(actual_test, self.expect_test)

    def test_data_frame(self):
        train = pd.DataFrame(self.train)
        test = pd.DataFrame(self.test)
        actual_train, actual_test = whiten_._whiten_array(train, test)
        pd.testing.assert_frame_equal(actual_train,
                                      pd.DataFrame(self.expect_train))
        pd.testing.assert_frame_equal(actual_test,
                                      pd.DataFrame(self.expect_test))


class TestWhiten2D(unittest.TestCase):

    def setUp(self):
        self.x = np.array([[1, 2], [3, 4]], dtype=np.float32)
        self.expect = np.array([[-1, -1], [1, 1]], dtype=np.float32)

    def test_ndarray(self):
        actual = whiten_._whiten_array(self.x)
        np.testing.assert_array_equal(actual, self.expect)

    def test_data_frame(self):
        x = pd.DataFrame(self.x)
        actual = whiten_._whiten_array(x)
        pd.testing.assert_frame_equal(actual, pd.DataFrame(self.expect))


class TestWhitenWithTestSet2D(unittest.TestCase):

    def setUp(self):
        self.train = np.array([[1, 2], [3, 4]], dtype=np.float32)
        self.test = np.array([[5, 6], [7, 8]], dtype=np.float32)
        mean = self.train.mean(axis=0)
        std = self.train.std(axis=0, ddof=0)
        self.expect_train = (self.train - mean) / std
        self.expect_test = (self.test - mean) / std

    def test_ndarray(self):
        actual_train, actual_test = whiten_._whiten_array(
            self.train, self.test)
        np.testing.assert_array_equal(actual_train, self.expect_train)
        np.testing.assert_array_equal(actual_test, self.expect_test)

    def test_data_frame(self):
        train = pd.DataFrame(self.train)
        test = pd.DataFrame(self.test)
        actual_train, actual_test = whiten_._whiten_array(train, test)
        pd.testing.assert_frame_equal(actual_train,
                                      pd.DataFrame(self.expect_train))
        pd.testing.assert_frame_equal(actual_test,
                                      pd.DataFrame(self.expect_test))
