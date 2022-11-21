import unittest

import numpy as np

from lib.util import util


class TestShuffle(unittest.TestCase):

    def setUp(self):
        self.a = np.array([1, 2, 3])
        self.b = np.array([1, 2, 3], dtype=np.float32)
        self.c = np.array([1, 2, 3], dtype=object)

    def test_normal(self):
        a, b, c = util.shuffle(self.a, self.b, self.c)
        self.assertEqual(len(a), 3)
        self.assertEqual(len(b), 3)
        self.assertEqual(len(c), 3)

    def test_empty_array(self):
        with self.assertRaises(AssertionError):
            util.shuffle()

    def test_different_length(self):
        b_invalid_length = np.array([1, 2, 3, 4])
        with self.assertRaises(AssertionError):
            util.shuffle(self.a, b_invalid_length)
