import copy
import unittest

from lib.dataset import metadata as M


def _create_metadata(n_feature=4, n_label=3):
    fns = ['feature_name_%d' % i for i in range(n_feature)]
    lns = ['labe_name_%d' % j for j in range(n_label)]
    return M.Metadata(fns, lns)


class TestMetadata(unittest.TestCase):

    def setUp(self):
        self.metadata = _create_metadata()

    def test_equal(self):
        other = copy.deepcopy(self.metadata)
        self.assertEqual(self.metadata, other)
