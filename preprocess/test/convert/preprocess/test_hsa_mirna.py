import unittest

import numpy as np
import pandas as pd

from lib.convert.preprocess import hsa_mirna as F


class TestUseHSAmiRNAFilter(unittest.TestCase):

    def test_filter(self):
        filter_ = F.use_hsa_mirna_only

        data_actual = {'G_Name': ['hsa-miR-28-3p', 'foo',
                                  'hsa-miR-1290', 'hsa-miR-451a',
                                  'hsa-miR-5100', 'hsa-miR-4448'],
                       '635nm': ['1.0', '1.0', '1.0',
                                 '1.0', '1.0', '1.0']}
        df = pd.DataFrame(data=data_actual)
        actual = filter_(df)

        data_expect = {'G_Name': ['hsa-miR-28-3p', 'hsa-miR-1290',
                                  'hsa-miR-451a', 'hsa-miR-5100',
                                  'hsa-miR-4448'],
                       '635nm': ['1.0', '1.0', '1.0',
                                 '1.0', '1.0']}
        expect = pd.DataFrame(data=data_expect)

        np.testing.assert_array_equal(expect.values, actual.values)
