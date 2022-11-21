import unittest

import pandas as pd

from lib.convert.preprocess import invalid_measurement as F


class TestInvalidemiRNARemovalFilter(unittest.TestCase):

    def test_filter(self):
        filter_ = F.remove_invalid_measurement_mirna

        data_actual = {'G_Name': ['hsa-miR-28-3p', 'foo',
                                  'hsa-miR-1290', 'hsa-miR-451a',
                                  'hsa-miR-5100', 'hsa-miR-4448'],
                       '635nm': ['1.0', '1.0', '1.0',
                                 '1.0', '1.0', '1.0']}
        df = pd.DataFrame(data=data_actual)
        actual = filter_(df)

        data_expect = {'G_Name': ['hsa-miR-28-3p', 'foo'],
                       '635nm': ['1.0', '1.0']}
        expect = pd.DataFrame(data=data_expect)

        pd.testing.assert_frame_equal(expect, actual)
