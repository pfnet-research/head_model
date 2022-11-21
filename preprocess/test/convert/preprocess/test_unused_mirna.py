import unittest

import pandas as pd

from lib.convert.preprocess import unused_mirna as F


class TestInvalidemiRNARemovalFilter(unittest.TestCase):

    def test_filter(self):
        filter_ = F.remove_unused_mirna

        data_actual = {'G_Name':
                       ['hsa-miR-28-3p',
                        'foo',
                        'hsa-miR-2467-3p',
                        'hsa-miR-4448',
                        'hsa-miR-4516',
                        'hsa-miR-4525',
                        'hsa-miR-4710',
                        'hsa-miR-4718',
                        'hsa-miR-614',
                        'hsa-miR-8059'],
                       '635nm': ['1.0', '1.0', '1.0', '1.0', '1.0',
                                 '1.0', '1.0', '1.0', '1.0', '1.0']}
        df = pd.DataFrame(data=data_actual)
        actual = filter_(df)

        data_expect = {'G_Name': ['hsa-miR-28-3p', 'foo'],
                       '635nm': ['1.0', '1.0']}
        expect = pd.DataFrame(data=data_expect)

        pd.testing.assert_frame_equal(expect, actual)
