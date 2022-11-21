import os
import unittest

import numpy as np
import pandas as pd

import lib.convert.preprocess.data
from lib.convert.preprocess import ngs_collated_mirna as F


class TestNGSCollatedmiRNAFilter(unittest.TestCase):

    def test_filter(self):
        filter_ = F.use_ngs_collated_mirna

        data = {'G_Name': ['hsa-miR-28-3p', 'foo', 'hsa-miR-9500'],
                '635nm': ['1.0', '2.0', '10.0']}

        df = pd.DataFrame(data=data)
        expect = pd.DataFrame(data={'G_Name': ['hsa-miR-28-3p'],
                                    '635nm': ['1.0']})
        actual = filter_(df)

        pd.testing.assert_frame_equal(expect, actual)

    def test_no_false_negative(self):
        filter_ = F.use_ngs_collated_mirna

        dir_ = os.path.dirname(lib.convert.preprocess.data.__file__)
        fname = os.path.join(dir_, 'miRList_NGS3Dgene_posicorr.csv')
        mirnas = pd.read_csv(fname, delimiter=',')
        mirnas['G_Name'] = mirnas['Name']
        mirnas['635nm'] = np.random.uniform(size=(len(mirnas),))

        actual = filter_(mirnas)
        np.testing.assert_array_equal(mirnas.values, actual.values)
