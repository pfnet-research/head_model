import os
import shutil
import tempfile
import unittest

from lib.convert import preprocess
from lib.io import load


def _create_dummy_measurement_file(fname):
    with open(fname, 'w+') as o:
        [o.write('header\n') for _ in range(6)]
        o.write('\t'.join(['G_Name', '635nm', 'dummy0', 'dummy1']))
        o.write('\n')
        o.write('\t'.join(['hsa-feature_name_0', '10.0', 'foo', 'bar']))
        o.write('\n')
        o.write('\t'.join(['hsa-feature_name_1', '0.0', 'foo', 'bar']))
        o.write('\n')
        o.write('\t'.join(['hsa-feature_name_1', '12.0', 'foo', 'bar']))
        o.write('\n')
        # hsa-miR-9500 is used to judge if the measurement file
        # is newer than v20.
        o.write('\t'.join(['hsa-miR-9500', '1.0', 'foo', 'bar']))
        o.write('\n')
        # These miRNAs should be removed.
        invalid_mirnas = ['hsa-miR-1290', 'hsa-miR-451a',
                          'hsa-miR-5100', 'hsa-miR-4448']
        for name in invalid_mirnas:
            o.write('\t'.join([name, '1.0', 'foo', 'bar']))
            o.write('\n')
        # Probes which do not start with "hsa-" should be removed
        o.write('\t'.join(['feature_name_0', '12.0', 'foo', 'bar']))
        o.write('\n')


def _create_dummy_v20_measurement_file(fname):
    # ignored as hsa-miR-9500 is absent
    with open(fname, 'w+') as o:
        [o.write('header\n') for _ in range(6)]
        o.write('\t'.join(['G_Name', '635nm', 'dummy0', 'dummy1']))
        o.write('\n')
        o.write('\t'.join(['hsa-feature_name_0', '10.0', 'foo', 'bar']))
        o.write('\n')


def _create_dummy_invalid_measurement_file(fname):
    open(fname, 'w+')


class DummyFeatureIDConverter(object):

    def __init__(self):
        self.name2id = {}

    def to_id(self, name):
        if name not in self.name2id:
            self.name2id[name] = len(self.name2id)
        return self.name2id[name]


class TestFetchDir(unittest.TestCase):

    def setUp(self):
        self.dir_name = tempfile.mkdtemp()
        _create_dummy_measurement_file(os.path.join(self.dir_name, '0.txt'))
        _create_dummy_measurement_file(os.path.join(self.dir_name, '1.txt'))
        _create_dummy_measurement_file(os.path.join(self.dir_name, '2.csv'))
        os.mkdir(os.path.join(self.dir_name, 'tmp'))
        _create_dummy_measurement_file(
            os.path.join(self.dir_name, 'tmp', '3.txt'))
        _create_dummy_invalid_measurement_file(
            os.path.join(self.dir_name, '4.txt'))
        _create_dummy_v20_measurement_file(
            os.path.join(self.dir_name, '5.txt'))
        self.converter = DummyFeatureIDConverter()
        self.filter_ = preprocess.DEFAULT_FILTERS

    def tearDown(self):
        if os.path.exists(self.dir_name):
            shutil.rmtree(self.dir_name)

    def test_normal(self):
        feature_vectors, instance_names = load.fetch_dir(
            self.dir_name, self.converter, self.filter_)

        self.assertEqual(len(feature_vectors), 2)
        self.assertEqual(feature_vectors[0],
                         {0: 10.0, 1: 0.0, 2: 12.0, 3: 1.0})
        self.assertEqual(feature_vectors[1],
                         {0: 10.0, 1: 0.0, 2: 12.0, 3: 1.0})

        self.assertEqual(len(instance_names), 2)
        self.assertEqual(instance_names[0],
                         os.path.join(self.dir_name, '0.txt'))
        self.assertEqual(instance_names[1],
                         os.path.join(self.dir_name, '1.txt'))

    def test_filter(self):
        def _filter(df):
            return df[df['G_Name'] == 'hsa-feature_name_0']

        feature_vectors, instance_names = load.fetch_dir(
            self.dir_name, self.converter, (_filter,))

        self.assertEqual(len(feature_vectors), 2)
        self.assertEqual(feature_vectors[0],
                         {0: 10.0})
        self.assertEqual(feature_vectors[1],
                         {0: 10.0})

        self.assertEqual(len(instance_names), 2)
        self.assertEqual(instance_names[0],
                         os.path.join(self.dir_name, '0.txt'))
        self.assertEqual(instance_names[1],
                         os.path.join(self.dir_name, '1.txt'))
