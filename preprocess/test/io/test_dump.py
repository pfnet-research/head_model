import os
import shutil
import tempfile
import unittest

import numpy as np

from lib.convert import postprocess
from lib.dataset import dataset as D
from lib.dataset import metadata as M
from lib.dataset import sample_set as S
from lib.io import dump


def _create_sample_set(row=4, col=5, n_label_type=10):
    fvs = np.random.randn(row, col)
    instance_names = np.random.randint(n_label_type, size=(row,))
    labels = np.random.randint(n_label_type, size=(row,))
    return S.SampleSet(fvs, instance_names, labels)


def _create_metadata(n_feature=5, n_label=10):
    fns = ['feature_name_%d' % i for i in range(n_feature)]
    lns = ['labe_name_%d' % j for j in range(n_label)]
    return M.Metadata(fns, lns)


class TestValidate(unittest.TestCase):

    def setUp(self):
        self.metadata = _create_metadata()
        self.sample_set = _create_sample_set()

    def test_normal(self):
        dump._validate(self.metadata, self.sample_set)  # do not raise errors

    def test_invalid_ndim(self):
        self.sample_set.feature_vectors = np.arange(20).reshape(2, 2, 5)
        with self.assertRaises(AssertionError):
            dump._validate(self.metadata, self.sample_set)

    def test_invalid_inconsistent_n_feature(self):
        metadata = _create_metadata(10)
        with self.assertRaises(AssertionError):
            dump._validate(metadata, self.sample_set)

    def test_invalid_inconsistent_label_value(self):
        self.sample_set.labels = np.arange(10) + 1
        with self.assertRaises(AssertionError):
            dump._validate(self.metadata, self.sample_set)


class TestDump(unittest.TestCase):

    def setUp(self):
        self.out_dir = tempfile.mkdtemp()
        self.feature_names = np.array(['f0', 'f1', 'f2'])
        self.label_names = np.array(['l0', 'l1'])
        self.feature_vectors = np.random.uniform(-1, 1, (6, 3))
        self.instance_names = np.array(['i0', 'i1', 'i2', 'i3', 'i4', 'i5'])
        self.labels = np.array([0, 1, 0, 1, 0, 0])
        self.num_fold = 3
        self.normalize = False

    def tearDown(self):
        if os.path.exists(self.out_dir):
            shutil.rmtree(self.out_dir)

    def check_dump(self, whiten):
        datasets = D.Dataset.load_datasets(self.out_dir)
        self.assertEqual(len(datasets), self.num_fold)

        expected_metadata = M.Metadata(self.feature_names, self.label_names)
        expected_sample_set = S.SampleSet(self.feature_vectors,
                                          self.instance_names,
                                          self.labels)

        for dataset in datasets:
            self.assertEqual(dataset.metadata, expected_metadata)
            sample_set = dataset.train + dataset.test
            sample_set.sort()
            expected_sample_set.sort()
            # TODO(Kenta Oono)
            # Do not skip feature vector check when we whiten fvs.
            self.assertTrue(sample_set.equal(expected_sample_set, whiten))

    def test_dump_with_whitening(self):
        filters = [postprocess.whiten]
        dump.dump_k_fold(self.out_dir, self.num_fold,
                         self.feature_names, self.label_names,
                         self.feature_vectors, self.instance_names,
                         self.labels, filters)
        self.check_dump(True)

    def test_dump_without_whitening(self):
        filters = []
        dump.dump_k_fold(self.out_dir, self.num_fold,
                         self.feature_names, self.label_names,
                         self.feature_vectors, self.instance_names,
                         self.labels, filters)
        self.check_dump(False)


class TestDumpWithNormalize(unittest.TestCase):

    def setUp(self):
        self.out_dir = tempfile.mkdtemp()
        # When normalize option is True, feature vectors
        # must have all control miRNAs and at least
        # one negative control miRNA.
        self.kept_feature_names = np.array(
            ['hsa-miR-0', 'hsa-miR-1', 'hsa-miR-2']
            + postprocess.normalize_.Normalizer.control_mirnas,
            dtype=str)
        self.neg_cons = np.array(
            ['Negative Control 2',
             'Negative Control 2_1',
             'Negative Control 2_2',
             'Negative Control 2_3'],
            dtype=str)
        self.feature_names = np.hstack(
            (self.kept_feature_names, self.neg_cons))
        self.label_names = np.array(['l0', 'l1'])
        self.feature_vectors = np.random.uniform(
            -10, 10, (6, len(self.feature_names)))
        self.instance_names = np.array(['i0', 'i1', 'i2', 'i3', 'i4', 'i5'])
        self.labels = np.array([0, 1, 0, 1, 0, 0])
        self.num_fold = 3
        self.normalize = True

    def tearDown(self):
        if os.path.exists(self.out_dir):
            shutil.rmtree(self.out_dir)

    def _is_all_samples_available(self, dataset):
        num_kept_samples = len(self.kept_feature_names)
        expected_sample_set = S.SampleSet(
            self.feature_vectors[:, :num_kept_samples],
            self.instance_names,
            self.labels)
        expected_sample_set.sort()
        sample_set = dataset.train + dataset.test
        sample_set.sort()
        self.assertTrue(sample_set.equal(expected_sample_set, True))
        self.assertEqual(sample_set.feature_vectors.shape,
                         expected_sample_set.feature_vectors.shape)

    def _is_normalized(self, dataset):
        def get_control_mirna_indices(x):
            return np.arange(len(x))[np.isin(x, control_mirnas)]

        x_train = dataset.train.feature_vectors
        feature_names = dataset.metadata.feature_names
        control_mirnas = postprocess.normalize_.Normalizer.control_mirnas
        indices = get_control_mirna_indices(feature_names)
        expected = postprocess.normalize_.Normalizer.preset_value
        actual = x_train[:, indices].mean()
        np.testing.assert_array_almost_equal(expected, actual)

    def check_dump(self, whiten):
        datasets = D.Dataset.load_datasets(self.out_dir)
        self.assertEqual(len(datasets), self.num_fold)

        expected_metadata = M.Metadata(self.kept_feature_names,
                                       self.label_names)
        for dataset in datasets:
            # TODO(Kenta Oono)
            # When we normlize along samples, we remove negative control
            # probes after the actual normalization.
            # That means the dump function reduces the feature dimension
            # of the design matrix.
            # This behavior of the dump function is counter-intuitive and
            # should be corrected.
            self.assertEqual(expected_metadata, dataset.metadata)
            if whiten:
                # TODO(Kenta Oono)
                # Do not skip feature vector check when we whiten fvs.
                self._is_all_samples_available(dataset)
            self._is_normalized(dataset)

    def test_dump_with_whitening(self):
        filters = [postprocess.whiten,
                   postprocess.normalize]
        dump.dump_k_fold(self.out_dir, self.num_fold,
                         self.feature_names, self.label_names,
                         self.feature_vectors, self.instance_names,
                         self.labels, filters)
        self.check_dump(True)

    def test_dump_without_whitening(self):
        filters = [postprocess.normalize]
        dump.dump_k_fold(self.out_dir, self.num_fold,
                         self.feature_names, self.label_names,
                         self.feature_vectors, self.instance_names,
                         self.labels, filters)
        self.check_dump(False)


class TestCopyMyself(unittest.TestCase):

    def setUp(self):
        self.src_dir = tempfile.mkdtemp()
        os.mkdir(os.path.join(self.src_dir, 'buz'))
        open(os.path.join(self.src_dir, 'foo.py'), 'w+')
        open(os.path.join(self.src_dir, 'bar.sh'), 'w+')
        open(os.path.join(self.src_dir, 'buz', 'buz.py'), 'w+')

        self.dst_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.src_dir):
            shutil.rmtree(self.src_dir)
        if os.path.exists(self.dst_dir):
            shutil.rmtree(self.dst_dir)

    def test_copy(self):
        dump.dump_myself(self.src_dir, self.dst_dir)
        files = [[os.path.join(dirpath, f) for f in file_names]
                 for (dirpath, _, file_names) in os.walk(self.dst_dir)]
        files = sum(files, [])  # flatten
        files.sort()
        self.assertEqual(len(files), 2)
        self.assertEqual(files[0], os.path.join(self.dst_dir, 'buz', 'buz.py'))
        self.assertEqual(files[1], os.path.join(self.dst_dir, 'foo.py'))
