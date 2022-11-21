import os
import shutil

import numpy as np

from lib.util import util


class SampleSet(object):

    def __init__(self, feature_vectors, instance_names, labels):
        self.feature_vectors = feature_vectors
        self.instance_names = instance_names
        self.labels = labels
        self.validate()

    def validate(self):
        N, D = self.feature_vectors.shape
        util.assert_equal(len(self.instance_names), N)
        util.assert_equal(len(self.labels), N)

    def __len__(self):
        return len(self.feature_vectors)

    def equal(self, other, ignore_fv):
        if ignore_fv:
            return (np.array_equal(self.instance_names, other.instance_names)
                    and np.array_equal(self.labels, other.labels))
        else:
            return (np.allclose(self.feature_vectors, other.feature_vectors)
                    and np.array_equal(self.instance_names,
                                       other.instance_names)
                    and np.array_equal(self.labels, other.labels))

    def __eq__(self, other):
        return self.equal(other, False)

    def __add__(self, other):
        feature_vectors = np.vstack(
            (self.feature_vectors, other.feature_vectors))
        instance_names = np.hstack(
            (self.instance_names, other.instance_names))
        labels = np.hstack((self.labels, other.labels))
        return SampleSet(feature_vectors, instance_names, labels)

    def split(self, train_idx, test_idx):
        N = len(self)
        all_idx = np.hstack((train_idx, test_idx))
        all_idx.sort()
        if not np.array_equal(np.arange(N), all_idx):
            raise ValueError('train_idx and test_idx are invalid')

        train = SampleSet(self.feature_vectors[train_idx],
                          self.instance_names[train_idx],
                          self.labels[train_idx])
        test = SampleSet(self.feature_vectors[test_idx],
                         self.instance_names[test_idx],
                         self.labels[test_idx])
        return train, test

    def sort(self):
        idx = np.argsort(self.instance_names)
        self.feature_vectors = self.feature_vectors[idx]
        self.instance_names = self.instance_names[idx]
        self.labels = self.labels[idx]

    @staticmethod
    def load(in_dir):
        feature_vectors = np.loadtxt(
            os.path.join(in_dir, 'feature_vectors.csv'),
            np.float32, delimiter=',', ndmin=2)
        instance_names = util.load_column(
            os.path.join(in_dir, 'instance_names.txt'))
        labels = util.load_column(
            os.path.join(in_dir, 'labels.txt'), np.int32)
        return SampleSet(feature_vectors, instance_names, labels)

    def dump(self, out_dir):
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.mkdir(out_dir)

        np.savetxt(os.path.join(out_dir, 'feature_vectors.csv'),
                   self.feature_vectors, delimiter=',')

        with open(os.path.join(out_dir, 'instance_names.txt'), 'w+') as o:
            o.write('\n'.join(self.instance_names))

        with open(os.path.join(out_dir, 'labels.txt'), 'w+') as o:
            o.write('\n'.join(map(str, self.labels)))
