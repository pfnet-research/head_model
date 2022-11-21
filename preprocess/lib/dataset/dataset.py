import os
import shutil

from lib.dataset import metadata as M
from lib.dataset import sample_set as S
from lib.util import util


class Dataset(object):

    def __init__(self, metadata, train, test):
        self.metadata = metadata
        self.train = train
        self.test = test
        self.validate()

    def validate(self):
        self.metadata.validate()
        self.train.validate()
        self.test.validate()

        N_train, D_train = self.train.feature_vectors.shape
        N_test, D_test = self.test.feature_vectors.shape
        util.assert_equal(D_train, D_test)
        util.assert_equal(len(self.metadata.feature_names), D_train)

        M = len(self.metadata.label_names)
        assert all([l < M for l in self.train.labels]), 'invalid labels'
        assert all([l < M for l in self.test.labels]), 'invalid labels'

    def __eq__(self, other):
        return (self.metadata == other.metadata
                and self.train == other.train
                and self.test == other.test)

    @staticmethod
    def load(in_dir):
        metadata = M.Metadata.load(in_dir)
        train = S.SampleSet.load(os.path.join(in_dir, 'train'))
        test = S.SampleSet.load(os.path.join(in_dir, 'test'))
        return Dataset(metadata, train, test)

    @staticmethod
    def load_datasets(in_dir):
        ret = []
        for d in os.listdir(in_dir):
            if not os.path.isdir(os.path.join(in_dir, d)):
                continue
            try:
                dataset = Dataset.load(os.path.join(in_dir, d))
            except Exception:
                continue

            ret.append(dataset)
        return ret

    def dump(self, out_dir):
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.mkdir(out_dir)

        self.metadata.dump(out_dir)
        self.train.dump(os.path.join(out_dir, 'train'))
        self.test.dump(os.path.join(out_dir, 'test'))
