import os

import numpy as np

from lib.util import util


class Metadata(object):

    def __init__(self, feature_names, label_names):
        self.feature_names = feature_names
        self.label_names = label_names
        self.validate()

    def validate(self):
        pass

    def __eq__(self, other):
        return (np.array_equal(self.feature_names, other.feature_names)
                and np.array_equal(self.label_names, other.label_names))

    @staticmethod
    def load(in_dir):
        feature_names = util.load_column(
            os.path.join(in_dir, 'feature_names.txt'))
        label_names = util.load_column(
            os.path.join(in_dir, 'label_names.txt'))
        return Metadata(feature_names, label_names)

    def dump(self, out_dir):
        assert os.path.exists(out_dir)
        with open(os.path.join(out_dir, 'feature_names.txt'), 'w+') as o:
            o.write('\n'.join(self.feature_names))

        with open(os.path.join(out_dir, 'label_names.txt'), 'w+') as o:
            o.write('\n'.join(self.label_names))
