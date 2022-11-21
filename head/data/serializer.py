import os
import sys
import logging
import numpy as np

from data import fold_indices
from data import io


class Serializer:

    def __init__(self, path_policy, seed=None, n_folds=None):
        self.path_policy = path_policy
        self.seed = seed
        self.n_folds = n_folds
        if seed is not None:
            self.fold_indices = fold_indices.FoldIndices(seed, n_folds, path_policy)

    def _load_single_fold(self, feature_name, target_fold, leaveout_folds):
        # Case 1: Supervised (Level > 0)
        x, info = None, None
        if self.seed is not None:
            path = self.path_policy.get_file_path_by_name(
                self.seed, self.n_folds, feature_name, target_fold, leaveout_folds)
            print('{}: {}'.format(path, os.path.exists(path)))
            if os.path.exists(path):
                x, info = io.load(path)

        # Case 2: Unsupervised (Level == 0)
        t = 'train' if isinstance(target_fold, int) else target_fold
        path = self.path_policy.get_file_path_by_name(
            None, None, feature_name, t, leaveout_folds, unsupervised=True)
        print('{}: {}'.format(path, os.path.exists(path)))
        if os.path.exists(path):
            assert(x is None)
            x, info = io.load(path)
            if isinstance(target_fold, int):
              x = x[self.fold_indices.fold_indices[target_fold]]

        assert x is not None, "Not found: {}, {}, {}, {}, {}".format(
            self.seed, self.n_folds, feature_name, target_fold, leaveout_folds)
        print(info, file=sys.stderr)  # TODO
        return x

    def _load_single_feature(self, feature_name, target_folds, leaveout_folds):
        # TODO: faster loading by merging training folds
        if isinstance(target_folds, list):
            return np.vstack(tuple(
                self._load_single_fold(feature_name, target_fold, leaveout_folds)
                for target_fold in target_folds
            ))
        else:
            return self._load_single_fold(feature_name, target_folds, leaveout_folds)

    def load(self, feature_names, target_folds, leaveout_folds=list()):
        return np.hstack(tuple(
            self._load_single_feature(feature_name, target_folds, leaveout_folds)
            for feature_name in feature_names
        ))

    def save(self, o, feature_name, target_folds, info=None, unsupervised=False):
        dir_path = self.path_policy.get_directory_path_by_name(
            self.seed, self.n_folds, feature_name, unsupervised)
        os.makedirs(dir_path, exist_ok=True)

        if len(target_folds) > 1:
            oo = self.fold_indices.split_to_fold_wise(o, target_folds)
        else:
            oo = [o]

        for i in range(len(target_folds)):
            leaveout_folds = target_folds[:]
            leaveout_folds.remove(target_folds[i])
            leaveout_folds.sort()

            path = self.path_policy.get_file_path_by_name(
                self.seed, self.n_folds, feature_name, target_folds[i], leaveout_folds, unsupervised)

            io.save(path, oo[i], info)

    def version_file_path(self, feature_name, unsupervised):
        dir_path = self.path_policy.get_directory_path_by_name(
            self.seed, self.n_folds, feature_name, unsupervised=unsupervised)
        return os.path.join(dir_path, 'version.txt')

    def load_version(self, feature_name):
        # Case 1: Unsupervised (level == 0)
        version_file_path = self.version_file_path(feature_name, unsupervised=True)
        if os.path.exists(version_file_path):
            with open(version_file_path) as f:
                return int(f.read())

        # Case 2: Supervised (level > 0)
        if self.seed is not None:
            version_file_path = self.version_file_path(feature_name, unsupervised=False)
            if os.path.exists(version_file_path):
                with open(version_file_path) as f:
                    return int(f.read())

        return -1

    def save_version(self, feature_name, version, unsupervised=False):
        version_file_path = self.version_file_path(feature_name, unsupervised)
        with open(version_file_path, 'w') as f:
            f.write(str(version))

    def is_cached(self, output_name, version=0):
        assert version >= 0
        if self.load_version(output_name) >= version:
            assert self.load_version(output_name) == version, "You are downgrading...??"
            logging.info('Skipping cached step: {}, version={}'.format(output_name, version))
            return True
        return False