import project
import os
import sys
import pickle
import sklearn.model_selection
from data import io


class FoldIndices:

    def __init__(self, seed, n_folds, path_policy):
        self.path_policy = path_policy
        if seed is not None:
            path = path_policy.get_fold_indices_file_path(seed, n_folds)
            if not os.path.exists(path):
                self.fold_indices = self._create_split_file(seed, n_folds)
            else:
                with open(path, 'rb') as f:
                    self.fold_indices = pickle.load(f)

        self.n_train = self._get_n_examples_train()
        self.n_test = self._get_n_examples_test()

    def n_examples_of_fold(self, fold):
        if fold == 'test':
            return self.n_test
        elif fold == 'train':
            return self.n_train
        else:
            return len(self.fold_indices[fold])

    def split_to_fold_wise(self, y, folds):
        ns = [self.n_examples_of_fold(fold) for fold in folds]
        assert(len(y) == sum(ns))
        i = 0
        ret = []
        for n in ns:
            ret.append(y[range(i, i + n)])
            i += n
        return ret

    def _split(self, seed, n_folds):
        path = self.path_policy.get_file_path_by_name(None, None, 'y', 'train', None, unsupervised=True)
        y, info = io.load(path)
        y = y.ravel()
        print(info, file=sys.stderr)
        skf = sklearn.model_selection.StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=seed)
        x_dummy = [None] * len(y)
        folds = [test_indices for _, test_indices in skf.split(x_dummy, y)]

        print([len(fold) for fold in folds])

        return folds

    def _create_split_file(self, seed, n_folds):
        fold_indices = self._split(seed, n_folds)
        d = self.path_policy.get_directory_path_by_seed(seed, n_folds)
        os.makedirs(d, exist_ok=True)
        with open(self.path_policy.get_fold_indices_file_path(seed, n_folds), 'wb') as f:
            pickle.dump(fold_indices, f)
        return fold_indices

    def _get_n_examples(self, data_name):
        # TODO: better way
        path = self.path_policy.get_file_path_by_name(None, None, 'raw', data_name, None, unsupervised=True)
        y, info = io.load(path)
        print(info, file=sys.stderr)
        return len(y)

    def _get_n_examples_train(self):
        return self._get_n_examples('train')

    def _get_n_examples_test(self):
        return self._get_n_examples('test')

