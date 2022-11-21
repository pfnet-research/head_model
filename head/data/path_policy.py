"""
File naming rules:
* splitting scheme
    * `level0`
    * `<seed>_<n_folds>`
* feature name (e.g., 'raw', 't_sne', 'knn_n=5')
* fold name
    * `test.h5`                          --- seed=None, target_fold='test'
    * `train.h5`                         --- seed=None, target_fold='train'
    * `train_<fold>.h5`                  --- seed=5, target_fold=0
    * `train_<fold>_leaveout=<folds>.h5` --- seed=5, target_fold=0, laveout_folds
"""
import os
import logging


class PathPolicy:

    def __init__(self, working_dir):
        self.working_dir = working_dir

    def get_label_name_path(self):
        return os.path.join(self.working_dir, 'label_names.txt')

    def get_directory_path_by_seed(self, seed, n_folds):
        return os.path.join(self.working_dir, "seed={}_folds={}".format(seed, n_folds))

    def get_directory_path_by_name(self, seed, n_folds, feature_name, unsupervised=False):
        if unsupervised or (seed is None and n_folds is None):
            if not unsupervised:
                assert False
            return os.path.join(
                self.working_dir,
                'unsupervised',
                feature_name)
        else:
            return os.path.join(
                self.get_directory_path_by_seed(seed, n_folds),
                feature_name)

    def get_file_path_by_name(self, seed, n_folds, feature_name, target_fold, leaveout_folds, unsupervised=False):
        if isinstance(target_fold, str):
            assert target_fold in ['train', 'test']
            file_name = "{}.h5".format(target_fold)
        elif leaveout_folds is None or len(leaveout_folds) == 0:
            file_name = "train_fold={}.h5".format(target_fold)
        else:
            if isinstance(leaveout_folds, list):
                leaveout_folds = ','.join(map(str, leaveout_folds))
            file_name = "train_fold={}_leaveout={}.h5".format(target_fold, leaveout_folds)

        return os.path.join(
            self.get_directory_path_by_name(seed, n_folds, feature_name, unsupervised),
            file_name)

    def get_fold_indices_file_path(self, seed, n_folds):
        return os.path.join(
            self.get_directory_path_by_seed(seed, n_folds),
            'fold_indices.p')

    def get_submission_file_path(self, seed, n_folds, feature_name):
        return os.path.join(
            self.get_directory_path_by_name(seed, n_folds, feature_name),
            'submission.txt')
