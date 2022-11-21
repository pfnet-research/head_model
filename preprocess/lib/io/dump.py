import glob
import os
import shutil
import sys

import numpy as np
from sklearn import model_selection

from lib.dataset import dataset as D
from lib.dataset import metadata as M
from lib.dataset import sample_set as S
from lib.util import util


def _validate(metadata, sample_set):
    util.assert_equal(len(sample_set.feature_vectors.shape), 2)
    N, D = sample_set.feature_vectors.shape

    util.assert_equal(len(metadata.feature_names), D)
    util.assert_equal(len(sample_set.instance_names), N)
    util.assert_equal(len(sample_set.labels), N)

    M = len(metadata.label_names)
    assert all([l < M for l in sample_set.labels]), 'invalid labels'


def dump_k_fold(out_dir, num_fold,
                feature_names, label_names,
                feature_vectors, instance_names, labels,
                filters):

    """Dumps k fold cross validation dataset

    Expected output
    out/
    ├── 0
    │   ├── feature_names.txt
    │   ├── label_names.txt
    │   ├── test
    │   │   ├── feature_vectors.csv
    │   │   ├── instance_names
    │   │   └── labels.txt
    │   └── train
    │       ├── feature_vectors.csv
    │       ├── instance_names
    │       └── labels.txt
    .
    .
    .
    └── 4
        ├── feature_names.txt
        ├── label_names.txt
        ├── test
        │   ├── feature_vectors.csv
        │   ├── instance_names
        │   └── labels.txt
        └── train
            ├── feature_vectors.csv
            ├── instance_names
            └── labels.txt
    """

    if os.path.exists(out_dir):
        print('Directory %s exists. Remove it.' % out_dir)
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    metadata = M.Metadata(feature_names, label_names)
    sample_set = S.SampleSet(feature_vectors, instance_names, labels)
    _validate(metadata, sample_set)

    skf = model_selection.StratifiedKFold(num_fold)
    place_holder = np.zeros_like(labels[:, None])
    for i, (train_idx, test_idx) in enumerate(skf.split(place_holder, labels)):
        train, test = sample_set.split(train_idx, test_idx)
        dataset = D.Dataset(metadata, train, test)
        for f in filters:
            dataset = f(dataset)
        out_dir_for_this_fold = os.path.join(out_dir, str(i))
        dataset.dump(out_dir_for_this_fold)

    dump_myself(sys.argv[0], os.path.join(out_dir, 'script'))


def dump_myself(src_dir, dst_dir):
    if os.path.exists(dst_dir):
        print('Directory %s exists. Remove it.' % dst_dir)
        shutil.rmtree(dst_dir)
    os.mkdir(dst_dir)

    for f in glob.glob(os.path.join(src_dir, '**/*.py'), recursive=True):
        rel_path = os.path.relpath(f, src_dir)
        dst_path = os.path.join(dst_dir, rel_path)
        dir_name = os.path.dirname(dst_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        shutil.copy2(f, dst_path)
