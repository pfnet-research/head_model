#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np

from lib.convert import postprocess
from lib.convert import preprocess
from lib.io import dump
from lib.io import load
from lib.io import sample_directory
from lib.util import util
from lib.util.id_converter import IDConverter
from lib.util.feature_converter import convert_feature_vectors


def fetch(root_dir, dataset, filters):
    feature_vectors = []
    instance_names = []
    labels = []

    feature_id_converter = IDConverter()
    label_id_converter = IDConverter()

    for dir_name, label_name in dir_names:
        dir_name = os.path.join(root_dir, dir_name)
        if not os.path.isdir(dir_name):
            print(dir_name + ' not found.')
            continue

        print('Processing ' + dir_name)

        feature_vectors_, instance_names_ = load.fetch_dir(
            dir_name, feature_id_converter, filters)
        lid = label_id_converter.to_id(label_name)
        labels_ = [lid] * len(feature_vectors_)

        util.assert_equal(len(feature_vectors_), len(instance_names_))
        util.assert_equal(len(feature_vectors_), len(labels_))

        feature_vectors += feature_vectors_
        instance_names += instance_names_
        labels += labels_

    # Convert data
    feature_names = feature_id_converter.id2name
    label_names = label_id_converter.id2name
    feature_vectors = convert_feature_vectors(
        feature_vectors, feature_id_converter.unique_num, True)
    instance_names = np.array(instance_names)
    labels = np.array(labels)

    # shuffle
    feature_vectors, instance_names, labels = util.shuffle(
        feature_vectors, instance_names, labels)

    return feature_names, label_names, feature_vectors, instance_names, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preprocessor')
    parser.add_argument('--in-dir', '-i', type=str, default='.')
    parser.add_argument('--out-dir', '-o', type=str, default='out')
    parser.add_argument('--num-fold', '-k', type=int, default=5)
    parser.add_argument('--dataset', '-d', type=str, default='full',
                        choices=('full', 'test_ternary_large',
                                 'test_binary_large', 'test_binary_small'))
    parser.add_argument('--use-ngs-collated-mirna-only', '-u',
                        action='store_true')
    parser.add_argument('--whiten-before-split', '-w', action='store_true',
                        help='Normalize the expression levels '
                        'along feature dimensions. '
                        'the Note that it whitens BEFORE splitting '
                        'the dataset into train and test.')
    parser.add_argument('--whiten-after-split', '-W', action='store_true',
                        help='Normalize the expression levels '
                        'along feature dimensions. '
                        'Note that it whitens AFTER splitting '
                        'the dataset into train and test.')
    parser.add_argument('--normalize', '-n', action='store_true',
                        help='Normalize the expression levels '
                        'along samples (after splitting).')
    parser.add_argument('--use-important-mirna-only', '-I',
                        type=int, default=-1,
                        help='# of adopted miRNAs based on importance score. '
                        'If it is a non-positive value, '
                        'it does not filter miRNAs '
                        'based on importance scores.')
    parser.add_argument('--remove-unused-mirnas', '-D', action='store_true',
                        help='If true, we remove reported unused miRNAs '
                        'from feature dimensions.')
    args = parser.parse_args()

    if args.dataset == 'full':
        dir_names = sample_directory.DIR_NAMES
    elif args.dataset == 'test_ternary_large':
        dir_names = sample_directory.TEST_DIR_NAMES_TERNARY_LARGE
    elif args.dataset == 'test_binary_large':
        dir_names = sample_directory.TEST_DIR_NAMES_BINARY_LARGE
    elif args.dataset == 'test_binary_small':
        dir_names = sample_directory.TEST_DIR_NAMES_BINARY_SMALL
    else:
        raise ValueError('invalid dataset type:{}'.format(args.dataset))

    preprocess_filters = list(preprocess.DEFAULT_FILTERS)
    if args.remove_unused_mirnas:
        preprocess_filters.append(
            preprocess.remove_unused_mirna)
    if args.normalize:
        # We must keep negative controls because the normalize
        # operation needs them.
        # After the normalizer performs normalization,
        # it drops feature dimensions other than hsa miRNAs.
        preprocess_filters.remove(preprocess.use_hsa_mirna_only)
    if args.use_ngs_collated_mirna_only:
        preprocess_filters.append(
            preprocess.use_ngs_collated_mirna)
    if args.whiten_before_split:
        preprocess_filters.append(preprocess.whiten)

    d = fetch(args.in_dir, dir_names, preprocess_filters)
    feature_names = d[0]
    label_names = d[1]
    feature_vectors = d[2]
    instance_names = d[3]
    labels = d[4]

    postprocess_filters = []
    if args.whiten_after_split:
        postprocess_filters.append(postprocess.whiten)
    if args.normalize:
        postprocess_filters.append(postprocess.normalize)
    if args.use_important_mirna_only > 0:
        postprocess_filters.append(
            postprocess.create_importance_score_filter(
                args.use_important_mirna_only))

    dump.dump_k_fold(args.out_dir, args.num_fold,
                     feature_names, label_names,
                     feature_vectors, instance_names, labels,
                     postprocess_filters)
