import glob
import os

import pandas as pd
import six

from lib.util.unique_name_generator import UniqueNameGenerator


def _create_feature_vector(df, feature_id_converter):
    fv = {}
    mirna_names = df['G_Name']
    values = df['635nm']

    gen = UniqueNameGenerator()
    for name, value in six.moves.zip(mirna_names, values):
        name = gen.make_unique(name)
        fid = feature_id_converter.to_id(name)
        if fid in fv:
            print('%s is duplicated' % name)
        fv[fid] = value
    return fv


def _is_older_than_v21(df):
    return 'hsa-miR-9500' in list(df['G_Name'])


def fetch_dir(dir_name, feature_id_converter, filters=()):
    feature_vectors = []
    instance_names = []

    for fname in glob.glob('%s/*.txt' % dir_name):
        if not (os.path.isfile(fname)):
            print(str(fname) + ' is not a measurement file.')
            continue

        try:
            df = pd.read_csv(fname, header=6, delimiter='\t')
        except Exception:
            print('Failed to parse: ' + str(fname))
            continue

        if not _is_older_than_v21(df):
            continue

        for _filter in filters:
            df = _filter(df)

        fv = _create_feature_vector(df, feature_id_converter)
        feature_vectors.append(fv)
        instance_names.append(fname)

    return feature_vectors, instance_names
