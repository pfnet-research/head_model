import math

import pandas as pd

from lib.dataset import dataset as D
from lib.dataset import metadata as M
from lib.dataset import sample_set as S


def normalize(dataset):
    train = dataset.train
    test = dataset.test
    metadata = dataset.metadata

    df_train = pd.DataFrame(train.feature_vectors,
                            columns=metadata.feature_names)
    df_test = pd.DataFrame(test.feature_vectors,
                           columns=metadata.feature_names)

    normalizer = Normalizer()
    df_train = normalizer.fit_transform(df_train)
    df_test = normalizer.transform(df_test)

    train = S.SampleSet(df_train.values,
                        train.instance_names,
                        train.labels)
    test = S.SampleSet(df_test.values,
                       test.instance_names,
                       test.labels)
    metadata = M.Metadata(normalizer.feature_names,
                          metadata.label_names)
    dataset = D.Dataset(metadata, train, test)
    return dataset


class Normalizer(object):

    preset_value = 10.9
    control_mirnas = ['hsa-miR-149-3p',
                      'hsa-miR-2861',
                      'hsa-miR-4463']

    def _transform(self, df, fit):
        missing = set(self.control_mirnas) - set(df.columns)
        if missing:
            msg = 'Input dataframe must have all '\
                  'control miRNAs. '\
                  '{} is missing'.format(missing)
            raise ValueError(msg)
        df_column_normalized = df.apply(samplewise_preprocess, axis=1)
        if fit:
            self.control_mirna_mean = \
                df_column_normalized[self.control_mirnas].mean().mean()
        assert self.control_mirna_mean is not None
        df_normalized = (df_column_normalized - self.control_mirna_mean
                         + self.preset_value)
        return df_normalized

    def fit_transform(self, df):
        df = self._transform(df, True)
        self.feature_names = df.columns
        return df

    def transform(self, df):
        return self._transform(df, False)


def compute_control_statistics(col):
    neg_con_indices = col.index.str.startswith('Negative Control 2')
    num_neg_con = sum(neg_con_indices)
    if num_neg_con < 3:
        # At least two negative controls are removed.
        # So, we need at least three controls to compute
        # mean and std.
        msg = 'col must have at least three negative control.'
        raise ValueError(msg)

    cutoff_num = max(num_neg_con // 20, 1)
    neg_con = col[neg_con_indices].sort_values()
    neg_con = neg_con[cutoff_num:-cutoff_num]
    mean = neg_con.mean()
    std = neg_con.std(ddof=0)
    return mean, std


def samplewise_preprocess(col):
    mean, std = compute_control_statistics(col)
    col = col.loc[col.index.map(lambda x: x.startswith('hsa-'))]

    col -= mean
    present = col > 2 * std
    col[present] = col[present].apply(math.log2)

    # In the original paper [1], they used the commented processing
    # instead of the uncommented one.
    # Following Kawauchi-san@Toray,
    # we change the default values filling to the absent probes.
    # [1] https://onlinelibrary.wiley.com/doi/full/10.1111/cas.12880
    col[~present] = 0.1
    col[col < 0] = 0.1
    # min_val = col[present].min()
    # col[~present] = min_val - 0.1
    # col[col < 0] = min_val - 0.1
    return col
