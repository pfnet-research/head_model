from lib.dataset import dataset as D
from lib.dataset import sample_set as S


def whiten(dataset):
    train = dataset.train
    test = dataset.test
    metadata = dataset.metadata
    train, test = _whiten_sample_set(train, test)
    return D.Dataset(metadata, train, test)


def _whiten_sample_set(train, test):
    train_fv, test_fv = _whiten_array(train.feature_vectors,
                                      test.feature_vectors)
    train_new = S.SampleSet(train_fv, train.instance_names, train.labels)
    test_new = S.SampleSet(test_fv, test.instance_names, test.labels)
    return train_new, test_new


def _whiten_array(train, test=None):
    train, mean, std = _whiten(train)
    if test is None:
        return train
    else:
        test, _, _ = _whiten(test, mean, std)
        return train, test


def _whiten(x, mean=None, std=None):
    if mean is None:
        mean = x.mean(axis=0)
        std = x.std(ddof=0, axis=0)
    return (x - mean) / std, mean, std
