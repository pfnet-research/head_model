import scipy.stats.mstats
import numpy as np


def fix_shape(predictions):
    # BE CAREFUL: Sometimes columns are tarinai...
    n_classes = max((len(p[0]) for p in predictions))
    for i in range(len(predictions)):
        p = predictions[i]
        if len(p[0]) < n_classes:
            predictions[i] = np.hstack(
                (p, np.zeros(shape=(len(p), n_classes - len(p[0])))))
    return predictions


def preprocess(predictions):
    fix_shape(predictions)
    assert type(predictions) == list
    return predictions
    #return list(map(normalize, predictions))


def geometric_mean(predictions):
    predictions = preprocess(predictions)
    for prediction in predictions:
        prediction += 1E-9  # Avoid division zero
    return scipy.stats.mstats.gmean(predictions)


def arithmetic_mean(predictions):
    predictions = preprocess(predictions)
    return np.array(predictions).mean(axis=0)


def mean_function(method='arithmetic'):
    return {
        'geometric': geometric_mean,
        'arithmetic': arithmetic_mean,
    }[method]


def mean(predictions, method='arithmetic'):
    return mean_function(method)(predictions)


def vote(predictions):
    return scipy.stats.mstats.mode(predictions)[0][0]


if __name__ == '__main__':
    print(mean([
        np.array([[0.3, 0.7], [0.5, 0.5], [0, 1]]),
        np.array([[4.0, 0, 0, 0], [600, 0, 0, 0], [0.9, 0, 0, 0]]),
    ], 'arithmetic'))
