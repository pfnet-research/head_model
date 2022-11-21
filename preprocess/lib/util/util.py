import numpy as np


def assert_equal(a, b):
    assert a == b, '%d != %d' % (a, b)


def shuffle(*arr):
    assert len(arr) > 0, 'arrs hould be non-empty'

    N = len(arr[0])
    for i, a in enumerate(arr):
        assert len(arr[i]) == N, '{}-th array has different length.'

    idx = np.random.permutation(N)
    return tuple(a[idx] for a in arr)


def load_column(fname, dtype=None):
    with open(fname, 'r') as i:
        lines = i.readlines()
    return np.array(list(map(lambda l: l.strip(), lines)), dtype)
