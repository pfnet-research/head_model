import numpy as np
import six


def convert_feature_vectors(fvs, col_num,
                            forbid_empty_entry=True):
    """Converts feature vectors to numpy ndarray.

    Args:
        fvs (list of dictionaries): list of feature vectors
            fvs[sample_id][feature_id] = value
        col_num: Feature dimension of converted feature vectors.
        forbid_empty_entry (bool): If ``True``, the resulting
        array should not have ``numpy.nan``

    Returns:
        numpy.ndarray: Shape is (row_num, col_num)
            where row_num = len(fvs).
            ret[sample_id][feature_id] = value
            If some feature value is not existent in `fvs`,
            the corresponding element in the output is
            filled with ``numpy.nan``.

    """

    row_num = len(fvs)
    ret = np.full((row_num, col_num), np.nan, dtype=np.float32)
    for i, fv in enumerate(fvs):
        for k, v in six.iteritems(fv):
            if not isinstance(k, int):
                raise ValueError('keys of dictionaries in '
                                 'fvs must be integer.')
            if k >= col_num:
                raise ValueError('keys of dictionaries in '
                                 'fvs must be smaller than col_num.')
            ret[i][k] = v
    if forbid_empty_entry and np.isnan(ret).any():
        raise ValueError('Non-existent feature found')
    return ret
