import tables as tb
import numpy as np
import sys
import os
import json


def save(path, matrix, info=None):
    filters = tb.Filters(complevel=5, complib='blosc')
    with tb.open_file(path, 'w') as f:
        data_ = f.create_earray(f.root, 'data', tb.Float32Atom(), shape=(0,), filters=filters)
        data_.append(matrix.ravel())
        shape_ = f.create_earray(f.root, 'shape', tb.Float32Atom(), shape=(0,))
        shape_.append(np.array(matrix.shape))

    if info is not None:
        with open(path + '.json', 'w') as f:
            json.dump(info, f, indent=2, default=str)


def load(path):
    with tb.open_file(path, 'r') as f:
        matrix_raveled = f.root.data[:]
        shape = f.root.shape[:]
        shape = shape.astype(np.int64)
    matrix = matrix_raveled.reshape(shape)

    info_path = path + '.json'
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
    else:
        info = None

    return matrix, info


if __name__ == '__main__':
    x = np.array([[1, 2, 3], [4, 5, 6]])
    print(x)
    save('tmp.h5', x, dict(name='hey', my_favorite_number=3, e='upwpoefaowpiefjaow'))
    #{
    #         'name': 'hoge',
    #         'my_favorite_number': 3
    #     })
    print(load('tmp.h5'))

    x = np.array([[[1, 2], [3, 3.5]], [[4, 5], [6, 7]]])
    print(x)
    save('tmp.h5', x)
    print(load('tmp.h5'))


#
# High-level IO
#
