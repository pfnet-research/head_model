import numpy as np


def whiten(df):
    print(df)
    dtype = df['635nm'].dtype
    val = df['635nm'].astype(np.float32)
    df['635nm'] = _whiten(val).astype(dtype)
    return df


def _whiten(x):
    mean = x.mean(axis=0)
    std = x.std(ddof=0, axis=0)
    return (x - mean) / std
