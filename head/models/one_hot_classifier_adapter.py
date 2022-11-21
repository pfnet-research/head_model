import numpy as np


class OneHotClassifierAdapter:

    def __init__(self, model):
        self.model = model

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict_proba(self, x):
        o = self.model.predict(x)
        n_classes = o.max() + 1
        ret = np.vstack(tuple(
            (o == c).astype(np.float64)
            for c in range(n_classes)
        ))
        ret = ret.transpose(1, 0)
        return ret