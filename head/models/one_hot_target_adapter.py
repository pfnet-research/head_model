import numpy as np

class OneHotTargetAdapter:

    def __init__(self, model):
        self.model = model
        self.n_classes = None

    def fit(self, x, y):
        n_examples = len(y)
        self.n_classes = y.max() + 1
        y_one_hot = np.ndarray((n_examples, self.n_classes))
        for c in range(self.n_classes):
            y_one_hot[:,c] = (y == c)
        self.model.fit(x, y_one_hot)

    def predict_proba(self, x):
        return self.model.predict(x)