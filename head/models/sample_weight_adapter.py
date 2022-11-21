import numpy as np


class SampleWeightAdapter():
    def __init__(self, model):
        self.model = model

    def fit_and_validate(self, tra_x, tra_y, val_x, val_y):
        n_examples = len(tra_y)
        classes, counts = np.unique(tra_y, return_counts=True)
        print(classes, counts)

        sample_weight = np.zeros(n_examples)
        for i in range(len(classes)):
            sample_weight[tra_y == classes[i]] = 1.0 / counts[i]
        sample_weight = np.sqrt(sample_weight)  # TODO: various normalization

        self.model.fit(tra_x, tra_y, sample_weight=sample_weight)

    def predict_proba(self, x):
        return self.model.predict_proba(x)
