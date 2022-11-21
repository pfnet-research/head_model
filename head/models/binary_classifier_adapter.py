import numpy as np
import sklearn.metrics


class BinaryClassifierAdapter:

    def __init__(self, model_gen_func, classes=[7]):
        self.classes = classes
        self.model_gen_func = model_gen_func
        self.models = None

    def fit_and_validate(self, tra_x, tra_y, val_x, val_y):
        self.models = []
        for c in self.classes:
            tra_y_c = (tra_y == c)
            val_y_c = (val_y == c)
            model = self.model_gen_func()

            if hasattr(model, 'fit_and_validate'):
                model.fit_and_validate(tra_x, tra_y_c, val_x, val_y_c)
            else:
                model.fit(tra_x, tra_y_c)
            self.models.append(model)

            val_o = model.predict_proba(val_x)
            val_p = val_o.argmax(axis=1)
            print(sklearn.metrics.classification_report(val_y_c, val_p, digits=4))

    def transform(self, x):
        return np.hstack(tuple(
            model.predict_proba(x)
            for model in self.models
        ))