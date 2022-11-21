import numpy as np


class Rotator:

    def __init__(self, serializer, input_names, output_name,
                 validation_folds, leaveout_folds=None):
        self.serializer = serializer
        self.input_names = input_names
        self.output_name = output_name

        self.validation_folds = validation_folds
        self.leaveout_folds = [] if leaveout_folds is None else leaveout_folds
        self.train_folds = list(sorted(set(range(serializer.n_folds)).difference(
            self.validation_folds + self.leaveout_folds)))

    def load_x(self, folds):
        lf = list(sorted(set(self.validation_folds + self.leaveout_folds).difference(folds)))
        return self.serializer.load(self.input_names, folds, lf)  # BE CAREFUL!!!!!!

    def load_y(self, folds):
        y = self.serializer.load(['y'], folds)
        return y.ravel().astype(np.int32)  # BE CAREFUL!!!!

    def load_train_x(self):
        return self.load_x(self.train_folds)

    def load_validation_x(self):
        return self.load_x(self.validation_folds)

    def load_test_x(self):
        return self.load_x(['test'])

    def load_train_y(self):
        return self.load_y(self.train_folds)

    def load_validation_y(self):
        return self.load_y(self.validation_folds)

    def load_test_y(self):
        return self.load_y(['test'])

    def save_validation_o(self, o, feature_name, info=None):
        self.serializer.save(o, feature_name, self.validation_folds, info)

    def save_test_o(self, o, feature_name, info=None):
        self.serializer.save(o, feature_name, ['test'], info)
