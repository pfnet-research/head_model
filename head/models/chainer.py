import sklearn.base
import chainer
from chainer import training, datasets
from chainer.training import extensions
from chainer import functions as F
from chainer import links as L
import numpy as np

import logger


class MLP3(chainer.Chain):

    def __init__(self, n_in, n_units=512, n_out=12):
        super(MLP3, self).__init__(
            fc1=L.Linear(n_in, n_units),
            fc2=L.Linear(n_units, n_units),
            fc3=L.Linear(n_units, n_out),
            bn1=L.BatchNormalization(n_units),
            bn2=L.BatchNormalization(n_units),
        )
        self.n_out = n_out

    def __call__(self, x):
        x = F.dropout(F.relu(self.bn1(self.fc1(x))))
        x = F.dropout(F.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)


class MLP4(chainer.Chain):

    def __init__(self, n_in, n_units=512, n_out=12):
        super(MLP4, self).__init__(
            fc1=L.Linear(n_in, n_units),
            fc2=L.Linear(n_units, n_units),
            fc3=L.Linear(n_units, n_units),
            fc4=L.Linear(n_units, n_out),
            bn1=L.BatchNormalization(n_units),
            bn2=L.BatchNormalization(n_units),
            bn3=L.BatchNormalization(n_units),
        )
        self.n_out = n_out

    def __call__(self, x):
        x = F.dropout(F.relu(self.bn1(self.fc1(x))))
        x = F.dropout(F.relu(self.bn2(self.fc2(x))))
        x = F.dropout(F.relu(self.bn3(self.fc3(x))))
        return self.fc4(x)


class ChainerClassifier(sklearn.base.BaseEstimator):
    def __init__(self, model_gen_func, gpu=-1, n_epoch=250, n_out=12):
        self.model = None
        self.model_gen_func = model_gen_func
        self.gpu = gpu
        self.n_epoch = n_epoch
        self.n_out=n_out

    def fit_and_validate(self, train_x, train_y, validate_x=None, validate_y=None):
        classes, _ = np.unique(train_y, return_inverse=True)
        self.model = self.model_gen_func(len(train_x[0]), n_out=self.n_out)
        if self.gpu >= 0:
            chainer.cuda.get_device(self.gpu).use()
            self.model.to_gpu()

        train_model = self.model
        validate_model = train_model.copy()

        batchsize = 100
        log_trigger = 1, 'epoch'  #100, 'iteration'
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(L.Classifier(train_model))

        train_y = train_y.astype(np.int32)
        train_dataset = datasets.TupleDataset(train_x, train_y)
        train_iter = chainer.iterators.SerialIterator(train_dataset, batchsize)
        updater = chainer.training.StandardUpdater(train_iter, optimizer, device=self.gpu)
        trainer = chainer.training.Trainer(updater, (self.n_epoch, 'epoch'), out='out')

        if validate_x is not None:
            assert validate_y is not None
            validate_y = validate_y.astype(np.int32)
            validate_dataset = datasets.TupleDataset(validate_x, validate_y)
            validate_iter = chainer.iterators.SerialIterator(
                validate_dataset, batch_size=1000, repeat=False, shuffle=False)
            trainer.extend(chainer.training.extensions.Evaluator(validate_iter, L.Classifier(validate_model), device=self.gpu))

        trainer.extend(chainer.training.extensions.LogReport(trigger=log_trigger))

        trainer.extend(extensions.PrintReport(
            ['time', 'epoch', 'iteration', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy', 'lr'
             ]), trigger=log_trigger)
        trainer.extend(extensions.ProgressBar(update_interval=10))
        trainer.extend(logger.observe_time(), trigger=log_trigger)
        trainer.extend(logger.observe_lr(optimizer), trigger=log_trigger)

        trainer.run()

    def fit(self, x, y):
        self.fit_and_validate(x, y, None, None)

    def predict_proba(self, x):
        result = np.ndarray((len(x), self.model.n_out))

        test_iter = chainer.iterators.SerialIterator(
            x, batch_size=1000, shuffle=False, repeat=False)

        i = 0
        for batch in test_iter:
            in_array = chainer.dataset.convert.concat_examples(batch, self.gpu)
            in_var = chainer.variable.Variable(in_array)

            with chainer.using_config('train', False):
                with chainer.using_config('enable_backprop', False):
                    out_var = F.softmax(self.model(in_var))
            out_array = out_var.data
            if self.gpu >= 0:
                out_array = out_array.get()

            for j in range(len(out_array)):
                result[i] = out_array[j]
                i += 1

        assert i == len(x)
        return result

    def predict(self, x):
        prob = self.predict_proba(x)
        pred = np.argmax(prob, axis=1)
        return pred
