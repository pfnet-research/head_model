import enum
import itertools
import logging

import numpy as np
import sklearn.metrics

from data import rotator as rotator_module, voting
from data import serializer as serializer_module
from data import voting
from logger import jlog


class _StepType(enum.Enum):
    transform = 0
    predict_proba = 1
    predict = 2
    decision_function = 3


class Pipeline:

    def __init__(self, path_policy, seed, n_folds,
                 submission_func_predict=None,
                 submission_func_predict_proba=None, submission_func_decision_function=None):
        self.path_policy = path_policy
        self.seed = seed
        self.n_folds = n_folds
        self.serializer = serializer_module.Serializer(self.path_policy, self.seed, self.n_folds)

        self.submission_func_predict = submission_func_predict
        self.submission_func_predict_proba = submission_func_predict_proba
        self.submission_func_decision_function = submission_func_decision_function
        self.label_names = []
        with open(path_policy.get_label_name_path(), 'r') as f:
            for row in f:
                label_name = row.strip()
                self.label_names.append(label_name)
        print('label: ', self.label_names)

    def transform(self, estimator, input_names, output_name,
                  validate_size=None, unsupervised=False, version=0):
        if unsupervised:
            print(estimator)
            self.serializer.is_cached(output_name, version)
            x = self.serializer.load(input_names, ['train', 'test'])
            o = estimator.fit_transform(x)
            self.serializer.save(o, output_name, ['train', 'test'], unsupervised=True)
            self.serializer.save_version(output_name, version, unsupervised=True)
        else:
            self._run(estimator, input_names, output_name,
                      validate_size, version, _StepType.transform, voting_func=voting.arithmetic_mean)

    def predict_proba(self, estimator, input_names, output_name,
                      validate_size=None, version=0):
        self._run(
            estimator, input_names, output_name,
            validate_size, version, _StepType.predict_proba, voting_func=voting.geometric_mean,
            submission_func=self.submission_func_predict_proba)

    def predict(self, estimator, input_names, output_name,
                validate_size=None, version=0):
        self._run(
            estimator, input_names, output_name,
            validate_size, version, _StepType.predict, voting_func=voting.vote,
            submission_func=self.submission_func_predict)

    def decision_function(self, estimator, input_names, output_name,
                          validate_size=None, version=0):
        self._run(
            estimator, input_names, output_name,
            validate_size, version, _StepType.decision_function, voting_func=voting.arithmetic_mean,
            submission_func=self.submission_func_decision_function)

    def _run(self, estimator, input_names, output_name,
             validate_size, version, step_type, voting_func, submission_func=None):
        with jlog.add_open('pipeline_step'):
            print(estimator)

            serializer = serializer_module.Serializer(self.path_policy, self.seed, self.n_folds)
            if self.serializer.is_cached(output_name, version):
                return

            validation_fold_sets = itertools.combinations(range(self.n_folds), validate_size)
            validation_fold_sets = list(map(list, validation_fold_sets))
            jlog.put('n_validation_fold_sets', len(validation_fold_sets))
            jlog.put('validation_fold_sets', validation_fold_sets)
            fold_indices = serializer.fold_indices

            all_val_o = [list() for _ in range(self.n_folds)]
            all_val_y = [None for _ in range(self.n_folds)]
            all_test_o = []

            def info():
                return {
                    'input_names': input_names,
                    'output_name': output_name,
                    'estimator': {
                        'type': str(type(estimator)),
                        'params': estimator.get_params(),
                    },
                    'jlog': jlog.stack[-1],
                }

            for validation_folds in validation_fold_sets:
                with jlog.add_open('bag'):
                    rotator = rotator_module.Rotator(
                        serializer, input_names, output_name, validation_folds)
                    jlog.put('validation_folds', validation_folds)

                    #
                    # Train
                    #
                    tra_x = rotator.load_train_x()
                    tra_y = rotator.load_train_y()
                    val_x = rotator.load_validation_x()
                    val_y = rotator.load_validation_y()

                    with jlog.put_benchmark('time.train'):
                        if hasattr(estimator, 'fit_and_validate'):
                            estimator.fit_and_validate(tra_x, tra_y, val_x, val_y)
                        else:
                            estimator.fit(tra_x, tra_y)

                    #
                    # Test
                    #
                    with jlog.put_benchmark('time.test'):
                        test_x = rotator.load_test_x()
                        if step_type is _StepType.transform:
                            test_o = estimator.transform(test_x)
                        elif step_type is _StepType.predict:
                            test_o = estimator.predict(test_x)
                        elif step_type is _StepType.predict_proba:
                            test_o = estimator.predict_proba(test_x)
                        elif step_type is _StepType.decision_function:
                            test_o = estimator.decision_function(test_x)
                        else:
                            assert False

                        all_test_o.append(test_o)

                    #
                    # Validation
                    #
                    with jlog.put_benchmark('time.validate'):
                        if step_type is _StepType.transform:
                            val_o = estimator.transform(val_x)
                        elif step_type in [_StepType.predict, _StepType.predict_proba, _StepType.decision_function]:
                            if step_type is _StepType.predict:
                                val_p = val_o = estimator.predict(val_x)
                            else:
                                if step_type is _StepType.predict_proba:
                                    val_o = estimator.predict_proba(val_x)
                                elif step_type is _StepType.decision_function:
                                    val_o = estimator.decision_function(val_x)
                                print("HEYYYYY")
                                print(val_o)
                                print(val_o.shape)
                                val_p = val_o.argmax(axis=1)

                            jlog.put('accuracy', sklearn.metrics.accuracy_score(val_y, val_p))
                            logging.info(sklearn.metrics.classification_report(val_y, val_p, target_names=self.label_names, digits=4))
                        else:
                            assert False

                    serializer.save(val_o, output_name, validation_folds, info())
                    val_os = fold_indices.split_to_fold_wise(val_o, validation_folds)
                    val_ys = fold_indices.split_to_fold_wise(val_y, validation_folds)
                    for i, f in enumerate(validation_folds):
                        all_val_o[f].append(val_os[i])
                        all_val_y[f] = val_ys[i]

            #
            # Total validation
            #
            all_val_y = np.concatenate(tuple(all_val_y))
            all_val_o = list(map(voting_func, all_val_o))
            if validate_size >= 2:
                for f in range(self.n_folds):
                    serializer.save(all_val_o[f], output_name, [f], info())

            all_val_o = np.concatenate(tuple(all_val_o))

            if step_type in [_StepType.predict, _StepType.predict_proba, _StepType.decision_function]:
                if step_type is _StepType.predict:
                    all_val_p = all_val_o
                else:
                    all_val_p = all_val_o.argmax(axis=1)

                jlog.put('accuracy', sklearn.metrics.accuracy_score(all_val_y, all_val_p))
                print(sklearn.metrics.classification_report(all_val_y, all_val_p, target_names=self.label_names, digits=4))

            #
            # Total test
            #
            test_o = voting_func(all_test_o)
            serializer.save(test_o, output_name, ['test'], info())
            serializer.save_version(output_name, version)
            if submission_func is not None:
                submission_file_path = self.path_policy.get_submission_file_path(
                    self.seed, self.n_folds, output_name)
                submission_func(test_o, submission_file_path)
