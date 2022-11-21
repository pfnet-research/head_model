import io
import unittest

import numpy as np
from sklearn import metrics as metrics_

import metrics


def convert(precision, recall, pos, neg):
    tp = pos * recall
    fp = (1 - precision) / precision * tp
    tn = neg - fp
    specificity = tn / neg
    sensitivity = recall
    return sensitivity, specificity


class TestSensitivityAndSpecificity(unittest.TestCase):

    def test_close_to_sklearn(self):
        K = 5
        N = 100
        y_true = np.random.randint(0, K, N)
        y_pred = np.random.randint(0, K, N)
        actual = metrics.sensitivity_and_specificity(
            y_true, y_pred)
        sk_result = metrics_.classification_report(
            y_true, y_pred, output_dict=True)
        labels = np.unique(y_true)
        for l in labels:
            pos = (y_true == l).sum()
            neg = (y_true != l).sum()
            precision = sk_result[str(l)]['precision']
            recall = sk_result[str(l)]['recall']
            if precision == 0 or neg == 0:
                continue
            expect = convert(precision, recall, pos, neg)
            np.testing.assert_array_almost_equal(
                actual[l]['sensitivity'], expect[0])
            np.testing.assert_array_almost_equal(
                actual[l]['specificity'], expect[1])
            np.testing.assert_equal(
                actual[l]['support'], sk_result[str(l)]['support'])

    def test_zero_support(self):
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 0])
        actual = metrics.sensitivity_and_specificity(y_true, y_pred)

        expect_0 = {'sensitivity': 0.,
                    'specificity': 0.75,
                    'support': 0}
        self.assertDictEqual(actual[0], expect_0)

        expect_1 = {'sensitivity': 0.75,
                    'specificity': 0.,
                    'support': 4}
        self.assertDictEqual(actual[1], expect_1)

    def test_zero_prediction(self):
        y_true = np.array([1, 1, 1, 0])
        y_pred = np.array([1, 1, 1, 1])
        actual = metrics.sensitivity_and_specificity(y_true, y_pred)

        expect_0 = {'sensitivity': 0.,
                    'specificity': 1.,
                    'support': 1}
        self.assertDictEqual(actual[0], expect_0)

        expect_1 = {'sensitivity': 1.,
                    'specificity': 0.,
                    'support': 3}
        self.assertDictEqual(actual[1], expect_1)

    def test_label_names(self):
        y_true = np.array([1, 1, 1, 0])
        y_pred = np.array([1, 1, 1, 1])
        label_names = ['foo', 'bar']
        actual = metrics.sensitivity_and_specificity(
            y_true, y_pred, label_names)

        expect_0 = {'sensitivity': 0.,
                    'specificity': 1.,
                    'support': 1}
        self.assertDictEqual(actual[label_names[0]], expect_0)

        expect_1 = {'sensitivity': 1.,
                    'specificity': 0.,
                    'support': 3}
        self.assertDictEqual(actual[label_names[1]], expect_1)


class TestToCSV(unittest.TestCase):

    def setUp(self):
        self.y_true = np.array([1, 1, 1, 0])
        self.y_pred = np.array([1, 1, 1, 1])
        self.col_names = ['sensitivity', 'specificity', 'support']

    def test_without_label_names(self):
        report = metrics.sensitivity_and_specificity(
            self.y_true, self.y_pred)
        stream = io.StringIO()
        metrics.to_csv(report, stream, self.col_names)
        actual = stream.getvalue()
        expect = """label,sensitivity,specificity,support
0,0.0,1.0,1
1,1.0,0.0,3
"""
        self.assertEqual(actual, expect)

    def test_with_label_names(self):
        label_names = ['foo', 'bar']
        report = metrics.sensitivity_and_specificity(
            self.y_true, self.y_pred, label_names)
        stream = io.StringIO()
        metrics.to_csv(report, stream, self.col_names)
        actual = stream.getvalue()
        expect = """label,sensitivity,specificity,support
bar,1.0,0.0,3
foo,0.0,1.0,1
"""
        self.assertEqual(actual, expect)
