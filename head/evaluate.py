import argparse
import os

import numpy as np
from sklearn import metrics as metrics_

import metrics

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--prediction', type=str,
                    default='./submission.txt')
parser.add_argument('-g', '--ground-truth', type=str,
                    default='./3d_gene/out/0/test/labels.txt')
parser.add_argument('-l', '--label-names', type=str,
                    default='./3d_gene/out/0/label_names.txt')
parser.add_argument('-o', '--out-dir', type=str,
                    default='./results/',
                    help='Path to output directory. '
                    'This script generates three files: '
                    'sensitivity_specificity.csv, '
                    'precision_recall.txt, '
                    'and confusion_matrix.npy')
args = parser.parse_args()

y_pred = np.loadtxt(args.prediction).astype(np.int32)
y_true = np.loadtxt(args.ground_truth).astype(np.int32)
err_msg = 'pred:{} != gt:{}'.format(len(y_pred), len(y_true))
assert len(y_pred) == len(y_true), err_msg

label_names = np.loadtxt(args.label_names, dtype=str)

# sensitivity and specificity
sensitivity_specificity = metrics.sensitivity_and_specificity(
    y_true, y_pred, label_names)
fname = os.path.join(args.out_dir, 'sensitivity_specificity.csv')
with open(fname, 'w') as f:
    metrics.to_csv(sensitivity_specificity, f,
                   ['sensitivity',
                    'specificity',
                    'support'])
print(sensitivity_specificity)

# precision and recall
precision_recall = metrics_.classification_report(
    y_true, y_pred, target_names=label_names)
fname = os.path.join(args.out_dir, 'precision_recall.txt')
with open(fname, 'w') as f:
    f.write(precision_recall)
print(precision_recall)

# confusion matrix
confusion_matrix = metrics_.confusion_matrix(y_true, y_pred)
fname = os.path.join(args.out_dir, 'confusion_matrix.npy')
np.save(fname, confusion_matrix)
np.set_printoptions(linewidth=200)
print(confusion_matrix)
