import numpy as np


def sensitivity_and_specificity(y_true, y_pred, label_names=None):
    report = {}
    labels = np.unique(np.hstack((y_true, y_pred)))
    for l in labels:
        pos = (y_true == l).sum()
        neg = (y_true != l).sum()
        tp = ((y_true == l) & (y_pred == l)).sum()
        tn = ((y_true != l) & (y_pred != l)).sum()

        sensitivity = 0. if pos == 0 else tp / pos
        specificity = 0. if neg == 0 else tn / neg
        key = l if label_names is None else label_names[l]
        report[key] = {'sensitivity': sensitivity,
                       'specificity': specificity,
                       'support': pos}
    return report


def to_csv(report, file_like, col_names):
    col_names = ['label'] + col_names
    header = ','.join(col_names)
    file_like.write(header)
    file_like.write('\n')
    for k in sorted(report.keys()):
        values = report[k]
        row = [k] + [values[c] for c in col_names[1:]]
        file_like.write(','.join(map(str, row)))
        file_like.write('\n')
