import numpy as np

from sklearn.metrics import average_precision_score, roc_curve, auc, f1_score, accuracy_score, roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning


def compute_eval_metrics(gt_labels, pred_labels,):

    # mean_fpr = np.linspace(0, 1, 100)
    if np.sum(gt_labels) != 0:
        # fpr, tpr, thresholds = roc_curve(gt_labels, pred_labels)
        roc_auc = roc_auc_score(gt_labels, pred_labels)
        pr_auc = average_precision_score(gt_labels, pred_labels)
        try:
            f1 = f1_score(gt_labels, pred_labels, average="weighted")
        except UndefinedMetricWarning:
            print("WARNING - UndefinedMetricWarning - sum(pred_labels)={}".format(np.sum(pred_labels)))
    else:
        roc_auc = 0
        pr_auc = 0
        f1 = -1.
    acc = accuracy_score(gt_labels, pred_labels)

    return f1, roc_auc, pr_auc, acc
