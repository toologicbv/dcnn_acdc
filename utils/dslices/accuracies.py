import numpy as np

from sklearn.metrics import average_precision_score, precision_recall_fscore_support, accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.exceptions import UndefinedMetricWarning


def compute_eval_metrics(gt_labels, pred_labels, probs_pos_cls=None):
    """

    :param gt_labels:
    :param pred_labels:
    :param probs_pos_cls: 1D numpy array. Assuming probs indicate the prob for positive class
    :return:
    """
    # mean_fpr = np.linspace(0, 1, 100)
    if np.sum(gt_labels) != 0:
        # fpr, tpr, thresholds = roc_curve(gt_labels, pred_labels)

        if probs_pos_cls is not None:
            roc_auc = roc_auc_score(gt_labels, probs_pos_cls)
            pr_auc = average_precision_score(gt_labels, probs_pos_cls)
        else:
            roc_auc = roc_auc_score(gt_labels, pred_labels)
            pr_auc = average_precision_score(gt_labels, pred_labels)
        try:
            prec, rec, f1, _ = precision_recall_fscore_support(gt_labels, pred_labels, beta=1, labels=1,
                                                               average="binary")
        except UndefinedMetricWarning:
            print("WARNING - UndefinedMetricWarning - sum(pred_labels)={}".format(np.sum(pred_labels)))
    else:
        roc_auc = 0
        pr_auc = 0
        f1 = -1.
        prec = -1
        rec = -1
    acc = accuracy_score(gt_labels, pred_labels)

    return f1, roc_auc, pr_auc, acc, prec, rec
