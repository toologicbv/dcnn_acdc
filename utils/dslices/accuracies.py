import numpy as np

from sklearn.metrics import average_precision_score, precision_recall_fscore_support, accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.exceptions import UndefinedMetricWarning


def compute_eval_metrics(gt_labels, pred_labels, probs_pos_cls=None):
    """
    Please see details on how scikit-learn implements "Classification metrics", specifically for
    precision-recall curve
    http://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics

    :param gt_labels:
    :param pred_labels:
    :param probs_pos_cls: 1D numpy array. Assuming probs indicate the prob for positive class
    :return:
    """

    if gt_labels.ndim > 1:
        gt_labels = gt_labels.flatten()
    if probs_pos_cls.ndim > 1:
        probs_pos_cls = probs_pos_cls.flatten()
    if pred_labels.ndim > 1:
        pred_labels = pred_labels.flatten()

    # In case we have no TP (degenerate label=1) or no TN (degenerate label 0) we can't compute AUC measure
    if np.sum(gt_labels) != 0 or np.sum(gt_labels) == gt_labels.shape[0]:
        if probs_pos_cls is not None:
            x_values = np.linspace(0, 1, 100)
            fpr, tpr, thresholds = roc_curve(gt_labels, probs_pos_cls)
            precision, recall, thresholds = precision_recall_curve(gt_labels, probs_pos_cls)
            roc_auc = roc_auc_score(gt_labels, probs_pos_cls)
            pr_auc = average_precision_score(gt_labels, probs_pos_cls)
            # different way of calculating AUC values
            # alt_roc_auc = auc(fpr, tpr)
            # alt_pr_auc = auc(recall, precision)
            # print("--- Compare: {:.2f}/{:.2f} - {:.2f}/{:.2f}".format(roc_auc, pr_auc, alt_roc_auc,
            #                                                          alt_pr_auc))
        else:
            fpr, tpr, thresholds = roc_curve(gt_labels, pred_labels)
            precision, recall, thresholds = precision_recall_curve(gt_labels, pred_labels)
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
        fpr, tpr, precision, recall = None, None, None, None
    # acc = accuracy_score(gt_labels, pred_labels)

    return f1, roc_auc, pr_auc, prec, rec, fpr, tpr, precision, recall
