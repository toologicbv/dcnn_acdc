import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.utils.fixes import signature

from common.detector.config import config_detector


def plot_detection_auc_curves():

    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})

    width = 20
    height = 20
    fig = plt.figure(figsize=(width, height))
    x_values = np.linspace(0, 1., 50)

    ax1 = plt.subplot2grid((6, 6), (0, 0), rowspan=2, colspan=2)
    perc_fp = threshold_fp_rd1[::-1] * 1. / num_of_grids * 100
    ax1.step(perc_fp, threshold_sensitivity_rd1[::-1], color='b', alpha=0.2, where='post')
    ax1.fill_between(perc_fp, threshold_sensitivity_rd1[::-1], alpha=0.2, color='b', **step_kwargs)
    ax1.set_ylim([0., 1.])
    ax1.set_xlabel("FP(% of all regions)", **config_detector.axis_font18)
    ax1.set_ylabel("Sensitivity", **config_detector.axis_font18)
    ax1.set_title("FROC regions", **config_detector.axis_font18)
    # ax1.plot(threshold_fp_rd3[::-1], threshold_sensitivity_rd3[::-1], c='r')
    ax2 = plt.subplot2grid((6, 6), (0, 2), rowspan=2, colspan=2)
    perc_fp_slice = slice_fp_rd1[::-1] * 1. / num_of_slices * 100
    ax2.step(perc_fp_slice, slice_sensitivity_rd1[::-1], alpha=0.2, color='b', where='post')
    ax2.fill_between(perc_fp_slice, slice_sensitivity_rd1[::-1], color='b', alpha=0.2, **step_kwargs)
    # ax2.set_xlim([0., 1.])
    ax2.set_ylim([0., 1.])
    ax2.set_xlabel("FP(% of all slices)", **config_detector.axis_font18)
    ax2.set_title("FROC slices", **config_detector.axis_font18)
    # precision/recall regions
    ax3 = plt.subplot2grid((6, 6), (2, 0), rowspan=2, colspan=2)
    ax3.step(threshold_sensitivity_rd1[::-1], threshold_precision_rd1[::-1], color='g', alpha=0.2, where='post')
    ax3.fill_between(threshold_sensitivity_rd1[::-1], threshold_precision_rd1[::-1], alpha=0.2, color='g',
                     **step_kwargs)
    ax3.plot([0, 1], [1, 0], color='navy', lw=2, linestyle='--')
    ax3.set_xlabel("Sensitivity", **config_detector.axis_font18)
    ax3.set_ylabel("Precision", **config_detector.axis_font18)
    ax3.set_xlim([0., 1.])
    ax3.set_ylim([0., 1.])
    ax3.set_title("PR-curve regions", **config_detector.axis_font18)
    # precision/recall slices
    ax4 = plt.subplot2grid((6, 6), (2, 2), rowspan=2, colspan=2)
    ax4.step(slice_sensitivity_rd1[::-1], slice_threshold_precision_rd1[::-1], alpha=0.2, color='g', where='post')
    ax4.fill_between(slice_sensitivity_rd1[::-1], slice_threshold_precision_rd1[::-1], alpha=0.2, color='g',
                     **step_kwargs)
    ax4.set_xlabel("Sensitivity", **config_detector.axis_font18)
    ax4.set_xlim([0., 1.])
    ax4.set_ylim([0., 1.])
    ax4.set_title("PR-curve slices", **config_detector.axis_font18)
    # ROC AUC
    ax5 = plt.subplot2grid((6, 6), (4, 0), rowspan=2, colspan=2)

    fpr, tpr, _ = roc_curve(gt_labels, pred_probs)

    # ax5.step(threshold_fp_rate_rd1[::-1], threshold_sensitivity_rd1[::-1], alpha=0.2, color='y', where='post')
    # ax5.fill_between(threshold_fp_rate_rd1[::-1], threshold_sensitivity_rd1[::-1], alpha=0.2, color='y',
    #                 **step_kwargs)
    ax5.step(fpr, tpr, alpha=0.2, color='y', where='post')
    ax5.fill_between(fpr, tpr, alpha=0.2, color='y', **step_kwargs)

    ax5.set_xlabel("False negative rate", **config_detector.axis_font18)
    ax5.set_ylabel("True positive rate", **config_detector.axis_font18)
    ax5.set_xlim([0., 1.])
    ax5.set_ylim([0., 1.])
    ax5.set_title("ROC AUC-curve regions", **config_detector.axis_font18)
    # ROC AUC
    ax6 = plt.subplot2grid((6, 6), (4, 2), rowspan=2, colspan=2)
    ax6.step(slice_threshold_fp_rate[::-1], slice_sensitivity_rd1[::-1], alpha=0.2, color='y', where='post')
    ax6.fill_between(slice_threshold_fp_rate[::-1], slice_sensitivity_rd1[::-1], alpha=0.2, color='y',
                     **step_kwargs)
    ax6.set_xlabel("False negative rate", **config_detector.axis_font18)
    ax6.set_xlim([0., 1.])
    ax6.set_ylim([0., 1.])
    ax6.set_title("ROC AUC-curve slices", **config_detector.axis_font18)

    # idx = np.where((perc_fp_rd1 >= 0.13) & (perc_fp_rd1 <= 0.14))
    # print(idx)
    # print(threshold_list_rd1[idx])
    # print(slice_sensitivity_rd1[idx])
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()
