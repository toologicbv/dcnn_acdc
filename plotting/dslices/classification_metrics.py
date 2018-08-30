from common.dslices.config import config
import matplotlib.pyplot as plt
import matplotlib
from pylab import MaxNLocator
import numpy as np
import os


def average_auc(expers, fig_name=None, height=8, width=6, save=False, show=True):

    """

    """
    if len(expers) > 4:
        raise ValueError("ERROR - currently max=4 experiments supported. Your array contains {}".format(len(expers)))
    color_code = ['b', 'r', 'g', 'b']
    model_markers = ["o", "s", "*", "x"]
    model_linestyles = ["-", ":", "-.", "--"]
    fig = plt.figure(figsize=(width, height))
    ax1 = plt.subplot2grid((4, 2), (0, 0), rowspan=2, colspan=2)
    ax2 = plt.subplot2grid((4, 2), (2, 0), rowspan=2, colspan=2)
    ax1.set_ylabel("AUC")
    ax1.set_xlabel("Iterations")
    ax1.set_ylim([0, 1.])
    p_title = "Average AUC-ROC/PR"
    ax1.set_title(p_title, **config.title_font_small)
    ax2.set_ylabel("f1")
    ax2.set_xlabel("Iterations")
    ax2.set_ylim([0, 1.])
    p_title = "Average F1-score"
    ax2.set_title(p_title, **config.title_font_small)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    for idx, e in enumerate(expers):
        type_of_map = "no-" + e.run_args.type_of_map if e.run_args.use_no_map else e.run_args.type_of_map
        x_values = e.val_stats["epoch_ids"]
        y_auc_roc = e.val_stats["roc_auc"]
        y_auc_pr = e.val_stats["pr_auc"]
        ax1.plot(x_values, y_auc_roc, 'r', label="AUC-ROC-{}".format(type_of_map), linestyle=model_linestyles[idx],
                 alpha=0.25, )
        ax1.plot(x_values, y_auc_pr, 'g', label="AUC-PR-{}".format(type_of_map), linestyle=model_linestyles[idx],
                 alpha=0.25,)
        # precision/recall
        y_f1 = e.val_stats["f1"]
        # y_precision = e.val_stats["prec"]
        ax2.plot(x_values, y_f1, 'r', label="f1-{}".format(type_of_map), linestyle=model_linestyles[idx],
                 c=color_code[idx], alpha=0.35, marker=model_markers[idx])
        # ax2.plot(x_values, y_precision, 'g', label="prec-{}".format(type_of_map), linestyle=model_linestyles[idx],
        #         alpha=0.35)
    if len(x_values) <= 10:
        plt.xticks(x_values)
    ax1.legend(loc="best")
    ax2.legend(loc="best")

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    if save:
        if fig_name is None:
            fig_name = os.path.join(e.root_directory, os.path.join(e.log_directory,
                                                                       "figures/mean_auc_roc_pr" + config.figure_ext))
        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)
    if show:
        plt.show()
    plt.close()
