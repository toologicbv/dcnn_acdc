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
    ax = plt.figure(figsize=(width, height)).gca()
    plt.ylabel("AUC")
    plt.xlabel("Iterations")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for idx, e in enumerate(expers):
        type_of_map = e.run_args.type_of_map
        x_values = e.val_stats["epoch_ids"]
        y_auc_roc = e.val_stats["roc_auc"]
        y_auc_pr = e.val_stats["pr_auc"]
        plt.plot(x_values, y_auc_roc, 'r', label="AUC-ROC-{}".format(type_of_map), linestyle=model_linestyles[idx],
                 alpha=0.35)
        plt.plot(x_values, y_auc_pr, 'g', label="AUC-PR-{}".format(type_of_map), linestyle=model_linestyles[idx],
                 alpha=0.35)
    if len(x_values) <= 10:
        plt.xticks(x_values)
    plt.legend(loc="best")
    p_title = "Average AUC-ROC/PR"
    plt.title(p_title, **config.title_font_small)

    if save:
        if fig_name is None:
            fig_name = os.path.join(e.root_directory, os.path.join(e.log_directory,
                                                                       "figures/mean_auc_roc_pr" + config.figure_ext))
        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)
    if show:
        plt.show()
    plt.close()
