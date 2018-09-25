from config.config import config
from common.hvsmr.config import config_hvsmr
import matplotlib.pyplot as plt
import pylab
import numpy as np
import os


def plot_coverage_risk_curve(list_of_sel_class, width=10, height=8, do_save=False, do_show=True):
    """

    :param list_of_sel_class: list of SelectiveClassification objects
    :param width:
    :param height:
    :param do_save:
    :param do_show:
    :return:
    """
    loss_function = None
    axis_label_size = {'fontname': 'Monospace', 'size': '28', 'color': 'black', 'weight': 'normal'}
    tick_size = 24
    legend_size = 24
    linewidth = 3
    p_colors = ['b', 'r', 'g', 'c']
    fig = plt.figure(figsize=(width, height))

    for idx, sel_class_obj in enumerate(list_of_sel_class):
        plt.plot(sel_class_obj.x_coverages, np.mean(sel_class_obj.mean_risks, axis=0),
                 c=p_colors[idx], label=sel_class_obj.type_of_map, alpha=0.4, linewidth=linewidth)
        if loss_function is None:
            loss_function = sel_class_obj.loss_function
    if sel_class_obj.optimal_curve is not None:
        plt.plot(np.array([1 - sel_class_obj.optimal_curve, 1]), np.array([0, 1]), c='k', alpha=0.4, label="optimal",
                 linewidth=linewidth)
    plt.legend(loc="best", prop={'size': legend_size})
    plt.xlabel("Coverage", **axis_label_size)
    plt.ylabel("Selective risk", **axis_label_size)
    plt.tick_params(axis='both', which='major', labelsize=tick_size)
    plt.title("Coverage risk curve {} loss".format(loss_function), **axis_label_size)
    plt.xlim([0., 1.0])
    plt.ylim([0, 1.])

    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    if do_save:
        fig_path = os.path.join(config.root_dir, config.figure_path)

        if not os.path.isdir(fig_path):
            os.makedirs(fig_path)
        fig_name = "coverage_risk_curve_" + loss_function
        fig_name = os.path.join(fig_path, fig_name + ".jpeg")

        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)

    if do_show:
        plt.show()