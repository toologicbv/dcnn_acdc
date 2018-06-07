import os
from config.config import config
import matplotlib.pyplot as plt
from pylab import MaxNLocator
import numpy as np
from collections import OrderedDict


def get_dice_diffs(org_dice_slices, dice_slices, num_of_slices, slice_stats, phase):
    """

    :param org_dice_slices:
    :param dice_slices:
    :param phase: can be 0=ES or 1=ED
    :param num_of_slices: OrderedDict with key #slices, value frequency
    :param slice_stats: OrderedDict with key #slices, value mean dice increase between org - ref dice coeff.
    :return:
    """
    diffs = np.sum(dice_slices[phase, 1:] - org_dice_slices[phase, 1:], axis=0)
    if np.any(diffs < 0.):
        print("Ref dice phase={}".format(phase))
        print(dice_slices[phase, 1:])
        print("Original")
        print(org_dice_slices[phase, 1:])
        print("Diffs")
        print(diffs)
    slices = diffs.shape[0]
    if slices in slice_stats.keys():
        num_of_slices[slices] += 1
        slice_stats[slices] += diffs
    else:
        num_of_slices[slices] = 1
        slice_stats[slices] = diffs

    return num_of_slices, slice_stats


def diff_slice_acc(dict_org_dice_slices, dict_ref_dice_slices):
    slice_stats_es = OrderedDict()
    num_of_slices_es = OrderedDict()
    slice_stats_ed = OrderedDict()
    num_of_slices_ed = OrderedDict()
    # layout grid plots
    columns = 3
    for patient_id, dice_slices in dict_ref_dice_slices.iteritems():
        org_dice_slices = dict_org_dice_slices[patient_id]
        num_of_slices_es, slice_stats_es = get_dice_diffs(org_dice_slices, dice_slices, num_of_slices_es,
                                                          slice_stats_es, phase=0)  # ES
        num_of_slices_ed, slice_stats_ed = get_dice_diffs(org_dice_slices, dice_slices, num_of_slices_ed,
                                                          slice_stats_ed, phase=1)  # ES


def histogram_slice_acc(dict_org_dice_slices, dict_ref_dice_slices, referral_threshold, width=12, height=9,
                        do_save=False, do_show=True, plot_title="Main title"):

    str_referral_threshold = str(referral_threshold).replace(".", "_")
    slice_stats_es = OrderedDict()
    num_of_slices_es = OrderedDict()
    slice_stats_ed = OrderedDict()
    num_of_slices_ed = OrderedDict()
    # layout grid plots
    columns = 3
    for patient_id, dice_slices in dict_ref_dice_slices.iteritems():
        org_dice_slices = dict_org_dice_slices[patient_id]
        num_of_slices_es, slice_stats_es = get_dice_diffs(org_dice_slices, dice_slices, num_of_slices_es,
                                                          slice_stats_es, phase=0)  # ES
        num_of_slices_ed, slice_stats_ed = get_dice_diffs(org_dice_slices, dice_slices, num_of_slices_ed,
                                                          slice_stats_ed, phase=1)  # ES

    unique_num_of_slices = slice_stats_es.keys()
    num_of_plots = len(unique_num_of_slices)
    if num_of_plots % columns != 0:
        add_row = 1
    else:
        add_row = 0
    rows = (num_of_plots / columns) + add_row
    fig = plt.figure(figsize=(width, height))
    ax = fig.gca()
    fig.suptitle(plot_title, **config.title_font_medium)
    row = 0
    column = 0
    unique_num_of_slices.sort()
    bar_width = 0.25
    print("Rows/columns {}/{}".format(rows, columns))
    new_row = True
    for num_slices in unique_num_of_slices:
        slice_stats_es[num_slices] = slice_stats_es[num_slices] * 1. / num_of_slices_es[num_slices]
        slice_stats_ed[num_slices] = slice_stats_ed[num_slices] * 1. / num_of_slices_ed[num_slices]
        x_ticks = np.arange(1, num_slices + 1)
        ax1 = plt.subplot2grid((rows, columns), (row, column), rowspan=1, colspan=1)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.bar(x_ticks, slice_stats_es[num_slices] * 100, bar_width, label="ES",
                color='b', alpha=0.2, align="center")
        ax1.tick_params(axis='y', colors='b')
        ax1b = ax1.twinx()

        ax1b.bar(x_ticks + bar_width, slice_stats_ed[num_slices] * 100, bar_width, label="ED",
                 color='g', alpha=0.2, align="center")
        ax1b.tick_params(axis='y', colors='g')
        ax1.legend(loc=1, prop={'size': 12})
        ax1b.legend(loc=2, prop={'size': 12})
        ax1.grid(False)
        ax1.set_title("#Slices: {} Freq: {}".format(num_slices, num_of_slices_ed[num_slices]))
        ax1.set_xlabel("Slice")
        # ax1.set_xticks(x_ticks)
        if new_row:
            ax1.set_ylabel("Sum dice increase (%)")
        if column == columns -1:
            column = 0
            row += 1
            new_row = True
        else:
            new_row = False
            column += 1
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    if do_save:
        fig_path = os.path.join(config.root_dir, "figures")
        if not os.path.isdir(fig_path):
            os.makedirs(fig_path)
        fig_name = "referral_slice_improvements_" + str_referral_threshold
        fig_name = os.path.join(fig_path, fig_name + ".pdf")
        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)

    if do_show:
        plt.show()
