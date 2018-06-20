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


def histogram_slice_acc(referral_results, referral_threshold, width=12,
                        do_save=False, do_show=True, plot_title="Main title"):

    str_referral_threshold = str(referral_threshold).replace(".", "_")
    # get all stats from the referral_result object
    slice_stats_es = referral_results.es_mean_slice_improvements[referral_threshold]
    num_of_slices_es = referral_results.es_slice_freqs[referral_threshold]
    slice_stats_ed = referral_results.ed_mean_slice_improvements[referral_threshold]
    num_of_slices_ed = referral_results.ed_slice_freqs[referral_threshold]
    # layout grid plots
    columns = 3
    unique_num_of_slices = slice_stats_es.keys()
    num_of_plots = len(unique_num_of_slices)
    if num_of_plots % columns != 0:
        add_row = 1
    else:
        add_row = 0
    rows = (num_of_plots / columns) + add_row
    height = rows * 5
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
        ax1.set_xlabel("Slice", **config.axis_font)
        ax1.tick_params(axis='both', which='major', labelsize=config.axis_ticks_font_size)
        ax1b.tick_params(axis='both', which='major', labelsize=config.axis_ticks_font_size)
        if new_row:
            ax1.set_ylabel("Sum dice increase (%)", **config.axis_font)
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


def histogram_slice_referral(referral_results, referral_threshold, width=12,
                             do_save=False, do_show=True, plot_title="Main title"):

    str_referral_threshold = str(referral_threshold).replace(".", "_")
    # get all stats from the referral_result object
    slice_stats_es = referral_results.es_mean_slice_improvements[referral_threshold]
    num_of_slices_es = referral_results.es_slice_freqs[referral_threshold]
    slice_stats_ed = referral_results.ed_mean_slice_improvements[referral_threshold]
    num_of_slices_ed = referral_results.ed_slice_freqs[referral_threshold]
    # layout grid plots
    columns = 3
    unique_num_of_slices = slice_stats_es.keys()
    num_of_plots = len(unique_num_of_slices)
    if num_of_plots % columns != 0:
        add_row = 1
    else:
        add_row = 0
    rows = (num_of_plots / columns) + add_row
    height = rows * 5
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
        ax1.set_xlabel("Slice", **config.axis_font)
        ax1.tick_params(axis='both', which='major', labelsize=config.axis_ticks_font_size)
        ax1b.tick_params(axis='both', which='major', labelsize=config.axis_ticks_font_size)
        if new_row:
            ax1.set_ylabel("Sum dice increase (%)", **config.axis_font)
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


def show_referral_results(ref_result_obj, referral_threshold, per_disease=False):

    ref_dice = np.concatenate((np.expand_dims(ref_result_obj.ref_dice_stats[referral_threshold][0][0], axis=0),
                                   np.expand_dims(ref_result_obj.ref_dice_stats[referral_threshold][1][0], axis=0)))
    ref_dice_std = np.concatenate((np.expand_dims(ref_result_obj.ref_dice_stats[referral_threshold][0][1], axis=0),
                               np.expand_dims(ref_result_obj.ref_dice_stats[referral_threshold][1][1], axis=0)))
    ref_hd = np.concatenate((np.expand_dims(ref_result_obj.ref_hd_stats[referral_threshold][0][0], axis=0),
                                 np.expand_dims(ref_result_obj.ref_hd_stats[referral_threshold][1][0], axis=0)))
    ref_hd_std = np.concatenate((np.expand_dims(ref_result_obj.ref_hd_stats[referral_threshold][0][1], axis=0),
                             np.expand_dims(ref_result_obj.ref_hd_stats[referral_threshold][1][1], axis=0)))
    org_dice = np.concatenate((np.expand_dims(ref_result_obj.org_dice_stats[referral_threshold][0][0], axis=0),
                               np.expand_dims(ref_result_obj.org_dice_stats[referral_threshold][1][0], axis=0)))

    print("----------------------Overall results for referral-threshold {:.2f} -----------------"
          "-----".format(referral_threshold))

    print("without referral - "
              "dice(RV/Myo/LV):\tES {:.2f}/{:.2f}/{:.2f}\t"
              "ED {:.2f}/{:.2f}/{:.2f}".format(org_dice[0, 1], org_dice[0, 2],
                                               org_dice[0, 3], org_dice[1, 1],
                                               org_dice[1, 2], org_dice[1, 3]))

    print("   with referral - "
          "dice(RV/Myo/LV):\tES {:.2f}/{:.2f}/{:.2f}\t"
          "ED {:.2f}/{:.2f}/{:.2f}".format(ref_dice[0, 1], ref_dice[0, 2],
                                           ref_dice[0, 3], ref_dice[1, 1],
                                           ref_dice[1, 2], ref_dice[1, 3]))
    print("")
    if per_disease:
        num_per_category = ref_result_obj.num_per_category[referral_threshold]
        mean_org_dice_per_dcat = ref_result_obj.org_dice_per_dcat[referral_threshold]
        num_of_slices_referred = ref_result_obj.num_of_slices_referred[referral_threshold]
        total_num_of_slices = ref_result_obj.total_num_of_slices[referral_threshold]
        mean_blob_uvalue_per_slice = ref_result_obj.mean_blob_uvalue_per_slice[referral_threshold]
        mean_ref_dice_per_dcat = ref_result_obj.ref_dice_per_dcat[referral_threshold]
        mean_org_hd_per_dcat = ref_result_obj.org_hd_per_dcat[referral_threshold]
        mean_ref_hd_per_dcat = ref_result_obj.ref_hd_per_dcat[referral_threshold]

        for disease_cat in num_per_category.keys():
            org_dice = mean_org_dice_per_dcat[disease_cat]
            print("------------------------------ Results for class {} -----------------"
                  "-----------------".format(disease_cat))

            perc_slices_referred = np.nan_to_num(num_of_slices_referred[disease_cat] *
                                                 100. / float(total_num_of_slices[disease_cat])).astype(np.int)

            print("ES & ED Mean/median u-value {:.2f}/{:.2f} & {:.2f}/{:.2f}"
                  "\t % slices referred {:.2f} & {:.2f}".format(mean_blob_uvalue_per_slice[disease_cat][0][0],
                                                                mean_blob_uvalue_per_slice[disease_cat][0][1],
                                                                mean_blob_uvalue_per_slice[disease_cat][1][0],
                                                                mean_blob_uvalue_per_slice[disease_cat][1][1],
                                                                perc_slices_referred[0],
                                                                perc_slices_referred[1]))
            print("without referral - "
                  "dice(RV/Myo/LV):\tES {:.2f}/{:.2f}/{:.2f}\t"
                  "ED {:.2f}/{:.2f}/{:.2f}".format(org_dice[0, 1], org_dice[0, 2],
                                                   org_dice[0, 3], org_dice[1, 1],
                                                   org_dice[1, 2], org_dice[1, 3]))
            ref_dice = mean_ref_dice_per_dcat[disease_cat]
            print("   with referral - "
                  "dice(RV/Myo/LV):\tES {:.2f}/{:.2f}/{:.2f}\t"
                  "ED {:.2f}/{:.2f}/{:.2f}".format(ref_dice[0, 1], ref_dice[0, 2],
                                                   ref_dice[0, 3], ref_dice[1, 1],
                                                   ref_dice[1, 2], ref_dice[1, 3]))
            org_hd = mean_org_hd_per_dcat[disease_cat]
            print("without referral - HD (RV/Myo/LV):\tES {:.2f}/{:.2f}/{:.2f}\t"
                  "ED {:.2f}/{:.2f}/{:.2f}".format(org_hd[0, 1], org_hd[0, 2],
                                                   org_hd[0, 3], org_hd[1, 1],
                                                   org_hd[1, 2], org_hd[1, 3]))

            ref_hd = mean_ref_hd_per_dcat[disease_cat]
            print("   with referral - HD (RV/Myo/LV):\tES {:.2f}/{:.2f}/{:.2f}\t"
                  "ED {:.2f}/{:.2f}/{:.2f}".format(ref_hd[0, 1], ref_hd[0, 2],
                                                   ref_hd[0, 3], ref_hd[1, 1],
                                                   ref_hd[1, 2], ref_hd[1, 3]))
            print(" ")