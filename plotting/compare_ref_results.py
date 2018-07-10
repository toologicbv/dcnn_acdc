from config.config import config
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.referral_results import ReferralResults


def compare_referral_results(ref_result_objects, width=16, height=10, plot_base=False,
                             do_save=False, do_show=True, plot_title="Model compare",
                             model_lbls=None, fig_name="compare_ref_results"):
    """
    :param ref_result_objects: List of ReferralResult objects for which ALL slices are referred
    :param

    :param model_lbls: list that contains exactly one entry for each model, used in legend
    :param do_show
    :param do_save
    :param plot_title
    :param width
    :param height
    :param fig_name
    :param plot_base: plot the dice results WITHOUT REFERRAL as a horizontal line
    """
    # Some general set-up stuff
    class_lbls = ["BG", "RV", "MYO", "LV"]
    color_code = ['b', 'r', 'g', 'b']
    model_markers = ["o", "D", "*"]
    model_linestyles = ["-", ":", "-."]
    columns = 2
    rows = 4

    # min/max values of y-axis
    max_dice = 1.01
    min_dice = 0.77
    x_min = 0
    x_max = 0.505
    x_axis = np.linspace(x_min, 0.5, 11)
    # prepare figure
    fig = plt.figure(figsize=(width, height))
    fig.suptitle(plot_title, **config.title_font_medium)
    ax1 = plt.subplot2grid((rows, columns), (0, 0), rowspan=2, colspan=2)
    ax1.set_xlim([x_min, x_max])
    ax1.set_ylim([min_dice, max_dice])
    ax1.set_xticks(x_axis)
    ax1.tick_params(axis='both', which='major', labelsize=config.axis_ticks_font_size)
    # same for axis 2
    ax2 = plt.subplot2grid((rows, columns), (2, 0), rowspan=2, colspan=2)
    ax2.set_xlim([x_min, x_max])
    ax2.set_ylim([min_dice, max_dice])
    ax2.set_xticks(x_axis)
    ax2.tick_params(axis='both', which='major', labelsize=config.axis_ticks_font_size)
    # loop over result objects
    for res_idx, ref_result_obj in enumerate(ref_result_objects):
        # % referred slices over ALL classes. Key=referral_threshold, value numpy [2] (for ES/ED)
        dice_ref = ref_result_obj.get_dice_referral_dict()
        mdl_marker = model_markers[res_idx]
        mdl_linestyle = model_linestyles[res_idx]
        mdl_lbl = model_lbls[res_idx]
        xs = np.array(dice_ref.keys())
        X = np.vstack(dice_ref.values())
        # dice_ref-values has shape [2, 4], but we
        X = np.reshape(X, (-1, 8))
        # actually we don't want the BG class to interfere with min/max, so set to zero
        X[:, 0], X[:, 4] = 0, 0
        # np array "org_dice_stats" contains [2, 2, 4] values dim0=ES/ED, dim1=mean/std
        # because these are the results without referral (dice/hd) we only need to get this for ONE referral
        # threshold, we take the last dictionary key here and pass the dice-values WITHOUT REFERRAL to object
        # dice_wo_ref
        dflt_ref_key = xs[-1]
        dice_wo_ref = np.concatenate((np.expand_dims(ref_result_obj.org_dice_stats[dflt_ref_key][0][0], axis=0),
                                      np.expand_dims(ref_result_obj.org_dice_stats[dflt_ref_key][1][0], axis=0)))

        # stacked everything so that dim0 is referral thresholds and dim1 is ES 4values + ED 4 values = [#refs, 8]

        wo_ref = np.zeros(x_axis.shape[0])
        for cls in np.arange(1, 8):
            if cls < 4:
                # will the baseline scalar over all thresholds for comparison
                wo_ref.fill(dice_wo_ref[0, cls])
                # plot class-dice for all thresholds
                ax1.plot(xs, X[:, cls], label="{}-u-map-{}".format(mdl_lbl, class_lbls[cls]), c=color_code[cls],
                             marker=mdl_marker, linestyle=mdl_linestyle, alpha=0.3, markersize=5)

                # plot our baseline performance
                if plot_base:
                    ax1.plot(x_axis, wo_ref,c=color_code[cls],
                                 linestyle=mdl_linestyle, alpha=0.05)
                    #  label="{}-base-{}".format(mdl_lbl, class_lbls[cls])
                ax1.set_ylabel("dice", **config.axis_font)
                ax1.set_xlabel("referral threshold", **config.axis_font)
                ax1.legend(loc=3)
                ax1.set_title("ES", **config.title_font_small)

            if 4 < cls <= 7:
                class_idx = cls - 4

                ax2.plot(xs, X[:, cls], label="{}-u-map-{}".format(mdl_lbl, class_lbls[class_idx]),
                             c=color_code[class_idx], marker=mdl_marker, markersize=5,
                             linestyle=mdl_linestyle, alpha=0.3)

                wo_ref.fill(dice_wo_ref[1, class_idx])
                # plot our baseline performance
                if plot_base:
                    ax2.plot(x_axis, wo_ref,
                             c=color_code[class_idx], linestyle=mdl_linestyle, alpha=0.05)
                    # label="{}-base-{}".format(mdl_lbl, class_lbls[class_idx])
                # ax2.set_ylabel("dice", **config.axis_font)
                ax2.set_xlabel("referral threshold", **config.axis_font)

                ax2.legend(loc=3)
                ax2.set_title("ED", **config.title_font_small)

    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    if do_save:
        fig_path = os.path.join(config.root_dir, "figures")

        if not os.path.isdir(fig_path):
            os.makedirs(fig_path)

        fig_name_pdf = os.path.join(fig_path, fig_name + ".pdf")
        fig_name_jpeg = os.path.join(fig_path, fig_name + ".jpeg")
        plt.savefig(fig_name_pdf, bbox_inches='tight')
        plt.savefig(fig_name_jpeg, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name_pdf)

    if do_show:
        plt.show()
