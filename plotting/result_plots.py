from config.config import config
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.referral_handler import ReferralResults


def plot_referral_results(ref_result_obj, ref_slice_results=None, width=16, height=14,
                          do_save=False, do_show=True, model_name="Main title"):
    """
    :param dice_ref: OrderedDict with key referral_threshold. Holds the referral results in which we refer
                     ALL SLICES.
    :param dice_slices_ref: OrderedDict with key referral_threshold. Holds the DICE referral results in which
                    we refer ONLY SPECIFIC SLICES.

                    shape of the values [

    :param model_name
    :param do_show
    :param do_save
    :param width
    :param height
    """

    if ref_slice_results is not None:
        slice_referral = True
        dice_slices_ref = ref_slice_results.get_dice_referral_dict()
        X_pos = np.vstack(dice_slices_ref.values())
        X_pos = np.reshape(X_pos, (-1, 8))
        X_pos[:, 0], X_pos[:, 4] = 0, 0
    else:
        slice_referral = False
    rows = 2
    columns = 4
    dice_wo_ref = ReferralResults.results_dice_wo_referral
    class_lbls = ["BG", "RV", "MYO", "LV"]
    color_code = ['b', 'r', 'g', 'b']

    dice_ref = ref_result_obj.get_dice_referral_dict()
    xs = np.array(dice_ref.keys())
    X = np.vstack(dice_ref.values())
    # dice_ref-values has shape [2, 4], but we
    X = np.reshape(X, (-1, 8))
    # actually we don't want the BG class to interfere with min/max, so set to zero
    X[:, 0], X[:, 4] = 0, 0
    max_dice = np.max(X)
    min_dice = np.min(dice_wo_ref[dice_wo_ref != 0])
    min_dice -= min_dice * 0.03
    max_dice += max_dice * 0.01
    # stacked everything so that dim0 is referral thresholds and dim1 is ES 4values + ED 4 values = [#refs, 8]
    fig = plt.figure(figsize=(width, height))
    main_title = "Uncertainty informed decision referral for {}".format(model_name)
    fig.suptitle(main_title, **config.title_font_medium)
    ax1 = plt.subplot2grid((rows, columns), (0, 0), rowspan=2, colspan=2)
    wo_ref = np.zeros(xs.shape[0])
    for cls in np.arange(1, 8):
        if cls < 4:
            # will the baseline scalar over all thresholds for comparison
            wo_ref.fill(dice_wo_ref[0, cls])
            # plot class-dice for all thresholds
            ax1.plot(xs, X[:, cls], label="ref all slices {}".format(class_lbls[cls]), c=color_code[cls], marker="*",
                     linestyle="--", alpha=0.6)
            # plot our baseline performance

            ax1.plot(xs, wo_ref, label="no referral {}".format(class_lbls[cls]), c=color_code[cls],
                     linestyle="-", alpha=0.2)
            ax1.set_ylabel("dice", **config.axis_font)
            ax1.set_xlabel("referral threshold", **config.axis_font)
            ax1.set_ylim([min_dice, max_dice])
            plt.tick_params(axis='both', which='major', labelsize=config.axis_ticks_font_size)
            ax1.set_xticks(xs)
            ax1.legend(loc="best")
            ax1.set_title("ES", **config.title_font_small)
            if slice_referral:
                ax1.plot(xs, X_pos[:, cls], label="slice-referral {}".format(class_lbls[cls]),
                         c=color_code[cls], marker="D",
                         linestyle=":", alpha=0.6)
        if cls == 4:
            ax2 = plt.subplot2grid((rows, columns), (0, 2), rowspan=2, colspan=2)
        if 4 < cls <= 7:
            class_idx = cls - 4
            ax2.plot(xs, X[:, cls], label="ref all slices {}".format(class_lbls[class_idx]),
                     c=color_code[class_idx], marker="*",
                     linestyle="--", alpha=0.6)
            wo_ref.fill(dice_wo_ref[1, class_idx])
            # plot our baseline performance
            ax2.plot(xs, wo_ref, label="no referral {}".format(class_lbls[class_idx]), c=color_code[class_idx],
                     linestyle="-", alpha=0.2)
            if slice_referral:
                ax2.plot(xs, X_pos[:, cls], label="slice-referral {}".format(class_lbls[class_idx]),
                         c=color_code[class_idx], marker="D",
                         linestyle=":", alpha=0.6)
            ax2.set_ylabel("dice", **config.axis_font)
            ax2.set_xlabel("referral threshold", **config.axis_font)
            plt.tick_params(axis='both', which='major', labelsize=config.axis_ticks_font_size)
            ax2.set_xticks(xs)
            ax2.set_ylim([min_dice, max_dice])
            ax2.legend(loc="best")
            ax2.set_title("ED", **config.title_font_small)

    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    if do_save:
        fig_path = os.path.join(config.root_dir, "figures")
        if not os.path.isdir(fig_path):
            os.makedirs(fig_path)
        fig_name = model_name + "_referral_results"
        fig_name_pdf = os.path.join(fig_path, fig_name + ".pdf")
        fig_name_jpeg = os.path.join(fig_path, fig_name + ".jpeg")
        plt.savefig(fig_name_pdf, bbox_inches='tight')
        plt.savefig(fig_name_jpeg, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name_pdf)

    if do_show:
        plt.show()
