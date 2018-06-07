from config.config import config
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.referral_handler import ReferralResults


def plot_referral_results(dice_ref, model_name, dice_ref_pos_only=None, width=16, height=14,
                          do_save=False, do_show=True):
    """
    :param dice_ref: OrderedDict with key referral_threshold

    """
    if dice_ref_pos_only is not None:
        pos_only = True
        X_pos = np.vstack(dice_ref_pos_only.values())
        X_pos = np.reshape(X_pos, (-1, 8))
        X_pos[:, 0], X_pos[:, 4] = 0, 0
    else:
        pos_only = False
    rows = 2
    columns = 4
    dice_wo_ref = ReferralResults.results_dice_wo_referral
    class_lbls = ["BG", "RV", "MYO", "LV"]
    color_code = ['b', 'r', 'g', 'b']

    xs = np.array(dice_ref.keys() )
    X = np.vstack(dice_ref.values())
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
            ax1.plot(xs, X[:, cls], label="with ref {}".format(class_lbls[cls]), c=color_code[cls], marker="*",
                     linestyle="--", alpha=0.6)
            # plot our baseline performance
            if cls == 1:
                print(wo_ref)
            ax1.plot(xs, wo_ref, label="wo ref {}".format(class_lbls[cls]), c=color_code[cls],
                     linestyle="-", alpha=0.2)
            ax1.set_ylabel("dice", **config.axis_font)
            ax1.set_xlabel("referral threshold", **config.axis_font)
            ax1.set_ylim([min_dice, max_dice])
            plt.tick_params(axis='both', which='major', labelsize=config.axis_ticks_font_size)
            ax1.set_xticks(xs)
            ax1.legend(loc="best")
            ax1.set_title("ES", **config.title_font_small)
            if pos_only:
                ax1.plot(xs, X_pos[:, cls], label="ref pos-only {}".format(class_lbls[cls]),
                         c=color_code[cls], marker="D",
                         linestyle=":", alpha=0.6)
        if cls == 4:
            ax2 = plt.subplot2grid((rows, columns), (0, 2), rowspan=2, colspan=2)
        if 4 < cls <= 7:
            class_idx = cls - 4
            ax2.plot(xs, X[:, cls], label="with ref {}".format(class_lbls[class_idx]),
                     c=color_code[class_idx], marker="*",
                     linestyle="--", alpha=0.6)
            wo_ref.fill(dice_wo_ref[1, class_idx])
            # plot our baseline performance
            ax2.plot(xs, wo_ref, label="wo ref {}".format(class_lbls[class_idx]), c=color_code[class_idx],
                     linestyle="-", alpha=0.2)
            if pos_only:
                ax2.plot(xs, X_pos[:, cls], label="ref pos-only {}".format(class_lbls[class_idx]),
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
        fig_name = os.path.join(fig_path, fig_name + ".pdf")
        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)

    if do_show:
        plt.show()
