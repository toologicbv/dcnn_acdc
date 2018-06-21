from config.config import config
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.referral_results import ReferralResults


def plot_referral_results(ref_result_obj, ref_slice_results=None, width=16, height=14,
                          do_save=False, do_show=True, model_name="Main title"):
    """
    :param ref_result_obj: ReferralResult object for which ALL slices are referred
    :param ref_slice_results:  ReferralResult object for which only SPECIFIC slices are referred

                    shape of the values [ ]

    :param model_name
    :param do_show
    :param do_save
    :param width
    :param height
    """
    dice_ref = ref_result_obj.get_dice_referral_dict()
    xs = np.array(dice_ref.keys())
    X = np.vstack(dice_ref.values())
    # dice_ref-values has shape [2, 4], but we
    X = np.reshape(X, (-1, 8))
    # actually we don't want the BG class to interfere with min/max, so set to zero
    X[:, 0], X[:, 4] = 0, 0
    num_of_thresholds = xs.shape[0]
    # % referred slices over ALL classes. Key=referral_threshold, value numpy [2] (for ES/ED)

    if ref_slice_results is not None:
        slice_referral = True
        dice_slices_ref = ref_slice_results.get_dice_referral_dict()
        X_pos = np.vstack(dice_slices_ref.values())
        X_pos = np.reshape(X_pos, (-1, 8))
        X_pos[:, 0], X_pos[:, 4] = 0, 0
        # percentages referred statistics
        # 1) overall
        total_perc_slices_referred = np.vstack(ref_slice_results.perc_slices_referred.values())
        total_perc_slices_referred = np.reshape(total_perc_slices_referred, (-1, 2))
        perc_slices_min = np.min(total_perc_slices_referred)
        perc_slices_max = np.max(total_perc_slices_referred)

        # per disease category
        # this numpy array will hold the % of referred slices per disease category
        # (dim1=5) for ES/ED (dim2=2)
        perc_slices_referred_per_dcat = np.zeros((num_of_thresholds, 5, 2))
        i = 0
        d_cat_idx = dict([(key_value, idx) for idx, key_value in enumerate(config.disease_categories)])
        for referral_threshold, perc_per_dcat in ref_slice_results.perc_slices_referred_per_dcat.iteritems():
            for disease_cat, percs in perc_per_dcat.iteritems():
                np_idx = d_cat_idx[disease_cat]
                perc_slices_referred_per_dcat[i, np_idx] = percs
                minp = np.min(percs)
                maxp = np.max(percs)
                if minp < perc_slices_min:
                    perc_slices_min = minp
                if maxp > perc_slices_max:
                    perc_slices_max = maxp
            i += 1
        perc_slices_min *= 100
        perc_slices_max *= 100
        perc_slices_min -= perc_slices_min * 0.03
        perc_slices_max += perc_slices_max * 0.01
        rows = 4
    else:
        slice_referral = False
        rows = 2

    columns = 4
    dice_wo_ref = ReferralResults.results_dice_wo_referral
    class_lbls = ["BG", "RV", "MYO", "LV"]
    color_code = ['b', 'r', 'g', 'b']
    dcat_colours = ['b', 'g', 'c', 'm', 'y']

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
            ax1.plot(xs, X[:, cls], label="ref all slices {}".format(class_lbls[cls]), c=color_code[cls], marker="o",
                     linestyle="--", alpha=0.6, markersize=9)
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
                         c=color_code[cls], marker="D", markersize=9,
                         linestyle=":", alpha=0.4)
        if cls == 4:
            ax2 = plt.subplot2grid((rows, columns), (0, 2), rowspan=2, colspan=2)
        if 4 < cls <= 7:
            class_idx = cls - 4
            ax2.plot(xs, X[:, cls], label="ref all slices {}".format(class_lbls[class_idx]),
                     c=color_code[class_idx], marker="o", markersize=9,
                     linestyle="--", alpha=0.6)
            wo_ref.fill(dice_wo_ref[1, class_idx])
            # plot our baseline performance
            ax2.plot(xs, wo_ref, label="no referral {}".format(class_lbls[class_idx]), c=color_code[class_idx],
                     linestyle="-", alpha=0.2)
            if slice_referral:
                ax2.plot(xs, X_pos[:, cls], label="slice-referral {}".format(class_lbls[class_idx]),
                         c=color_code[class_idx], marker="D", markersize=9,
                         linestyle=":", alpha=0.4)
            # ax2.set_ylabel("dice", **config.axis_font)
            ax2.set_xlabel("referral threshold", **config.axis_font)
            plt.tick_params(axis='both', which='major', labelsize=config.axis_ticks_font_size)
            ax2.set_xticks(xs)
            ax2.set_ylim([min_dice, max_dice])
            ax2.legend(loc="best")
            ax2.set_title("ED", **config.title_font_small)

    if slice_referral:
        # ES percentage referred slices
        ax3a = plt.subplot2grid((rows, columns), (2, 0), rowspan=2, colspan=2)
        ax3a.plot(xs, total_perc_slices_referred[:, 0] * 100, label="% referred slices",
                 c='r', marker="*", linestyle="--", alpha=0.6, markersize=12)
        ax3a.set_title("% slices referred", **config.title_font_small)
        ax3a.tick_params(axis='both', which='major', labelsize=config.axis_ticks_font_size)
        # ED percentage referred slices
        ax3b = plt.subplot2grid((rows, columns), (2, 2), rowspan=2, colspan=2)
        ax3b.plot(xs, total_perc_slices_referred[:, 1] * 100, label="% referred slices",
                  c='r', marker="*", linestyle="--", alpha=0.6, markersize=12)
        ax3b.tick_params(axis='both', which='major', labelsize=config.axis_ticks_font_size)
        # plot % referral per disease category
        for disease_cat, idx in d_cat_idx.iteritems():
            percs_es = perc_slices_referred_per_dcat[:, idx, 0]
            percs_ed = perc_slices_referred_per_dcat[:, idx, 1]
            ax3a.plot(xs, percs_es * 100, label="% referred {}".format(disease_cat),
                      c=dcat_colours[idx], marker="D", linestyle="--", alpha=0.15, markersize=6)
            ax3b.plot(xs, percs_ed * 100, label="% referred {}".format(disease_cat),
                      c=dcat_colours[idx], marker="D", linestyle="--", alpha=0.15, markersize=6)
        ax3a.legend(loc="best")
        ax3a.set_ylim([perc_slices_min, perc_slices_max])
        ax3a.set_xlabel("referral threshold", **config.axis_font)
        ax3a.set_ylabel("% slices referred", **config.axis_font)

        ax3b.legend(loc="best")
        ax3b.set_title("% slices referred", **config.title_font_small)
        ax3b.set_xlabel("referral threshold", **config.axis_font)
        ax3b.set_ylim([perc_slices_min, perc_slices_max])

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
