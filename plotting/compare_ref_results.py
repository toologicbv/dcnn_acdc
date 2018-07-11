from config.config import config
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import ScalarFormatter


def compare_referral_results(ref_result_objects, width=16, height=10, plot_base=False, x_range=None, y_range=None,
                             do_save=False, do_show=True, plot_title="Model compare", plot_type="ref",
                             fig_name="compare_ref_results"):
    """
    :param ref_result_objects: List of ReferralResult objects for which ALL slices are referred
    :param

    :param y_range:
    :param do_show
    :param do_save
    :param plot_type: determines which units to plot on the x-axis: ref=ref-thresholds, perc=mean % voxels referred
                                                                                             per volume
    :param plot_title
    :param width
    :param height
    :param fig_name
    :param x_range: list with x-value [min, max] values, currently only used for plot_type=perc
    :param plot_base: plot the dice results WITHOUT REFERRAL as a horizontal line
    """
    if plot_type not in ["perc", "ref"]:
        raise  ValueError("ERROR - argument plot_type must be in [perc, ref]")

    # Some general set-up stuff
    class_lbls = ["BG", "RV", "MYO", "LV"]
    color_code = ['b', 'r', 'g', 'b']
    model_markers = ["o", "P", "*"]
    model_linestyles = ["-", ":", "-."]
    columns = 4
    rows = 4

    # min/max values of y-axis
    max_dice = 1.01
    min_dice = 0.77
    x_min = 0
    x_max = 0.505
    eps = 1e-5
    x_axis = np.linspace(x_min, 0.5, 11)
    # prepare figure
    fig = plt.figure(figsize=(width, height))
    fig.suptitle(plot_title, **config.title_font_medium)

    ax1 = plt.subplot2grid((rows, columns), (0, 0), rowspan=2, colspan=2)
    ax1.set_ylim([min_dice, max_dice])

    ax1.tick_params(axis='both', which='major', labelsize=config.axis_ticks_font_size)
    ax1.set_ylabel("dice", **config.axis_font)
    ax1.set_title("ES", **config.title_font_small)
    # same for axis 2
    ax2 = plt.subplot2grid((rows, columns), (0, 2), rowspan=2, colspan=2)

    ax2.set_ylim([min_dice, max_dice])
    ax2.tick_params(axis='both', which='major', labelsize=config.axis_ticks_font_size)
    if plot_type == "perc":
        if x_range is not None:
            ax1.set_xlim(x_range)
            ax2.set_xlim(x_range)
        else:
            x_range = [1e-4, 1]
            ax1.set_xlim(x_range)
            ax2.set_xlim(x_range)
        if y_range is not None:
            ax1.set_ylim(y_range)
            ax2.set_ylim(y_range)
        ax1.set_xscale('log')
        ax2.set_xscale('log')
        x_t = np.linspace(x_range[0], x_range[1], 11)
        ax1.plot(x_t, np.ones(x_t.shape[0]), linestyle=":", c="gray")
        ax2.plot(x_t, np.ones(x_t.shape[0]), linestyle=":", c="gray")
        ax1.xaxis.set_major_formatter(ScalarFormatter())
        ax2.xaxis.set_major_formatter(ScalarFormatter())
        ax1.set_xlabel("mean % of voxels referred (per volume)", **config.axis_font)
        ax2.set_xlabel("mean % of voxels referred (per volume)", **config.axis_font)
    else:
        ax1.set_xlim([x_max, x_min])
        ax2.set_xlim([x_max, x_min])
        ax2.set_xlabel("referral threshold", **config.axis_font)
        ax1.set_xticks(x_axis)
        ax1.set_xlabel("referral threshold", **config.axis_font)
    # loop over result objects
    for res_idx, ref_result_obj in enumerate(ref_result_objects):
        # % referred slices over ALL classes. Key=referral_threshold, value numpy [2] (for ES/ED)
        if ref_result_obj.use_entropy_maps:
            type_of_map = "ent-map"
        else:
            type_of_map = "u-map"
        loss_function = ref_result_obj.loss_function
        dice_ref = ref_result_obj.get_dice_referral_dict()
        mdl_marker = model_markers[res_idx]
        mdl_linestyle = model_linestyles[res_idx]

        if plot_type == "perc":
            all_per_values = np.vstack(ref_result_obj.mean_uvoxels_per_vol.values())
        else:
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
        dflt_ref_key = dice_ref.keys()[-1]
        dice_wo_ref = np.concatenate((np.expand_dims(ref_result_obj.org_dice_stats[dflt_ref_key][0][0], axis=0),
                                      np.expand_dims(ref_result_obj.org_dice_stats[dflt_ref_key][1][0], axis=0)))

        # stacked everything so that dim0 is referral thresholds and dim1 is ES 4values + ED 4 values = [#refs, 8]

        wo_ref = np.zeros(x_axis.shape[0])
        for cls in np.arange(1, 8):
            if cls < 4:
                if plot_type == "perc":
                    # REMEMBER: because we're on log scale we add a "tiny" eps to the values, because some of them
                    # are equal to zero and we won't see this values on the x-axis
                    xs = all_per_values[:, 0] + eps  # ES % values per referrence threshold
                # will the baseline scalar over all thresholds for comparison
                wo_ref.fill(dice_wo_ref[0, cls])
                # plot class-dice for all thresholds
                ax1.plot(xs, X[:, cls], label="{}-{}-{}".format(loss_function, type_of_map, class_lbls[cls]),
                         c=color_code[cls],
                         marker=mdl_marker, linestyle=mdl_linestyle, alpha=0.3, markersize=9)
                # plot our baseline performance
                if plot_base:
                    ax1.plot(x_axis, wo_ref,c=color_code[cls], linestyle=mdl_linestyle, alpha=0.05)
                    #  label="{}-base-{}".format(mdl_lbl, class_lbls[cls])
                ax1.legend(loc=4, fontsize=14)
            if 4 < cls <= 7:
                class_idx = cls - 4
                if plot_type == "perc":
                    xs = all_per_values[:, 1] + eps
                ax2.plot(xs, X[:, cls], label="{}-{}-{}".format(loss_function, type_of_map, class_lbls[class_idx]),
                             c=color_code[class_idx], marker=mdl_marker, markersize=9,
                             linestyle=mdl_linestyle, alpha=0.3)

                wo_ref.fill(dice_wo_ref[1, class_idx])
                # plot our baseline performance
                if plot_base:
                    ax2.plot(x_axis, wo_ref,
                             c=color_code[class_idx], linestyle=mdl_linestyle, alpha=0.05)
                    # label="{}-base-{}".format(mdl_lbl, class_lbls[class_idx])
                # ax2.set_ylabel("dice", **config.axis_font)
                ax2.legend(loc=4, fontsize=14)
                ax2.set_title("ED", **config.title_font_small)

    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    if do_save:
        fig_path = os.path.join(config.root_dir, "figures")

        if not os.path.isdir(fig_path):
            os.makedirs(fig_path)
        fig_name += "_" + plot_type
        fig_name_pdf = os.path.join(fig_path, fig_name + ".pdf")
        fig_name_jpeg = os.path.join(fig_path, fig_name + ".jpeg")
        # plt.savefig(fig_name_pdf, bbox_inches='tight')
        plt.savefig(fig_name_jpeg, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name_pdf)

    if do_show:
        plt.show()
