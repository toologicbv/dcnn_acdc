from config.config import config
import matplotlib.pyplot as plt
import pylab
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

    # if we compare all three models we need loss-function in legend
    if len(ref_result_objects) >= 3:
        all_losses = True
    else:
        all_losses = False
    # Some general set-up stuff
    class_lbls = ["BG", "RV", "MYO", "LV"]
    color_code = ['b', 'r', 'g', 'b']
    model_markers = ["o", "s", "*"]
    model_linestyles = ["-", ":", "-."]
    model_linestyle_base = "--"
    mdl_marker_size = 18
    columns = 4
    rows = 4

    # min/max values of y-axis
    max_dice = 1.01
    min_dice = 0.77
    x_min = 0
    x_max = 0.505
    eps = 1e-5

    # prepare figure
    fig = plt.figure(figsize=(width, height))
    fig.suptitle(plot_title, **config.title_font_large)

    ax2 = plt.subplot2grid((rows, columns), (0, 0), rowspan=2, colspan=2)
    ax1 = plt.subplot2grid((rows, columns), (0, 2), rowspan=2, colspan=2, sharey=ax2)
    ax1.set_ylim([min_dice, max_dice])
    ax2.set_ylabel("dice", **{'fontname': 'Monospace', 'size': '36', 'color': 'black', 'weight': 'normal'})
    pylab.setp(ax1.get_yticklabels(), visible=False)
    ax1.set_title("ES", **config.title_font_large)
    # same for axis 2

    ax2.set_title("ED", **config.title_font_large)
    ax2.set_ylim([min_dice, max_dice])
    ax1.tick_params(axis='both', which='major', labelsize=30)
    ax2.tick_params(axis='both', which='major', labelsize=30)
    ax2.yaxis.tick_right()
    if plot_type == "perc":
        if x_range is not None:
            ax1.set_xlim(x_range)
            ax2.set_xlim(x_range)
        else:
            x_range = [0.0, 100]
            ax1.set_xlim(x_range)
            ax2.set_xlim(x_range)
        if y_range is not None:
            ax1.set_ylim(y_range)
            ax2.set_ylim(y_range)
        x_axis = np.linspace(x_range[0], x_range[1], 11)
        ax1.set_xscale('log')
        ax2.set_xscale('log')
        x_t = np.linspace(x_range[0], x_range[1], 11)
        # plot the horizontal line that indicates a dice score of 1 e.g. 100%
        ax1.plot(x_t, np.ones(x_t.shape[0]), linestyle=":", c="gray", linewidth=6)
        ax2.plot(x_t, np.ones(x_t.shape[0]), linestyle=":", c="gray", linewidth=6)
        ax1.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax1.set_xticks(np.array([0.001, 0.1, 1, 10, 100.]))
        ax2.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax2.set_xticks(np.array([0.001, 0.1, 1, 10, 100.]))
        ax1.set_xlabel("mean % of voxels", **config.axis_font28)
        ax2.set_xlabel("mean % of voxels", **config.axis_font28)
    else:
        x_axis = np.linspace(x_min, x_max, 11)
        ax1.set_xlim([x_max, x_min])
        ax2.set_xlim([x_max, x_min])
        ax2.set_xlabel("referral threshold", **config.axis_font28)
        ax1.set_xticks(x_axis)
        ax1.set_xlabel("referral threshold", **config.axis_font28)
    # loop over result objects
    for res_idx, ref_result_obj in enumerate(ref_result_objects):
        # % referred slices over ALL classes. Key=referral_threshold, value numpy [2] (for ES/ED)
        if ref_result_obj.use_entropy_maps:
            type_of_map = "e-map"
        else:
            type_of_map = "u-map"
        loss_function = ref_result_obj.loss_function
        dice_ref = ref_result_obj.get_dice_referral_dict()
        mdl_marker = model_markers[res_idx]
        mdl_linestyle = model_linestyles[res_idx]

        """
            IMPORTANT: Dictionary keys are sorted from low uncertainty threshold to high, which implies
                        that corresponding dice results are descending from high scores to low scores!
            In case we're plotting the mean percentage referred values on the x-axis, the all_per_values
            object will contain the mean average % referred voxels, which also starts with high values for
            tiny uncertainty thresholds 
        """
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
            if 4 <= cls <= 7:
                class_idx = cls - 4
            else:
                class_idx = cls
            # if the max of xs is below 100%, we plot a line from the last point to the right most side of the
            # figure in order to visualize the achieved referral performance
            max_cls_dice = None
            if plot_type == "perc":
                # we use the max dice to plot a horizontal line to the right, to indicate the max dice achieved
                max_cls_dice = np.max(X[:, cls])
            if all_losses:
                l_label = "{}-{}-{}".format(loss_function, type_of_map, class_lbls[class_idx])
            else:
                l_label = "{}-{}".format(type_of_map, class_lbls[class_idx])

            if cls < 4:
                if plot_type == "perc":
                    # REMEMBER: because we're on log scale we add a "tiny" eps to the values, because some of them
                    # are equal to zero and we won't see this values on the x-axis
                    xs = 100 * (all_per_values[:, 0] + eps)  # ES % values per referrence threshold
                    # xs = np.concatenate((xs, [0]))
                    # y_values = np.concatenate((X[:, cls], [dice_wo_ref[0, cls]]))
                    y_values = X[:, cls]
                    if np.max(xs) < 100.:
                        # we take index 0 from xs because the sequence is ascending, starting with high values to low
                        xs_max_line = np.array([xs[0], 100.])
                        y_max_line = np.array([max_cls_dice, max_cls_dice])
                    # if cls != 0:
                    #     print("------- {} {} ---------".format(cls, ref_result_obj.use_entropy_maps))
                    #     print(xs)
                    #     print(y_values)
                else:
                    y_values = X[:, cls]
                # will the baseline scalar over all thresholds for comparison
                wo_ref.fill(dice_wo_ref[0, cls])
                # plot class-dice for all thresholds
                ax1.plot(xs, y_values, label=l_label, c=color_code[cls],
                         marker=mdl_marker, linestyle=mdl_linestyle, alpha=0.25, linewidth=6, markersize=mdl_marker_size)
                # when plotting % on the x-axis, we extend the last y-value with a line to the right, to visualize
                # the best referral results.
                if max_cls_dice is not None:
                    ax1.plot(xs_max_line, y_max_line, c=color_code[cls], linestyle=mdl_linestyle,
                             alpha=0.15, linewidth=6)
                # plot our baseline performance
                if plot_base:
                    ax1.plot(x_axis, wo_ref, c=color_code[cls], linestyle=model_linestyle_base, alpha=0.05, linewidth=6)

                ax1.legend(loc=4, fontsize=24)
                ax1.grid("on")
            if 4 < cls <= 7:
                class_idx = cls - 4
                if plot_type == "perc":
                    xs = 100 * (all_per_values[:, 1] + eps)
                    if np.max(xs) < 100.:
                        # we take index 0 from xs because the sequence is ascending, starting with high values to low
                        xs_max_line = np.array([xs[0], 100.])
                        y_max_line = np.array([max_cls_dice, max_cls_dice])
                ax2.plot(xs, X[:, cls], label=l_label, c=color_code[class_idx], marker=mdl_marker,
                         markersize=mdl_marker_size,
                         linestyle=mdl_linestyle, alpha=0.25, linewidth=6)
                # when plotting % on the x-axis, we extend the last y-value with a line to the right, to visualize
                # the best referral results.
                if max_cls_dice is not None:
                    ax2.plot(xs_max_line, y_max_line, c=color_code[class_idx], linestyle=mdl_linestyle,
                             alpha=0.15, linewidth=6)
                wo_ref.fill(dice_wo_ref[1, class_idx])
                # plot our baseline performance
                if plot_base:
                    ax2.plot(x_axis, wo_ref, c=color_code[class_idx], linestyle=model_linestyle_base,
                             alpha=0.05, linewidth=6)

                ax2.legend(loc=4, fontsize=24)
                ax2.grid("on")
        # increase markersize for next result object for visibility
        # mdl_marker_size += 6
    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    if do_save:
        fig_path = os.path.join(config.root_dir, "figures")

        if not os.path.isdir(fig_path):
            os.makedirs(fig_path)
        fig_name += "_" + plot_type
        # fig_name_pdf = os.path.join(fig_path, fig_name + ".pdf")
        fig_name_jpeg = os.path.join(fig_path, fig_name + ".jpeg")
        # plt.savefig(fig_name_pdf, bbox_inches='tight')
        plt.savefig(fig_name_jpeg, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name_jpeg)

    if do_show:
        plt.show()
