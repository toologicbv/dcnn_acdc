from config.config import config
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_cross(p_axis, x_lims, y_lims):
    p_axis.plot([0.5] * 10, np.linspace(y_lims[0], y_lims[1], 10), 'r', alpha=0.3)
    p_axis.plot(np.linspace(x_lims[0], x_lims[1], 10), [0.5] * 10, 'r', alpha=0.3)


def plot_slices_nums(p_axis, x_vals, y_vals, font_size=12):
    idx = 1
    for x, y in zip(x_vals, y_vals):
        p_axis.text(x, y, str(idx), color="red", fontsize=font_size)
        idx += 1


def analyze_slices(exper_handler, width=18, height=12, do_save=False, do_show=False, image_range=None,
                   u_type="bald", fig_name=None):

    # model_name = exper_handler.exper.model_name
    fig_output_dir = os.path.join(exper_handler.exper.config.root_dir, exper_handler.exper.output_dir)
    fig_output_dir = os.path.join(fig_output_dir, exper_handler.exper.config.figure_path)
    image_ids = exper_handler.test_results.image_ids
    if image_range is None:
        image_range = np.arange(len(exper_handler.test_results.images))

    """
        NOTE: test_results.uncertainty_stats is a list, containing dictionaries 1) "bald" and 2) "stddev"
        for each test image.
        Each dictionary contains a numpy array of shape [2, 4, #slices] where 2 = ES + ED and 4 classes
        The 4 measures are: 1) total number of uncertainty 2) number of seg-errors 3) number of pixels with 
        uncertainty above "eps"=0.01 and 4) number of pixels with uncertainty above u_threshold (which is actually 
        also in the dictionary "u_threshold")

        Also note that dice score and hd are averaged over classes

    """
    x_lims = [0, 1.]
    y_lims = [-0.1, 1.1]

    for img_idx in image_range:
        es_u_stats = []
        ed_u_stats = []
        es_hd = []
        ed_hd = []
        es_dice = []
        ed_dice = []
        img_name = image_ids[img_idx][:image_ids[img_idx].find("_")]
        img_acc = exper_handler.test_results.test_accuracy[img_idx]
        img_hd = exper_handler.test_results.test_hd[img_idx]
        es_performance = "Dice \ HD (RV/Myo/LV):\t ES {:.2f}/{:.2f}/{:.2f} " \
                         "\\ {:.2f}/{:.2f}/{:.2f}".format(img_acc[1], img_acc[2], img_acc[3],
                                                          img_hd[1], img_hd[2], img_hd[3])
        ed_performance = "\t\tED: {:.2f}/{:.2f}/{:.2f} " \
                         "\\ {:.2f}/{:.2f}/{:.2f}".format(img_acc[5], img_acc[6], img_acc[7],
                                                          img_hd[5], img_hd[6], img_hd[7])
        es_performance = es_performance.expandtabs()
        ed_performance = ed_performance.expandtabs()

        u_stats = exper_handler.test_results.uncertainty_stats[img_idx][u_type]  # [2, 4 measures, #slices]
        num_of_slices = u_stats.shape[2]
        es_u_stats.append(u_stats[0])
        ed_u_stats.append(u_stats[1])
        img_dice = exper_handler.test_results.test_accuracy_slices[img_idx]  # [2, 4 measures, #slices]
        es_dice.append(np.mean(img_dice[0], axis=0, keepdims=True))
        ed_dice.append(np.mean(img_dice[1], axis=0, keepdims=True))
        img_hd = exper_handler.test_results.test_hd_slices[img_idx]
        es_hd.append(np.mean(img_hd[0], axis=0, keepdims=True))
        ed_hd.append(np.mean(img_hd[1], axis=0, keepdims=True))

        es_u_stats = np.concatenate(es_u_stats, axis=1)
        ed_u_stats = np.concatenate(ed_u_stats, axis=1)
        es_hd = np.concatenate(es_hd, axis=1)
        ed_hd = np.concatenate(ed_hd, axis=1)
        es_dice = np.concatenate(es_dice, axis=1)
        ed_dice = np.concatenate(ed_dice, axis=1)
        # print("---------------------------- Image performance --------------------------------------")
        # print(es_performance)
        # print(ed_performance)
        # RESCALE TOTAL UNCERTAINTY VALUE TO INTERVAL [0,1]
        total_uncert_es = (es_u_stats[0] - np.min(es_u_stats[0])) / (np.max(es_u_stats[0]) - np.min(es_u_stats[0]))
        total_uncert_ed = (ed_u_stats[0] - np.min(ed_u_stats[0])) / (np.max(ed_u_stats[0]) - np.min(ed_u_stats[0]))

        fig = plt.figure(figsize=(width, height))
        fig.suptitle("Image: " + img_name + "\n\n" + es_performance + ed_performance, **config.title_font_medium)
        # ---------------------------- PLOT (0,0) ----------------------------------------------------
        ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=1)
        ax1.scatter(es_dice.squeeze(), total_uncert_es, s=es_hd * 10, alpha=0.3, c='g')
        plot_slices_nums(ax1, es_dice.squeeze(), total_uncert_es)
        ax1.set_ylabel("total uncertainty")
        ax1.set_xlabel("mean dice")
        ax1.set_title("ES slice uncertainties (area=mean-hd) (#slices={})".format(num_of_slices))
        ax1.set_xlim(x_lims)
        ax1.set_ylim(y_lims)
        plot_cross(ax1, x_lims, y_lims)
        # ---------------------------- PLOT (0,1) ----------------------------------------------------
        ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1)
        ax2.scatter(ed_dice.squeeze(), total_uncert_ed, s=ed_hd * 10, alpha=0.3, c='b')
        plot_slices_nums(ax2, ed_dice.squeeze(), total_uncert_ed)
        ax2.set_ylabel("total uncertainty")
        ax2.set_xlabel("mean dice")
        ax2.set_title("ED slice uncertainties (area=mean-hd) (#slices={})".format(num_of_slices))
        ax2.set_xlim(x_lims)
        ax2.set_ylim(y_lims)
        plot_cross(ax2, x_lims, y_lims)
        # ---------------------------------------------------------------------------------------------------------
        # Use number of seg errors as area of the marker
        # ---------------------------------------------------------------------------------------------------------
        # ---------------------------- PLOT (1,0) ----------------------------------------------------
        ax3 = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1)
        ax3.scatter(es_dice.squeeze(), total_uncert_es, s=es_u_stats[1] * 0.1, alpha=0.3, c='g')
        plot_slices_nums(ax3, es_dice.squeeze(), total_uncert_es)

        ax3.set_ylabel("total uncertainty")
        ax3.set_xlabel("mean dice")
        ax3.set_title("ES slice uncertainties (area=seg-errors) (#slices={})".format(num_of_slices))
        ax3.set_xlim(x_lims)
        ax3.set_ylim(y_lims)
        plot_cross(ax3, x_lims, y_lims)
        # ---------------------------- PLOT (1,1) ----------------------------------------------------
        ax4 = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1)
        ax4.scatter(ed_dice.squeeze(), total_uncert_ed, s=ed_u_stats[1] * 0.1, alpha=0.3, c='b')
        plot_slices_nums(ax4, ed_dice.squeeze(), total_uncert_ed)
        ax4.set_ylabel("total uncertainty")
        ax4.set_xlabel("mean dice")
        ax4.set_title("ED slice uncertainties (area=seg-errors) (#slices={})".format(num_of_slices))
        ax4.set_xlim(x_lims)
        ax4.set_ylim(y_lims)
        plot_cross(ax4, x_lims, y_lims)

        #   fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        if do_save:
            if fig_name is None:
                fig_name = "slice_analysis_{}images".format(len(image_range))

            fig_name = os.path.join(fig_output_dir, fig_name + ".pdf")

            plt.savefig(fig_name, bbox_inches='tight')
            print("INFO - Successfully saved fig %s" % fig_name)
        if do_show:
            plt.show()
        plt.close()
