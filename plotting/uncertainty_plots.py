from config.config import config
import matplotlib.pyplot as plt
import numpy as np
import os

from utils.generate_uncertainty_maps import UncertaintyMapsGenerator, ImageUncertainties


def create_figure_dir(fig_path):
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)


def plot_cross(p_axis, x_values, y_values, x_lims, y_lims, m_y=None, y_std=None):
    # m_x: mean value x-axis, m_y: mean value y-axis
    m_x, x_std = np.mean(x_values), np.std(x_values)
    if m_y is None and y_std is None:
        m_y, y_std = np.mean(y_values), np.std(y_values)

    # print("u-values {}".format(np.array_str(y_values, precision=5)))
    # print("stdev of stdev: {:.3f}".format(y_std))
    p_axis.plot([m_x] * 10, np.linspace(y_lims[0], y_lims[1], 10), 'r', alpha=0.3)
    p_axis.fill_betweenx(np.linspace(y_lims[0], y_lims[1], 10), [m_x + x_std] * 10, [m_x - x_std] * 10,
                         color='orange', alpha=0.15)
    p_axis.plot(np.linspace(x_lims[0], x_lims[1], 10), [m_y] * 10, 'r', alpha=0.3)
    p_axis.fill_between(np.linspace(x_lims[0], x_lims[1], 10), [m_y + y_std] * 10,
                        [m_y - y_std] * 10, alpha=0.05, color='r')


def plot_slices_nums(p_axis, x_vals, y_vals, font_size=12):
    idx = 1
    for x, y in zip(x_vals, y_vals):
        p_axis.text(x, y, str(idx), color="red", fontsize=font_size)
        idx += 1


def compute_mean_std_per_class(test_results, u_type, image_range):
    u_es_per_class = {1: [], 2: [], 3: []}
    u_ed_per_class = {1: [], 2: [], 3: []}

    for img_idx in image_range:
        u_stats = test_results.uncertainty_stats[img_idx][u_type]
        # 1st measure is total uncertainty for all slices!
        for cls in np.arange(1, u_stats.shape[1]):
            es_u_stats_cls = u_stats[0][cls][0]
            ed_u_stats_cls = u_stats[1][cls][0]
            u_es_per_class[cls].extend(es_u_stats_cls)
            u_ed_per_class[cls].extend(ed_u_stats_cls)

    # shape [2 phases (es/ed), 3 classes, 4 stat-values]
    stats = np.zeros((2, 3, 4))
    for cls in np.arange(1, 4):
        u_es_per_class[cls] = np.array(u_es_per_class[cls])
        # overall stats for ES classes
        stats[0, cls - 1] = np.array([np.mean(u_es_per_class[cls]), np.std(u_es_per_class[cls]), np.min(u_es_per_class[cls]), \
        np.max(u_es_per_class[cls])])
        # overall stats for ED classes
        u_ed_per_class[cls] = np.array(u_ed_per_class[cls])
        stats[1, cls - 1] = np.array(
            [np.mean(u_ed_per_class[cls]), np.std(u_ed_per_class[cls]), np.min(u_ed_per_class[cls]), \
             np.max(u_ed_per_class[cls])])

    return stats


def analyze_slices(exper_handler, width=18, height=12, do_save=False, do_show=False, image_range=None,
                   u_type="bald", use_high_threshold=False):

    model_name = exper_handler.exper.model_name
    fig_root_dir = os.path.join(exper_handler.exper.config.root_dir, exper_handler.exper.output_dir)
    fig_root_dir = os.path.join(fig_root_dir, exper_handler.exper.config.figure_path)
    image_names = exper_handler.test_results.image_names

    if image_range is None:
        image_range = exper_handler.test_results.image_ids
        patientID_range = exper_handler.test_results.image_names
        print("INFO - No image range specified using test_result object.image_ids {}".format(image_range))
    else:
        patientID_range = []
        np_test_img_ids = np.array(exper_handler.test_results.image_ids)
        # translate imageIDs to patientIDs and check for inconsistencies:
        if np.min(np.array(image_range)) < np.min(np_test_img_ids) or \
            np.max(np.array(image_range)) > np.max(np_test_img_ids):
            raise ValueError("ERROR - Invalid image range. Test set contains the following"
                             "image IDs {}".format(exper_handler.test_results.image_ids))
        # translate image range
        for id in image_range:
            img_name = exper_handler.test_results.image_names[id]
            if exper_handler.test_results.trans_dict[img_name] != id:
                raise ValueError("ERROR - Inconsistency detected between test_result.trans_dict"
                                 "object and test_result.image_names, unfortunately...")
            else:
                patientID_range.append(img_name)
    print("patientID_range ", patientID_range)
    """
        NOTE: test_results.uncertainty_stats is a list, containing dictionaries 1) "bald" and 2) "stddev"
        for each test image.
        Each dictionary contains a numpy array of shape [2, 4, #slices] where 2 = ES + ED and 4 classes
        The 4 measures are: 1) total number of uncertainty 2) number of seg-errors 3) number of pixels with 
        uncertainty above "eps"=0.01 and 4) number of pixels with uncertainty above u_threshold (which is actually 
        also in the dictionary "u_threshold")
        Also note that dice score and hd are averaged over classes
    """
    x_lims = [-0.1, 1.]
    y_lims = [-0.1, 1.1]
    class_labels = ["BG", "RV", "MYO", "LV"]
    # translate image_range to indices in test_result object
    # if translate_img_range:
    #     new_image_range = []
    #     for idx in image_range:
    #         try:
    #             i = exper_handler.test_results.image_ids.index(idx)
    #             new_image_range.append(i)
    #         except ValueError:
    #             print("WARNING - Can't find image with index {} in "
    #                   "test_results.image_ids. Discarding!".format(idx))
    #     image_range = new_image_range
    # double height of figure because we're plotting 4 instead of 2 rows
    if u_type == "stddev":
        height *= 3

    img_uncert_obj = ImageUncertainties.create_from_testresult(exper_handler.test_results)
    img_outliers = img_uncert_obj.get_outlier_obj(use_high_threshold=use_high_threshold)

    for idx, img_idx in enumerate(image_range):
        # IMPORTANT: we filled the list patiendID_range successively with the patientIDs belonging to the
        # image-range that we passed and hence we can use a simpel loop index counter to access the patientID from
        # the list
        patientID = patientID_range[idx]
        img_name = image_names[img_idx]
        img_id = exper_handler.test_results.image_ids[img_idx]
        img_acc = exper_handler.test_results.test_accuracy[img_idx]
        img_hd = exper_handler.test_results.test_hd[img_idx]
        img_seg_errors = exper_handler.test_results.seg_errors[img_idx]
        u_threshold = exper_handler.test_results.uncertainty_stats[img_idx]["u_threshold"]
        es_performance = "(RV/Myo/LV):\t ES {:.2f}/{:.2f}/{:.2f} " \
                         " HD: {:.2f}/{:.2f}/{:.2f}".format(img_acc[1], img_acc[2], img_acc[3],
                                                          img_hd[1], img_hd[2], img_hd[3])
        ed_performance = "\t\tED: {:.2f}/{:.2f}/{:.2f} " \
                         " HD: {:.2f}/{:.2f}/{:.2f}".format(img_acc[5], img_acc[6], img_acc[7],
                                                          img_hd[5], img_hd[6], img_hd[7])
        # make sure the \tab delimiters get displayed correctly
        es_performance = es_performance.expandtabs()
        ed_performance = ed_performance.expandtabs()
        img_dice = exper_handler.test_results.test_accuracy_slices[img_idx]  # [2, 4 classes, #slices]
        img_hd = exper_handler.test_results.test_hd_slices[img_idx]  # [2, 4 classes, #slices]
        if u_type == "bald":
            u_stats = exper_handler.test_results.uncertainty_stats[img_idx][u_type]  # [2, 4 measures, #slices]
            u_stats_cls = None
            es_dice_cls = None
            ed_dice_cls = None
            es_hd_cls = None
            ed_hd_cls = None
            rows = 2
            columns = 2
        else:
            # for stddev we store the u-stats per class. we'll use 3 classes below (RV/MYO/LV)
            # [2, 4classes, 4 measures, #slices], so we average over classes dim1 here
            u_stats = np.mean(exper_handler.test_results.uncertainty_stats[img_idx][u_type], axis=1)
            u_stats_cls = exper_handler.test_results.uncertainty_stats[img_idx][u_type]
            es_dice_cls = img_dice[0]
            ed_dice_cls = img_dice[1]
            es_hd_cls = img_hd[0]
            ed_hd_cls = img_hd[1]
            rows = 5  # 1row for mean-uncertainties over classes, 2-4 for class uncertainties, 5th-row mean seg-errors
            columns = 2
        num_of_slices = u_stats.shape[2]

        es_dice = np.mean(img_dice[0], axis=0, keepdims=True)
        ed_dice = np.mean(img_dice[1], axis=0, keepdims=True)
        es_hd = np.mean(img_hd[0], axis=0, keepdims=True)
        ed_hd = np.mean(img_hd[1], axis=0, keepdims=True)
        es_seg_errors = img_seg_errors[:, :4]
        ed_seg_errors = img_seg_errors[:, 4:]
        es_mean_seg_errors = np.mean(es_seg_errors, axis=1)
        ed_mean_seg_errors = np.mean(ed_seg_errors, axis=1)

        # print("---------------------------- Image performance --------------------------------------")
        # print(es_performance)
        # print(ed_performance)
        # RESCALE TOTAL UNCERTAINTY VALUE TO INTERVAL [0,1]. stats_per_phase contains overall statistics for
        # each phase (ES/ED 1st index). 2nd index specifies stat-measure (0-3) mean/std/min/max
        total_uncert_es = img_uncert_obj.norm_uncertainty_per_phase[patientID][0]
        total_uncert_ed = img_uncert_obj.norm_uncertainty_per_phase[patientID][1]
        print("--------- Outlier slices -------------")
        print(np.array(img_outliers.outliers_per_img[patientID]) + 1)
        fig = plt.figure(figsize=(width, height))
        fig.suptitle("Model " + model_name + "\n" + "Image: " + img_name + "(id={})".format(img_id) +
                     "\n\n" + es_performance + ed_performance,
                     **config.title_font_small)
        # ---------------------------- PLOT (0,0) ----------------------------------------------------
        ax1 = plt.subplot2grid((rows, columns), (0, 0), rowspan=1, colspan=1)
        ax1.scatter(es_dice.squeeze(), total_uncert_es, s=es_hd * 10, alpha=0.3, c='g')
        plot_slices_nums(ax1, es_dice.squeeze(), total_uncert_es)
        ax1.set_ylabel("total uncertainty")
        ax1.set_xlabel("mean dice")
        ax1.set_title("ES mean-slice-uncertainties (area=mean-hd) (#slices={})".format(num_of_slices))
        ax1.set_xlim(x_lims)
        ax1.set_ylim(y_lims)
        plot_cross(ax1, es_dice.squeeze(), total_uncert_es, x_lims, y_lims)
        # ---------------------------- PLOT (0,1) ----------------------------------------------------
        ax2 = plt.subplot2grid((rows, columns), (0, 1), rowspan=1, colspan=1)
        ax2.scatter(ed_dice.squeeze(), total_uncert_ed, s=ed_hd * 10, alpha=0.3, c='b')
        plot_slices_nums(ax2, ed_dice.squeeze(), total_uncert_ed)
        ax2.set_ylabel("total uncertainty")
        ax2.set_xlabel("mean dice")
        ax2.set_title("ED mean-slice-uncertainties (area=mean-hd) (#slices={})".format(num_of_slices))
        ax2.set_xlim(x_lims)
        ax2.set_ylim(y_lims)
        plot_cross(ax2, ed_dice.squeeze(), total_uncert_ed, x_lims, y_lims)
        # we need an offset so that the figures following the class fig-details will be plotted in the correct
        # position
        row_offset = 1
        if u_type == "stddev":
            # a couple of plots per class (RV/MYO/LV)
            for cls in np.arange(1, u_stats_cls.shape[1]):

                total_uncert_es_cls = img_uncert_obj.norm_uncertainty_per_cls[patientID][0, cls]
                total_uncert_ed_cls = img_uncert_obj.norm_uncertainty_per_cls[patientID][1, cls]
                es_u_mean = img_uncert_obj.norm_stats_per_cls[0, cls, 0]
                es_u_stddev = img_uncert_obj.norm_stats_per_cls[0, cls, 1]
                # ---------------------------- PLOT (cls, 0) ----------------------------------------------------
                ax1b = plt.subplot2grid((rows, columns), (cls, 0), rowspan=1, colspan=1)
                ax1b.scatter(es_dice_cls[cls].squeeze(), total_uncert_es_cls, s=es_hd_cls[cls] * 10, alpha=0.3, c='g')
                plot_slices_nums(ax1b, es_dice_cls[cls].squeeze(), total_uncert_es_cls)
                ax1b.set_ylabel("total uncertainty")
                ax1b.set_xlabel("mean dice")
                ax1b.set_title("ES - class {} slice uncertainties (area=mean-hd) (#slices={})".format(class_labels[cls],
                                                                                               num_of_slices))
                ax1b.set_xlim(x_lims)
                ax1b.set_ylim(y_lims)
                plot_cross(ax1b, es_dice_cls[cls].squeeze(), total_uncert_es_cls, x_lims, y_lims, m_y=es_u_mean,
                           y_std=es_u_stddev)
                # ---------------------------- PLOT (cls,1) ----------------------------------------------------
                ed_u_mean = img_uncert_obj.norm_stats_per_cls[1, cls, 0]
                ed_u_stddev = img_uncert_obj.norm_stats_per_cls[1, cls, 1]
                ax2b = plt.subplot2grid((rows, columns), (cls, 1), rowspan=1, colspan=1)
                ax2b.scatter(ed_dice_cls[cls].squeeze(), total_uncert_ed_cls, s=ed_hd_cls[cls] * 10, alpha=0.3, c='b')
                plot_slices_nums(ax2b, ed_dice_cls[cls].squeeze(), total_uncert_ed_cls)
                ax2b.set_ylabel("total uncertainty")
                ax2b.set_xlabel("mean dice")
                ax2b.set_title("ED - class {} slice uncertainties (area=mean-hd) (#slices={})".format(class_labels[cls],
                                                                                              num_of_slices))
                ax2b.set_xlim(x_lims)
                ax2b.set_ylim(y_lims)
                plot_cross(ax2b, ed_dice_cls[cls].squeeze(), total_uncert_ed_cls, x_lims, y_lims, m_y=ed_u_mean,
                           y_std=ed_u_stddev)
                # add one to offset for the next figures ax3 (below)
                row_offset += 1

        # ---------------------------------------------------------------------------------------------------------
        # Use number of seg errors as area of the marker
        # ---------------------------------------------------------------------------------------------------------
        # ---------------------------- PLOT (1,0) ----------------------------------------------------
        ax3 = plt.subplot2grid((rows, columns), (row_offset, 0), rowspan=1, colspan=1)
        ax3.scatter(es_dice.squeeze(), total_uncert_es, s=es_mean_seg_errors * 1.5, alpha=0.3, c='g')
        plot_slices_nums(ax3, es_dice.squeeze(), total_uncert_es)

        ax3.set_ylabel("total uncertainty")
        ax3.set_xlabel("mean dice")
        ax3.set_title("ES mean-slice-uncertainties (area=pixels) (#slices={})".format(num_of_slices))
        ax3.set_xlim(x_lims)
        ax3.set_ylim(y_lims)
        plot_cross(ax3, es_dice.squeeze(), total_uncert_es, x_lims, y_lims)
        # ---------------------------- PLOT (1,1) ----------------------------------------------------
        ax4 = plt.subplot2grid((rows, columns), (row_offset, 1), rowspan=1, colspan=1)
        ax4.scatter(ed_dice.squeeze(), total_uncert_ed, s=ed_mean_seg_errors * 1.5, alpha=0.3, c='b')
        plot_slices_nums(ax4, ed_dice.squeeze(), total_uncert_ed)
        ax4.set_ylabel("total uncertainty")
        ax4.set_xlabel("mean dice")
        ax4.set_title("ED mean-slice-uncertainties (area=seg-errors) (#slices={})".format(num_of_slices))
        ax4.set_xlim(x_lims)
        ax4.set_ylim(y_lims)
        plot_cross(ax4, ed_dice.squeeze(), total_uncert_ed, x_lims, y_lims)

        #   fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        if do_save:
            fig_dir = os.path.join(fig_root_dir, img_name)
            create_figure_dir(fig_dir)
            u_threshold = str(u_threshold).replace(".", "_")
            fig_name = "slice_analysis_img_{}_threshold{}".format(img_id, u_threshold)
            fig_name = os.path.join(fig_dir, fig_name + ".pdf")

            plt.savefig(fig_name, bbox_inches='tight')
            print("INFO - Successfully saved fig %s" % fig_name)
        if do_show:
            plt.show()
        plt.close()
