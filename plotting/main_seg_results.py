from config.config import config
import matplotlib.pyplot as plt
from pylab import MaxNLocator
import numpy as np
import os
import copy

from matplotlib import cm
from common.common import to_rgb1a, set_error_pixels, create_mask_uncertainties, detect_seg_contours
from common.common import load_pred_labels, prepare_referrals, generate_std_hist_corr_err
from common.common import generate_std_hist_corr_err_per_class, overlay_seg_mask
from common.common import get_exper_objects


def plot_seg_erros_uncertainties(exper_handler, test_set, patient_id=None, width=16,
                                 test_results=None, slice_filter_type=None,
                                 info_type=None, referral_threshold=0., do_show=False, model_name="",
                                 do_save=False, slice_range=None, errors_only=False,
                                 load_base_model_pred_labels=False, umaps_without_postprocessing=False):
    """

    :param patient_id:
    :param width:
    :param slice_filter_type: None or M, MD, MS. If not None implies that we referred only specific slices
            during referral.
    :param test_set:
    :param exper_handler:
    :param info_type: default is None. Only used if we want extra histogram figures related to softmax probabilities
           info_type="probs"   OR  uncertainties. info_type="uncertainty"
    :param referral_threshold:

    :param do_show:
    :param model_name:
    :param do_save:
    :param slice_range:
    :param errors_only:
    :param umaps_without_postprocessing: when overlaying u-maps use the thresholded maps WITHOUT POST-PROCESSING
                                         IF FALSE: use filtered+post-processing
    :param load_base_model_pred_labels: load the predicted labels for Jelmer's baseline model
           The files should be stored in the specific fold-directory that you're currently evaluating
           e.g:     data/Folds/fold1/pred_lbls/
    :param plot_detailed_hists: if False then we plot only 5 rows: 1-2 images, 3-4 huge uncertainty maps,
                                                                   5: uncertainty maps per class
                                                                   NO HISTOGRAMS!
    :return:
    """

    def transparent_cmap(cmap, N=255):
        "Copy colormap and set alpha values"

        mycmap = cmap
        mycmap._init()
        mycmap._lut[:, -1] = np.linspace(0, 0.8, N + 4)
        return mycmap

    # Use base cmap to create transparent
    mycmap = transparent_cmap(plt.get_cmap('jet'))

    if test_set is None:
        raise ValueError("ERROR - test_set cannot be None.")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Ubuntu'
    plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    column_lbls = ["bg", "RV", "MYO", "LV"]

    image, label = test_set.get_test_pair(patient_id)
    # VERY IMPORTANT: get rid off padding around image
    image = image[:, config.pad_size:-config.pad_size, config.pad_size:-config.pad_size, :]
    image_num = test_set.trans_dict[patient_id]
    image_name = patient_id
    if exper_handler.patients is None:
        exper_handler.get_patients(use_four_digits=test_set.generate_flipped_images)
    if exper_handler.referral_umaps is None:
        exper_handler.get_referral_maps(referral_threshold, per_class=False, aggregate_func="max")
    # get the matrix of shape [2, #slices] that indicates which slices were referred for this image (ES/ED)
    if slice_filter_type is None:
        # unfortunately we need a boolean array shape [2, #slices] although we referred ALL SLICES
        referred_slices = np.ones((2, image.shape[3]))
    else:
        if exper_handler.referred_slices is None:
            referred_slices = exper_handler.get_referred_slices(referral_threshold, patient_id=patient_id,
                                                                slice_filter_type=slice_filter_type)
        elif patient_id in exper_handler.referred_slices.keys():
            referred_slices = exper_handler.referred_slices[patient_id]
        else:
            referred_slices = exper_handler.get_referred_slices(referral_threshold, patient_id=patient_id,
                                                                    slice_filter_type=slice_filter_type)

    ref_u_map_blobs = exper_handler.ref_map_blobs[patient_id]
    # get disease category
    patient_category = exper_handler.patients[patient_id]

    if test_results is not None:
        pred_labels = test_results.pred_labels[image_num]
        uncertainty_map = test_results.stddev_maps[image_num]
        umap_dir = test_results.umap_dir
        pred_labels_input_dir = test_results.pred_labels_input_dir
        test_results._create_figure_dir(patient_id)
        fig_path = os.path.join(test_results.fig_output_dir, image_name)
    else:
        # get necessary objects with the help of the exper_handler object
        # uncertainty_map is RAW map per class
        umap_dir, pred_labels_input_dir, fig_path, pred_labels, uncertainty_map = \
            get_exper_objects(exper_handler, patient_id)

    # Set some general things we need for the figures:
    num_of_classes = label.shape[0]
    half_classes = num_of_classes / 2
    num_of_slices_img = image.shape[3]
    columns = half_classes
    # ------------------------------- Referral preparations --------------------------------------
    if referral_threshold != 0.:
        # formerly we used object "referral_pred_labels_filter_slices" where we only referred positive segmented voxes
        # we dropped this functionality but haven't eliminated all references to it. hence we set it here to None
        # object "filtered_cls_std_map_wpostp" is filtered/thresholded but without post-processing
        # filtered_cls_std_map_wpostp has shape [2, width, height, #slices]
        filtered_cls_std_map, filtered_std_map, filtered_cls_std_map_wpostp, referral_pred_labels = \
            prepare_referrals(image_name, referral_threshold, umap_dir, pred_labels_input_dir)
    else:
        # Normal non-referral functionality
        filtered_std_map = None
        referral_pred_labels = None
        filtered_stddev_bin_edges = None
        filtered_stddev_bin_values = None
    # ------------------------------------ LOADING pred_labels -----------------------------------------
    # Finally we need the predictions of the base-model if...
    if load_base_model_pred_labels:
        base_pred_labels_input_dir = os.path.join(test_set.abs_path_fold + str(test_set.fold_ids[0]), config.pred_lbl_dir)
        search_path = os.path.join(base_pred_labels_input_dir, image_name + "_pred_labels.npz")
        # the predicted labels we obtained with the MC model when NOT using dropout during inference!
        pred_labels_base_model = load_pred_labels(search_path)
    else:
        pred_labels_base_model = None
    # ---------------------------- If REFERRAL and we only want to refer pos-labels we filter here ------------
    # if referral_threshold != 0.:
    #     mask = pred_labels == 0
    #     overall_mask = create_mask_uncertainties(pred_labels)
    #     overall_mask_es = overall_mask[0]
    #     overall_mask_ed = overall_mask[1]
    #     filtered_std_map_pos_only = copy.deepcopy(filtered_std_map)
    #     filtered_cls_std_map_pos_only = copy.deepcopy(filtered_cls_std_map)
    #     filtered_std_map_pos_only[0][overall_mask_es] = 0.
    #     filtered_std_map_pos_only[1][overall_mask_ed] = 0.
    #     filtered_cls_std_map_pos_only[mask] = 0
    # else:
    # we don't use this functionality (referral of pos-only voxes) anymore
    filtered_std_map_pos_only = None

    if referral_threshold != 0:
        filtered_stddev_bin_edges, filtered_stddev_bin_values = \
            generate_std_hist_corr_err(filtered_std_map, label, pred_labels, referral_threshold=referral_threshold)

    if slice_range is None:
        slice_range = np.arange(0, image.shape[3])
    num_of_slices = len(slice_range)

    if errors_only:
        rows = 4  # multiply with 4 because two rows ES, two rows ED for each slice
        height = 20
    elif info_type is None:
        if referral_threshold != 0:
            rows = 18
            height = 70
        else:
            rows = 14
            height = 50
    else:
        rows = 22
        height = 86
    print("Rows/columns/height {}/{}/{}".format(rows, columns, height))

    _ = rows * num_of_slices * columns
    if errors_only:
        height = 20

    for img_slice in slice_range:
        if referral_threshold > 0.:
            # rank = np.where(sorted_u_value_list == total_uncertainty_per_slice[img_slice])[0][0]
            main_title = r"Model {} - Test image: {} ({}) - slice: {}" "\n" \
                         r"$\sigma_{{Tr}}={:.2f}$".format(model_name, image_name, patient_category,
                                                          img_slice + 1, referral_threshold)
        else:
            main_title = "Model {} - Test image: {} - slice: {} \n".format(model_name, image_name, img_slice + 1)

        fig = plt.figure(figsize=(width, height))
        ax = fig.gca()
        fig.suptitle(main_title, **config.title_font_medium)
        row = 0
        # get the slice and then split ED and ES slices
        image_slice = image[:, :, :, img_slice]

        if info_type is not None:
            if filtered_stddev_bin_edges is not None:
                # stddev_bin_edges has shape [phases, #slices, #bins+1] and stddev_bin_values [phases, #slices, 2, bins]
                # when we use the filtered u-maps we omit all the zero values in the first bin!
                #
                stddev_bin_edges = filtered_stddev_bin_edges
                stddev_bin_values = filtered_stddev_bin_values
            else:
                stddev_bin_edges, stddev_bin_values = generate_std_hist_corr_err(uncertainty_map, label, pred_labels)
            # using raw/unfiltered u-maps
            cls_stddev_bin_edges, cls_stddev_bin_values = \
                generate_std_hist_corr_err_per_class(uncertainty_map, label, pred_labels, referral_threshold=None)

        for phase in np.arange(2):
            cls_offset = phase * half_classes
            img = image_slice[phase]  # INDEX 0 = end-systole image
            rgb_img_saved = to_rgb1a(img)
            slice_pred_labels = copy.deepcopy(pred_labels[:, :, :, img_slice])
            slice_true_labels = label[:, :, :, img_slice]
            slice_stddev = uncertainty_map[:, :, :, img_slice]
            if referral_threshold != 0.:
                filtered_slice_stddev = filtered_cls_std_map[:, :, :, img_slice]

            ax1 = plt.subplot2grid((rows, columns), (row, 0), rowspan=2, colspan=2)
            if phase == 0:
                ax1.set_title("Slice {}/{}: End-systole".format(img_slice + 1, num_of_slices_img),
                              **config.title_font_medium)
                str_phase = "ES"
            else:
                ax1.set_title("Slice {}/{}: End-diastole".format(img_slice + 1, num_of_slices_img),
                              **config.title_font_medium)
                str_phase = "ED"
            # -------------------------------- the original image we are segmenting -----------------------
            img_with_contours = detect_seg_contours(img, slice_true_labels, cls_offset)
            ax1.imshow(img_with_contours, cmap=cm.gray)
            ax1.set_aspect('auto')
            plt.axis('off')
            # --------------------  Plot segmentation errors for BASELINE (Jelmer) MODEL ------------------------
            if load_base_model_pred_labels:
                slice_pred_labels_wo_sampling = pred_labels_base_model[:, :, :, img_slice]
                rgb_img = copy.deepcopy(rgb_img_saved)
                ax_pred = plt.subplot2grid((rows, columns), (row, 2), rowspan=2, colspan=2)
                rgb_img_w_pred, cls_errors_base = set_error_pixels(rgb_img, slice_pred_labels_wo_sampling,
                                                                   slice_true_labels, cls_offset)
                ax_pred.text(20, 20, 'yellow: RV ({}), blue: Myo ({}), '
                                     'red: LV ({}), green: Myo/LV ({})'.format(cls_errors_base[1], cls_errors_base[2],
                                                                               cls_errors_base[3], cls_errors_base[0]),
                             bbox={'facecolor': 'white', 'pad': 18})
                ax_pred.imshow(rgb_img_w_pred, interpolation='nearest')
                ax_pred.set_aspect('auto')
                ax_pred.set_title("Prediction errors BASELINE model", **config.title_font_medium)
                plt.axis('off')
            # -------------------------- Plot segmentation per class on original image with sampling -----------

            row += 2
            ax1b = plt.subplot2grid((rows, columns), (row, 0), rowspan=2, colspan=2)
            rgb_img = copy.deepcopy(rgb_img_saved)
            rgb_img_with_seg_masks = overlay_seg_mask(rgb_img, slice_pred_labels, cls_offset)
            ax1b.imshow(rgb_img_with_seg_masks, interpolation='nearest')
            ax1b.set_aspect('auto')
            ax1b.set_title("Predicted labels WITH sampling", **config.title_font_medium)
            plt.axis('off')
            # -------------------------- Plot segmentation ERRORS per class on original image -----------
            # we also construct an image with the segmentation errors, placing it next to the original img
            ax1b = plt.subplot2grid((rows, columns), (row, 2), rowspan=2, colspan=2)
            rgb_img = copy.deepcopy(rgb_img_saved)
            rgb_img_with_sampling_pred, cls_errors = set_error_pixels(rgb_img, slice_pred_labels, slice_true_labels,
                                                                      cls_offset)
            ax1b.imshow(rgb_img_with_sampling_pred, interpolation='nearest')
            ax1b.set_aspect('auto')
            ax1b.text(20, 20, 'yellow: RV ({}), blue: Myo ({}), red: LV ({})'
                              ', green: Myo/LV ({})'.format(cls_errors[1],
                                                                                     cls_errors[2],
                                                                                     cls_errors[3], cls_errors[0]),
                      bbox={'facecolor': 'white', 'pad': 18})
            ax1b.set_title("Prediction errors WITH sampling", **config.title_font_medium)
            plt.axis('off')

            # ------------------- Next two figures are only plotted in case of REFERRAL -----------------------
            # show the remaining seg-errors after referral only positive uncertain pixels and another to the right
            # with remaining seg-errors after referral of all uncertain pixels
            if not errors_only:
                # plot unfiltered or filtered u-map (for all uncertain pixels) to the right of the u-map
                # We plot the filtered u-map WHICH ALL UNCERTAIN PIXELS not just the positive-ones only
                row += 2
                if umaps_without_postprocessing:
                    mean_slice_stddev = filtered_cls_std_map_wpostp[phase, :, :, img_slice]
                else:
                    mean_slice_stddev = filtered_std_map[phase, :, :, img_slice]
                total_uncertainty = np.count_nonzero(mean_slice_stddev)
                sub_title = "Slice {} {}: Filtered U-map (#u={})".format(img_slice + 1, str_phase,
                                                                                    total_uncertainty)
                ax4a = plt.subplot2grid((rows, columns), (row, 0), rowspan=2, colspan=2)
                if referral_threshold != 0:
                    u_map_blobs_after_erosion = ref_u_map_blobs[phase, img_slice]
                    u_map_blobs_after_erosion = u_map_blobs_after_erosion[u_map_blobs_after_erosion != 0]
                    # patch_sizes, remaining_patch = detect_largest_umap_areas_slice(mean_slice_stddev, structure)
                    print(u_map_blobs_after_erosion)
                    if len(u_map_blobs_after_erosion) != 0:
                        blobs_remaining = ", ".join([str(a) for a in u_map_blobs_after_erosion])
                        ax4a.text(20, 20, "u-blobs: {}".format(blobs_remaining), bbox={'facecolor': 'white', 'pad': 18})

                max_mean_stddev = np.max(mean_slice_stddev)
                ax4a.imshow(img, cmap=cm.gray)
                ax4aplot = ax4a.imshow(mean_slice_stddev, cmap=mycmap,
                                       vmin=0., vmax=max_mean_stddev)
                ax4a.set_aspect('auto')
                fig.colorbar(ax4aplot, ax=ax4a, fraction=0.046, pad=0.04)
                ax4a.set_title(sub_title, **config.title_font_small)
                plt.axis('off')

            if referral_pred_labels is not None:
                rgb_img = copy.deepcopy(rgb_img_saved)
                slice_pred_labels_referred = referral_pred_labels[:, :, :, img_slice]
                slice_referred = referred_slices[phase, img_slice]
                ax_pred_ref = plt.subplot2grid((rows, columns), (row, 2), rowspan=2, colspan=2)
                rgb_img_ref_pred, cls_errors_ref = set_error_pixels(rgb_img, slice_pred_labels_referred,
                                                                    slice_true_labels, cls_offset)

                ax_pred_ref.text(20, 20, 'yellow: RV ({}), blue: Myo ({}), '
                                         'red: LV ({}), green Myo/LV ({})'.format(cls_errors_ref[1], cls_errors_ref[2],
                                                                                  cls_errors_ref[3], cls_errors_ref[0]),
                                 bbox={'facecolor': 'white', 'pad': 18})
                ax_pred_ref.imshow(rgb_img_ref_pred, interpolation='nearest')
                ax_pred_ref.set_aspect('auto')
                ax_pred_ref.set_title("Prediction errors after referral (referred={})".format(slice_referred),
                                      **config.title_font_medium)
                plt.axis('off')
            # ARE WE SHOWING THE FILTERED (Positive-only) HEAT MAP?
            if filtered_std_map_pos_only is not None:
                row += 2
                # add all uncertainties from the separate STDDEV maps per class
                filtered_std_slice_map = filtered_std_map_pos_only[phase, :, :, img_slice]
                filtered_std_slice_max = np.max(filtered_std_slice_map)
                ax4 = plt.subplot2grid((rows, columns), (row, 0), rowspan=2, colspan=2)
                ax4.imshow(img, cmap=cm.gray)
                ax4plot = ax4.imshow(filtered_std_slice_map, cmap=mycmap, vmin=0.,
                                     vmax=filtered_std_slice_max)
                ax4.set_aspect('auto')
                fig.colorbar(ax4plot, ax=ax4, fraction=0.046, pad=0.04)
                total_uncertainty = np.count_nonzero(filtered_std_slice_map)
                ax4.set_title("Slice {} {}: Filtered-pos-only STDDEV u-map (#u={})".format(img_slice + 1, str_phase,
                                                                                           total_uncertainty),
                              **config.title_font_small)
                plt.axis('off')

            if info_type is not None:
                # Last two rows: histogram of uncertainties (pos-only or raw ones). Covers 2 rows/columns
                # create histogram
                if referral_threshold != 0:
                    title_suffix = " (filtered)"
                else:
                    title_suffix = ""
                stddev_corr = stddev_bin_values[phase, img_slice, 0]
                stddev_err = stddev_bin_values[phase, img_slice, 1]
                xs = stddev_bin_edges[phase, img_slice]
                bar_width = xs[1] - xs[0]
                ax5 = plt.subplot2grid((rows, columns), (row + 3, 0), rowspan=2, colspan=2)
                # print("Phase {} row {} - BALD histogram".format(phase, row + 2))
                # if stddev_err is not None:
                if np.sum(stddev_err) > 0:
                    ax5.yaxis.set_major_locator(MaxNLocator(integer=True))
                    ax5.bar(xs[:-1], stddev_err, bar_width,
                            label=r"$\sigma_{{pred(fp+fn)}}({})$".format(np.sum(stddev_err).astype(np.int)),
                            color='b', alpha=0.2, align="center")
                    ax5.legend(loc=1, prop={'size': 12})
                    ax5.tick_params(axis='y', colors='b')
                    ax5.grid(False)
                # if stddev_corr is not None:
                if np.sum(stddev_corr) > 0:
                    ax5b = ax5.twinx()
                    ax5b.yaxis.set_major_locator(MaxNLocator(integer=True))
                    ax5b.bar(xs[:-1], stddev_corr, bar_width,
                            label=r"$\sigma_{{pred(tp)}}({})$".format(np.sum(stddev_corr).astype(np.int)),
                            color='g', alpha=0.2, align="center")
                    ax5b.legend(loc=2, prop={'size': 12})
                    ax5b.tick_params(axis='y', colors='g')
                    ax5b.grid(False)
                ax5.set_xlabel("predictive uncertainty", **config.axis_font)
                sub_title = "Slice {} {}: Distribution of uncertainties" + title_suffix
                ax5.set_title(sub_title.format(img_slice + 1, str_phase)
                              , **config.title_font_small)

            # The row underneath the TWO MRI+U-map figures, containing 4 MRI-u-map figures, one per CLASS
            # In case we're only showing the error segmentation map we skip the next part (histograms and
            # stddev uncertainty maps per class. If we only skip the histograms, we visualize the uncertainty
            # maps for each class (at least for the stddev maps).
            if not errors_only:
                row += 2
                row_offset = 1
                col_offset = 0
                counter = 0
                # Remember object slice_stddev contains the original (raw) unfiltered uncertainty values
                if phase == 0:
                    max_stdddev_over_classes = np.max(slice_stddev[:half_classes])
                else:
                    max_stdddev_over_classes = np.max(slice_stddev[half_classes:])

                for cls in np.arange(half_classes):
                    # in the next subplot row we visualize the uncertainties per class
                    # print("phase {} row {} counter {}".format(phase, row, counter))
                    ax3 = plt.subplot2grid((rows, columns), (row, counter), colspan=1)
                    if referral_threshold != 0.:
                        std_map_cls = filtered_slice_stddev[cls + cls_offset]
                        sub_title_prefix = "filtered"
                    else:
                        std_map_cls = slice_stddev[cls + cls_offset]
                        sub_title_prefix = ""
                    ax3.imshow(img, cmap=cm.gray)
                    ax3plot = ax3.imshow(std_map_cls, vmin=0.,
                                         vmax=max_stdddev_over_classes, cmap=mycmap)
                    ax3.set_aspect('auto')
                    if cls == half_classes - 1:
                        fig.colorbar(ax3plot, ax=ax3, fraction=0.046, pad=0.04)
                    sub_title = "{} " + sub_title_prefix + " stddev: {} "
                    ax3.set_title(sub_title.format(str_phase, column_lbls[cls]),
                                  **config.title_font_small)
                    plt.axis("off")
                    # finally in the next row we plot the uncertainty densities per class
                    if info_type is not None:
                        if cls != 0:
                            # we're using the histogram values generated based on the UNFILTERED u-maps
                            #
                            p_corr_std = cls_stddev_bin_values[phase, cls, img_slice, 0]
                            p_err_std = cls_stddev_bin_values[phase, cls, img_slice, 1]
                            xs = cls_stddev_bin_edges[phase, cls, img_slice]
                            bar_width = xs[1] - xs[0]
                            ax2 = plt.subplot2grid((rows, columns), (row + row_offset, 2 + col_offset), colspan=1)
                            if np.sum(p_err_std) != 0:
                                ax2b = ax2.twinx()
                                ax2b.yaxis.set_major_locator(MaxNLocator(integer=True))
                                # xs[:-1] because the array contains the bin-edges, which is always one more than bins
                                ax2b.bar(xs[:-1], p_err_std, bar_width,
                                         label=r"$\sigma_{{pred(fp+fn)}}({})$".format(np.sum(p_err_std).astype(np.int))
                                         , color="b", alpha=0.2)
                                ax2b.legend(loc=2, prop={'size': 9})
                                ax2b.tick_params(axis='y', colors='b')

                            if np.sum(p_corr_std) != 0:
                                ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
                                ax2.bar(xs[:-1], p_corr_std, bar_width,
                                        label=r"$\sigma_{{pred(tp)}}({})$".format(np.sum(p_corr_std).astype(np.int)),
                                        color="g", alpha=0.4)
                                ax2.tick_params(axis='y', colors='g')
                                ax2.legend(loc=1, prop={'size': 9})

                            if info_type == "uncertainty":
                                ax2.set_xlabel("predictive uncertainty", **config.axis_font)
                            else:
                                ax2.set_xlabel(r"softmax $p(c|x)$", **config.axis_font)
                            ax2.set_title("{} slice-{}: {}".format(str_phase, img_slice + 1, column_lbls[cls]),
                                          **config.title_font_small)
                        col_offset += 1
                        if cls == 1:
                            row_offset += 1
                            col_offset = 0
                    counter += 1
            if errors_only:
                row += 2
            elif info_type is None:
                row += 1
            else:
                row += 3
        # fig.tight_layout()
        if not errors_only:
            fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        if do_save:
            if not errors_only:
                fig_suffix = "_mc"
            else:
                fig_suffix = "_err_only"
            fig_name = "analysis_seg_err_slice{}".format(img_slice + 1) \
                       + fig_suffix
            if referral_threshold > 0.:
                tr_string = "_tr" + str(referral_threshold).replace(".", "_")
                fig_name += tr_string

            if errors_only:
                fig_name = fig_name + "_w_uncrty"
            fig_name = os.path.join(fig_path, fig_name + ".pdf")
            if not os.path.isdir(fig_path):
                os.makedirs(fig_path)
            plt.savefig(fig_name, bbox_inches='tight')
            print("INFO - Successfully saved fig %s" % fig_name)
        if do_show:
            plt.show()
        plt.close()
