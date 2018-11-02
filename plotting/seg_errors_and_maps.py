from config.config import config
import matplotlib.pyplot as plt
from pylab import MaxNLocator
import numpy as np
import os
from matplotlib import cm
from common.hvsmr.helper import detect_seg_errors
from common.common import convert_to_multiclass


def plot_slices(exper_handler, patient_id, do_show=True, do_save=False, threshold=None, left_column_overlay="map",
                slice_range=None, type_of_map="e_map", aggregate_func=None, right_column_overlay="error"):

    """

    :param exper_handler:
    :param patient_id:
    :param do_show:
    :param do_save:
    :param threshold: related to bayesian u-map. we always use 0.001
    :param slice_range:  e.g. [0, 5]
    :param type_of_map: e_map or u_map
    :param aggregate_func:
    :param left_column_overlay: "map" = uncertainty maps or "error_roi" = seg error regions that needs to be detected
    :param right_column_overlay: ["ref", "error", "auto"] type of segmentation mask that we plot in the right figure
    :return:
    """

    def transparent_cmap(cmap, N=255):
        """ Copy colormap and set alpha values """
        mycmap = cmap
        mycmap._init()
        mycmap._lut[:, -1] = np.linspace(0, 0.8, N + 4)
        return mycmap

    if right_column_overlay not in ["ref", "error", "auto"]:
        raise ValueError("ERROR - seg_mask_type must be ref, error or auto! (and not {})".format(seg_mask_type))

    if type_of_map not in ["e_map", "u_map"]:
        raise ValueError("ERROR - type_of_map must be e_map or u_map! (and not {})".format(type_of_map))
    # Use base cmap to create transparent
    mycmap = transparent_cmap(plt.get_cmap('jet'))
    if type_of_map == "u_map":
        mc_dropout = True
        if exper_handler.test_set.__class__.__name__ != "HVSMRTesthandler":
            umap = exper_handler.get_referral_maps(0.001, per_class=False, aggregate_func=aggregate_func, use_raw_maps=True,
                                                   patient_id=patient_id, load_ref_map_blobs=False)
        else:
            umap = exper_handler.get_bayes_umaps(patient_id=patient_id, aggregate_func=aggregate_func)

    else:
        mc_dropout = False
        umap = exper_handler.get_entropy_maps(patient_id=patient_id)

    exper_handler.get_pred_labels(patient_id=patient_id, mc_dropout=mc_dropout)
    pred_labels = exper_handler.pred_labels[patient_id]
    exper_args = exper_handler.exper.run_args

    if exper_handler.test_set is None:
        exper_handler.get_test_set()
    mri_image, labels = exper_handler.test_set.get_test_pair(patient_id)
    if exper_handler.test_set.__class__.__name__ != "HVSMRTesthandler":
        is_acdc = True
        num_of_phases = 2
        phase_labels = ["ES", "ED"]
        num_of_classes = 4
        mri_image = mri_image[:, config.pad_size:-config.pad_size, config.pad_size:-config.pad_size, :]
        errors_to_detect = exper_handler.get_target_roi_maps(patient_id=patient_id, mc_dropout=mc_dropout)

    else:
        is_acdc = False
        num_of_classes = 3
        num_of_phases = 1
        errors_to_detect = None
    if slice_range is None:
        num_of_slices = umap.shape[-1]
        str_slice_range = "_s1_" + str(num_of_slices)
        slice_range = np.arange(num_of_slices)
    else:
        str_slice_range = "_s" + "_".join([str(i) for i in slice_range])
        slice_range = np.arange(slice_range[0], slice_range[1])
        num_of_slices = len(slice_range)

    columns = 4
    width = 16
    height = 14 * num_of_slices   # num_of_classes * 2 * num_of_slices
    row = 0

    model_info = "{} p={:.2f} fold={} loss={}".format(exper_args.model, exper_args.drop_prob,
                                                      exper_args.fold_ids[0],
                                                      exper_args.loss_function)
    if threshold is not None:
        str_threshold = str(threshold).replace(".", "_")
        model_info += " (thr={:.2f})".format(threshold)
    fig = plt.figure(figsize=(width, height))
    fig.suptitle(model_info + ": " + patient_id, **config.title_font_medium)
    for slice_id in slice_range:
        for phase in np.arange(num_of_phases):

            cls_offset = phase * num_of_classes
            # num_of_phases > 1 means we're dealing with ACDC dataset, otherwise HVSMR
            if num_of_phases > 1:
                rows = 4 * num_of_slices
                umap_slice = umap[phase, :, :, slice_id]
                img_slice = mri_image[phase, :, :, slice_id]
                # IMPORTANT: (already verified) Assuming labels AND pred_labels has shape [8, w, h, #slices]
                labels_slice = labels[cls_offset:cls_offset+num_of_classes, :, :, slice_id]
                pred_labels_slice = pred_labels[cls_offset:cls_offset+num_of_classes, :, :, slice_id]
                errors_slice = detect_seg_errors(labels_slice, pred_labels_slice, is_multi_class=False)
                errors_slice_to_detect = errors_to_detect[cls_offset:cls_offset+num_of_classes, :, :, slice_id]
            else:
                rows = 2 * num_of_slices
                umap_slice = umap[:, :, slice_id]
                img_slice = mri_image[:, :, slice_id]
                pred_labels_slice = pred_labels[:, :, :, slice_id]
                labels_slice = labels[:, :, slice_id]
                errors_slice = detect_seg_errors(labels_slice, pred_labels_slice, is_multi_class=True)
                errors_slice_to_detect = None
            if threshold is not None:
                # set everything below threshold to zero
                umap_slice[umap_slice < threshold] = 0

            # print("Min/max values {:.2f}/{:.2f}".format(np.min(entropy_slice_map),
            #                                            np.max(entropy_slice_map)))
            ax1 = plt.subplot2grid((rows, columns), (row, 0), rowspan=2, colspan=2)
            ax1.imshow(img_slice, cmap=cm.gray)
            if left_column_overlay == "map":
                _ = ax1.imshow(umap_slice, cmap=mycmap, vmin=0., vmax=0.4)
                left_title_suffix = " (uncertainties)"
            elif left_column_overlay == "error_roi":
                errors_slice_to_detect = convert_to_multiclass(errors_slice_to_detect)
                _ = ax1.imshow(errors_slice_to_detect, cmap=mycmap)
                left_title_suffix = " (error rois)"

            # ax1.set_aspect('auto')
            # fig.colorbar(ax1plot, ax=ax1, fraction=0.046, pad=0.04)
            plt.axis("off")
            if is_acdc:
                p_title = "{} {} slice {}: ".format(type_of_map, phase_labels[phase], slice_id + 1)
            else:
                p_title = "{} slice {}: ".format(type_of_map, slice_id + 1)
            ax1.set_title(p_title + left_title_suffix, **config.title_font_small)
            ax2 = plt.subplot2grid((rows, columns), (row, 2), rowspan=2, colspan=2)
            ax2.imshow(img_slice, cmap=cm.gray)
            # What do we plot in the right column?
            if right_column_overlay == "ref":
                multi_label_slice = convert_to_multiclass(labels_slice)
                ax2.imshow(multi_label_slice, cmap=mycmap)
                ax2.set_title("Reference (r=LV/y=myo/b=RV)", **config.title_font_small)
            elif right_column_overlay == "error":
                ax2.imshow(errors_slice, cmap=mycmap)
                ax2.set_title("Segmentation errors (r=LV/y=myo/b=RV)", **config.title_font_small)
            else:
                # automatic seg-mask
                multi_pred_labels = convert_to_multiclass(pred_labels_slice)
                ax2.imshow(multi_pred_labels, cmap=mycmap)
                ax2.set_title("Automatic mask (r=LV/y=myo/b=RV)", **config.title_font_small)
            plt.axis("off")
            row += 2

    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    if do_save:
        fig_path = os.path.join(exper_handler.exper.config.root_dir,
                                      os.path.join(exper_handler.exper.output_dir, config.figure_path))
        fig_path = os.path.join(fig_path, patient_id)
        if not os.path.isdir(fig_path):
            os.makedirs(fig_path)
        fig_name = patient_id + "_" + type_of_map
        if type_of_map == "u_map":
            fig_name += "_" + aggregate_func
        fig_name += str_slice_range
        fig_name = os.path.join(fig_path, fig_name + ".pdf")

        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)
    if do_show:
        plt.show()
