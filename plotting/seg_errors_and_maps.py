from config.config import config
import matplotlib.pyplot as plt
from pylab import MaxNLocator
import numpy as np
import os
from matplotlib import cm
from common.hvsmr.helper import detect_seg_errors


def plot_slices(exper_handler, patient_id, do_show=True, do_save=False, threshold=None,
                slice_range=None, type_of_map="emap", aggregate_func=None):

    def transparent_cmap(cmap, N=255):
        "Copy colormap and set alpha values"
        mycmap = cmap
        mycmap._init()
        mycmap._lut[:, -1] = np.linspace(0, 0.8, N + 4)
        return mycmap

    if type_of_map not in ["emap", "umap"]:
        raise ValueError("ERROR - type_of_map must be emap or umap! (and not {})".format(type_of_map))
    # Use base cmap to create transparent
    mycmap = transparent_cmap(plt.get_cmap('jet'))
    if type_of_map == "umap":
        umap = exper_handler.get_bayes_umaps(patient_id=patient_id, aggregate_func=aggregate_func)
    else:
        umap = exper_handler.get_entropy_maps(patient_id=patient_id)

    exper_handler.get_pred_labels(patient_id=patient_id, mc_dropout=False)
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
    else:
        is_acdc = False
        num_of_classes = 3
        num_of_phases = 1

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
    height = 14 * num_of_slices / 2  # num_of_classes * 2 * num_of_slices
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
            if num_of_phases > 1:
                rows = 4 * num_of_slices
                umap_slice = umap[phase, :, :, slice_id]
                img_slice = mri_image[phase, :, :, slice_id]
                # TODO never tested this part for ACDC. Assuming labels has shape [8, w, h, #slices]
                labels_slice = labels[cls_offset:cls_offset+num_of_classes, :, :, slice_id]
                pred_labels_slice = pred_labels[phase, :, :, slice_id]
                errors_slice = detect_seg_errors(labels_slice, pred_labels_slice, is_multi_class=False)
            else:

                rows = 2 * num_of_slices
                umap_slice = umap[:, :, slice_id]
                img_slice = mri_image[:, :, slice_id]
                pred_labels_slice = pred_labels[:, :, :, slice_id]
                labels_slice = labels[:, :, slice_id]
                errors_slice = detect_seg_errors(labels_slice, pred_labels_slice, is_multi_class=True)
            if threshold is not None:
                # set everything below threshold to zero
                umap_slice[umap_slice < threshold] = 0

            # print("Min/max values {:.2f}/{:.2f}".format(np.min(entropy_slice_map),
            #                                            np.max(entropy_slice_map)))
            ax1 = plt.subplot2grid((rows, columns), (row, 0), rowspan=2, colspan=2)
            ax1.imshow(img_slice, cmap=cm.gray)
            ax1plot = ax1.imshow(umap_slice, cmap=mycmap,
                                   vmin=0., vmax=0.4)
            # ax1.set_aspect('auto')
            # fig.colorbar(ax1plot, ax=ax1, fraction=0.046, pad=0.04)
            plt.axis("off")
            if is_acdc:
                p_title = "{} {} slice {}: ".format(type_of_map, phase_labels[phase], slice_id + 1)
            else:
                p_title = "{} slice {}: ".format(type_of_map, slice_id + 1)
            ax1.set_title(p_title, **config.title_font_small)
            ax2 = plt.subplot2grid((rows, columns), (row, 2), rowspan=2, colspan=2)
            ax2.imshow(img_slice, cmap=cm.gray)
            ax2.imshow(errors_slice, cmap=mycmap)
            ax2.set_title("Segmentation errors (r=LV/g=myo)", **config.title_font_small)
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
        if type_of_map == "umap":
            fig_name += "_" + aggregate_func
        fig_name += str_slice_range
        fig_name = os.path.join(fig_path, fig_name + ".pdf")

        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)
    if do_show:
        plt.show()