from config.config import config
import matplotlib.pyplot as plt
from pylab import MaxNLocator
import numpy as np
import os
from matplotlib import cm


def plot_entropy_map_for_patient(exper_handler, patient_id, do_show=True, do_save=False):

    def transparent_cmap(cmap, N=255):
        "Copy colormap and set alpha values"
        mycmap = cmap
        mycmap._init()
        mycmap._lut[:, -1] = np.linspace(0, 0.8, N + 4)
        return mycmap

    # Use base cmap to create transparent
    mycmap = transparent_cmap(plt.get_cmap('jet'))
    if exper_handler.entropy_maps is None:
        exper_handler.get_entropy_maps()
    exper_args = exper_handler.exper.run_args

    exper_handler.get_test_set()
    mri_image, labels = exper_handler.test_set.get_test_pair(patient_id)
    mri_image = mri_image[:, config.pad_size:-config.pad_size, config.pad_size:-config.pad_size, :]
    entropy_map = exper_handler.entropy_maps[patient_id]
    num_of_slices = entropy_map.shape[3]
    num_of_phases = 2

    phase_labels = ["ES", "ED"]
    slice_range = np.arange(num_of_slices)

    rows = 2 * num_of_slices
    columns = 4
    width = 16
    height = 14 * num_of_slices / 2  # num_of_classes * 2 * num_of_slices
    row = 0

    model_info = "{} p={:.2f} fold={} loss={}".format(exper_args.model, exper_args.drop_prob,
                                                      exper_args.fold_ids[0],
                                                      exper_args.loss_function)
    fig = plt.figure(figsize=(width, height))
    fig.suptitle(model_info + ": " + patient_id, **config.title_font_medium)
    for slice_id in slice_range:
        for phase in np.arange(num_of_phases):
            entropy_slice_map = entropy_map[phase, :, :, slice_id]
            img_slice = mri_image[phase, :, :, slice_id]
            # print("Min/max values {:.2f}/{:.2f}".format(np.min(entropy_slice_map),
            #                                            np.max(entropy_slice_map)))
            ax1 = plt.subplot2grid((rows, columns), (row, phase * 2), rowspan=2, colspan=2)
            ax1.imshow(img_slice, cmap=cm.gray)
            ax1plot = ax1.imshow(entropy_slice_map, cmap=mycmap,
                                   vmin=0., vmax=0.4)
            ax1.set_aspect('auto')
            fig.colorbar(ax1plot, ax=ax1, fraction=0.046, pad=0.04)

            plt.axis("off")
            ax1.set_title("{} slice {}: ".format(phase_labels[phase], slice_id + 1), **config.title_font_small)

        row += 2

    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    if do_save:
        fig_path = os.path.join(exper_handler.exper.config.root_dir,
                                      os.path.join(exper_handler.exper.output_dir, config.figure_path))
        fig_path = os.path.join(fig_path, patient_id)
        if not os.path.isdir(fig_path):
            os.makedirs(fig_path)
        fig_name = patient_id + "_entropy_map"
        fig_name = os.path.join(fig_path, fig_name + ".pdf")

        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)
    if do_show:
        plt.show()
