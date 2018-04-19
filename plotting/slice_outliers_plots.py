from config.config import config
import matplotlib.pyplot as plt
from pylab import MaxNLocator
import numpy as np
import os


def plot_dice_normal_outliers(exper, width=16, height=8, do_save=False, do_show=False, do_average=True,
                              epoch_range=None, statistic="dice_coeff", include_outliers=False):

    statistic_outlier = statistic + "_outliers"
    if epoch_range is not None:
        first_epoch = int(epoch_range[0])
        last_epoch = int(epoch_range[1])
        # x_values = np.arange(first_epoch, last_epoch)
    else:
        first_epoch = 1
        last_epoch = exper.epoch_id
        # x_values = np.arange(first_epoch, last_epoch)
    class_colors = ['g', 'b', 'r']
    class_labels = ["RV", "MYO", "LV"]

    if include_outliers:
        dice_out_stats = exper.epoch_stats[statistic_outlier][first_epoch:last_epoch]
    dice_stats = exper.epoch_stats[statistic][first_epoch:last_epoch]
    # get epochIDs with normal and outlier batches
    if include_outliers:
        epochs_out = np.argwhere(dice_out_stats[:, 0] != 0).squeeze()
    epochs_norm = np.argwhere(dice_stats[:, 0] != 0).squeeze()
    # separate ES/ED stats for normal and outlier epochs
    es_dice_stats_norm = dice_stats[epochs_norm, 0:3]
    ed_dice_stats_norm = dice_stats[epochs_norm, 3:]
    if include_outliers:
        es_dice_stats_out = dice_out_stats[epochs_out, 0:3]
        ed_dice_stats_out = dice_out_stats[epochs_out, 3:]

    rows = 2
    columns = 2
    fig = plt.figure(figsize=(width, height))
    ax = fig.gca()
    fig.suptitle("Dice coefficient during training iterations", **config.title_font_medium)

    ax5 = plt.subplot2grid((rows, columns), (0, 0), rowspan=1, colspan=2)
    num_of_classes = es_dice_stats_norm.shape[1]
    window_len = 10
    w = np.ones(window_len, 'd')
    # Plot ES performance
    for cls in np.arange(num_of_classes):
        es_norm = es_dice_stats_norm[:, cls]
        if include_outliers:
            es_outlier = es_dice_stats_out[:, cls]
        # Normal batches
        if do_average:
            es_norm = np.convolve(w / w.sum(), es_norm, mode='same')
            if include_outliers:
                es_outlier = np.convolve(w / w.sum(), es_outlier, mode='same')
        ax5.plot(epochs_norm + first_epoch, es_norm, label=r"{}".format(class_labels[cls]),
                 color=class_colors[cls], alpha=0.2)
        if include_outliers:
            ax5.plot(epochs_out + first_epoch, es_outlier, label=r"{} outlier".format(class_labels[cls]),
                     color=class_colors[cls], alpha=0.6, linestyle=':')

    ax5.legend(loc="best", prop={'size': 12})
    ax5.set_xlabel("EpochID", **config.axis_font)
    # ax5.set_xticks(x_values)
    ax5.set_ylabel("Dice coefficient")
    ax5.set_title("Dice coefficient ES classes", **config.title_font_small)

    ax6 = plt.subplot2grid((rows, columns), (1, 0), rowspan=1, colspan=2)
    for cls in np.arange(num_of_classes):
        ed_norm = ed_dice_stats_norm[:, cls]
        if include_outliers:
            ed_outlier = ed_dice_stats_out[:, cls]
        # Normal batches
        if do_average:
            ed_norm = np.convolve(w / w.sum(), ed_norm, mode='same')
            if include_outliers:
                ed_outlier = np.convolve(w / w.sum(), ed_outlier, mode='same')
        # Normal batches
        ax6.plot(epochs_norm + first_epoch, ed_norm, label=r"{}".format(class_labels[cls]), color=class_colors[cls],
                 alpha=0.2)
        if include_outliers:
            ax6.plot(epochs_out + first_epoch, ed_outlier, label=r"{} outlier".format(class_labels[cls]), color=class_colors[cls],
                     alpha=0.6, linestyle=':')
    ax6.legend(loc="best", prop={'size': 12})
    ax6.set_xlabel("EpochID", **config.axis_font)
    # ax6.set_xticks(x_values)
    ax6.set_ylabel("Dice coefficient")
    ax6.set_title("Dice coefficient ED classes", **config.title_font_small)

    if do_save:
        # we add 1 to first epoch, because we're dealing with numpy index-slicing, so last_epoch is not included
        # and first needs to be 0 (or 5000 instead of 5001)
        str_epoch_range = str(first_epoch + 1) + "_" + str(last_epoch)
        filename = "dice_train_ep" + str_epoch_range + config.figure_ext
        fig_name = os.path.join(exper.root_directory,
                                os.path.join(exper.log_directory, os.path.join(exper.config.figure_path,
                                                                               filename)))
        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)

    if do_show:
        plt.show()


def plot_outlier_slice_hists(exper, width=16, height=5.5, do_save=False, do_show=False, epochs=None):

    def slice_frequenties(dict_of_slices, slice_freq_dataset):
        slices = []
        for key, slice_list in dict_of_slices.iteritems():
            slices.extend(slice_list)
        slices.sort()
        slices = np.array(slices)
        unique_slices = np.unique(slices)
        num_of_bins = np.max(unique_slices)
        slice_freq, _ = np.histogram(slices, bins=num_of_bins+1)
        # remove zeros from slice_freq, twist, because the unique_slices are not contiguous, and we set num_of_bins
        # to max slices, we'll probably end up with zero-holes in the frequenties, which need to be removed
        slice_freq = slice_freq[np.nonzero(slice_freq)]
        # we need to rescale the outlier slice freqs with the freq their occur in the dataset
        slice_freq = slice_freq.astype(np.float32)
        for idx, s in enumerate(unique_slices):
            slice_freq[idx] = float(slice_freq[idx]) * float(slice_freq_dataset[s])

        return slice_freq, unique_slices

    model_name = exper.model_name
    if epochs is None:
        num_of_plots = len(exper.outliers_per_epoch)
    else:
        num_of_plots = len(epochs)

    columns = 2
    if num_of_plots % columns == 0:
        rows = num_of_plots / columns
    else:
        rows = 1 + (num_of_plots / columns)
    height = rows * height

    # exper.outliers_per_epoch is a dictionary, key is epochID. Value is a tuple [0]=dictionary (key patientxxx)
    # and values are lists which contain the images slices (first slice is 0) that were labeled as outliers
    counter, row, col = 0, 0, 0
    fig = plt.figure(figsize=(width, height))
    fig.suptitle("Histogram of outlier sliceIDs", **config.title_font_medium)
    ax5 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
    bar_width = 0.9/num_of_plots

    for i, epoch in enumerate(epochs):
        slice_freq, unique_slices = slice_frequenties(exper.outliers_per_epoch[epoch][0],
                                                      exper.batch_stats.slice_frequencies_dataset)

        ax5.bar(unique_slices + 1 + (i * bar_width), slice_freq, bar_width, label=r"Densities{}".format(epoch),
                align="center", alpha=0.5)
    ax5.legend(loc="best", prop={'size': 12})
    ax5.set_xlabel("slices", **config.axis_font)
    ax5.set_xticks(unique_slices.astype(np.int) + 1 + bar_width)
    ax5.axes.set_xticklabels(unique_slices.astype(np.int) + 1)
    ax5.set_ylabel("Density", **config.axis_font)
    ax5.set_title("Histogram outlier slices ({})".format(model_name), **config.title_font_small)
    counter += 1

    if do_save:
        pass

    if do_show:
        plt.show()
