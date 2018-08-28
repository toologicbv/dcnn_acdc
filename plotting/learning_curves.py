from config.config import config
import matplotlib.pyplot as plt
import matplotlib
from pylab import MaxNLocator
import numpy as np
import os


def loss_plot(exper, fig_name=None, height=8, width=6, save=False, show=True, validation=False,
              log_scale=False, epoch_range=None, do_average=False, window_size=10):

    """
    Simple plots of training and validation loss (if enabled).
    Figure can be saved in log directory of experiment (log directory + figures)

    Args:
        epoch_range: list containing two indices in order to plot a range of iterations
        do_average: smooth learning curve with simple convolution
        fig_name: default is learning_curve.png
        log_scale: if you want y-axis in log scale
        exper: object that contains the experimental details

    Shape:

    Examples:
        loss_plot(exper, width=18, validation=True, save=False, do_average=True, epoch_range=[40001, 50000])
    """

    ax = plt.figure(figsize=(width, height)).gca()

    train_loss = exper.get_loss()
    if do_average:
        w = np.ones(window_size, 'd')
        train_loss = np.convolve(w / w.sum(), train_loss, mode='same')
        print(len(train_loss))
    last_epoch = exper.epoch_id
    if epoch_range is not None:
        offset = int(epoch_range[0])
        last_epoch = int(epoch_range[1])
    else:
        offset = 10
    epochs = np.arange(offset, last_epoch)
    plt.ylabel("Loss")
    plt.xlabel("Iterations")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if log_scale:
        if np.any(train_loss <= 0):
            # applying modulus transfer if values are negative
            train_loss = np.sign(train_loss) * (np.log10(abs(train_loss) + 1))
            plt.plot(epochs.astype(int), train_loss[offset:last_epoch], 'r', label="train")
        else:
            plt.semilogy(epochs.astype(int), train_loss[offset:last_epoch], 'r', label="train")
    else:
        plt.plot(epochs.astype(int), train_loss[offset:last_epoch], 'r', label="train")

    if len(epochs) <= 10:
        plt.xticks(epochs.astype(int))

    if validation:
        validation_loss = exper.get_loss(validation=True)
        epochs = exper.validation_epoch_ids
        if offset != 0:
            array_idx = (epochs >= offset) & (epochs <= last_epoch)
        else:
            array_idx = (epochs >= offset)
        epochs = epochs[array_idx]
        if log_scale:
            if np.any(validation_loss <= 0):
                # applying modulus transfer if values are negative
                validation_loss = np.sign(validation_loss) * (np.log10(abs(validation_loss) + 1))
                plt.plot(epochs.astype(int), validation_loss[array_idx], 'b', label="validation")
            else:
                plt.semilogy(epochs.astype(int), validation_loss[array_idx], 'b', label="validation")
        else:
            # print(len(epochs), len(validation_loss[array_idx]))
            plt.plot(epochs.astype(int), validation_loss[array_idx], 'b', label="validation")

    plt.legend(loc="best")
    p_title = "Learning curve "

    plt.title(p_title, **config.title_font_small)

    if save:
        if fig_name is None:
            fig_name = os.path.join(exper.root_directory, os.path.join(exper.log_directory,
                                                                        "figures/learning_curve" + config.figure_ext))
        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)
    if show:
        plt.show()
    plt.close()



