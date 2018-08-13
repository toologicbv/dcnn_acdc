from config.config import config
import matplotlib.pyplot as plt
import pylab
import numpy as np
from scipy.stats import binned_statistic
import os
import glob

from utils.experiment import ExperimentHandler


def collect_exper_handlers(exper_dict):
    exper_handlers = []
    log_base = os.path.join(config.root_dir, config.log_root_path)
    for exper_id in exper_dict.values():
        exp_model_path = os.path.join(log_base, exper_id)
        exper_handler = ExperimentHandler()
        exper_handler.load_experiment(exp_model_path, use_logfile=False)
        exper_handler.set_root_dir(config.root_dir)
        exper_handlers.append(exper_handler)

    return exper_handlers


def compute_expected_calibration_error(pred_probs_cls, acc_per_bin, counts_pred_probs, num_of_bins=10):
    """

    :param pred_probs_cls: numpy flat array containing all softmax probabilities for this class/patient
    :param acc_per_bin: numpy array size [num_of_bins] specifying the accuracy for each probability bin
    :param counts_pred_probs: np array size [num_of_bin] specifying the counts in each prob bin
    :param num_of_bins:

    :return: ECE (expected calibration error), scalar
    """
    _ = np.seterr(divide='ignore')
    mean_prob_in_bin, _, _ = binned_statistic(pred_probs_cls, pred_probs_cls, statistic='mean',
                                              bins=num_of_bins, range=(0, 1.))
    # weighted average: numerator=number of probs in each of the bins,
    bin_scaler = counts_pred_probs * 1./np.sum(counts_pred_probs)
    # print("acc & conf")
    # print(acc_per_bin, mean_prob_in_bin)
    ece = np.sum(bin_scaler * np.abs(acc_per_bin - mean_prob_in_bin))

    return ece


def compute_calibration_terms(exper_dict, patient_id=None, mc_dropout=False, force_reload=False, num_of_bins=10,
                              do_save=False, with_bg=False):
    """
    Compute the ingredients we need for the ECE (Expected Calibration Error) and the Reliability Diagrams
    Taken from paper https://arxiv.org/abs/1706.04599

    :param exper_dict:
    :param patient_id:
    :param mc_dropout:
    :param force_reload: force reload of probability maps that model generated (property of exp-handler)
    :param num_of_bins:
    :param with_bg: include background class. By default we don't include the class (because we also don't report
                    results on segmentation performance for this class).
    :param do_save:
    :return:
    """
    if with_bg:
        class_range = [0, 1, 2, 3]
        print("INFO - WITH background class")
    else:
        class_range = [1, 2, 3]
    exper_handlers = collect_exper_handlers(exper_dict)
    # prepare: get probability maps and test_set which contains
    # prob_bins: contains bin-edges for the probabilities (default 10)
    prob_bins = np.linspace(0, 1, num_of_bins + 1)
    # acc_bins: accuracy per bin. numerator="# of positive predicted class labels for this bin";
    #                             denominator="# of predictions in this bin"
    acc_bins = np.zeros((2, 4, num_of_bins))
    acc_bins_used = np.zeros((2, 4, num_of_bins))
    mean_counts_per_bin = np.zeros((2, 4, num_of_bins))
    # confidence measure = mean predicted probability in this bin
    ece_per_class = np.zeros((2, 4))
    ece_counts_per_class = np.zeros((2, 4))
    num_of_classes = 4

    for exper_handler in exper_handlers:
        exper_args = exper_handler.exper.run_args
        info_str = "{} p={:.2f} fold={} loss={}".format(exper_args.model, exper_args.drop_prob, exper_args.fold_ids,
                                                        exper_args.loss_function)
        print("INFO - Experimental details extracted:: " + info_str)
        if force_reload or exper_handler.pred_prob_maps is None or len(exper_handler.pred_prob_maps) == 0:
            exper_handler.get_pred_prob_maps(patient_id=patient_id, mc_dropout=mc_dropout)
        exper_handler.get_test_set()
        if force_reload or exper_handler.pred_labels is None or len(exper_handler.pred_labels) == 0:
            exper_handler.get_pred_labels(patient_id=patient_id, mc_dropout=mc_dropout)
        if patient_id is not None:
            patients = [patient_id]
        else:
            patients = exper_handler.test_set.trans_dict.keys()
        for p_id in patients:
            image_num = exper_handler.test_set.trans_dict[p_id]
            gt_labels = exper_handler.test_set.labels[image_num]
            # pred_labels = exper_handler.pred_labels[p_id]
            pred_probs = exper_handler.pred_prob_maps[p_id]
            for phase in [0, 1]:
                for cls_idx in np.arange(num_of_classes):
                    cls_offset = phase * num_of_classes
                    if cls_idx != 0 and cls_idx in class_range:
                        gt_labels_cls = gt_labels[cls_offset + cls_idx, :, :, :].flatten()
                        pred_probs_cls = pred_probs[cls_offset + cls_idx, :, :, :].flatten()
                        gt_labels_cls = np.atleast_1d(gt_labels_cls.astype(np.bool))
                        # determine indices of the positive/class voxels
                        pos_voxels_idx = gt_labels_cls == 1
                        # get all predicted probabilities that "predicted" correctly the pos-class label
                        pred_probs_cls_cor = pred_probs_cls[pos_voxels_idx]
                        counts_pred_probs, _ = np.histogram(pred_probs_cls, bins=prob_bins)
                        # bin the predicted probabilities with correct predictions (class labels)
                        counts_pred_probs_cor, _ = np.histogram(pred_probs_cls_cor, bins=prob_bins)
                        acc_per_bin = np.zeros(num_of_bins)
                        for bin_idx in np.arange(len(acc_per_bin)):
                            if counts_pred_probs[bin_idx] != 0:
                                acc_per_bin[bin_idx] = counts_pred_probs_cor[bin_idx] * 1./counts_pred_probs[bin_idx]
                                acc_bins[phase, cls_idx, bin_idx] += acc_per_bin[bin_idx]

                                if counts_pred_probs_cor[bin_idx] != 0:
                                    acc_bins_used[phase, cls_idx, bin_idx] += 1
                            else:
                                acc_bins[phase, cls_idx, bin_idx] += 0
                        # compute ECE for this patient/class
                        ece = compute_expected_calibration_error(pred_probs_cls, acc_per_bin, counts_pred_probs,
                                                                 num_of_bins=num_of_bins)
                        ece_per_class[phase, cls_idx] += ece
                        ece_counts_per_class[phase, cls_idx] += 1
                        mean_counts_per_bin += counts_pred_probs
    # print(acc_bins_used)
    for phase in [0, 1]:
        for cls_idx in np.arange(num_of_classes):
            for bin_idx in np.arange(len(prob_bins[1:])):
                if acc_bins_used[phase, cls_idx, bin_idx] != 0:
                    acc_bins[phase, cls_idx, bin_idx] *= 1./acc_bins_used[phase, cls_idx, bin_idx]

    # compute final mean ECE value per class, omit the BACKGROUND class
    mean_ece_per_class = np.nan_to_num(np.divide(ece_per_class[:, 1:], ece_counts_per_class[:, 1:]))
    # compute mean counts per probability bin
    mean_counts_per_bin *= mean_counts_per_bin
    if do_save:
        try:
            e_suffix = ""
            if with_bg:
                e_suffix = "_wbg"
            file_name = "calibration_" + exper_args.model + "_" + exper_args.loss_function + e_suffix + ".npz"
            file_name = os.path.join(config.data_dir, file_name)
            np.savez(file_name, prob_bins=prob_bins, acc_bins=acc_bins, mean_ece_per_class=mean_ece_per_class)
            print("INFO - Successfully saved numpy arrays to location {}".format(file_name))
        except IOError:
            raise IOError("ERROR - can't save numpy arrays to location {}".format(file_name))

    return prob_bins, acc_bins, mean_ece_per_class


def plot_reliability_diagram(cal_data, height=None, width=16, do_show=True, do_save=False, per_class=False):

    """

    :param cal_data: CalibrationData object (see below for definition)
           NOTE:
           acc_per_bin is a numpy array of shape [phases, 4, #of_bins] phases/classes
    :param do_show:
    :param do_save:
    :param per_class: plot per class of one figure over mean class values
    :return:
    """
    phase_labels = ["ES", "ED"]
    cls_labels = ["BG", "RV", "MYO", "LV"]
    if per_class:
        rows = 2 * 3
        columns = 4
        if height is None:
            height = 14
        the_range = np.arange(1, 4)
        acc_bins = cal_data.acc_per_bin
    else:
        # 2 figures (ES/ED) average values over classes
        rows = 4
        columns = 4
        if height is None:
            height = 10
        the_range = np.arange(0, 1)
        # average per phase over classes ignoring background class
        acc_bins = np.mean(cal_data.acc_per_bin[:, 1:], axis=1)

    row = 0
    bar_width = 0.09
    tick_size = 42
    legend_size = 42
    axis_label_size = {'fontname': 'Monospace', 'size': '46', 'color': 'black', 'weight': 'normal'}
    sub_title_size = {'fontname': 'Monospace', 'size': '46', 'color': 'black', 'weight': 'normal'}
    prob_bins = cal_data.prob_bin_edges

    model_info = "{}".format(cal_data.loss_function.title())

    fig = plt.figure(figsize=(width, height))
    fig.suptitle(model_info, **{'fontname': 'Monospace', 'size': '50', 'color': 'black', 'weight': 'normal'})

    for cls_idx in the_range:
        ax1 = plt.subplot2grid((rows, columns), (row, 0), rowspan=2, colspan=2)
        if per_class:
            acc_ed = acc_bins[1, cls_idx]
            ax1.set_title("{} - class {}".format(phase_labels[1], cls_labels[cls_idx]), **sub_title_size)
        else:
            acc_ed = acc_bins[1]
            # print(acc_ed)
            ax1.set_title("{} ".format(phase_labels[1]), **sub_title_size)
        # compute the gap between fraction of forcast and identiy. Set to zero if bin was empty (2nd line)
        y_gaps = np.abs(prob_bins[1:] - acc_ed)
        y_gaps[y_gaps == prob_bins[1:]] = 0
        ax1.bar(prob_bins[1:], acc_ed, bar_width, color="b", alpha=0.4)
        ax1.bar(prob_bins[1:], y_gaps, bar_width, bottom=acc_ed, color="r", alpha=0.3, hatch='/',
                label="Miscalibration", edgecolor='red')
        ax1.set_ylim([0, 1])
        ax1.set_xlim([0, 1.1])
        ax1.set_ylabel("Fraction of positive cases", **axis_label_size)
        ax1.set_xlabel("Probability", **axis_label_size)
        ax1.set_xticks(np.array([0.2, 0.4, 0.6, 0.8, 1.0]))
        ax1.tick_params(axis='both', which='major', labelsize=tick_size)
        # plot the identify function i.e. bisector line
        ax1.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), "--", c="gray", linewidth=6)
        ax1.legend(loc=0, prop={'size': legend_size})
        ax2 = plt.subplot2grid((rows, columns), (row, 2), rowspan=2, colspan=2, sharey=ax1)
        if per_class:
            acc_es = acc_bins[1, cls_idx]
            ax2.set_title("{} - class {}".format(phase_labels[1], cls_labels[cls_idx]), **config.title_font_medium)
        else:
            acc_es = acc_bins[0]
            # print(acc_es)
            ax2.set_title("{} ".format(phase_labels[0]), **sub_title_size)
        # compute the gap between fraction of forcast and identiy. Set to zero if bin was empty (2nd line)
        y_gaps = np.abs(prob_bins[1:] - acc_es)
        y_gaps[y_gaps == prob_bins[1:]] = 0
        ax2.bar(prob_bins[1:], acc_es, bar_width, color="g", alpha=0.4)  # yerr=y_gaps,
        ax2.bar(prob_bins[1:], y_gaps, bar_width, bottom=acc_es, color="r", alpha=0.3, label="Miscalibration",
                hatch='/', edgecolor="red")
        ax2.set_ylim([0, 1])
        ax2.set_xlim([0, 1.1])
        ax2.set_xlabel("Probability", **axis_label_size)
        ax2.tick_params(axis='both', which='major', labelsize=tick_size)
        ax2.set_yticks(np.array([0.2, 0.4, 0.6, 0.8, 1.0]))
        ax2.set_xticks(np.array([0.2, 0.4, 0.6, 0.8, 1.0]))
        pylab.setp(ax2.get_yticklabels(), visible=False)
        # plot the identify function i.e. bisector line
        ax2.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), "--", c="gray", linewidth=6)
        ax2.legend(loc=0, prop={'size': legend_size})
        row += 2

    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    if do_save:
        fig_path = os.path.join(config.root_dir, config.figure_path)

        if not os.path.isdir(fig_path):
            os.makedirs(fig_path)
        fig_name = cal_data.model_name + "_" + cal_data.loss_function + "_reliability_probs"
        fig_name = os.path.join(fig_path, fig_name + ".jpeg")

        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)
    if do_show:
        plt.show()


def plot_loss_functions_for_true_label(do_show=True, do_save=False, width=6, height=6):

    eps = 1e-4
    probs = np.linspace(0, 1, 50)
    log_loss = -np.log(probs + eps)  # for positive label
    brier_loss = (1 - probs) ** 2
    softdice = -(probs) / (1 + probs)
    softdice -= np.min(softdice)

    fig = plt.figure(figsize=(width, height))
    fig.suptitle("Loss for a true label", **config.title_font_large)
    plt.plot(probs, log_loss, c='g', label="binary entropy loss")
    plt.plot(probs, brier_loss, c='b', label="brier loss")
    plt.plot(probs, softdice, c='r', label="soft-dice loss")
    plt.ylim([-1, 4])
    plt.ylabel("Loss", **config.axis_font22)
    plt.xlabel("Probability", **config.axis_font22)
    plt.tick_params(axis='both', which='major', labelsize=22)
    plt.legend(loc=0, prop={'size': 20})
    plt.grid("on")
    if do_save:
        fig_path = os.path.join(config.root_dir, config.figure_path)
        if not os.path.isdir(fig_path):
            os.makedirs(fig_path)
        fig_name = os.path.join(fig_path, "compare_loss_functions.jpeg")

        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)
    if do_show:
        plt.show()


class CalibrationData(object):

    file_prefix = "calibration_"

    def __init__(self, model_name, loss_function, with_bg=False):
        self.model_name = model_name
        self.loss_function = loss_function
        self.with_bg = with_bg
        self.prob_bin_edges = None
        self.acc_per_bin = None
        self.mean_ece_per_class = None

    def load(self):

        file_name = CalibrationData.file_prefix + self.model_name + "_" + self.loss_function + ".npz"
        file_name = os.path.join(config.data_dir, file_name)
        if len(glob.glob(file_name)) != 1:
            raise ValueError("ERROR - found {} files with name {}. Must be one".format(len(glob.glob(file_name)),
                                                                                       file_name))

        try:
            data = np.load(file_name)
            self.prob_bin_edges = data["prob_bins"]
            self.acc_per_bin = data["acc_bins"]
            self.mean_ece_per_class = data["mean_ece_per_class"]
        except (IOError, KeyError) as e:
            print("ERROR - can't load numpy archive {}".format(file_name))
            print(e)

    def save(self, prob_bins, acc_bins, mean_ece_per_class, fig_suffix=None):
        if fig_suffix is None:
            fig_suffix = self.model_name + "_" + self.loss_function
        if self.with_bg:
            file_name = CalibrationData.file_prefix + fig_suffix + "_wbg.npz"
        else:
            file_name = CalibrationData.file_prefix + fig_suffix + ".npz"
        file_name = os.path.join(config.data_dir, file_name)

        try:
            np.savez(file_name, prob_bins=prob_bins, acc_bins=acc_bins, mean_ece_per_class=mean_ece_per_class)
            print("INFO - Successfully saved numpy arrays to location {}".format(file_name))
        except IOError:
            raise IOError("ERROR - can't save numpy arrays to location {}".format(file_name))