from config.config import config
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic
import os

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
                              do_save=False):
    """
    Compute the ingredients we need for the ECE (Expected Calibration Error) and the Reliability Diagrams
    Taken from paper https://arxiv.org/abs/1706.04599

    :param exper_dict:
    :param patient_id:
    :param mc_dropout:
    :param force_reload:
    :param num_of_bins:
    :return:
    """
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
                    if cls_idx != 0 and cls_idx in [1, 2, 3]:
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
            file_name = "calibration_" + exper_args.model + "_" + exper_args.loss_function + ".npz"
            file_name = os.path.join(config.data_dir, file_name)
            np.savez(file_name, prob_bins=prob_bins, acc_bins=acc_bins, mean_ece_per_class=mean_ece_per_class)
            print("INFO - Successfully saved numpy arrays to location {}".format(file_name))
        except IOError:
            raise IOError("ERROR - can't save numpy arrays to location {}".format(file_name))

    return prob_bins, acc_bins, mean_ece_per_class


def plot_reliability_diagram(exper_handler, prob_bins, acc_bins, do_show=True, do_save=False):
    exper_args = exper_handler.exper.run_args
    phase_labels = ["ES", "ED"]
    cls_labels = ["BG", "RV", "MYO", "LV"]
    rows = 2 * 3
    columns = 4
    width = 16
    height = 14
    row = 0
    bar_width = 0.09

    model_info = "{} p={:.2f} loss={}".format(exper_args.model, exper_args.drop_prob, exper_args.loss_function)

    fig = plt.figure(figsize=(width, height))
    fig.suptitle(model_info, **config.title_font_medium)

    for cls_idx in np.arange(1, 4):
        ax1 = plt.subplot2grid((rows, columns), (row, 0), rowspan=2, colspan=2)
        acc_es = acc_bins[0, cls_idx]
        ax1.bar(prob_bins[1:], acc_es, bar_width, color="b", alpha=0.4)
        ax1.set_ylim([0, 1])
        ax1.set_xlim([0, 1.1])
        ax1.set_ylabel("Accuracy")
        ax1.set_xlabel("Probability")
        ax1.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), "--", c="gray")
        ax1.set_title("{} - class {}".format(phase_labels[0], cls_labels[cls_idx]))
        ax2 = plt.subplot2grid((rows, columns), (row, 2), rowspan=2, colspan=2)
        acc_ed = acc_bins[1, cls_idx]
        ax2.bar(prob_bins[1:], acc_ed, bar_width, color="g", alpha=0.4)
        ax2.set_ylim([0, 1])
        ax2.set_xlim([0, 1.1])
        ax2.set_xlabel("Probability")
        ax2.set_title("{} - class {}".format(phase_labels[1], cls_labels[cls_idx]))
        ax2.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), "--", c="gray")
        row += 2

    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    if do_save:
        fig_path = os.path.join(exper_handler.exper.config.root_dir, config.figure_path)

        if not os.path.isdir(fig_path):
            os.makedirs(fig_path)
        fig_name = exper_args.model + "_" + exper_args.loss_function + "_reliability_probs"
        fig_name = os.path.join(fig_path, fig_name + ".jpeg")

        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)
    if do_show:
        plt.show()