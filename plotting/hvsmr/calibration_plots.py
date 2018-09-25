import os
import numpy as np
import matplotlib.pyplot as plt
import pylab
from common.hvsmr.config import config_hvsmr
from utils.hvsmr.exper_hdl_ensemble import ExperHandlerEnsemble
from plotting.calibration_plots import CalibrationData


def compute_calibration_terms(exper_dict, patient_id=None, mc_dropout=False, force_reload=False, num_of_bins=10,
                              do_save=False, class_range=[1, 2], cardiac_phases=1):
    """
    Compute the ingredients we need for the ECE (Expected Calibration Error) and the Reliability Diagrams
    Taken from paper https://arxiv.org/abs/1706.04599

    :param exper_dict:
    :param patient_id:
    :param mc_dropout:
    :param force_reload: force reload of probability maps that model generated (property of exp-handler)
    :param num_of_bins:
    :param class_range: list of class indices (we would normally omit the background class 0
    :param cardiac_phases: 1 (HVSMR dataset) or 2 (ACDC dataset)
    :param do_save:
    :return:
    """

    exper_hdl_ensemble = ExperHandlerEnsemble(exper_dict)
    num_of_classes = len(class_range)
    # if we omit the bg-class we add 1 to the number of classes because this makes life much easier
    if class_range[0] != 0:
        num_of_classes += 1
    # prepare: get probability maps and test_set which contains
    # prob_bins: contains bin-edges for the probabilities (default 10)
    prob_bins = np.linspace(0, 1, num_of_bins + 1)
    # acc_bins: accuracy per bin. numerator="# of positive predicted class labels for this bin";
    #                             denominator="# of predictions in this bin"
    acc_bins = np.zeros((cardiac_phases, num_of_classes, num_of_bins))
    probs_per_bin = np.zeros((cardiac_phases, num_of_classes, num_of_bins))
    probs_per_bin_denom = np.zeros((cardiac_phases, num_of_classes, num_of_bins))
    acc_bins_used = np.zeros((cardiac_phases, num_of_classes, num_of_bins))
    mean_counts_per_bin = np.zeros((cardiac_phases, num_of_classes, num_of_bins))
    # confidence measure = mean predicted probability in this bin

    for fold_id, exper_handler in exper_hdl_ensemble.seg_exper_handlers.iteritems():
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
            gt_labels = exper_handler.test_set.get_labels_per_class(image_num)
            # pred_labels = exper_handler.pred_labels[p_id]
            pred_probs = exper_handler.pred_prob_maps[p_id]
            for phase in np.arange(cardiac_phases):
                for cls_idx in class_range:
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
                        cls_idx_probs_per_bin = np.digitize(pred_probs_cls, bins=prob_bins)
                        # bin the predicted probabilities with correct predictions (class labels)
                        counts_pred_probs_cor, _ = np.histogram(pred_probs_cls_cor, bins=prob_bins)
                        acc_per_bin = np.zeros(num_of_bins)
                        for bin_idx in np.arange(len(acc_per_bin)):
                            # do the stuff to compute the conf(B_m) metric from the Guo & Pleiss paper
                            cls_probs_per_bin = pred_probs_cls[cls_idx_probs_per_bin == bin_idx + 1]
                            probs_per_bin[phase, cls_idx, bin_idx] += np.sum(cls_probs_per_bin)
                            probs_per_bin_denom[phase, cls_idx, bin_idx] += cls_probs_per_bin.shape[0]
                            # do the stuff to compute the acc(B_m) metric from the Guo&Pleiss paper
                            if counts_pred_probs[bin_idx] != 0:
                                acc_per_bin[bin_idx] = counts_pred_probs_cor[bin_idx] * 1./counts_pred_probs[bin_idx]
                                acc_bins[phase, cls_idx, bin_idx] += acc_per_bin[bin_idx]

                                if counts_pred_probs_cor[bin_idx] != 0:
                                    acc_bins_used[phase, cls_idx, bin_idx] += 1
                            else:
                                acc_bins[phase, cls_idx, bin_idx] += 0

                        mean_counts_per_bin += counts_pred_probs
    # print(acc_bins_used)
    for phase in np.arange(cardiac_phases):
        for cls_idx in np.arange(num_of_classes):
            for bin_idx in np.arange(len(prob_bins[1:])):
                if acc_bins_used[phase, cls_idx, bin_idx] != 0:
                    acc_bins[phase, cls_idx, bin_idx] *= 1./acc_bins_used[phase, cls_idx, bin_idx]
                if probs_per_bin_denom[phase, cls_idx, bin_idx] != 0:
                    probs_per_bin[phase, cls_idx, bin_idx] *= 1./probs_per_bin_denom[phase, cls_idx, bin_idx]

    # compute mean counts per probability bin
    mean_counts_per_bin *= mean_counts_per_bin
    if do_save:
        try:
            loss_function = exper_args.loss_function.replace("-", "")
            file_name = CalibrationData.file_prefix + exper_args.model + "_" + loss_function + ".npz"
            file_name = os.path.join(config_hvsmr.data_dir, file_name)
            np.savez(file_name, prob_bins=prob_bins, acc_bins=acc_bins, mean_ece_per_class=None,
                     probs_per_bin=probs_per_bin)
            print("INFO - Successfully saved numpy arrays to location {}".format(file_name))
        except IOError:
            raise IOError("ERROR - can't save numpy arrays to location {}".format(file_name))

    return prob_bins, acc_bins, probs_per_bin


def plot_reliability_diagram(cal_data, height=None, width=16, do_show=True, do_save=False,
                             phase_labels=["ES", "ED"]):

    """

    :param cal_data: CalibrationData object (see below for definition)
           NOTE:
           acc_per_bin is a numpy array of shape [phases, 4, #of_bins] phases/classes
    :param do_show:
    :param do_save:
    :param phase_labels: plot per class of one figure over mean class values
    :return:
    """

    # For ACDC, two figures (ES/ED) average values over classes
    # For HVSMR, one figure
    if "hvsmr" in cal_data.model_name:
        num_of_plots = 1
        phase_idx = 0
        tick_size = 24
        legend_size = 24
        axis_label_size = {'fontname': 'Monospace', 'size': '24', 'color': 'black', 'weight': 'normal'}
        sub_title_size = {'fontname': 'Monospace', 'size': '24', 'color': 'black', 'weight': 'normal'}
        title_size = {'fontname': 'Monospace', 'size': '30', 'color': 'black', 'weight': 'normal'}
    else:
        num_of_plots = 2
        phase_idx = 1
        tick_size = 42
        legend_size = 42
        axis_label_size = {'fontname': 'Monospace', 'size': '46', 'color': 'black', 'weight': 'normal'}
        sub_title_size = {'fontname': 'Monospace', 'size': '46', 'color': 'black', 'weight': 'normal'}
        title_size = {'fontname': 'Monospace', 'size': '50', 'color': 'black', 'weight': 'normal'}

    rows = 2 * num_of_plots
    columns = 2 * num_of_plots
    if height is None:
        height = 10

    # average per phase over classes ignoring background class
    acc_bins = np.mean(cal_data.acc_per_bin[:, 1:], axis=1)
    probs_per_bin = np.mean(cal_data.probs_per_bin[:, 1:], axis=1)

    row = 0
    bar_width = 0.09

    prob_bins = cal_data.prob_bin_edges - 0.05

    model_info = "{}".format(cal_data.loss_function.title())

    fig = plt.figure(figsize=(width, height))
    fig.suptitle(model_info, **title_size)

    ax1 = plt.subplot2grid((rows, columns), (row, 0), rowspan=2, colspan=2)
    acc_ed = acc_bins[phase_idx]
    # print(acc_ed)
    ax1.set_title("{} ".format(phase_labels[phase_idx]), **sub_title_size)
    # compute the gap between fraction of forcast and identiy. Set to zero if bin was empty (2nd line)
    # old version how we computed miscalibration (wrong): y_gaps = np.abs(prob_bins[1:] - acc_ed)
    y_gaps = np.abs(probs_per_bin[phase_idx] - acc_ed)
    y_gaps[y_gaps == prob_bins[1:]] = 0
    ax1.bar(prob_bins[1:], acc_ed, bar_width, color="b", alpha=0.4)
    ax1.bar(prob_bins[1:], y_gaps, bar_width, bottom=acc_ed, color="r", alpha=0.3, hatch='/',
            label="Miscalibration", edgecolor='red')
    ax1.set_ylim([0, 1])
    ax1.set_xlim([0, 1])
    ax1.set_ylabel("Fraction of positive cases", **axis_label_size)
    ax1.set_xlabel("Probability", **axis_label_size)
    ax1.set_xticks(np.array([0.2, 0.4, 0.6, 0.8, 1.0]))
    ax1.tick_params(axis='both', which='major', labelsize=tick_size)
    # plot the identify function i.e. bisector line
    ax1.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), "--", c="gray", linewidth=6)
    ax1.legend(loc=0, prop={'size': legend_size})

    if num_of_plots == 2:
        ax2 = plt.subplot2grid((rows, columns), (row, 2), rowspan=2, colspan=2, sharey=ax1)

        acc_es = acc_bins[phase_idx - 2]
        # print(acc_es)
        ax2.set_title("{} ".format(phase_labels[phase_idx - 2]), **sub_title_size)
        # compute the gap between fraction of forcast and identiy. Set to zero if bin was empty (2nd line)
        # old version how we computed miscalibration (wrong): y_gaps = np.abs(prob_bins[1:] - acc_es)
        y_gaps = np.abs(probs_per_bin[phase_idx - 2] - acc_es)
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

    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    if do_save:
        fig_path = os.path.join(config_hvsmr.root_dir, config_hvsmr.figure_path)

        if not os.path.isdir(fig_path):
            os.makedirs(fig_path)
        fig_name = cal_data.model_name + "_" + cal_data.loss_function + "_reliability_probs"
        fig_name = os.path.join(fig_path, fig_name + ".jpeg")

        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)
    if do_show:
        plt.show()