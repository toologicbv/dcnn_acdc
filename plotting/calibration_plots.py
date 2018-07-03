from config.config import config
import matplotlib.pyplot as plt
import numpy as np
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


def reliability_diagram(exper_dict, patient_id=None, mc_dropout=False, force_reload=False):
    exper_handlers = collect_exper_handlers(exper_dict)
    # prepare: get probability maps and test_set which contains

    prob_bins = np.linspace(0, 1, 11)
    acc_bins = np.zeros((2, 4, 10))
    acc_bins_used = np.zeros((2, 4, 10))
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
                        # num_of_positives = np.count_nonzero(gt_labels)
                        # pred_labels_cls = pred_labels[cls_offset + cls_idx, :, :, :].flatten()
                        pred_probs_cls = pred_probs[cls_offset + cls_idx, :, :, :].flatten()
                        gt_labels_cls = np.atleast_1d(gt_labels_cls.astype(np.bool))
                        # pred_labels_cls = np.atleast_1d(pred_labels_cls.astype(np.bool))
                        pos_voxels_idx = gt_labels_cls == 1
                        pred_probs_cls_cor = pred_probs_cls[pos_voxels_idx]
                        bin_pred_probs, _ = np.histogram(pred_probs_cls, bins=prob_bins)
                        bin_pred_probs_cor, _ = np.histogram(pred_probs_cls_cor, bins=prob_bins)
                        for bin_idx in np.arange(len(prob_bins[1:])):
                            if bin_pred_probs[bin_idx] != 0:
                                acc_bins[phase, cls_idx, bin_idx] += bin_pred_probs_cor[bin_idx] * \
                                                                     1./bin_pred_probs[bin_idx]
                                if bin_pred_probs_cor[bin_idx] != 0:
                                    acc_bins_used[phase, cls_idx, bin_idx] += 1
                            else:
                                acc_bins[phase, cls_idx, bin_idx] += 0

    # print(acc_bins_used)
    for phase in [0, 1]:
        for cls_idx in np.arange(num_of_classes):
            for bin_idx in np.arange(len(prob_bins[1:])):
                if acc_bins_used[phase, cls_idx, bin_idx] != 0:
                    acc_bins[phase, cls_idx, bin_idx] *= 1./acc_bins_used[phase, cls_idx, bin_idx]

    return prob_bins, acc_bins


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