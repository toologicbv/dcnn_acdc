from config.config import config
import matplotlib.pyplot as plt
import numpy as np

arr_diff_lv_ej, arr_diff_rv_ej = [], []
arr_lv_ej, arr_rv_ej = [], []
arr_errors_lv, arr_errors_rv = [], []
arr_errors_reg_lv, arr_errors_reg_rv = [], []


def plot_error_dependencies(exper_handler, height, width, do_show=True, do_save=False, mc_dropout=False,
                            prepare_handler=True, second_yaxis=""):

    global arr_diff_lv_ej, arr_diff_rv_ej, arr_lv_ej, arr_rv_ej, arr_errors_lv, arr_errors_rv, arr_errors_reg_lv, \
        arr_errors_reg_rv

    analyze_cardiac_indices(exper_handler, mc_dropout=mc_dropout, prepare_handler=prepare_handler)
    # LV
    np_diff_lv_ej = np.array(arr_diff_lv_ej)
    np_errors_lv = np.array(arr_errors_lv)
    np_errors_reg_lv = np.array(arr_errors_reg_lv)
    # RV
    np_diff_rv_ej = np.array(arr_diff_rv_ej)
    np_errors_rv = np.array(arr_errors_rv)
    np_errors_reg_rv = np.array(arr_errors_reg_rv)

    # setup
    axis_label_size = {'fontname': 'Monospace', 'size': '20', 'color': 'black', 'weight': 'normal'}
    sub_title_size = {'fontname': 'Monospace', 'size': '24', 'color': 'black', 'weight': 'normal'}
    tick_size = 16
    legend_size = 16

    rows = 4
    columns = 4

    fig = plt.figure(figsize=(width, height))
    ax1 = plt.subplot2grid((rows, columns), (0, 0), rowspan=2, colspan=4)
    ax1.plot(np_errors_lv, np_errors_reg_lv, color="red", linestyle='', marker='o', markersize=10, alpha=0.4,
             label="errors-to-detect")
    ax1.set_xlabel("# segmentation errors", **axis_label_size)
    ax1.set_ylabel("# errors-to-detect", **axis_label_size)
    ax1.tick_params(axis='both', which='major', labelsize=tick_size)
    ax1.tick_params(axis='y', colors='red')
    ax1.set_title("Relationship between errors", **sub_title_size)
    # ax1.legend(loc=1,prop={'size': legend_size})
    if second_yaxis[:3] == "ejf":
        ax1b = ax1.twinx()
        if second_yaxis == "ejf_LV":
            y_axis_error = np_diff_lv_ej
            c_phase = "LV"
        elif second_yaxis == "ejf_RV":
            y_axis_error = np_diff_rv_ej
            c_phase = "RV"
        else:
            raise ValueError("{} is not supported".format(second_yaxis))
        ax1b.plot(np_errors_lv, y_axis_error, color="blue", linestyle='', marker='d', markersize=10, alpha=0.4,
                  label="error EJF {}".format(c_phase))
        ax1b.set_ylabel("# error EJF ({})".format(c_phase), **axis_label_size)
        ax1b.tick_params(axis='both', which='major', labelsize=tick_size)
        ax1b.tick_params(axis='y', colors='blue')
        # ax1b.legend(loc=2, prop={'size': legend_size})
    if do_show:
        plt.show()


def plot_ejection_fraction(exper_handler, height, width, do_show=True, do_save=False, mc_dropout=False,
                           prepare_handler=True, bars="both"):
    """

    :param exper_handler:
    :param height:
    :param width:
    :param do_show:
    :param do_save:
    :param mc_dropout:
    :param prepare_handler:
    :param bars: ["both", "ejf", "errors"]:
    :return:
    """
    global arr_diff_lv_ej, arr_diff_rv_ej, arr_lv_ej, arr_rv_ej, arr_errors_lv, arr_errors_rv, arr_errors_reg_lv, \
           arr_errors_reg_rv

    analyze_cardiac_indices(exper_handler, mc_dropout=mc_dropout, prepare_handler=prepare_handler)
    np_lv_ej = np.array(arr_lv_ej)
    np_diff_lv_ej = np.array(arr_diff_lv_ej)
    np_errors_lv = np.array(arr_errors_lv)
    np_errors_reg_lv = np.array(arr_errors_reg_lv)
    num_of_patients = np_lv_ej.shape[0]
    np_patients = []
    for patient_id in exper_handler.test_set.cardiac_indices.keys():
        np_patients.append(int(patient_id.strip("patient")))
    np_patients = np.array(np_patients)

    # setup
    bar_width = 0.3
    rows = 4
    columns = 4
    x_values = np.arange(1, num_of_patients+1)
    legend_size = 16
    tick_size = 16
    axis_label_size = {'fontname': 'Monospace', 'size': '20', 'color': 'black', 'weight': 'normal'}
    sub_title_size = {'fontname': 'Monospace', 'size': '24', 'color': 'black', 'weight': 'normal'}

    fig = plt.figure(figsize=(width, height))
    ax1 = plt.subplot2grid((rows, columns), (0, 0), rowspan=2, colspan=4)
    if bars == "both" or bars == "ejf":
        ax1.bar(x_values, np_lv_ej, bar_width, color="b", alpha=0.4, label="LV EJF", align="center")
        ax1.bar(x_values, np_diff_lv_ej, bar_width, bottom=np_lv_ej, color="r", alpha=0.3, hatch='/',
                label="LV_EJF error", edgecolor='red', align="center")

        ax1.set_ylabel("LV EJF (%)", **axis_label_size)
        ax1.legend(loc=1, prop={'size': legend_size})
        ax1.set_ylim([0, 100])
    ax1.set_xlabel("Patient ID", **axis_label_size)
    ax1.set_xticks(x_values)
    ax1.set_xticklabels(np_patients)
    ax1.tick_params(axis='both', which='major', labelsize=tick_size)
    if bars == "both" or bars == "errors":
        ax1b = ax1.twinx()
        ax1b.bar(x_values + bar_width, np_errors_lv, bar_width, color="g", alpha=0.4, label="LV #seg-errors",
                 align="center")
        ax1b.bar(x_values + bar_width, np_errors_reg_lv, bar_width, bottom=np_errors_lv, color="orange", alpha=0.5,
                 hatch='/', label="LV errors-to-detect", edgecolor='red', align="center")
        ax1b.set_ylabel("LV #seg-errors", **axis_label_size)
        ax1b.tick_params(axis='both', which='major', labelsize=tick_size)
        ax1b.legend(loc=2, prop={'size': legend_size})
        if bars == "errors":
            ax1.set_title("LV segmentation errors", **sub_title_size)
        else:
            ax1.set_title("LV ejection fraction & seg-errors", **sub_title_size)
    else:
        ax1.set_title("LV ejection fraction", **sub_title_size)

    #       THE SAME FOR RV Ejection fraction
    np_rv_ej = np.array(arr_rv_ej)
    np_diff_rv_ej = np.array(arr_diff_rv_ej)
    np_errors_rv = np.array(arr_errors_rv)
    np_errors_reg_rv = np.array(arr_errors_reg_rv)
    ax2 = plt.subplot2grid((rows, columns), (2, 0), rowspan=2, colspan=4)
    if bars == "both" or bars == "ejf":
        ax2.bar(x_values, np_rv_ej, bar_width, color="b", alpha=0.4, label="RV EJF", align="center")
        ax2.bar(x_values, np_diff_rv_ej, bar_width, bottom=np_rv_ej, color="r", alpha=0.3, hatch='/',
                label="RV_EJF error", edgecolor='red', align="center")

        ax2.set_ylabel("RV EJF (%)", **axis_label_size)
        ax2.legend(loc=1, prop={'size': legend_size})
        ax2.set_ylim([0, 100])
    ax2.set_xlabel("Patient ID", **axis_label_size)
    ax2.set_xticks(x_values)
    ax2.set_xticklabels(np_patients)
    ax2.set_title("RV ejection fraction", **sub_title_size)
    ax2.tick_params(axis='both', which='major', labelsize=tick_size)
    if bars == "both" or bars == "errors":
        ax2b = ax2.twinx()
        ax2b.bar(x_values + bar_width, np_errors_rv, bar_width, color="g", alpha=0.4, label="RV #seg-errors",
                 align="center")
        ax2b.bar(x_values + bar_width, np_errors_reg_rv, bar_width, bottom=np_errors_rv, color="orange", alpha=0.5,
                 hatch='/', label="RV errors-to-detect", edgecolor='red', align="center")
        ax2b.set_ylabel("RV #seg-errors", **axis_label_size)
        ax2b.tick_params(axis='both', which='major', labelsize=tick_size)
        ax2b.legend(loc=2, prop={'size': legend_size})

    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])

    if do_save:
        pass
    if do_show:
        plt.show()


def analyze_cardiac_indices(exper_handler, prepare_handler=True, mc_dropout=False):
    global arr_diff_lv_ej, arr_diff_rv_ej, arr_lv_ej, arr_rv_ej, arr_errors_lv, arr_errors_rv, arr_errors_reg_lv, \
        arr_errors_reg_rv

    arr_diff_lv_ej, arr_diff_rv_ej, arr_lv_ej, arr_rv_ej, arr_errors_lv, arr_errors_rv, arr_errors_reg_lv, \
        arr_errors_reg_rv = [], [], [], [], [], [], [], []

    if prepare_handler:
        exper_handler.get_patients()
        exper_handler.compute_cardiac_indices(mc_dropout=mc_dropout)
        exper_handler.test_set.compute_cardiac_indices()
        _ = exper_handler.get_pred_labels_errors(mc_dropout=mc_dropout)
        _ = exper_handler.get_target_roi_maps(mc_dropout=mc_dropout)

    for patient_id, c_indices in exper_handler.test_set.cardiac_indices.iteritems():
        arr_lv_ej.append(c_indices[0, 2])
        arr_rv_ej.append(c_indices[1, 2])
        test_img_idx = exper_handler.test_set.trans_dict[patient_id]
        disease_cat = exper_handler.patients[patient_id]
        gt_labels = exper_handler.test_set.labels[test_img_idx]
        gt_labels_es = gt_labels[:4]
        gt_labels_ed = gt_labels[4:]
        gt_labels_lv = np.count_nonzero(gt_labels_es[3]) + np.count_nonzero(gt_labels_ed[3])
        gt_labels_rv = np.count_nonzero(gt_labels_es[1]) + np.count_nonzero(gt_labels_ed[1])
        errors = exper_handler.pred_labels_errors[patient_id]
        errors_regions = exper_handler.target_roi_maps[patient_id]
        errors_es = errors[:4]
        errors_ed = errors[4:]
        errors_regions_es = errors_regions[:4]
        errors_regions_ed = errors_regions[4:]
        cardiac_indices_pred = exper_handler.cardiac_indices[patient_id]
        diff_lv_ej = abs(c_indices[0, 2] - cardiac_indices_pred[0, 2])
        diff_rv_ej = abs(c_indices[1, 2] - cardiac_indices_pred[1, 2])
        arr_diff_lv_ej.append(diff_lv_ej)
        arr_diff_rv_ej.append(diff_rv_ej)
        errors_lv = int(np.count_nonzero(errors_es[3]) + np.count_nonzero(errors_ed[3]))
        errors_rv = int(np.count_nonzero(errors_es[1]) + np.count_nonzero(errors_ed[1]))
        arr_errors_lv.append(errors_lv)
        arr_errors_rv.append(errors_rv)
        errors_lv_perc = errors_lv / float(gt_labels_lv) * 100
        errors_rv_perc = errors_rv / float(gt_labels_rv) * 100
        errors_reg_lv = int(np.count_nonzero(errors_regions_es[3]) + np.count_nonzero(errors_regions_ed[3]))
        errors_reg_rv = int(np.count_nonzero(errors_regions_es[1]) + np.count_nonzero(errors_regions_ed[1]))
        arr_errors_reg_lv.append(errors_reg_lv)
        arr_errors_reg_rv.append(errors_reg_rv)



