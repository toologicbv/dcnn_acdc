import numpy as np
import os
import traceback
from config.config import config as config_acdc


def compute_tp_tn_fn_fp(result, reference):
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    tp = np.count_nonzero(result & reference)
    tp_idx = result & reference
    fn = np.count_nonzero(~result & reference)
    fn_idx = ~result & reference
    tn = np.count_nonzero(~result & ~reference)
    tn_idx = ~result & ~reference
    fp = np.count_nonzero(result & ~reference)
    fp_idx = result & ~reference

    return tuple((tp, tp_idx)), tuple((fn, fn_idx)), tuple((tn, tn_idx)), tuple((fp, fp_idx))


def compute_froc(pred_probs, gt_labels, nbr_of_thresholds=40, range_threshold=None, slice_probs_dict=None,
                 slice_labels_dict=None, exper_handler=None, base_apex_labels=None, loc=1):
    """

    :param pred_probs: [#predictions] assuming the np.array only contains the probs for the true class (binary 0/1)
    :param gt_labels:  [#predictions] numpy array that comprises the ground truth binary labels
    :param nbr_of_thresholds: integer
    :param range_threshold: list e.g. [0, 0.5]
    :param slice_probs_dict: dictionary with probs for each slice (key=slice_id)
    :param slice_labels_dict: dictionary with slice_labels (key=slice_id)
    :param exper_handler: ExperimentHandler. If not None, we save the results to a numpy file in the stats subdirectory
                          of the corresponding model.
    :param base_apex_labels: compute the statistics only for apex/base slices
    :return:   (1) numpy array with sensitivity for each threshold
               (2) numpy array with FP values for each threshold
    """
    def convert_slice_probs_to_labels(slice_probs_dict, threshold):

        threshold_slice_labels = []
        for slice_id, slice_probs in slice_probs_dict.iteritems():
            thresholded_slice_pred_labels = np.zeros_like(slice_probs)
            thresholded_slice_pred_labels[slice_probs >= threshold] = 1
            threshold_slice_labels.append(1 if np.count_nonzero(thresholded_slice_pred_labels) != 0 else 0)

        return np.array(threshold_slice_labels)

    if np.count_nonzero(gt_labels) == 0:
        raise Warning("WARNING - the ground truth labels of len={} does not contain any TP.".format(len(gt_labels)))
    if slice_probs_dict is not None:
        gt_slice_labels = np.hstack(slice_labels_dict.values())
        num_of_slices = gt_slice_labels.shape[0]
        num_of_grids = pred_probs.shape[0]
    else:
        num_of_slices = 0
        num_of_grids = 0

    # rescale ground truth and proba map between 0 and 1
    pred_probs = pred_probs.astype(np.float32)
    # define the thresholds
    if range_threshold is None:
        threshold_list = (np.linspace(np.min(pred_probs), np.max(pred_probs), nbr_of_thresholds)).tolist()
    else:
        threshold_list = (np.linspace(range_threshold[0], range_threshold[1], nbr_of_thresholds)).tolist()

    threshold_sensitivity = []
    threshold_precision = []
    threshold_fp_rate = []  # 1 - specificity: False negative rate, used on ROC AUC curve x-axis
    threshold_fp = []
    if slice_probs_dict is not None:
        slice_threshold_sensitivity = []
        slice_threshold_precision = []
        slice_threshold_fp_rate = []
        slice_threshold_fp = []
    else:
        slice_threshold_sensitivity = None
        slice_threshold_precision = None
        slice_threshold_fp_rate = None
        slice_threshold_fp = None
    # loop over thresholds
    for i, threshold in enumerate(threshold_list):
        thresholded_pred_labels = np.zeros_like(pred_probs)
        thresholded_pred_labels[pred_probs >= threshold] = 1
        # compute TP, FN, TN, FP. The function actually returns a tuple for each of these measures
        # index 0 = integer / index 1 = np array of indices
        tp_2, fn_2, tn_2, fp_2 = compute_tp_tn_fn_fp(thresholded_pred_labels, gt_labels)
        if slice_probs_dict is not None:
            threshold_slice_labels = convert_slice_probs_to_labels(slice_probs_dict, threshold)
            if base_apex_labels is not None:
                target_idx = base_apex_labels == loc
                s_tp_2, s_fn_2, s_tn_2, s_fp_2 = compute_tp_tn_fn_fp(threshold_slice_labels[target_idx],
                                                                     gt_slice_labels[target_idx])
            else:
                s_tp_2, s_fn_2, s_tn_2, s_fp_2 = compute_tp_tn_fn_fp(threshold_slice_labels, gt_slice_labels)
            if s_tp_2[0] != 0:
                slice_threshold_sensitivity.append((float(s_tp_2[0]) / (float(s_tp_2[0]) + s_fn_2[0])))
                slice_threshold_precision.append((float(s_tp_2[0]) / (float(s_tp_2[0]) + s_fp_2[0])))
                slice_threshold_fp_rate.append((float(s_fp_2[0]) / (float(s_tn_2[0]) + s_fp_2[0])))
                slice_threshold_fp.append(s_fp_2[0])
        # check that ground truth contains at least one positive
        threshold_sensitivity.append((float(tp_2[0]) / (float(tp_2[0]) + fn_2[0])))
        threshold_precision.append((float(tp_2[0]) / (float(tp_2[0]) + fp_2[0])))
        threshold_fp_rate.append((float(fp_2[0]) / (float(tn_2[0]) + fp_2[0])))
        threshold_fp.append(fp_2[0])

    threshold_sensitivity = np.array(threshold_sensitivity)
    threshold_precision = np.array(threshold_precision)
    threshold_fp_rate = np.array(threshold_fp_rate)
    threshold_fp = np.array(threshold_fp)
    if slice_probs_dict is not None:
        slice_threshold_sensitivity = np.array(slice_threshold_sensitivity)
        slice_threshold_precision = np.array(slice_threshold_precision)
        slice_threshold_fp_rate = np.array(slice_threshold_fp_rate)
        slice_threshold_fp = np.array(slice_threshold_fp)

    if exper_handler is not None:
        output_dir = os.path.join(exper_handler.exper.config.root_dir, os.path.join(exper_handler.exper.output_dir,
                                                                                    config_acdc.stats_path))
        if base_apex_labels is not None:
            if loc == 1:
                file_name = "froc_data_apex_base"
            else:
                file_name = "froc_data_mid"
        else:
            file_name = "froc_data"
        abs_file_name = os.path.join(output_dir, file_name)
        try:
            np.savez(abs_file_name, threshold_sensitivity=threshold_sensitivity, threshold_precision=threshold_precision,
                     threshold_fp=threshold_fp, threshold_fp_rate=threshold_fp_rate,
                     slice_threshold_sensitivity=slice_threshold_sensitivity,
                     slice_threshold_precision=slice_threshold_precision,
                     slice_threshold_fp=slice_threshold_fp,
                     slice_threshold_fp_rate=slice_threshold_fp_rate,
                     threshold_list=threshold_list, num_of_slices=num_of_slices, num_of_grids=num_of_grids)
            print("INFO - Saved froc data successfully to {}".format(abs_file_name))
        except Exception as e:
            print(traceback.format_exc())

    if slice_probs_dict is None:
        return threshold_sensitivity, threshold_precision, threshold_fp, threshold_fp_rate, threshold_list, num_of_slices, num_of_grids
    else:
        return threshold_sensitivity, threshold_precision, threshold_fp, threshold_fp_rate,\
               slice_threshold_sensitivity, slice_threshold_precision, slice_threshold_fp, slice_threshold_fp_rate, \
               threshold_list, num_of_slices, num_of_grids


def load_froc_data(exper_handler):
    output_dir = os.path.join(exper_handler.exper.config.root_dir, os.path.join(exper_handler.exper.output_dir,
                                                                                config_acdc.stats_path))
    abs_file_name = os.path.join(output_dir, "froc_data.npz")
    threshold_sensitivity, threshold_fp, slice_threshold_sensitivity, slice_threshold_fp, threshold_list = None, None, \
                                                                                                           None, None, None

    try:
        froc_data = np.load(abs_file_name)
        threshold_sensitivity = froc_data["threshold_sensitivity"]
        threshold_precision = froc_data["threshold_precision"]
        threshold_fp = froc_data["threshold_fp"]
        slice_threshold_sensitivity = froc_data["slice_threshold_sensitivity"]
        slice_threshold_fp = froc_data["slice_threshold_fp"]
        threshold_list = froc_data["threshold_list"]
        slice_threshold_precision = froc_data["slice_threshold_precision"]
        num_of_grids = froc_data["num_of_grids"]
        if "num_of_slices" in froc_data.keys():
            num_of_slices = froc_data["num_of_slices"]
        else:
            num_of_slices = 0
        print("INFO - Loaded froc data successfully fro, {}".format(abs_file_name))
    except Exception as e:
        print(traceback.format_exc())

    return threshold_sensitivity, threshold_precision, threshold_fp, slice_threshold_sensitivity, slice_threshold_precision, \
           slice_threshold_fp, threshold_list, num_of_slices, num_of_grids


class RegionDetectionEvaluation(object):

    def __init__(self, pred_probs, gt_labels, slice_probs_dict, slice_labels_dict, exper_handler,
                 nbr_of_thresholds=50, base_apex_labels=None, loc=0, range_threshold=None):

        self.pred_probs = pred_probs
        self.gt_labels = gt_labels
        self.slice_probs_dict = slice_probs_dict
        self.slice_labels_dict = slice_labels_dict
        self.exper_handler = exper_handler
        self.nbr_of_thresholds = nbr_of_thresholds
        self.base_apex_labels = base_apex_labels
        # loc can have 2 values: 0=non-base-apex    1=base-apex
        self.loc = loc
        self.range_threshold = range_threshold

    def generate_auc_curves(self):
        pass

