import os
import numpy as np
from common.hvsmr.config import config_hvsmr
from utils.hvsmr.exper_handler import HVSMRExperimentHandler


def convert_to_multilabel(labels, bg_cls_idx=[0, 4]):
    """
    Assuming label_slice has shape [#classes, w, h, #slices].

    AND cls_idx=0 is background class.

    :param label_slice:
    :param bg_cls_idx: indices in dim0 that are background classes (0 and 4 for ACDC dataset)
    :return:
                converted volume where each slice contains all seg errors indicated by different scalar values
                has shape [2, w, h, #slices]
    """
    nclasses, w, h, num_slices = labels.shape
    multilabel_volume = np.zeros((2, w, h, num_slices))
    for slice_id in np.arange(num_slices):
        for cls_idx in np.arange(nclasses):
            # assuming first bg class idx = 0
            if cls_idx not in bg_cls_idx:
                lbl_slice[labels[cls_idx, :, :, slice_id] == 1] = cls_idx
            else:
                # when starting with ED we save the seg-error labels for ES
                if cls_idx == bg_cls_idx[1]:
                    multilabel_volume[0, :, :, slice_id] = lbl_slice
                lbl_slice = np.zeros((w, h))
        # ED slice seg-errors
        multilabel_volume[1, :, :, slice_id] = lbl_slice
    return multilabel_volume


def detect_seg_errors(labels, pred_labels, is_multi_class=False):

    """

    :param labels: if is_multi_class then [w, h] otherwise [num_of_classes, w, h]
    :param pred_labels: always [num_of_classes, w, h]
    :param is_multi_class: indicating that ground truth labels have shape [w, h]
    :return: [w, h] multiclass errors. so each voxels not equal to zero is an error. {1...nclass} indicates
                the FP-class the voxels belongs to. Possibly more than one class, but we only indicate the last one
                meaning, in the sequence of the classes
    """
    num_of_classes, w, h = pred_labels.shape
    errors = np.zeros((w, h))
    for cls in np.arange(num_of_classes):

        if is_multi_class:
            gt_labels_cls = (labels == cls).astype('int16')
        else:
            gt_labels_cls = labels[cls]
        errors_cls = gt_labels_cls != pred_labels[cls]
        errors[errors_cls] = cls

    return errors


def create_experiment(exper_id, verbose=False):

    log_dir = os.path.join(config_hvsmr.root_dir, config_hvsmr.log_root_path)
    exp_model_path = os.path.join(log_dir, exper_id)
    exper_handler = HVSMRExperimentHandler()
    exper_handler.load_experiment(exp_model_path, use_logfile=False)
    exper_handler.set_root_dir(config_hvsmr.root_dir)
    exper_args = exper_handler.exper.run_args
    info_str = "{} p={:.2f} fold={} loss={}".format(exper_args.model, exper_args.drop_prob, exper_args.fold_ids,
                                                    exper_args.loss_function)
    if verbose:
        print("INFO - Experimental details extracted:: " + info_str)
    return exper_handler


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

