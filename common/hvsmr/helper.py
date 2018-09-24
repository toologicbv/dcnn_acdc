import os
import numpy as np
from common.hvsmr.config import config_hvsmr
from utils.hvsmr.exper_handler import HVSMRExperimentHandler


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
