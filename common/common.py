from config.config import config
import logging
import os
import copy
from datetime import datetime
import numpy as np

import torch
from pytz import timezone


def create_mask_uncertainties(labels):
    """
    We need a mask to filter out all uncertainties for which we don't predict a positive label.
    This makes referral more sparse.
    But the background class has a vast amount of true positives, hence we switch for this class the labels
    from 0 to 1 and vice versa so that the objects (LV, RV, myo) get the positive labels instead of the real bg.

    :param labels: contains predicted labels, shape can be 3 or 4 dim:
            [8classes, width, height] or with additional last dim of #slices. both should work
    :return: return a mask for ES and ED [2, width, height] plus optional #slices in last dim
    """
    # assuming labels has shape [8classes, width, height, with or without slices]
    num_of_classes = labels.shape[0] / 2
    mask_es, mask_ed = None, None
    # first flip background labels
    cp_labels = copy.deepcopy(labels)
    for phase in range(2):
        cls_offset = phase * num_of_classes
        # switch labels for bg class only
        # bg0 = cp_labels[0 + cls_offset] == 0
        # bg1 = cp_labels[0 + cls_offset] == 1
        # cp_labels[0 + cls_offset][bg0] = 1
        # cp_labels[0 + cls_offset][bg1] = 0

        if phase == 0:
            mask_es = np.sum(cp_labels[1:num_of_classes], axis=0) == 0
        else:
            mask_ed = np.sum(cp_labels[num_of_classes+1:], axis=0) == 0

    del cp_labels
    return np.concatenate((np.expand_dims(mask_es, axis=0), np.expand_dims(mask_ed, axis=0))).astype(np.bool)


def to_rgb1a(im):

    w, h = im.shape
    ret = np.zeros((w, h, 3), dtype=np.uint8)

    rgba = ((im - np.min(im)) / (np.max(im) - np.min(im))) * 255
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = rgba
    return ret


def set_error_pixels(img_rgb, pred_labels, true_labels, cls_offset, stddev=None, std_threshold=0.):
    """

    :param img_rgb: The original MRI scan extended to three channels (RGB)
    :param pred_labels: The predicted segmentation mask
    :param true_labels: The GT segmentation mask
    :param cls_offset:  0 or 4 depending on the phase ES resp. ED
    :param stddev: Uncertainty map for the image
    :param std_threshold: Threshold which can be used to simulate the "expert" mode. If uncertainty is higher
    than threshold we assume the EXPERT would adjust the pixel to the correct seg-label.
    :return:
    """
    #                       RV:YELLOW           MYO:BLUE    LV:RED
    rgb_error_codes = [[], [204, 204, 0], [0, 120, 255], [255, 0, 0]]
    num_of_errors = np.zeros(4).astype(np.int)

    for cls in np.arange(pred_labels.shape[0] // 2):
        error_idx = pred_labels[cls + cls_offset] != true_labels[cls + cls_offset]
        errors_after = None
        if std_threshold > 0.:
            if cls == 1:
                # filter for RV predictions with high stddev
                # pred_cls_labels_filtered = np.copy(pred_labels[cls + cls_offset])
                referral_idx = stddev[cls + cls_offset] > std_threshold
                # IMPORTANT: HERE we set the pixels of high uncertainty to the EXPERT ground truth!!!
                pred_labels[cls + cls_offset][referral_idx] = true_labels[cls + cls_offset][referral_idx]
                errors_after = true_labels[cls + cls_offset] != pred_labels[cls + cls_offset]
                print("WARNING - RV-errors using {:.2f} before/after {} / {}".format(std_threshold,
                                                                                     np.count_nonzero(error_idx),
                                                                                     np.count_nonzero(errors_after)))
        if errors_after is not None:
            if cls != 0:
                img_rgb[errors_after, :] = rgb_error_codes[cls]
            num_of_errors[cls] = np.count_nonzero(errors_after)
        else:
            if cls != 0:
                img_rgb[error_idx, :] = rgb_error_codes[cls]
            num_of_errors[cls] = np.count_nonzero(error_idx)
    return img_rgb, num_of_errors


def datestr(withyear=True):
    jetzt = datetime.now(timezone('Europe/Berlin')).strftime('%Y_%m_%d_%H_%M_%S.%f')[:-10]
    jetzt = jetzt.replace(" ", "_")
    if not withyear:
        jetzt = jetzt[5:]
    return jetzt


def create_logger(exper=None, file_handler=False, output_dir=None):
    # create logger
    if exper is None and output_dir is None:
        raise ValueError("Parameter -experiment- and -output_dir- cannot be both equal to None")
    logger = logging.getLogger('experiment logger')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    if file_handler:
        if output_dir is None:
            output_dir = exper.output_dir
        fh = logging.FileHandler(os.path.join(output_dir, config.logger_filename))
        # fh.setLevel(logging.INFO)
        fh.setLevel(logging.DEBUG)
        formatter_fh = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter_fh)
        logger.addHandler(fh)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers

    formatter_ch = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(formatter_ch)
    # add the handlers to the logger
    logger.addHandler(ch)

    return logger


def create_exper_label(exper):

    # exper_label = exper.run_args.model + exper.run_args.version + "_" + str(exper.run_args.epochs) + "E"
    if exper.run_args.model == "dcnn":
        exper_label = exper.run_args.model + "_f" + str(exper.run_args.fold_ids[0]) + "_" + \
                      str(exper.run_args.epochs / 1000) + "KE"
    elif exper.run_args.model == "dcnn_mc":
        prob = "p" + str(exper.run_args.drop_prob).replace(".", "")
        if exper.run_args.loss_function == "brier":
            prob += "_" + exper.run_args.loss_function
        exper_label = exper.run_args.model + "_f" + str(exper.run_args.fold_ids[0]) + \
                       prob + "_" + str(exper.run_args.epochs / 1000) + "KE"
    else:
        raise ValueError("ERROR - model name {} is not supported.".format(exper.run_args.model))

    return exper_label


def logsumexp(a):
    a_max = a.max(axis=0)
    return np.log(np.sum(np.exp(a - a_max), axis=0)) + a_max


def setSeed(seed=None, use_cuda=False):
    if seed is None:
        seed = 4325
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if use_cuda:
        torch.backends.cudnn.enabled = True

    np.random.seed(seed)


def compute_batch_freq_outliers(run_args, outlier_dataset, dataset):
    # the size of the outlier dataset can never exceed the size of the original training set. In the worst
    # case it's equal to 1 (all slices are outliers!). Current heuristic for how often do we make
    # training batches from the outlier set: epochs / outlier_freq
    # So e.g. if there're twice as much training slices as outlier slices we take every second time
    # slices from outliers. NOTE: we're not removing the outliers from the original dataset!!!
    # which means, they have an extra chance of being trained on with the danger of overfitting to these
    # slices. We'll see what happens.
    outlier_freq = int(run_args.epochs / (run_args.epochs *
                                                          (float(outlier_dataset.num_of_slices) / float(
                                                              dataset.train_num_slices))))
    return outlier_freq
