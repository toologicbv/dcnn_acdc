from config.config import config
import logging
import os
import copy
from datetime import datetime
import numpy as np
import glob
import torch
from pytz import timezone
from skimage import segmentation
from skimage import exposure
import matplotlib.pyplot as plt


def get_dice_diffs(diffs, num_of_slices, slice_stats, phase):
    """

    :param org_dice_slices:
    :param ref_dice_slices:
    :param phase: can be 0=ES or 1=ED
    :param num_of_slices: OrderedDict with key #slices, value frequency
    :param slice_stats: OrderedDict with key #slices, value mean dice increase between org - ref dice coeff.
    :return:
    """

    if np.any(diffs < 0.):
        print("------------------ WRONG negative diff between ref and org slice dice ---------------------")
        print("Ref dice phase={}".format(phase))
        print(diffs[phase, 1:])
        print("Original")
        print(diffs[phase, 1:])
        print("Diffs")
        print(diffs)
    slices = diffs.shape[0]
    if slices in slice_stats.keys():
        num_of_slices[slices] += 1
        slice_stats[slices] += diffs
    else:
        num_of_slices[slices] = 1
        slice_stats[slices] = diffs

    return num_of_slices, slice_stats


def generate_std_hist_corr_err(img_stds, labels, pred_labels, referral_threshold=None):
    """

    Note that for detecting the correct and incorrect classified pixels, we only need to look at the background
    class, because the incorrect pixels in this class are the summation of the other 3 classes (for ES and ED
    separately).


    :param img_stds: standard deviation for image pixels. One for ES and one for ED [classes, width, height, slices]
    :param labels: true labels with [8classes, width, height, #slices]
    :param pred_labels: [8classes, width, height, #slices]
    :param referral_threshold: indicating whether the u-maps is thresholded (then we omit the 0 values)
    :return:
    """
    half_classes = labels.shape[0] / 2
    num_of_slices = labels.shape[3]
    num_of_bins = 20
    slice_bin_edges = np.zeros((2, num_of_slices, num_of_bins + 1))
    # we store per phase, #slices, 2 hist arrays: one correct pixels, one error pixels
    slice_bin_values = np.zeros((2, num_of_slices, 2, num_of_bins))
    for img_slice in np.arange(num_of_slices):
        for phase in np.arange(2):
            # start with ES -> IMPORTANT, we skip the background class because we're only interested in the
            # errors and correct segmentations we made for the classes we report the dice/hd on
            union_errors = []
            union_correct = []
            for cls in np.arange(1, half_classes):
                class_idx = (phase * half_classes) + cls
                s_labels_ph = labels[class_idx, :, :, img_slice].flatten()
                s_pred_labels_ph = pred_labels[class_idx, :, :, img_slice].flatten()
                s_labels_ph = np.atleast_1d(s_labels_ph.astype(np.bool))
                s_pred_labels_ph = np.atleast_1d(s_pred_labels_ph.astype(np.bool))
                errors = np.argwhere((~s_pred_labels_ph & s_labels_ph) | (s_pred_labels_ph & ~s_labels_ph))
                correct = np.argwhere((s_pred_labels_ph & s_labels_ph))
                union_correct.extend(correct)
                union_errors.extend(errors)
            if img_stds.shape[0] == 8:
                # so img_stds is a u-map per class (first dim is 8)
                if phase == 0:
                    # compute mean stddev (over 4 classes)
                    slice_stds_ph = np.mean(img_stds[:half_classes, :, :, img_slice], axis=0).flatten()
                else:
                    # compute mean stddev (over 4 classes)
                    slice_stds_ph = np.mean(img_stds[half_classes:, :, :, img_slice], axis=0).flatten()
            else:
                # so img_stds is a tensor of [2, width, height, #slices], so basically not per class u-map
                slice_stds_ph = img_stds[phase, :, :, img_slice].flatten()

            # at this moment slice_probs_ph should be [half_classes, num_of_pixels]
            # and slice_stds_ph contains the pixel stddev for ES or ED [num_of_pixels]
            union_errors = np.array(union_errors)
            union_correct = np.array(union_correct)
            if union_errors.ndim > 1:
                err_std = slice_stds_ph[union_errors].flatten()
            else:
                err_std = None
            if union_correct.ndim > 1:
                pos_std = slice_stds_ph[union_correct].flatten()
            else:
                pos_std = None
            std_max = max(err_std.max() if err_std is not None else 0,
                          pos_std.max() if pos_std is not None else 0.)
            # print("# in err_std ", err_std.shape[0])
            if referral_threshold is not None:
                if std_max != 0:
                    std_min = referral_threshold
                else:
                    std_min = 0
            else:
                std_min = 0

            if pos_std is not None and np.count_nonzero(pos_std) != 0:
                # arr_stddev_cor, _, _ = plt.hist(pos_std, bins=xs)
                arr_stddev_cor, xs_cor = np.histogram(pos_std, bins=num_of_bins, range=(std_min, std_max))
                slice_bin_values[phase, img_slice, 0, :] = arr_stddev_cor
            if err_std is not None and np.count_nonzero(err_std) != 0:
                # arr_stddev_err, _, _ = plt.hist(err_std, bins=xs)
                arr_stddev_err, xs_cor = np.histogram(err_std, bins=num_of_bins, range=(std_min, std_max))
                slice_bin_values[phase, img_slice, 1, :] = arr_stddev_err
            if (pos_std is not None and np.count_nonzero(pos_std) != 0) or \
                    (err_std is not None and np.count_nonzero(err_std) != 0):
                slice_bin_edges[phase, img_slice] = xs_cor

    return slice_bin_edges, slice_bin_values


def generate_std_hist_corr_err_per_class(img_stds, labels, pred_labels, referral_threshold=None):
    """

    :param img_stds: standard deviation for image pixels. One for ES and one for ED [classes, width, height, slices]
    :param labels: true labels with [8classes, width, height, #slices]
    :param pred_labels: [8classes, width, height, #slices]
    :param referral_threshold: indicating whether the u-maps is thresholded (then we omit the 0 values)
    :return: slice_bin_edges: see shape below
             slice_bin_values: see shape below, NOTE: we store stddev densities for correct and incorrectly
             segmented voxels
    """
    half_classes = labels.shape[0] / 2
    num_of_slices = labels.shape[3]
    num_of_bins = 20
    slice_bin_edges = np.zeros((2, half_classes, num_of_slices, num_of_bins + 1))
    # we store per phase, #slices, 2 hist arrays: one correct pixels, one error pixels
    slice_bin_values = np.zeros((2, half_classes, num_of_slices, 2, num_of_bins))
    for img_slice in np.arange(num_of_slices):
        for phase in np.arange(2):
            # start with ES -> IMPORTANT, we skip the background class because we're only interested in the
            # errors and correct segmentations we made for the classes we report the dice/hd on
            for cls in np.arange(1, half_classes):
                class_idx = (phase * half_classes) + cls
                s_labels_ph = labels[class_idx, :, :, img_slice].flatten()
                s_pred_labels_ph = pred_labels[class_idx, :, :, img_slice].flatten()
                s_labels_ph = np.atleast_1d(s_labels_ph.astype(np.bool))
                s_pred_labels_ph = np.atleast_1d(s_pred_labels_ph.astype(np.bool))
                errors = np.argwhere((~s_pred_labels_ph & s_labels_ph) | (s_pred_labels_ph & ~s_labels_ph))
                correct = np.argwhere((s_pred_labels_ph & s_labels_ph))
                slice_stds_ph = img_stds[class_idx, :, :, img_slice].flatten()
                if np.count_nonzero(errors) != 0:
                    err_std = slice_stds_ph[errors].flatten()
                else:
                    err_std = None
                if np.count_nonzero(correct) != 0:
                    pos_std = slice_stds_ph[correct].flatten()
                else:
                    pos_std = None
                if not (err_std is None and pos_std is None):
                    std_max = max(err_std.max() if err_std is not None else 0,
                                  pos_std.max() if pos_std is not None else 0.)
                    if referral_threshold is not None:
                        # check whether std_max is zero, no values, then we make sure std_min is also zero
                        if std_max != 0:
                            std_min = referral_threshold
                        else:
                            std_min = 0.
                    else:
                        std_min = 0
                    xs = np.linspace(std_min, std_max, num_of_bins + 1)
                    if pos_std is not None and np.count_nonzero(pos_std) != 0:
                        # arr_stddev_cor, _, _ = plt.hist(pos_std, bins=xs)
                        arr_stddev_cor, xs_cor = np.histogram(pos_std, bins=num_of_bins, range=(std_min, std_max))
                        slice_bin_values[phase, cls, img_slice, 0, :] = arr_stddev_cor
                    if err_std is not None and np.count_nonzero(err_std) != 0:
                        # arr_stddev_err, _, _ = plt.hist(err_std, bins=xs)
                        arr_stddev_err, xs_cor = np.histogram(err_std, bins=num_of_bins, range=(std_min, std_max))
                        slice_bin_values[phase, cls, img_slice, 1, :] = arr_stddev_err
                    if pos_std is not None or err_std is not None:
                        slice_bin_edges[phase, cls, img_slice] = xs_cor

    return slice_bin_edges, slice_bin_values


def get_exper_objects(exper_handler, patient_id):

    # if not yet done, get raw uncertainty maps
    if exper_handler.u_maps is None:
        exper_handler.get_u_maps()

    umap_dir = os.path.join(exper_handler.exper.config.root_dir,
                                 os.path.join(exper_handler.exper.output_dir, config.u_map_dir))

    pred_labels_input_dir = os.path.join(exper_handler.exper.config.root_dir,
                                              os.path.join(exper_handler.exper.output_dir, config.pred_lbl_dir))
    fig_output_dir = os.path.join(exper_handler.exper.config.root_dir,
                                       os.path.join(exper_handler.exper.output_dir, config.figure_path))

    search_path = os.path.join(pred_labels_input_dir, patient_id + "_pred_labels_mc.npz")
    pred_labels = load_pred_labels(search_path)
    uncertainty_map = exper_handler.u_maps[patient_id]
    # in this case uncertainty_map has shape [2, 4, width, height, #slices] but we need [8, width, heiht, #slices]
    uncertainty_map = np.concatenate((uncertainty_map[0], uncertainty_map[1]))
    fig_path = os.path.join(fig_output_dir, patient_id)
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    return umap_dir, pred_labels_input_dir, fig_path, pred_labels, uncertainty_map


def prepare_referrals(image_name, referral_threshold, umap_dir, pred_labels_input_dir, aggregate_func="max"):
    # REFERRAL functionality.
    referral_threshold = str(referral_threshold).replace(".", "_")
    search_path = os.path.join(umap_dir,
                               image_name + "*" + "_filtered_cls_umaps_" + aggregate_func + referral_threshold + ".npz")
    filtered_cls_std_map = load_referral_umap(search_path, per_class=True)

    search_path = os.path.join(umap_dir,
                               image_name + "*" + "_filtered_umaps_" + aggregate_func + referral_threshold + ".npz")
    filtered_std_map = load_referral_umap(search_path, per_class=False)
    # filtered but without post-processing (filtered means referral threshold applied
    filtered_cls_std_map_wpostp = load_referral_umap(search_path, per_class=True, without_post_process=True)

    search_path = os.path.join(pred_labels_input_dir,
                               image_name + "_filtered_pred_labels_mc" + referral_threshold + ".npz")
    try:
        referral_pred_labels = load_pred_labels(search_path)
    except IOError:
        referral_pred_labels = None

    return filtered_cls_std_map, filtered_std_map, filtered_cls_std_map_wpostp, referral_pred_labels


def load_referral_umap(search_path, per_class=True, without_post_process=False):

    data = None
    files = glob.glob(search_path)
    if len(files) == 0:
        raise ImportError("ERROR - no referral u-map found in {}".format(search_path))
    fname = files[0]

    try:
        data = np.load(fname)
    except IOError:
        print("Unable to load uncertainty map from {}".format(fname))
    if per_class:
        if without_post_process:
            return data["filtered_raw_umap"]
        else:
            return data["filtered_cls_umap"]
    else:
        return data["filtered_umap"]


def load_pred_labels(search_path, get_slice_referral=False):

    files = glob.glob(search_path)
    if len(files) == 0:
        raise ImportError("ERROR - no pred labels found in {}".format(search_path))
    fname = files[0]

    try:
        data = np.load(fname)
    except IOError:
        print("Unable to load pred-labels from {}".format(fname))
        return None
    try:
        pred_labels = data["pred_labels"]
    except KeyError:
        # for backward compatibility
        pred_labels = data["filtered_pred_label"]

    if get_slice_referral:
        try:
            # boolean vector indicating which slices were referred
            referred_slices = data["referred_slices"]
        except KeyError:
            print("WARNING - common.load_pred_labels - Archive does not contain object referred_slices")

    if get_slice_referral:
        return pred_labels, referred_slices
    else:
        return pred_labels


def detect_seg_contours(img, lbls, cls_offset):
    #                 YELLOW        BLUE            RED
    gb_error_codes = [(1, 1, 0.), (0., 0.3, 0.7), (1, 0, 0)]
    rv_lbls = lbls[1 + cls_offset]
    myo_lbls = lbls[2 + cls_offset]
    lv_lbls = lbls[3 + cls_offset]
    img_rescaled = exposure.rescale_intensity(img)
    clean_border_rv = segmentation.clear_border(rv_lbls).astype(np.int)
    img_lbl = segmentation.mark_boundaries(img_rescaled, clean_border_rv, mode="outer",
                                           color=gb_error_codes[0], background_label=0)
    clean_border_myo = segmentation.clear_border(myo_lbls).astype(np.int)
    img_lbl = segmentation.mark_boundaries(img_lbl, clean_border_myo, mode="inner",
                                           color=gb_error_codes[1], background_label=0)
    clean_border_lv = segmentation.clear_border(lv_lbls).astype(np.int)
    img_lbl = segmentation.mark_boundaries(img_lbl, clean_border_lv, mode="inner",
                                           color=gb_error_codes[2], background_label=0)
    img_lbl = exposure.rescale_intensity(img_lbl, out_range=(0, 1))
    return img_lbl


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


def set_error_pixels(img_rgb, pred_labels, true_labels, cls_offset):
    """

    :param img_rgb: The original MRI scan extended to three channels (RGB)
    :param pred_labels: The predicted segmentation mask
    :param true_labels: The GT segmentation mask
    :param cls_offset:  0 or 4 depending on the phase ES resp. ED

    :return:
    """
    #                     MYO/LV:green    RV:YELLOW           MYO:BLUE    LV:RED
    rgb_error_codes = [[0, 190, 0], [204, 204, 0], [0, 120, 255], [255, 0, 0]]
    num_of_errors = np.zeros(4).astype(np.int)
    cls_range = np.arange(pred_labels.shape[0] // 2)
    # cls_range.sort()
    for cls in cls_range:
        error_idx = pred_labels[cls + cls_offset] != true_labels[cls + cls_offset]
        if cls == 2:
            error_idx_myo = copy.deepcopy(error_idx)
        elif cls == 3:
            error_idx_lv = copy.deepcopy(error_idx)
        if cls != 0:
            img_rgb[error_idx, :] = rgb_error_codes[cls]
        num_of_errors[cls] = np.count_nonzero(error_idx)
    error_mix = np.logical_and(error_idx_myo, error_idx_lv)
    img_rgb[error_mix, :] = rgb_error_codes[0]
    # making slightly mis-use of BG error count, which we overwrite with mixed errors Myo/LV
    num_of_errors[0] = np.count_nonzero(error_mix)

    return img_rgb, num_of_errors


def overlay_seg_mask(img_rgb, pred_labels, cls_offset):
    """

    :param img_rgb: The original MRI scan extended to three channels (RGB)
    :param pred_labels: The predicted segmentation mask
    :param cls_offset:  0 or 4 depending on the phase ES resp. ED

    :return:
    """
    #                       RV:YELLOW           MYO:BLUE    LV:RED
    rgb_error_codes = [[], [204, 204, 0], [0, 120, 255], [255, 0, 0]]
    for cls in np.arange(pred_labels.shape[0] // 2):
        seg_idx = pred_labels[cls + cls_offset] == 1
        if cls != 0:
            if np.count_nonzero(seg_idx) != 0:
                color = rgb_error_codes[cls]
                img_rgb[seg_idx, :] = color
    return img_rgb


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
        fh = logging.FileHandler(os.path.join(output_dir, exper.config.logger_filename))
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
    if exper.run_args.loss_function == "soft-dice":
        loss_func_name = "sdice"
    elif exper.run_args.loss_function == "brier":
        loss_func_name = "brier"
    elif exper.run_args.loss_function == "cross-entropy":
        loss_func_name = "entrpy"
    else:
        raise ValueError("ERROR - {} as loss functional is not supported!".format(exper.run_args.loss_function))

    if exper.run_args.model == "dcnn":
        exper_label = exper.run_args.model + "_f" + str(exper.run_args.fold_ids[0])
        exper_label += "_" + loss_func_name
        exper_label += "_" + str(exper.run_args.epochs / 1000) + "KE"

    elif exper.run_args.model[:7] == "dcnn_mc" or exper.run_args.model[:13] == "dcnn_hvsmr_mc":
        prob = "p" + str(exper.run_args.drop_prob).replace(".", "")
        prob += "_" + loss_func_name
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
