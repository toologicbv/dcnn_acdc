import numpy as np
from scipy.stats import gaussian_kde
import os
import dill
import sys
from tqdm import tqdm
from collections import OrderedDict
from config.config import config
from utils.dice_metric import dice_coefficient
from utils.medpy_metrics import hd
if "/home/jogi/.local/lib/python2.7/site-packages" in sys.path:
    sys.path.remove("/home/jogi/.local/lib/python2.7/site-packages")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from common.common import datestr
import copy
from scipy.stats import wilcoxon, ttest_ind, mannwhitneyu


def set_error_pixels(img_err, pred_labels, true_labels, cls_offset, stddev=None, std_threshold=0.):

    #                       RV:RED           MYO:YELLOW    LV:GREEN
    rgb_error_codes = [[], [255, 0, 0], [204, 204, 0], [0, 204, 0]]
    num_of_errors = np.zeros(4).astype(np.int)

    for cls in np.arange(pred_labels.shape[0] // 2):
        error_idx = pred_labels[cls + cls_offset] != true_labels[cls + cls_offset]
        errors_after = None
        if std_threshold > 0.:
            if cls == 1:
                # filter for RV predictions with high stddev
                # pred_cls_labels_filtered = np.copy(pred_labels[cls + cls_offset])
                referral_idx = stddev[cls + cls_offset] > std_threshold
                pred_labels[cls + cls_offset][referral_idx] = 0
                errors_after = true_labels[cls + cls_offset] != pred_labels[cls + cls_offset]
                print("WARNING - RV-errors using {:.2f} before/after {} / {}".format(std_threshold,
                                                                                     np.count_nonzero(error_idx),
                                                                                     np.count_nonzero(errors_after)))
        if errors_after is not None:
            if cls != 0:
                img_err[errors_after, :] = rgb_error_codes[cls]
            num_of_errors[cls] = np.count_nonzero(errors_after)
        else:
            if cls != 0:
                img_err[error_idx, :] = rgb_error_codes[cls]
            num_of_errors[cls] = np.count_nonzero(error_idx)
    return img_err, num_of_errors


def to_rgb1a(im):

    w, h = im.shape
    ret = np.zeros((w, h, 3), dtype=np.uint8)

    rgba = ((im - np.min(im)) / (np.max(im) - np.min(im))) * 255
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = rgba
    return ret


def get_accuracies_all(pred_labels, true_labels):

    total = pred_labels.flatten().shape[0]
    tp = np.count_nonzero(pred_labels & true_labels)
    fn = np.count_nonzero(~pred_labels & true_labels)
    fp = np.count_nonzero(pred_labels & ~true_labels)
    tn = np.count_nonzero(~pred_labels & ~true_labels)
    return total, tp, fn, fp, tn


def compute_p_values(err_group, corr_group, phase, stats, uncertainty_type="std"):

    u_type = uncertainty_type
    if err_group.shape[0] > 0 and corr_group.shape[0] > 0:
        _, p_value_ttest = ttest_ind(err_group, corr_group)
        _, p_value_mannwhitneyu = mannwhitneyu(err_group, corr_group)
    else:
        p_value_ttest = np.NaN
        p_value_mannwhitneyu = np.NaN

    if phase == 0:
        dict_key = "es_pvalue_ttest_" + u_type
        stats[dict_key] = p_value_ttest
        dict_key = "es_pvalue_mwhitu_" + u_type
        stats[dict_key] = p_value_mannwhitneyu
        # can't compute Wilcoxon p-value because the two groups have very different sample sizes
        # _, stats["es_pvalue_wilc_" + u_type] = wilcoxon(err_group, corr_group)
    else:
        stats["ed_pvalue_ttest_" + u_type] = p_value_ttest
        stats["ed_pvalue_mwhitu_" + u_type] = p_value_mannwhitneyu
        # _, stats["ed_pvalue_wilc_" + u_type] = wilcoxon(err_group, corr_group)


def get_mean_pred_per_slice(img_slice, img_probs, img_stds, img_balds, labels, pred_labels, half_classes, stats):
    """

    Note that for detecting the correct and incorrect classified pixels, we only need to look at the background
    class, because the incorrect pixels in this class are the summation of the other 3 classes (for ES and ED
    separately).

    :param img_probs: predicted probabilities for image pixels for the 8 classes [classes, width, height, slices]
            0-3: ES and 4-7 ED
    :param img_stds: standard deviation for image pixels. One for ES and one for ED [classes, width, height, slices]
    :param labels: true labels with [num_of_classes, width, height, slices]
    :param pred_labels: [num_of_classes, width, height, slices]
    :param half_classes: in this case 4
    :param stats: object that holds the dictionaries that capture results of test run
    :return:
    """

    for phase in np.arange(2):
        # start with ES
        bg_class_idx = phase * half_classes
        s_labels_ph = labels[bg_class_idx, :, :, img_slice].flatten()
        s_pred_labels_ph = pred_labels[bg_class_idx, :, :, img_slice].flatten()
        s_labels_ph = np.atleast_1d(s_labels_ph.astype(np.bool))
        s_pred_labels_ph = np.atleast_1d(s_pred_labels_ph.astype(np.bool))
        errors = np.argwhere((~s_pred_labels_ph & s_labels_ph) | (s_pred_labels_ph & ~s_labels_ph))
        correct = np.argwhere((s_pred_labels_ph & s_labels_ph))

        if phase == 0:
            # compute mean stddev (over 4 classes)
            slice_stds_ph = np.mean(img_stds[:half_classes, :, :, img_slice], axis=0).flatten()
            slice_probs_ph = img_probs[:half_classes, :, :, img_slice]
            slice_bald_ph = img_balds[phase, :, :, img_slice].flatten()
        else:
            # compute mean stddev (over 4 classes)
            slice_stds_ph = np.mean(img_stds[half_classes:, :, :, img_slice], axis=0).flatten()
            slice_bald_ph = img_balds[phase, :, :, img_slice].flatten()
            slice_probs_ph = img_probs[half_classes:, :, :, img_slice]
        slice_probs_ph = np.reshape(slice_probs_ph, (slice_probs_ph.shape[0],
                                                     slice_probs_ph.shape[1] *
                                                     slice_probs_ph.shape[2]))
        # at this moment slice_probs_ph should be [half_classes, num_of_pixels]
        # and slice_stds_ph contains the pixel stddev for ES or ED [num_of_pixels]

        err_probs = slice_probs_ph.T[errors].flatten()
        err_std = slice_stds_ph[errors].flatten()
        err_bald = slice_bald_ph[errors].flatten()
        pos_probs = slice_probs_ph.T[correct].flatten()
        pos_std = slice_stds_ph[correct].flatten()
        pos_bald = slice_bald_ph[correct].flatten()
        if phase == 0:
            stats["es_mean_err_p"] = err_probs
            stats["es_mean_cor_p"] = pos_probs
            stats["es_mean_err_std"] = err_std
            stats["es_mean_cor_std"] = pos_std
            stats["es_err_bald"] = err_bald
            stats["es_cor_bald"] = pos_bald
        else:
            stats["ed_mean_err_p"] = err_probs
            stats["ed_mean_cor_p"] = pos_probs
            stats["ed_mean_err_std"] = err_std
            stats["ed_mean_cor_std"] = pos_std
            stats["ed_err_bald"] = err_bald
            stats["ed_cor_bald"] = pos_bald

        # compute hypothesis test p-values for group (1) correct classified versus (2) incorrect classified
        # using the uncertainty measures for both groups (TO DO: group (1) DOES NOT INCORPORATE "true negatives"
        # is that CORRECT?
        # (1) for stddev
        compute_p_values(err_std, pos_std, phase, stats, uncertainty_type="std")
        # (2) BALD values
        compute_p_values(err_bald, pos_bald, phase, stats, uncertainty_type="bald")


def get_img_stats_per_slice_class(img_slice_probs, img_slice, phase, half_classes,
                                  p_err_std, p_corr_std, p_err_prob, p_corr_prob):

    for cls in np.arange(half_classes):

        if img_slice == 0:
            if phase == 0:
                p_err_prob = np.array(img_slice_probs["es_err_p"][cls])
                p_corr_prob = np.array(img_slice_probs["es_cor_p"][cls])
                p_err_std = np.array(img_slice_probs["es_err_std"][cls])
                p_corr_std = np.array(img_slice_probs["es_cor_std"][cls])
            else:
                p_err_prob = np.array(img_slice_probs["ed_err_p"][cls])
                p_corr_prob = np.array(img_slice_probs["ed_cor_p"][cls])
                p_err_std = np.array(img_slice_probs["ed_err_std"][cls])
                p_corr_std = np.array(img_slice_probs["ed_cor_std"][cls])
        else:
            if phase == 0:
                p_err_prob = np.concatenate((p_err_prob, np.array(img_slice_probs["es_err_p"][cls])))
                p_corr_prob = np.concatenate((p_corr_prob, np.array(img_slice_probs["es_cor_p"][cls])))
                p_err_std = np.concatenate((p_err_std, np.array(img_slice_probs["es_err_std"][cls])))
                p_corr_std = np.concatenate((p_corr_std, np.array(img_slice_probs["es_cor_std"][cls])))
            else:
                p_err_prob = np.concatenate((p_err_prob, np.array(img_slice_probs["ed_err_p"][cls])))
                p_corr_prob = np.concatenate((p_corr_prob, np.array(img_slice_probs["ed_cor_p"][cls])))
                p_err_std = np.concatenate((p_err_std, np.array(img_slice_probs["ed_err_std"][cls])))
                p_corr_std = np.concatenate((p_corr_std, np.array(img_slice_probs["ed_cor_std"][cls])))

    return p_err_std, p_corr_std, p_err_prob, p_corr_prob


def get_img_stats_per_slice(img_slice_probs, img_slice, phase, p_err_std, p_corr_std, p_err_prob, p_corr_prob):

    if img_slice == 0:
        if phase == 0:
            p_err_prob = img_slice_probs["es_mean_err_p"]
            p_corr_prob = img_slice_probs["es_mean_cor_p"]
            p_err_std = img_slice_probs["es_mean_err_std"]
            p_corr_std = img_slice_probs["es_mean_cor_std"]
        else:
            p_err_prob = img_slice_probs["ed_mean_err_p"]
            p_corr_prob = img_slice_probs["ed_mean_cor_p"]
            p_err_std = img_slice_probs["ed_mean_err_std"]
            p_corr_std = img_slice_probs["ed_mean_cor_std"]
    else:
        if phase == 0:
            p_err_prob = np.concatenate((p_err_prob, np.array(img_slice_probs["es_mean_err_p"])))
            p_corr_prob = np.concatenate((p_corr_prob, np.array(img_slice_probs["es_mean_cor_p"])))
            p_err_std = np.concatenate((p_err_std, np.array(img_slice_probs["es_mean_err_std"])))
            p_corr_std = np.concatenate((p_corr_std, np.array(img_slice_probs["es_mean_cor_std"])))
        else:
            p_err_prob = np.concatenate((p_err_prob, np.array(img_slice_probs["ed_mean_err_p"])))
            p_corr_prob = np.concatenate((p_corr_prob, np.array(img_slice_probs["ed_mean_cor_p"])))
            p_err_std = np.concatenate((p_err_std, np.array(img_slice_probs["ed_mean_err_std"])))
            p_corr_std = np.concatenate((p_corr_std, np.array(img_slice_probs["ed_mean_cor_std"])))

    return p_err_std, p_corr_std, p_err_prob, p_corr_prob


class TestResults(object):

    def __init__(self, exper, use_dropout=False, mc_samples=1):
        """
            numpy arrays in self.pred_probs have shape [slices, classes, width, height]
                            self.pred_labels [classes, width, height, slices]
                            self.images [classes, width, height, slices]
                            self.pred_probs: [mc_samples, classes, width, height, slices]
        """
        self.use_dropout = use_dropout
        self.mc_samples = mc_samples

        self.images = []
        self.image_ids = []
        self.image_names = []
        self.labels = []
        self.pred_labels = []
        self.mc_pred_probs = []
        self.stddev_maps = []
        self.seg_errors = []
        self.bald_maps = []
        self.uncertainty_stats = []
        self.test_accuracy = []
        self.test_hd = []
        self.test_accuracy_slices = []
        self.test_hd_slices = []
        self.dice_results = None
        self.hd_results = None
        self.num_of_samples = []
        # for each image we are using during testing we append ONE LIST, which contains for each image slice
        # and ordered dictionary with the following keys a) es_err b) es_corr c) ed_err d) ed_corr
        # this object is used for the detailed analysis of the uncertainties per image-slice, distinguishing
        # the correct and wrongs labeled pixels for the two cardiac phase ES and ED.
        self.image_probs_categorized = []

        # set path in order to save results and figures
        self.fig_output_dir = os.path.join(exper.config.root_dir,
                                           os.path.join(exper.output_dir, exper.config.figure_path))
        self.save_output_dir = os.path.join(exper.config.root_dir,
                                            os.path.join(exper.output_dir, exper.config.stats_path))

    def add_results(self, batch_image, batch_labels, image_id, pred_labels, b_predictions, stddev_map,
                    test_accuracy, test_hd, seg_errors, store_all=False, bald_maps=None, uncertainty_stats=None,
                    test_hd_slices=None, test_accuracy_slices=None, image_name=None,
                    repeated_run=False):
        """

        :param batch_image: [2, width, height, slices]
        :param batch_labels:
        :param pred_labels:
        :param b_predictions:
        :param seg_errors: [#slices, #classes(8)]
        :param stddev_map: [#classes (8), width, height, slices]
        :param bald_maps: [2, width, height, slices]
        :param uncertainty_stats: dictionary with keys "bald", "stddev" and "u_threshold"
                                  the first 2 contain numpy arrays of shape [2, half classes (4), #slices]
        :param test_accuracy:
        :param test_hd:
        :param repeated_run: when we use multiple checkpoints aka models during testing, we don't want to
               save the image details everytime we run the same model ensemble on the test set although we want
               to average the final result over the ensemble runs (hence we save the performance measures but not
               the image details.

        :return:
        """
        # get rid off padding around image
        batch_image = batch_image[:, config.pad_size:-config.pad_size, config.pad_size:-config.pad_size, :]
        # during ensemble testing we only want to store the images and ground truth labels once
        if store_all:
            if not repeated_run:
                self.images.append(batch_image)
                self.labels.append(batch_labels)
        if store_all:
            self.pred_labels.append(pred_labels)
            self.mc_pred_probs.append(b_predictions)
            self.stddev_maps.append(stddev_map)
            self.bald_maps.append(bald_maps)
        # when using multiple models for evaluation during testing, we only want to store the image IDs and
        # names once.
        if not repeated_run:
            self.image_ids.append(image_id)
            self.image_names.append(image_name)
            self.num_of_samples.append(b_predictions.shape[0])

        self.test_accuracy.append(test_accuracy)
        self.test_hd.append(test_hd)
        if test_hd_slices is not None:
            self.test_hd_slices.append(test_hd_slices)
        if test_accuracy_slices is not None:
            self.test_accuracy_slices.append(test_accuracy_slices)
        # segmentation errors is a numpy error with size [#slices, #classes (8)]
        self.seg_errors.append(seg_errors)
        if self.num_of_samples[-1] > 1:
            self.uncertainty_stats.append(uncertainty_stats)

    def compute_mean_stats(self):

        N = len(self.test_accuracy)
        # print("#Images {}".format(N))
        if len(self.test_accuracy) == 0 or len(self.test_hd) == 0:
            raise ValueError("ERROR - there's no data that could be used to compute statistics!")

        columns_dice = self.test_accuracy[0].shape[0]
        columns_hd = self.test_hd[0].shape[0]
        mean_dice = np.empty((0, columns_dice))
        mean_hd = np.empty((0, columns_hd))

        for img_idx in np.arange(N):
            # test_accuracy and test_hd are a vectors of 8 values, 0-3: ES, 4:7: ED
            dice = self.test_accuracy[img_idx]
            hausd = self.test_hd[img_idx]
            mean_dice = np.vstack([mean_dice, dice]) if mean_dice.size else dice
            mean_hd = np.vstack([mean_hd, hausd]) if mean_hd.size else hausd

        # so we stack the image results and should end up with a matrix [#images, 8] for dice and hd

        if N > 1:
            self.dice_results = np.array([np.mean(mean_dice, axis=0), np.std(mean_dice, axis=0)])
            self.hd_results = np.array([np.mean(mean_hd, axis=0), np.std(mean_hd, axis=0)])
        else:
            # only one image tested, there is no mean or stddev
            self.dice_results = np.array([mean_dice, np.zeros(mean_dice.shape[0])])
            self.hd_results = np.array([mean_hd, np.zeros(mean_hd.shape[0])])

    def generate_all_statistics(self):
        for image_num in np.arange(self.N):
            self.generate_slice_statistics(image_num=image_num)

    def generate_slice_statistics(self, image_num=0):
        """

        For a detailed analysis of the likelihoods p(y* | x*, theta) we split the predicted probabilities
        into (1) ED and ES (2) slices (3) segmentation classes.

        :param image_num: index of the test image we want to process
        :return:
        """
        # Reset object
        if len(self.image_probs_categorized) != 0:
            try:
                if image_num in self.image_probs_categorized[image_num]:
                    del self.image_probs_categorized[image_num]
                    insert_idx = image_num
                else:
                    del self.image_probs_categorized[0]
                    insert_idx = 0
            except IndexError:
                insert_idx = image_num
        else:
            insert_idx = image_num

        pred_labels = self.pred_labels[image_num]
        labels = self.labels[image_num]
        # this tensors contains all probs for the samples, but we will work with the means
        mc_pred_probs = self.mc_pred_probs[image_num]
        mean_pred_probs = np.mean(mc_pred_probs, axis=0)
        std_pred_probs = self.stddev_maps[image_num]
        bald_maps = self.bald_maps[image_num]
        num_slices = labels.shape[3]
        num_classes = labels.shape[0]
        half_classes = num_classes / 2
        probs_per_img_slice = []
        for slice in np.arange(num_slices):
            # object that holds probabilities per image slice
            probs_per_cls = {"es_err_p": OrderedDict((i, []) for i in np.arange(half_classes)),
                             "es_cor_p": OrderedDict((i, []) for i in np.arange(half_classes)),
                             "ed_err_p": OrderedDict((i, []) for i in np.arange(half_classes)),
                             "ed_cor_p": OrderedDict((i, []) for i in np.arange(half_classes)),
                             "es_err_std": OrderedDict((i, []) for i in np.arange(half_classes)),
                             "es_cor_std": OrderedDict((i, []) for i in np.arange(half_classes)),
                             "ed_err_std": OrderedDict((i, []) for i in np.arange(half_classes)),
                             "ed_cor_std": OrderedDict((i, []) for i in np.arange(half_classes)),
                             "es_mean_err_p": np.zeros(half_classes), "es_mean_cor_p": np.zeros(half_classes),
                             "ed_mean_err_p": np.zeros(half_classes), "ed_mean_cor_p": np.zeros(half_classes),
                             "es_mean_err_std": np.zeros(half_classes), "es_mean_cor_std": np.zeros(half_classes),
                             "ed_mean_err_std": np.zeros(half_classes), "ed_mean_cor_std": np.zeros(half_classes),
                             "es_err_bald": np.zeros(half_classes),
                             "es_cor_bald": np.zeros(half_classes),
                             "ed_err_bald": np.zeros(half_classes),
                             "ed_cor_bald": np.zeros(half_classes),
                             "es_pvalue_ttest_std": 0, "es_pvalue_mwhitu_std": 0,
                             "ed_pvalue_ttest_std": 0, "ed_pvalue_mwhitu_std": 0,
                             "es_pvalue_ttest_bald": 0, "es_pvalue_mwhitu_bald": 0,
                             "ed_pvalue_ttest_bald": 0, "ed_pvalue_mwhitu_bald": 0,
                             }

            # mean per class
            get_mean_pred_per_slice(slice, mean_pred_probs, std_pred_probs, bald_maps, labels, pred_labels,
                                    half_classes, probs_per_cls)

            for cls in np.arange(num_classes):
                s_labels = labels[cls, :, :, slice].flatten()
                s_pred_labels = pred_labels[cls, :, :, slice].flatten()
                s_slice_cls_probs = mc_pred_probs[:, cls, :, :, slice]
                # we don't need to calculate stddev again, we did that already in experiment.test() method
                s_slice_cls_probs_std = std_pred_probs[cls, :, :, slice].flatten()
                s_slice_cls_probs = np.reshape(s_slice_cls_probs, (s_slice_cls_probs.shape[0],
                                                                   s_slice_cls_probs.shape[1] *
                                                                   s_slice_cls_probs.shape[2]))

                # get the indices for the errors and the correct classified pixels in this slice/per class
                s_pred_labels = np.atleast_1d(s_pred_labels.astype(np.bool))
                s_labels = np.atleast_1d(s_labels.astype(np.bool))

                # errors: fn + fp
                # correct: tp
                errors = np.argwhere((~s_pred_labels & s_labels) | (s_pred_labels & ~s_labels))
                # correct = np.invert(errors)
                correct = np.argwhere((s_pred_labels & s_labels))
                # print("cls {} total/tp/fn/fp/tn {} / {} / {} / {} / {}".format(cls, total_l, tp, fn, fp, tn))
                # print("\t tp / fn+fp {} {}".format(correct.shape[0], errors.shape[0]))
                err_probs = s_slice_cls_probs.T[errors].flatten()
                err_std = s_slice_cls_probs_std[errors].flatten()
                pos_probs = s_slice_cls_probs.T[correct].flatten()
                pos_std = s_slice_cls_probs_std[correct].flatten()
                if cls < half_classes:
                    if cls not in probs_per_cls["es_err_p"]:
                        probs_per_cls["es_err_p"][cls] = list(err_probs)
                        probs_per_cls["es_err_std"][cls] = list(err_std)
                    else:
                        probs_per_cls["es_err_p"][cls].extend(list(err_probs))
                        probs_per_cls["es_err_std"][cls].extend(list(err_std))
                    if cls not in probs_per_cls["es_cor_p"]:
                        probs_per_cls["es_cor_p"][cls] = list(pos_probs)
                        probs_per_cls["es_cor_std"][cls] = list(pos_std)
                    else:
                        probs_per_cls["es_cor_p"][cls].extend(list(pos_probs))
                        probs_per_cls["es_cor_std"][cls].extend(list(pos_std))
                else:
                    # print("ED: correct/error {}/{}".format(pos_probs.shape[0], err_probs.shape[0]))
                    if cls - half_classes not in probs_per_cls["ed_err_p"]:
                        probs_per_cls["ed_err_p"][cls - half_classes] = list(err_probs)
                        probs_per_cls["ed_err_std"][cls - half_classes] = list(err_std)
                    else:
                        probs_per_cls["ed_err_p"][cls - half_classes].extend(list(err_probs))
                        probs_per_cls["ed_err_std"][cls - half_classes].extend(list(err_std))
                    if cls - half_classes not in probs_per_cls["ed_cor_p"]:
                        probs_per_cls["ed_cor_p"][cls - half_classes] = list(pos_probs)
                        probs_per_cls["ed_cor_std"][cls - half_classes] = list(pos_std)
                    else:
                        probs_per_cls["ed_cor_p"][cls - half_classes].extend(list(pos_probs))
                        probs_per_cls["ed_cor_std"][cls - half_classes].extend(list(pos_std))
            # finally store probs_per_cls for this slice
            probs_per_img_slice.append(probs_per_cls)

        self.image_probs_categorized.insert(insert_idx, probs_per_img_slice)

    def visualize_uncertainty_stats(self, image_num=0, width=16, height=10, info_type="uncertainty",
                                    use_class_stats=False, do_save=False, fig_name=None,
                                    model_name="", do_show=False):

        if info_type not in ["probs", "uncertainty"]:
            raise ValueError("Parameter info_type must be probs or uncertainty and not {}".format(info_type))
        try:
            image_num = self.image_ids.index(image_num)
        except ValueError:
            print("WARNING - Can't find image with index {} in "
                  "test_results.image_ids. Discarding!".format(image_num))
        image_probs = self.image_probs_categorized[image_num]
        label = self.labels[image_num]
        num_of_classes = label.shape[0]
        half_classes = num_of_classes / 2
        mc_samples = self.mc_pred_probs[image_num].shape[0]
        num_of_slices = label.shape[3]
        num_of_subplots = 2
        columns = 2
        counter = 1
        kde = True
        if not kde:
            num_of_subplots = 4
            columns = 2

        fig = plt.figure(figsize=(width, height))
        fig.suptitle("Densities of Uncertainties - model {}".format(model_name) , **config.title_font_medium)

        for phase in np.arange(2):
            if phase == 0:
                str_phase = "ES"
            else:
                str_phase = "ED"

            p_err_std, p_corr_std, p_err_prob, p_corr_prob = None, None, None, None
            for img_slice in np.arange(num_of_slices):

                img_slice_probs = image_probs[img_slice]
                if use_class_stats:
                    p_err_std, p_corr_std, p_err_prob, p_corr_prob = \
                        get_img_stats_per_slice_class(img_slice_probs, img_slice, phase, half_classes, p_err_std,
                                                      p_corr_std, p_err_prob, p_corr_prob)
                else:
                    p_err_std, p_corr_std, p_err_prob, p_corr_prob = \
                        get_img_stats_per_slice(img_slice_probs, img_slice, phase, p_err_std, p_corr_std, p_err_prob,
                                            p_corr_prob)

            print("{} correct/error(fp+fn) {} / {}".format(str_phase, p_corr_std.shape, p_err_std.shape))
            ax2 = plt.subplot(num_of_subplots, columns, counter)
            if p_err_std is not None:
                ax2b = ax2.twinx()
                if kde:
                    if info_type == "uncertainty":

                        density_err = gaussian_kde(p_err_std)
                        xs_err = np.linspace(0, p_err_std.max(), 200)
                        density_err.covariance_factor = lambda: .25
                        density_err._compute_covariance()
                        p_err = ax2b.fill_between(xs_err, density_err(xs_err), label="$\sigma_{pred(fp+fn)}$",
                                         color="b", alpha=0.2)
                    else:
                        density_err = gaussian_kde(p_err_prob)
                        xs_err = np.linspace(0, p_err_prob.max(), 200)
                        density_err.covariance_factor = lambda: .25
                        density_err._compute_covariance()
                        p_err = ax2b.fill_between(xs_err, density_err(xs_err), label="$p_{pred(fp+fn)}(c|x)$",
                                         color="b", alpha=0.2)
                else:
                    xs_err = np.linspace(0, p_err_std.max(), 200)
                    p_err = ax2b.hist(p_err_std, bins=xs_err, label=r"$\sigma_{pred(fp+fn)}$", color="b", alpha=0.2)

            if p_corr_std is not None:
                if kde:
                    if info_type == "uncertainty":
                        density_cor = gaussian_kde(p_corr_std)
                        xs_cor = np.linspace(0, p_corr_std.max(), 200)
                        density_cor.covariance_factor = lambda: .25
                        density_cor._compute_covariance()
                        p_corr = ax2.fill_between(xs_cor, density_cor(xs_cor), label=r"$\sigma_{pred(tp)}$",
                                         color="g", alpha=0.2)
                    else:
                        density_cor = gaussian_kde(p_corr_prob)
                        xs_cor = np.linspace(0, p_corr_prob.max(), 200)
                        density_cor.covariance_factor = lambda: .25
                        density_cor._compute_covariance()
                        p_corr = ax2.fill_between(xs_cor, density_cor(xs_cor), label="$p_{pred(tp)}(c|x)$",
                                         color="g", alpha=0.2)
                else:
                    counter += 1
                    ax3 = plt.subplot(num_of_subplots, columns, counter)
                    xs_cor = np.linspace(0, p_corr_std.max(), 200)
                    p_corr = ax3.hist(p_corr_std, bins=xs_cor, label=r"$\sigma_{pred(tp)}$", color="g", alpha=0.2)
                    ax3.set_ylabel("density")
                    ax3.set_xlabel("model uncertainty")
                    ax3.legend(loc="best")

                if info_type == "uncertainty":
                    ax2.set_xlabel("model uncertainty", **config.axis_font)
                else:
                    ax2.set_xlabel(r"softmax $p(c|x)$", **config.axis_font)
                ax2.set_title("{}: all classes ({}/{})".format(str_phase, p_corr_std.shape[0],
                                                               p_err_std.shape[0]),
                              **config.title_font_medium)
                plots = [p_corr, p_err]
                ax2.legend(plots, [l.get_label() for l in plots], loc="best", prop={'size': 16})
                # ax2.legend(loc="best")
                ax2.set_ylabel("density", **config.axis_font)

            counter += 1
        # fig.tight_layout()
        if do_save:
            fig_out_dir = self._create_figure_dir(image_num)
            if fig_name is None:
                fig_name = info_type + "_densities_mc" + str(mc_samples) \
                           + "_" + str(use_class_stats)
            fig_name = os.path.join(fig_out_dir, fig_name + ".png")

            plt.savefig(fig_name, bbox_inches='tight')
            print("INFO - Successfully saved fig %s" % fig_name)
        if do_show:
            plt.show()
        plt.close()

    def visualize_test_slices(self, image_num=0, width=8, height=6, slice_range=None, do_save=False,
                              fig_name=None):
        """

        Remember that self.image is a list, containing images with shape [2, height, width, depth]
        NOTE: self.b_pred_labels only contains the image that we just processed and NOT the complete list
        of images as in self.images and self.labels!!!

        NOTE: we only visualize 1 image (given by image_idx)

        """
        column_lbls = ["bg", "RV", "MYO", "LV"]
        image = self.images[image_num]
        img_labels = self.labels[image_num]
        img_pred_labels = self.pred_labels[image_num]
        num_of_classes = img_labels.shape[0]
        half_classes = num_of_classes / 2
        num_of_slices = img_labels.shape[3]

        if slice_range is None:
            slice_range = np.arange(0, num_of_slices // 2)

        fig = plt.figure(figsize=(width, height))
        counter = 1
        columns = half_classes + 1
        if img_pred_labels is not None:
            rows = 4
            plot_preds = True
        else:
            rows = 2
            plot_preds = False

        num_of_subplots = rows * 1 * columns  # +1 because the original image is included
        if len(slice_range) * num_of_subplots > 100:
            print("WARNING: need to limit number of subplots")
            slice_range = slice_range[:5]
        str_slice_range = [str(i) for i in slice_range]
        print("Number of subplots {} columns {} rows {} slices {}".format(num_of_subplots, columns, rows,
                                                                          ",".join(str_slice_range)))
        for idx in slice_range:
            # get the slice and then split ED and ES slices
            img = image[:, :, :, idx]
            labels = img_labels[:, :, :, idx]

            if plot_preds:
                pred_labels = img_pred_labels[:, :, :, idx]
            img_ed = img[0]  # INDEX 0 = end-diastole image
            img_es = img[1]  # INDEX 1 = end-systole image

            ax1 = plt.subplot(num_of_subplots, columns, counter)
            ax1.set_title("End-systole image", **config.title_font_medium)
            plt.imshow(img_ed, cmap=cm.gray)
            plt.axis('off')
            counter += 1
            for cls1 in np.arange(half_classes):
                ax2 = plt.subplot(num_of_subplots, columns, counter)
                plt.imshow(labels[cls1], cmap=cm.gray)
                ax2.set_title(column_lbls[cls1] + " (true labels)", **config.title_font_medium)
                plt.axis('off')
                if plot_preds:
                    ax3 = plt.subplot(num_of_subplots, columns, counter + columns)
                    plt.imshow(pred_labels[cls1], cmap=cm.gray)
                    plt.axis('off')
                    ax3.set_title(column_lbls[cls1] + " (pred labels)", **config.title_font_medium)
                counter += 1

            cls1 += 1
            counter += columns
            ax2 = plt.subplot(num_of_subplots, columns, counter)
            ax2.set_title("End-diastole image", **config.title_font_medium)
            plt.imshow(img_es, cmap=cm.gray)
            plt.axis('off')
            counter += 1
            for cls2 in np.arange(half_classes):
                ax4 = plt.subplot(num_of_subplots, columns, counter)
                plt.imshow(labels[cls1 + cls2], cmap=cm.gray)
                ax4.set_title(column_lbls[cls2] + " (true labels)", **config.title_font_medium)
                plt.axis('off')
                if plot_preds:
                    ax5 = plt.subplot(num_of_subplots, columns, counter + columns)
                    plt.imshow(pred_labels[cls1 + cls2], cmap=cm.gray)
                    ax5.set_title(column_lbls[cls2] + " (pred labels)", **config.title_font_medium)
                    plt.axis('off')
                counter += 1

            counter += columns

        fig.tight_layout()
        if do_save:
            file_suffix = "_".join(str_slice_range)
            if fig_name is None:
                fig_name = "test_img{}".format(image_num) + "_vis_pred_" + file_suffix
            fig_name = os.path.join(self.fig_output_dir, fig_name + ".png")

            plt.savefig(fig_name, bbox_inches='tight')
            print("INFO - Successfully saved fig %s" % fig_name)
        plt.show()

    def visualize_prediction_uncertainty(self, image_num=0, width=12, height=12, slice_range=None, do_save=False,
                                         fig_name=None, std_threshold=None, verbose=False):

        if std_threshold is None:
            use_uncertainty = False
        else:
            use_uncertainty = True

        column_lbls = ["bg", "RV", "MYO", "LV"]
        image = self.images[image_num]
        labels = self.labels[image_num]
        img_pred_labels = self.pred_labels[image_num]
        uncertainty_map = self.stddev_maps[image_num]
        num_of_classes = labels.shape[0]
        half_classes = num_of_classes / 2
        num_of_slices = labels.shape[3]

        if slice_range is None:
            slice_range = np.arange(num_of_slices)
            num_of_slices = len(slice_range)

        height = height * num_of_slices
        fig = plt.figure(figsize=(width, height))
        counter = 1
        columns = half_classes + 1  # currently only original image and uncertainty map
        rows = 2
        num_of_slices = len(slice_range)
        num_of_subplots = rows * num_of_slices * columns
        str_slice_range = [str(i) for i in slice_range]
        print("Number of subplots {} columns {} rows {} slices {}".format(num_of_subplots, columns, rows,
                                                                          ",".join(str_slice_range)))
        for idx in slice_range:
            # get the slice and then split ED and ES slices
            image_slice = image[:, :, :, idx]
            true_labels = labels[:, :, :, idx]
            pred_labels = img_pred_labels[:, :, :, idx]
            uncertainty = uncertainty_map[:, :, :, idx]
            for phase in np.arange(2):

                img = image_slice[phase]  # INDEX 0 = end-systole image
                ax1 = plt.subplot(num_of_subplots, columns, counter)
                if phase == 0:
                    ax1.set_title("Slice {}: End-systole".format(idx+1), **config.title_font_medium)
                    phase_str = "ES"
                else:
                    ax1.set_title("Slice {}: End-diastole".format(idx+1), **config.title_font_medium)
                    phase_str = "ED"

                plt.imshow(img, cmap=cm.gray)
                plt.axis('off')
                counter += 1
                # we use the cls_offset to plot ES and ED images in one loop (phase variable)
                cls_offset = phase * half_classes
                mean_stddev = 0.
                for cls in np.arange(half_classes):
                    std = uncertainty[cls + cls_offset]
                    mean_stddev += std
                    ax2 = plt.subplot(num_of_subplots, columns, counter)
                    x2_plot = ax2.imshow(std, cmap=cm.coolwarm, vmin=0.0, vmax=0.6)
                    # plt.colorbar(x2_plot, cax=ax2)
                    ax2.set_title(r"$\sigma_{{pred}}$ {}".format(column_lbls[cls]), **config.title_font_medium)
                    plt.axis('off')
                    counter += 1
                    true_cls_labels = true_labels[cls + cls_offset]
                    pred_cls_labels = pred_labels[cls + cls_offset]
                    errors = true_cls_labels != pred_cls_labels
                    ax3 = plt.subplot(num_of_subplots, columns, counter + half_classes)
                    dice_before = dice_coefficient(true_cls_labels.flatten(), pred_cls_labels.flatten())
                    if 0 != np.count_nonzero(pred_cls_labels) and 0 != np.count_nonzero(true_cls_labels):
                        hausdorff_before = hd(pred_cls_labels, true_cls_labels, voxelspacing=1.4, connectivity=1)
                    else:
                        hausdorff_before = 0.
                    if use_uncertainty:
                        error_std = np.copy(std)
                        error_std[~errors] = 0.
                        pred_cls_labels_filtered = np.copy(pred_cls_labels)
                        error_idx = error_std > std_threshold
                        pred_cls_labels_filtered[error_idx] = 0
                        error_std[error_idx] = 0.
                        errors_after = true_cls_labels != pred_cls_labels_filtered
                        dice_after = dice_coefficient(true_cls_labels, pred_cls_labels_filtered)
                        if 0 != np.count_nonzero(pred_cls_labels_filtered) and 0 != np.count_nonzero(true_cls_labels):
                            hausdorff_after = hd(pred_cls_labels_filtered, true_cls_labels, voxelspacing=1.4,
                                                 connectivity=1)
                        else:
                            hausdorff_after = 0.

                        if False:
                            plt.imshow(error_std, cmap=cm.coolwarm, vmin=0.0, vmax=std_threshold)
                        else:
                            plt.imshow(errors_after, cmap=cm.gray)
                        ax3.set_title("{} errors: {}".format(column_lbls[cls], np.count_nonzero(errors_after)),
                                      **config.title_font_medium)
                        if cls != 0 and verbose:
                            print("Slice {} - {} - Class {} before/after: errors {} {} || dice {:.2f} / {:.2f} || "
                                  "hd  {:.2f} / {:.2f}".format(idx + 1, phase_str, cls, np.count_nonzero(errors),
                                                               np.count_nonzero(errors_after), dice_before, dice_after,
                                                               hausdorff_before, hausdorff_after))
                    else:
                        plt.imshow(errors, cmap=cm.gray)
                        ax3.set_title("{} errors: {}".format(column_lbls[cls], np.count_nonzero(errors)),
                                      **config.title_font_medium)
                        if cls != 0 and verbose:
                            print("Slice {} - {} - Class {}: errors {} || dice {:.2f} || "
                                  "hd  {:.2f}".format(idx + 1, phase_str, cls, np.count_nonzero(errors),
                                                      dice_before, hausdorff_before))
                    plt.axis('off')

                # plot the average uncertainty per pixel (over classes)
                mean_stddev = 1./float(half_classes) * mean_stddev
                ax4 = plt.subplot(num_of_subplots, columns, counter )
                ax4_plot = ax4.imshow(mean_stddev, cmap=cm.coolwarm, vmin=0.0, vmax=0.6)
                # plt.colorbar(ax4_plot, cax=ax4_plot)
                ax4.set_title(r"$\sigma_{{pred}}$ {}".format("mean"), **config.title_font_medium)
                plt.axis('off')
                # move to the correct subplot space for the next phase (ES/ED)
                counter += half_classes + 1  # move counter forward in subplot
        fig.tight_layout()
        if do_save:
            file_suffix = "_".join(str_slice_range)
            if fig_name is None:
                fig_name = "test_img{}".format(image_num) + "_vis_pred_uncertainty_" + file_suffix
            fig_name = os.path.join(self.fig_output_dir, fig_name + ".pdf")

            plt.savefig(fig_name, bbox_inches='tight')
            print("INFO - Successfully saved fig %s" % fig_name)

    def visualize_uncertainty_histograms(self, image_num=0, width=16, height=10, info_type="uncertainty",
                                         std_threshold=0., do_show=False, model_name="", use_bald=True,
                                         do_save=False, fig_name=None, slice_range=None, errors_only=False):
        """

        :param image_num:
        :param width:
        :param height:
        :param info_type:
        :param std_threshold:
        :param do_show:
        :param model_name:
        :param use_bald: use BALD as an uncertainty measure and show heat maps & histgrams for this measure per slice
        :param do_save:
        :param fig_name:
        :param slice_range:
        :param errors_only:
        :return:
        """

        if errors_only and use_bald:
            # need to set BALD to False as well
            print("WARNING - setting use_bald to False")
            use_bald = False
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = 'Ubuntu'
        plt.rcParams['font.monospace'] = 'Ubuntu Mono'
        column_lbls = ["bg", "RV", "MYO", "LV"]
        # image_num = self._translate_image_range(image_num)
        try:
            image_num = self.image_ids.index(image_num)
        except ValueError:
            print("WARNING - Can't find image with index {} in "
                  "test_results.image_ids. Discarding!".format(image_num))
        image = self.images[image_num]
        image_probs = self.image_probs_categorized[image_num]
        pred_labels = self.pred_labels[image_num]
        label = self.labels[image_num]
        uncertainty_map = self.stddev_maps[image_num]
        if use_bald:
            bald_map = self.bald_maps[image_num]  # shape [2, width, height, #slices] 0=ES balds, 1=ED balds so per phase
            bald_min, bald_max = np.min(bald_map), np.max(bald_map)
        else:
            bald_map = None

        num_of_classes = label.shape[0]
        half_classes = num_of_classes / 2
        mc_samples = self.mc_pred_probs[image_num].shape[0]
        max_stdddev = 0.5  # the maximum stddev we're dealing with (used in colorbar) because probs are between [0,1]

        image_name = self.image_names[image_num]
        if slice_range is None:
            slice_range = np.arange(0, image.shape[3])
        num_of_slices = len(slice_range)
        str_slice_range = [str(i) for i in slice_range]
        str_slice_range = "_".join(str_slice_range)
        columns = half_classes
        if errors_only:
            rows = 4  # multiply with 4 because two rows ES, two rows ED for each slice
        elif not use_bald:
            rows = 8  # multiply with 8 because four rows ES, four rows ED for each slice
        else:
            rows = 14  # multiply with 2 because five rows ES, five rows ED for each slice
        print("Rows/columns {}/{}".format(rows, columns))

        _ = rows * num_of_slices * columns
        if errors_only:
            height = 20

        for img_slice in slice_range:

            if std_threshold > 0.:
                main_title = r"Model {} - Test image: {} - slice: {}" \
                             r" - ($\sigma_{{Tr}}={:.2f}$) \n".format(model_name, image_name, img_slice+1, std_threshold)
            else:
                main_title = "Model {} - Test image: {} - slice: {} \n".format(model_name, image_name, img_slice+1)
            fig = plt.figure(figsize=(width, height))
            ax = fig.gca()
            fig.suptitle(main_title, **config.title_font_large)
            row = 0
            # get the slice and then split ED and ES slices
            image_slice = image[:, :, :, img_slice]
            img_slice_probs = image_probs[img_slice]
            # print("INFO - Slice {}".format(img_slice+1))
            for phase in np.arange(2):
                cls_offset = phase * half_classes
                img = image_slice[phase]  # INDEX 0 = end-systole image
                slice_pred_labels = copy.deepcopy(pred_labels[:, :, :, img_slice])
                slice_true_labels = label[:, :, :, img_slice]
                slice_stddev = uncertainty_map[:, :, :, img_slice]
                # ax1 = plt.subplot(num_of_subplots, columns, counter)

                # print("INFO-1 - row/counter {} / {}".format(row, counter))
                ax1 = plt.subplot2grid((rows, columns), (row, 0), rowspan=2, colspan=2)
                if phase == 0:
                    ax1.set_title("Slice {}: End-systole".format(img_slice+1), **config.title_font_large)
                    str_phase = "ES"
                else:
                    ax1.set_title("Slice {}: End-diastole".format(img_slice+1), **config.title_font_large)
                    str_phase = "ED"
                # the original image we are segmenting
                ax1.imshow(img, cmap=cm.gray)
                ax1.set_aspect('auto')
                plt.axis('off')
                # -------------------------- Plot segmentation ERRORS per class on original image -----------
                # we also construct an image with the segmentation errors, placing it next to the original img
                ax1b = plt.subplot2grid((rows, columns), (row, 2), rowspan=2, colspan=2)
                rgb_img = to_rgb1a(img)
                rgb_img, cls_errors = set_error_pixels(rgb_img, slice_pred_labels, slice_true_labels,
                                                                          cls_offset,
                                                                          slice_stddev, std_threshold=std_threshold)
                ax1b.imshow(rgb_img, interpolation='nearest')
                ax1b.set_aspect('auto')
                ax1b.text(20, 20, 'red: RV ({}), yellow: Myo ({}), green: LV ({})'.format(cls_errors[1],
                                                                                          cls_errors[2],
                                                                                          cls_errors[3]),
                          bbox={'facecolor': 'white', 'pad': 18})
                ax1b.set_title("Prediction errors", **config.title_font_medium)
                plt.axis('off')
                # ARE WE SHOWING THE BALD value heatmap?
                if use_bald:
                    row += 2
                    # show image for BALD measure (1) BALD per pixel, on the left side
                    slice_bald = bald_map[phase, :, :, img_slice]
                    # print("Phase {} row {} - BALD heatmap".format(phase, row))
                    ax4 = plt.subplot2grid((rows, columns), (row, 0), rowspan=2, colspan=2)
                    ax4plot = ax4.imshow(slice_bald, cmap=plt.get_cmap('jet'), vmin=bald_min, vmax=bald_max)
                    # divider = make_axes_locatable(ax)
                    # cax = divider.append_axes("right", size="5%", pad=0.05)
                    # fig.colorbar(ax4plot, cax=cax)
                    ax4.set_aspect('auto')
                    fig.colorbar(ax4plot, ax=ax4, fraction=0.046, pad=0.04)
                    # cb = Colorbar(ax=ax4, mappable=ax4plot, orientation='vertical', ticklocation='right')
                    # cb.set_label(r'Colorbar !', labelpad=10)
                    ax4.set_title("Slice {} {}: BALD-values".format(img_slice+1, str_phase),
                                  **config.title_font_medium)
                    plt.axis('off')

                if not errors_only:
                    # plot (2) MEAN STD (over classes) next to BALD heatmap, so we can compare the two measures
                    # get the stddev value for the first 4 or last 4 classes (ES/ED) and average over classes (dim0)
                    if phase == 0:
                        mean_slice_stddev = np.mean(slice_stddev[:half_classes], axis=0)
                    else:
                        mean_slice_stddev = np.mean(slice_stddev[half_classes:], axis=0)
                    # print("Phase {} row {} - MEAN STDDEV heatmap".format(phase, row))
                    ax4a = plt.subplot2grid((rows, columns), (row, 2), rowspan=2, colspan=2)
                    ax4aplot = ax4a.imshow(mean_slice_stddev, cmap=plt.get_cmap('jet'),
                                           vmin=0., vmax=max_stdddev)
                    ax4a.set_aspect('auto')
                    fig.colorbar(ax4aplot, ax=ax4a, fraction=0.046, pad=0.04)
                    ax4a.set_title("Slice {} {}: MEAN stddev-values".format(img_slice + 1, str_phase),
                                   **config.title_font_medium)
                    plt.axis('off')

                if use_bald:
                    # create histogram
                    if phase == 0:
                        bald_corr = img_slice_probs["es_cor_bald"]
                        bald_err = img_slice_probs["es_err_bald"]
                        p_value_ttest = img_slice_probs["es_pvalue_ttest_bald"]
                        p_value_mannwhitu = img_slice_probs["es_pvalue_mwhitu_bald"]
                    else:
                        bald_corr = img_slice_probs["ed_cor_bald"]
                        bald_err = img_slice_probs["ed_err_bald"]
                        p_value_ttest = img_slice_probs["ed_pvalue_ttest_bald"]
                        p_value_mannwhitu = img_slice_probs["ed_pvalue_mwhitu_bald"]
                        print("p-values ttest/Mann-Withney-U {:.2E}/{:.2E} ".format(p_value_ttest, p_value_mannwhitu))
                    if p_value_ttest >= 0.001 or p_value_mannwhitu >= 0.001:

                        str_p_value = "p={:.3f}/{:.3f}".format(p_value_ttest, p_value_mannwhitu)
                    else:
                        str_p_value = None

                    xs = np.linspace(0, bald_max, 20)
                    ax5 = plt.subplot2grid((rows, columns), (row + 3, 0), rowspan=2, colspan=2)
                    # print("Phase {} row {} - BALD histogram".format(phase, row + 2))
                    if bald_err is not None:

                        ax5.hist(bald_err[bald_err >= std_threshold], bins=xs,
                                 label=r"$bald_{{pred(fp+fn)}}({})$".format(bald_err.shape[0])
                                 , color='b', alpha=0.2, histtype='stepfilled')
                        ax5.legend(loc="best", prop={'size': 12})
                        ax5.grid(False)
                    if bald_corr is not None:
                        ax5b = ax5.twinx()
                        ax5b.hist(bald_corr[bald_corr >= std_threshold], bins=xs,
                                 label=r"$bald_{{pred(tp)}}({})$".format(bald_corr.shape[0]),
                                 color='g', alpha=0.4, histtype='stepfilled')
                        ax5b.legend(loc=2, prop={'size': 12})
                        ax5b.grid(False)
                    ax5.set_xlabel("BALD value", **config.axis_font)
                    if str_p_value is not None:
                        title_suffix = "(" + str_p_value + ")"
                    else:
                        title_suffix = ""
                    ax5.set_title("Slice {} {}: Distribution of BALD values ".format(img_slice + 1, str_phase)
                                  + title_suffix, ** config.title_font_medium)

                # In case we're only showing the error segmentation map we skip the next part (histgrams and
                # stddev uncertainty maps per class
                if not errors_only:
                    row += 2
                    row_offset = 1
                    col_offset = 0
                    counter = 0
                    for cls in np.arange(half_classes):
                        if phase == 0:
                            p_err_prob = np.array(img_slice_probs["es_err_p"][cls])
                            p_corr_prob = np.array(img_slice_probs["es_cor_p"][cls])
                            p_err_std = np.array(img_slice_probs["es_err_std"][cls])
                            p_corr_std = np.array(img_slice_probs["es_cor_std"][cls])
                        else:
                            p_err_prob = np.array(img_slice_probs["ed_err_p"][cls])
                            p_corr_prob = np.array(img_slice_probs["ed_cor_p"][cls])
                            p_err_std = np.array(img_slice_probs["ed_err_std"][cls])
                            p_corr_std = np.array(img_slice_probs["ed_cor_std"][cls])

                        # in the next subplot row we visualize the uncertainties per class
                        ax3 = plt.subplot2grid((rows, columns), (row, counter), colspan=1)
                        std_map_cls = slice_stddev[cls + cls_offset]
                        cmap = plt.get_cmap('jet')
                        std_rgba_img = cmap(std_map_cls)
                        std_rgb_img = np.delete(std_rgba_img, 3, 2)
                        ax3plot = ax3.imshow(std_rgb_img, vmin=0., vmax=max_stdddev)
                        ax3.set_aspect('auto')
                        if cls == half_classes - 1:
                            fig.colorbar(ax3plot, ax=ax3, fraction=0.046, pad=0.04)
                        ax3.set_title("{} stddev: {} ".format(str_phase, column_lbls[cls]),
                                      **config.title_font_medium)
                        plt.axis("off")
                        # finally in the next row we plot the uncertainty densities per class
                        # print("cls {} col_offset {}".format(cls, col_offset))
                        ax2 = plt.subplot2grid((rows, columns), (row + row_offset, 2 + col_offset), colspan=1)
                        col_offset += 1
                        std_max = max(p_err_std.max() if p_err_std.shape[0] > 0 else 0,
                                      p_corr_std.max() if p_corr_std.shape[0] > 0 else 0.)

                        xs = np.linspace(0, std_max, 20)
                        if p_err_std is not None:
                            ax2b = ax2.twinx()
                            ax2b.hist(p_err_std[p_err_std >= std_threshold], bins=xs,
                                     label=r"$\sigma_{{pred(fp+fn)}}({})$".format(cls_errors[cls])
                                     ,color="b", alpha=0.2)
                            ax2b.legend(loc=2, prop={'size': 9})

                        if p_corr_std is not None:
                            ax2.hist(p_corr_std[p_corr_std >= std_threshold], bins=xs,
                                     label=r"$\sigma_{{pred(tp)}}({})$".format(p_corr_std.shape[0]),
                                     color="g", alpha=0.4)

                        if info_type == "uncertainty":
                            ax2.set_xlabel("model uncertainty", **config.axis_font)
                        else:
                            ax2.set_xlabel(r"softmax $p(c|x)$", **config.axis_font)
                        ax2.set_title("{} slice-{}: {}".format(str_phase, img_slice+1, column_lbls[cls]),
                                      **config.title_font_medium)
                        ax2.legend(loc="best", prop={'size': 9})
                        # ax2.set_ylabel("density", **config.axis_font)
                        if cls == 1:
                            row_offset += 1
                            col_offset = 0
                        counter += 1
                if errors_only:
                    row += 2
                else:
                    row += 3
            # fig.tight_layout()
            if not errors_only:
                fig.tight_layout(rect=[0, 0.03, 1, 0.97])
            if do_save:

                fig_img_dir = self._create_figure_dir(image_num)
                fig_name = "analysis_seg_err_slice{}".format(img_slice+1) \
                            + "_mc" + str(mc_samples)
                if std_threshold > 0.:
                    tr_string = "_tr" + str(std_threshold).replace(".", "_")
                    fig_name += tr_string
                if errors_only:
                    fig_name = fig_name + "_w_uncrty"
                fig_name = os.path.join(fig_img_dir, fig_name + ".pdf")

                plt.savefig(fig_name, bbox_inches='tight')
                print("INFO - Successfully saved fig %s" % fig_name)
            if do_show:
                plt.show()
            plt.close()

    @property
    def N(self):
        return len(self.pred_labels)

    def save_results(self, fold_ids=None, outfile=None, epoch_id=None):

        # Saving the list with dictionaries for the slice statistics (probs, stddev) is amazingly slow.
        # So we don't save this object(s), but we can generate these stats when loading the object (see below).
        # We temporary save the object here and assign it back to the object property after we saved.
        image_probs_categorized = self.image_probs_categorized
        images = self.images
        labels = self.labels
        pred_labels = self.pred_labels
        self.images = []
        self.labels = []
        self.pred_labels = []
        self.image_probs_categorized = []

        if outfile is None:
            num_of_images = len(self.image_names)

            if self.use_dropout:
                outfile = "test_results_{}imgs_mc{}".format(num_of_images, self.mc_samples)
            else:
                outfile = "test_results_{}imgs".format(num_of_images)
            if fold_ids is not None:
                str_fold_ids = "_folds" + "_".join([str(i) for i in fold_ids])
                outfile += str_fold_ids
            if epoch_id is not None:
                str_epoch = "_ep" + str(epoch_id)
                outfile += str_epoch

            if epoch_id is None:
                jetzt = datestr(withyear=False)
                outfile = outfile + "_" + jetzt

        outfile = os.path.join(self.save_output_dir, outfile + ".dll")

        try:
            with open(outfile, 'wb') as f:
                dill.dump(self, f)
            print("INFO - Saved results to {}".format(outfile))
        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
            print("ERROR - can't save results to {}".format(outfile))

        self.images = images
        self.labels = labels
        self.pred_labels = pred_labels
        self.image_probs_categorized = image_probs_categorized
        del images
        del labels
        del pred_labels
        del image_probs_categorized

    @staticmethod
    def load_results(path_to_exp, generate_stats=False):

        print("INFO - Loading results from file {}".format(path_to_exp))
        try:
            with open(path_to_exp, 'rb') as f:
                test_results = dill.load(f)

        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
            print("ERROR - Can't open file {}".format(path_to_exp))
            raise IOError
        if generate_stats:
            print("INFO - Generating slice statistics for all {} images.".format(test_results.N))
            for image_num in tqdm(np.arange(len(test_results.N))):
                test_results.generate_slice_statistics(image_num)

        print("INFO - Successfully loaded TestResult object.")
        return test_results

    def show_results(self):
        if self.dice_results is not None:
            mean_dice = self.dice_results[0]
            stddev = self.dice_results[1]
            print("Test accuracy: \t "
                  "dice(RV/Myo/LV): ES {:.2f} ({:.2f})/{:.2f} ({:.2f})/{:.2f} ({:.2f}) --- "
                  "ED {:.2f} ({:.2f})/{:.2f} ({:.2f})/{:.2f} ({:.2f})".format(mean_dice[1], stddev[1], mean_dice[2],
                                                                              stddev[2], mean_dice[3], stddev[3],
                                                                              mean_dice[5], stddev[5], mean_dice[6],
                                                                              stddev[6], mean_dice[7], stddev[7]))
        if self.hd_results is not None:
            mean_hd = self.hd_results[0]
            stddev = self.hd_results[1]
            print("Test accuracy: \t "
                  "Hausdorff(RV/Myo/LV): ES {:.2f} ({:.2f})/{:.2f} ({:.2f})/{:.2f} ({:.2f}) --- "
                  "ED {:.2f} ({:.2f})/{:.2f} ({:.2f})/{:.2f} ({:.2f})".format(mean_hd[1], stddev[1], mean_hd[2],
                                                                              stddev[2], mean_hd[3], stddev[3],
                                                                              mean_hd[5], stddev[5], mean_hd[6],
                                                                              stddev[6], mean_hd[7], stddev[7]))

    def _create_figure_dir(self, image_num):
        image_name = self.image_names[image_num]
        fig_path = os.path.join(self.fig_output_dir, image_name)
        if not os.path.isdir(fig_path):
            os.makedirs(fig_path)

        return fig_path

    def _translate_image_range(self, image_num):

        try:
            idx = self.image_ids.index(image_num)
        except ValueError:
            print("WARNING - Can't find image with index {} in "
                  "test_results.image_ids. Discarding!".format(image_num))
        return idx

