import numpy as np
from scipy.stats import gaussian_kde
import os
import dill
import glob
from tqdm import tqdm
from collections import OrderedDict
from config.config import config
import matplotlib.pyplot as plt
from matplotlib import cm
from common.common import datestr, to_rgb1a, set_error_pixels, create_mask_uncertainties, detect_seg_contours
from common.common import load_referral_umap, load_pred_labels
import copy
from scipy.stats import wilcoxon, ttest_ind, mannwhitneyu
from utils.post_processing import filter_connected_components


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


def get_mean_pred_per_slice(img_slice, img_probs, img_stds, img_balds, labels, pred_labels, half_classes, stats,
                            compute_p_values=False):
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
        # print("slice id {} # errors {}".format(img_slice, np.count_nonzero(union_errors)))
        # print("slice id {} # correct{}".format(img_slice, np.count_nonzero(union_correct)))

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
        union_errors = np.array(union_errors)
        union_correct = np.array(union_correct)
        if union_errors.ndim > 1:
            err_probs = slice_probs_ph.T[union_errors].flatten()
            err_std = slice_stds_ph[union_errors].flatten()
            err_bald = slice_bald_ph[union_errors].flatten()
        else:
            err_probs = None
            err_std = None
            err_bald = None
        if union_correct.ndim > 1:
            pos_probs = slice_probs_ph.T[union_correct].flatten()
            pos_std = slice_stds_ph[union_correct].flatten()
            pos_bald = slice_bald_ph[union_correct].flatten()
        else:
            pos_probs = None
            pos_std = None
            pos_bald = None
        # print("# in err_std ", err_std.shape[0])
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
        if compute_p_values:
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
        # contains the patientxx IDs of images. IMPORTANT for linking the results to the original images on disk!!!
        # we will actually combine image_ids and image_names (which both come from TestHandler object) into a
        # dictionary that acts as a translation between key=patientID and imageID (value) as loaded by the TestHandler
        self.image_names = []
        self.trans_dict = OrderedDict()
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
        # the following two measure are probably temporary, they capture the mean dice-coeff for an image based
        # on the dice-coeff of the individual slices (so no post-processing 6 most connected components).
        # At least these two measure (referral and non_referral_accuray) should differ significantly because we
        # filter the "referral" on outliers slices (taking into account the class)
        self.referral_accuracy = []
        self.referral_hd = []
        self.referral = False
        # end temporaray
        self.test_hd_slices = []
        self.dice_results = None
        self.hd_results = None
        self.hd_referral_results = None
        self.dice_referral_results = None
        self.num_of_samples = []
        self.referral_threshold = 0.
        self.referral_stats = []
        self.referral_stats_results = None
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
        self.umap_dir = os.path.join(exper.config.root_dir,
                                                  os.path.join(exper.output_dir, config.u_map_dir))

        self.pred_labels_input_dir = os.path.join(exper.config.root_dir,
                                                  os.path.join(exper.output_dir, config.pred_lbl_dir))

    def add_results(self, batch_image, batch_labels, image_id, pred_labels, mc_pred_probs, stddev_map,
                    test_accuracy, test_hd, seg_errors, store_all=False, bald_maps=None, uncertainty_stats=None,
                    test_hd_slices=None, test_accuracy_slices=None, image_name=None, referral_accuracy=None,
                    referral_hd=None, referral_stats=None):
        """

        :param batch_image: [2, width, height, slices]
        :param batch_labels:
        :param pred_labels:
        :param mc_pred_probs: the actual softmax probabilities [#samples, 8classes, width, height, #slices]
        :param seg_errors: [#slices, #classes(8)]
        :param stddev_map: [#classes (8), width, height, slices]
        :param bald_maps: [2, width, height, slices]
        :param uncertainty_stats: dictionary with keys "bald", "stddev" and "u_threshold"
                                  the first 2 contain numpy arrays of shape [2, half classes (4), #slices]
        :param referral_accuracy: dice score per image (6 classes) after filtering the outliers
        :param referral_stats: for ES/ED the referral percentages and error reduction percentages
                               array has shape [2, 4classes,  3, #slices]
                               the 3 measures: ref%, reduction%, #true positives in gt
        :param test_accuracy:
        :param test_hd:
        :return:
        """
        # get rid off padding around image
        batch_image = batch_image[:, config.pad_size:-config.pad_size, config.pad_size:-config.pad_size, :]
        # during ensemble testing we only want to store the images and ground truth labels once
        if store_all:
            self.images.append(batch_image)
            self.labels.append(batch_labels)
        if store_all:
            self.pred_labels.append(pred_labels)
            self.mc_pred_probs.append(mc_pred_probs)
            self.stddev_maps.append(stddev_map)
            self.bald_maps.append(bald_maps)

        self.image_ids.append(image_id)
        self.image_names.append(image_name)
        self.num_of_samples.append(mc_pred_probs.shape[0])
        self.trans_dict[image_name] = image_id

        self.test_accuracy.append(test_accuracy)
        self.test_hd.append(test_hd)
        if test_hd_slices is not None:
            self.test_hd_slices.append(test_hd_slices)
        if test_accuracy_slices is not None:
            self.test_accuracy_slices.append(test_accuracy_slices)
        if referral_accuracy is not None:
            self.referral = True
            self.referral_accuracy.append(referral_accuracy)
        if referral_hd is not None:
            self.referral_hd.append(referral_hd)
        if referral_stats is not None:
            self.referral_stats.append(referral_stats)
        # segmentation errors is a numpy error with size [2, 4classes, #slices]
        self.seg_errors.append(seg_errors)
        if self.num_of_samples[-1] > 1:
            self.uncertainty_stats.append(uncertainty_stats)

    def get_uncertainty_stats(self):
        """
        We convert the TestResults.uncertainty_stats object (which is a list of dictionaries with two numpy arrays)
        (1) ["stddev"] and (2) ["bald"], into a dictionary with keys (patientID) and as value the original numpy array.
        We only convert the "stddev" entry.
        :return:
        """
        # remember self.image_names contains the patientxxx IDs
        u_stats_dict = OrderedDict([tuple((self.image_names[i], u_dict)) for i, u_dict in
                                    enumerate(self.uncertainty_stats)])
        return u_stats_dict

    def get_image_patientID(self):
        return self.image_names

    def compute_mean_stats(self):

        N = len(self.test_accuracy)
        # print("#Images {}".format(N))
        if len(self.test_accuracy) == 0 or len(self.test_hd) == 0:
            raise ValueError("ERROR - there's no data that could be used to compute statistics!")

        columns_dice = self.test_accuracy[0].shape[0]
        columns_hd = self.test_hd[0].shape[0]
        mean_dice = np.empty((0, columns_dice))
        mean_hd = np.empty((0, columns_hd))
        # We need 2 lists for each measure, one for ES and for ED
        mean_ref_perc, mean_reduc_perc, mean_prec, mean_recall = [[], []], [[], []],  [[], []],  \
                                                                 [[], []]
        # temporary for referral stuff
        if self.referral:
            mean_ref_dice = np.empty((0, columns_dice))
            mean_hd_ref = np.empty((0, columns_hd))
            # summarize the referral percentages, error reduction percentages for both phases ES/ED
            # we store mean, stddev and median for both measures (percentages)
            self.referral_stats_results = np.zeros((2, 4, 3))

        for img_idx in np.arange(N):
            # test_accuracy and test_hd are a vectors of 8 values, 0-3: ES, 4:7: ED
            dice = self.test_accuracy[img_idx]
            hausd = self.test_hd[img_idx]
            mean_dice = np.vstack([mean_dice, dice]) if mean_dice.size else dice
            mean_hd = np.vstack([mean_hd, hausd]) if mean_hd.size else hausd
            if self.referral:
                ref_stats = self.referral_stats[img_idx]
                ref_dice_img = self.referral_accuracy[img_idx]
                mean_ref_dice = np.vstack((mean_ref_dice, ref_dice_img))
                ref_hd_img = self.referral_hd[img_idx]
                mean_hd_ref = np.vstack((mean_hd_ref, ref_hd_img))
                # append the referral percentages for ES and ED for class 1, 2 and 3 (2nd index).
                # Please REMEMBER that the referral_stats has only 3 dimensions for the classes (we omit BG),
                # so 0=RV, 1=MYO, 2=LV
                # Do the same for the reduction percentages (last index 1 = reduction%), precision=index7,
                # recall=index=8
                mean_ref_perc[0].extend(list(ref_stats[0, 0, 0])), mean_ref_perc[0].extend(list(ref_stats[0, 1, 0]))
                mean_ref_perc[0].extend(list(ref_stats[0, 2, 0]))
                mean_ref_perc[1].extend(list(ref_stats[1, 0, 0])), mean_ref_perc[1].extend(list(ref_stats[1, 1, 0]))
                mean_ref_perc[1].extend(list(ref_stats[1, 2, 0]))
                mean_reduc_perc[0].extend(list(ref_stats[0, 0, 1])), mean_reduc_perc[0].extend(list(ref_stats[0, 1, 1]))
                mean_reduc_perc[0].extend(list(ref_stats[0, 2, 1]))
                mean_reduc_perc[1].extend(list(ref_stats[1, 0, 1])), mean_reduc_perc[1].extend(list(ref_stats[1, 1, 1]))
                mean_reduc_perc[1].extend(list(ref_stats[1, 2, 1]))
                mean_prec[0].extend(list(ref_stats[0, 0, 7])), mean_prec[0].extend(list(ref_stats[0, 1, 7]))
                mean_prec[0].extend(list(ref_stats[0, 2, 7]))
                mean_prec[1].extend(list(ref_stats[1, 0, 7])), mean_prec[1].extend(list(ref_stats[1, 1, 7]))
                mean_prec[1].extend(list(ref_stats[1, 2, 7]))
                mean_recall[0].extend(list(ref_stats[0, 0, 8])), mean_recall[0].extend(list(ref_stats[0, 1, 8]))
                mean_recall[0].extend(list(ref_stats[0, 2, 8]))
                mean_recall[1].extend(list(ref_stats[1, 0, 8])), mean_recall[1].extend(list(ref_stats[1, 1, 8]))
                mean_recall[1].extend(list(ref_stats[1, 2, 8]))
        if self.referral:
            # ES referral statistics
            self.referral_stats_results[0, 0] = np.array([np.mean(mean_ref_perc[0]), np.std(mean_ref_perc[0]),
                                                          np.median(mean_ref_perc[0])])
            # ED referral statistics
            self.referral_stats_results[1, 0] = np.array([np.mean(mean_ref_perc[1]), np.std(mean_ref_perc[1]),
                                                          np.median(mean_ref_perc[1])])
            # ES reduction percentages
            self.referral_stats_results[0, 1] = np.array([np.mean(mean_reduc_perc[0]), np.std(mean_reduc_perc[0]),
                                                          np.median(mean_reduc_perc[0])])
            # ED reduction percentages
            self.referral_stats_results[1, 1] = np.array([np.mean(mean_reduc_perc[1]), np.std(mean_reduc_perc[1]),
                                                          np.median(mean_reduc_perc[1])])
            # ES referral precision
            self.referral_stats_results[0, 2] = np.array([np.mean(mean_prec[0]), np.std(mean_prec[0]),
                                                          np.median(mean_prec[0])])
            # ED referral precision
            self.referral_stats_results[1, 2] = np.array([np.mean(mean_prec[1]), np.std(mean_prec[1]),
                                                          np.median(mean_prec[1])])
            # ES referral recall
            self.referral_stats_results[0, 3] = np.array([np.mean(mean_recall[0]), np.std(mean_recall[0]),
                                                          np.median(mean_recall[0])])
            # ED referral recall
            self.referral_stats_results[1, 3] = np.array([np.mean(mean_recall[1]), np.std(mean_recall[1]),
                                                          np.median(mean_recall[1])])

        # so we stack the image results and should end up with a matrix [#images, 8] for dice and hd

        if N > 1:
            self.dice_results = np.array([np.mean(mean_dice, axis=0), np.std(mean_dice, axis=0)])
            self.hd_results = np.array([np.mean(mean_hd, axis=0), np.std(mean_hd, axis=0)])
            if self.referral:
                self.hd_referral_results = np.array([np.mean(mean_hd_ref, axis=0),
                                                           np.std(mean_hd_ref, axis=0)])
                self.dice_referral_results = np.array([np.mean(mean_ref_dice, axis=0), np.std(mean_ref_dice, axis=0)])
        else:
            # only one image tested, there is no mean or stddev
            self.dice_results = np.array([mean_dice, np.zeros(mean_dice.shape[0])])
            self.hd_results = np.array([mean_hd, np.zeros(mean_hd.shape[0])])
            if self.referral:
                mean_hd_ref = mean_hd_ref.squeeze()
                mean_ref_dice = mean_ref_dice.squeeze()
                self.hd_referral_results = np.array([mean_hd_ref,
                                                           np.zeros(mean_hd_ref.shape[0])])
                self.dice_referral_results = np.array([mean_ref_dice, np.zeros(mean_ref_dice.shape[0])])

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
                correct = np.argwhere((s_pred_labels & s_labels))
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
            image_name = self.image_names[image_num]
            fig_out_dir = self._create_figure_dir(image_name)
            if fig_name is None:
                fig_name = info_type + "_densities_mc" + str(mc_samples) \
                           + "_" + str(use_class_stats)
            fig_name = os.path.join(fig_out_dir, fig_name + ".png")

            plt.savefig(fig_name, bbox_inches='tight')
            print("INFO - Successfully saved fig %s" % fig_name)
        if do_show:
            plt.show()
        plt.close()

    def visualize_uncertainty_histograms(self, image_num=None, width=16, height=10, info_type="uncertainty",
                                         std_threshold=0., do_show=False, model_name="", use_bald=True,
                                         do_save=False, slice_range=None, errors_only=False,
                                         plot_detailed_hists=False, image_data=None, load_referral=False,
                                         ref_positives_only=False):
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
        :param ref_positives_only: only consider voxels with high uncertainties that we predicted as positive (1)
        and ignore the ones we predicted as non-member of the specific class (mostly background).
        This reduces the number of voxels to be referred without hopefully impairing the referral results
        significantly. We can then probably lower the referral threshold.
        :param image_data: tuple containing the tensors we need to construct the figures. We use this in case
        we don't want to "store" all the large test_result objects (e.g. images, labels, pred_labels etc) but
        nevertheless want to construct these images (e.g. we call this method in "generate_uncertainty_maps.py")
        :param plot_detailed_hists: if False then we plot only 5 rows: 1-2 images, 3-4 huge uncertainty maps,
                                                                       5: uncertainty maps per class
                                                                       NO HISTOGRAMS!
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

        try:
            image_num = self.image_ids.index(image_num)
        except ValueError:
            print("WARNING - Can't find image with index {} in "
                  "test_results.image_ids. Discarding!".format(image_num))
        image = self.images[image_num]
        num_of_slices_img = image.shape[3]
        pred_labels = self.pred_labels[image_num]
        label = self.labels[image_num]
        uncertainty_map = self.stddev_maps[image_num]
        bald_map = self.bald_maps[image_num]
        image_name = self.image_names[image_num]
        mc_samples = self.mc_pred_probs[image_num].shape[0]

        num_of_classes = label.shape[0]
        if std_threshold != 0.:
            # REFERRAL functionality.
            if load_referral:
                referral_threshold = str(std_threshold).replace(".", "_")
                search_path = os.path.join(self.umap_dir,
                                           image_name + "*" + "_filtered_cls_umaps" + referral_threshold + ".npz")
                filtered_cls_std_map = load_referral_umap(search_path, per_class=True)
                # filtered_add_std_map_es = np.max(filtered_cls_std_map[:4], axis=0)
                # filtered_add_std_map_ed = np.max(filtered_cls_std_map[4:], axis=0)

                search_path = os.path.join(self.umap_dir,
                                           image_name + "*" + "_filtered_umaps" + referral_threshold + ".npz")
                filtered_std_map = load_referral_umap(search_path, per_class=False)
                if ref_positives_only:
                    referral_threshold += "_mc_pos_only"
                # predicted labels we obtained after referral with this threshold
                search_path = os.path.join(self.pred_labels_input_dir,
                                           image_name + "_filtered_pred_labels" + referral_threshold + ".npz")
                print("Found label file {}".format(search_path))
                referral_pred_labels = load_pred_labels(search_path)

            else:
                filtered_std_map = copy.deepcopy(uncertainty_map)
                for cls in np.arange(0, num_of_classes):
                    if cls != 0 and cls != 4:
                        u_3dmaps_cls = filtered_std_map[cls]
                        u_3dmaps_cls[u_3dmaps_cls < std_threshold] = 0
                        filtered_std_map[cls] = filter_connected_components(u_3dmaps_cls, threshold=std_threshold)

        else:
            # Normal non-referral functionality
            filtered_std_map = None
            referral_pred_labels = None
        if std_threshold != 0:
            search_path = os.path.join(self.pred_labels_input_dir, image_name + "_pred_labels.npz")
            # the predicted labels we obtained with the MC model when NOT using dropout during inference!
            pred_labels_wo_sampling = load_pred_labels(search_path)
        else:
            pred_labels_wo_sampling = None
        if ref_positives_only and std_threshold != 0.:
            if pred_labels_wo_sampling is not None:
                print("---- pred_labels_wo_sampling is not none ------")
                mask = pred_labels_wo_sampling == 0
                # overall_mask = create_mask_uncertainties(pred_labels_wo_sampling)
                overall_mask = create_mask_uncertainties(pred_labels)
                overall_mask_es = overall_mask[0]
                overall_mask_ed = overall_mask[1]
            else:
                mask = pred_labels == 0
                overall_mask = create_mask_uncertainties(pred_labels)
                overall_mask_es = overall_mask[0]
                overall_mask_ed = overall_mask[1]

            filtered_std_map[0][overall_mask_es] = 0.
            filtered_std_map[1][overall_mask_ed] = 0.

            # filtered_add_std_map_es[overall_mask_es] = 0.
            # filtered_add_std_map_ed[overall_mask_ed] = 0.
            filtered_cls_std_map[mask] = 0
        # sum uncertainties over slice-pixels, to give us an indication about the amount of uncertainty
        # and whether we can visually inspect why tht model is uncertain
        if filtered_std_map is None:

            es_count_nonzero = np.count_nonzero(uncertainty_map[:4], axis=(0, 1, 2))
            ed_count_nonzero = np.count_nonzero(uncertainty_map[4:], axis=(0, 1, 2))
        else:
            es_count_nonzero = np.count_nonzero(filtered_std_map[0], axis=(0, 1))
            ed_count_nonzero = np.count_nonzero(filtered_std_map[1], axis=(0, 1))
        total_uncertainty_per_slice = es_count_nonzero + ed_count_nonzero
        print(total_uncertainty_per_slice)
        dice_scores_slices = self.test_accuracy_slices[image_num]
        print(es_count_nonzero)
        print("ES {}".format(np.array_str(np.mean(dice_scores_slices[0, 1:], axis=0), precision=2)))
        print(ed_count_nonzero)
        print("ED {}".format(np.array_str(np.mean(dice_scores_slices[1, 1:], axis=0), precision=2)))
        max_u_value = np.max(total_uncertainty_per_slice)
        sorted_u_value_list = np.sort(total_uncertainty_per_slice)[::-1]

        if errors_only or not plot_detailed_hists:
            image_probs = None
        else:
            image_probs = self.image_probs_categorized[image_num]

        if use_bald:
            bald_min, bald_max = np.min(bald_map), np.max(bald_map)
        else:
            bald_map = None
            bald_min, bald_max = None, None

        half_classes = num_of_classes / 2

        if slice_range is None:
            slice_range = np.arange(0, image.shape[3])
        num_of_slices = len(slice_range)

        columns = half_classes
        if errors_only:
            rows = 4  # multiply with 4 because two rows ES, two rows ED for each slice
            height = 20
        elif not plot_detailed_hists:
            rows = 10
            height = 50
        elif not use_bald:
            rows = 8  # multiply with 8 because four rows ES, four rows ED for each slice
            height = 50
        else:
            # we show all plots! 14 rows for ES and ED figures
            if pred_labels_wo_sampling is None:
                rows = 14
                height = 65
            else:
                rows = 18
                height = 80
        print("Rows/columns/height {}/{}/{}".format(rows, columns, height))

        _ = rows * num_of_slices * columns
        if errors_only:
            height = 20
        if not plot_detailed_hists:
            height = 50

        for img_slice in slice_range:

            if std_threshold > 0.:
                rank = np.where(sorted_u_value_list == total_uncertainty_per_slice[img_slice])[0][0]
                main_title = r"Model {} - Test image: {} - slice: {}" "\n" \
                             r"$\sigma_{{Tr}}={:.2f}$; u-value={}/{}; " \
                             r"rank={}".format(model_name, image_name, img_slice+1, std_threshold,
                                               total_uncertainty_per_slice[img_slice], max_u_value,
                                               rank + 1)
            else:
                main_title = "Model {} - Test image: {} - slice: {} \n".format(model_name, image_name, img_slice+1)

            fig = plt.figure(figsize=(width, height))
            ax = fig.gca()
            fig.suptitle(main_title, **config.title_font_medium)
            row = 0
            # get the slice and then split ED and ES slices
            image_slice = image[:, :, :, img_slice]
            if errors_only or not plot_detailed_hists:
                img_slice_probs = None
            else:
                img_slice_probs = image_probs[img_slice]

            # print("INFO - Slice {}".format(img_slice+1))
            for phase in np.arange(2):
                cls_offset = phase * half_classes
                img = image_slice[phase]  # INDEX 0 = end-systole image
                slice_pred_labels = copy.deepcopy(pred_labels[:, :, :, img_slice])
                slice_true_labels = label[:, :, :, img_slice]
                slice_stddev = uncertainty_map[:, :, :, img_slice]
                if std_threshold != 0.:
                    filtered_slice_stddev = filtered_cls_std_map[:, :, :, img_slice]

                # ax1 = plt.subplot(num_of_subplots, columns, counter)

                # print("INFO-1 - row/counter {} / {}".format(row, counter))
                ax1 = plt.subplot2grid((rows, columns), (row, 0), rowspan=2, colspan=2)
                if phase == 0:
                    ax1.set_title("Slice {}/{}: End-systole".format(img_slice+1, num_of_slices_img),
                                  **config.title_font_medium)
                    str_phase = "ES"
                else:
                    ax1.set_title("Slice {}/{}: End-diastole".format(img_slice+1, num_of_slices_img),
                                  **config.title_font_medium)
                    str_phase = "ED"
                # the original image we are segmenting
                img_with_contours = detect_seg_contours(img, slice_true_labels, cls_offset)
                ax1.imshow(img_with_contours, cmap=cm.gray)
                ax1.set_aspect('auto')
                plt.axis('off')
                # -------------------------- Plot segmentation ERRORS per class on original image -----------
                # we also construct an image with the segmentation errors, placing it next to the original img
                ax1b = plt.subplot2grid((rows, columns), (row, 2), rowspan=2, colspan=2)
                rgb_img = to_rgb1a(img)
                # IMPORTANT: we disables filtering of RV pixels based on high uncertainties, hence we set
                # std_threshold to zero! We use the threshold only to filter the uncertainty maps!
                rgb_img_w_pred, cls_errors = set_error_pixels(rgb_img, slice_pred_labels, slice_true_labels,
                                                              cls_offset)
                ax1b.imshow(rgb_img_w_pred, interpolation='nearest')
                ax1b.set_aspect('auto')
                ax1b.text(20, 20, 'yellow: RV ({}), blue: Myo ({}), red: LV ({})'.format(cls_errors[1],
                                                                                          cls_errors[2],
                                                                                          cls_errors[3]),
                          bbox={'facecolor': 'white', 'pad': 18})
                ax1b.set_title("Prediction errors with sampling", **config.title_font_medium)
                plt.axis('off')
                if pred_labels_wo_sampling is not None:
                    slice_pred_labels_wo_sampling = pred_labels_wo_sampling[:, :, :, img_slice]
                    row += 2
                    rgb_img = to_rgb1a(img)
                    ax_pred = plt.subplot2grid((rows, columns), (row, 0), rowspan=2, colspan=2)
                    rgb_img_w_pred, cls_errors = set_error_pixels(rgb_img, slice_pred_labels_wo_sampling,
                                                                  slice_true_labels, cls_offset)
                    ax_pred.text(20, 20, 'yellow: RV ({}), blue: Myo ({}), '
                                 'red: LV ({}) '.format(cls_errors[1], cls_errors[2], cls_errors[3]),
                                 bbox={'facecolor': 'white', 'pad': 18})
                    ax_pred.imshow(rgb_img_w_pred, interpolation='nearest')
                    ax_pred.set_aspect('auto')
                    ax_pred.set_title("Prediction errors wo sampling", **config.title_font_medium)
                    plt.axis('off')
                if referral_pred_labels is not None:
                    rgb_img = to_rgb1a(img)
                    slice_pred_labels_referred = referral_pred_labels[:, :, :, img_slice]
                    if pred_labels_wo_sampling is None:
                        row += 2
                    ax_pred_ref = plt.subplot2grid((rows, columns), (row, 2), rowspan=2, colspan=2)
                    rgb_img_ref_pred, cls_errors = set_error_pixels(rgb_img, slice_pred_labels_referred,
                                                                    slice_true_labels, cls_offset)

                    ax_pred_ref.text(20, 20, 'yellow: RV ({}), blue: Myo ({}), '
                                     'red: LV ({})'.format(cls_errors[1], cls_errors[2], cls_errors[3]),
                                     bbox={'facecolor': 'white', 'pad': 18})
                    ax_pred_ref.imshow(rgb_img_ref_pred, interpolation='nearest')
                    ax_pred_ref.set_aspect('auto')
                    ax_pred_ref.set_title("Prediction errors after referral", **config.title_font_medium)
                    plt.axis('off')
                # ARE WE SHOWING THE BALD value heatmap?
                if use_bald and filtered_std_map is not None:
                    row += 2
                    # CONFUSING!!! not using BALD here anymore (dead code) using it now for u-map where we
                    # add all uncertainties from the separate STDDEV maps per class
                    slice_bald = filtered_std_map[phase, :, :, img_slice]
                    bald_max = np.max(slice_bald)
                    ax4 = plt.subplot2grid((rows, columns), (row, 0), rowspan=2, colspan=2)
                    ax4plot = ax4.imshow(slice_bald, cmap=plt.get_cmap('jet'), vmin=0., vmax=bald_max)
                    # divider = make_axes_locatable(ax)
                    # cax = divider.append_axes("right", size="5%", pad=0.05)
                    # fig.colorbar(ax4plot, cax=cax)
                    ax4.set_aspect('auto')
                    fig.colorbar(ax4plot, ax=ax4, fraction=0.046, pad=0.04)
                    # cb = Colorbar(ax=ax4, mappable=ax4plot, orientation='vertical', ticklocation='right')
                    # cb.set_label(r'Colorbar !', labelpad=10)
                    total_uncertainty = np.count_nonzero(slice_bald)
                    ax4.set_title("Slice {} {}: Max STDDEV u-map (#u={})".format(img_slice+1, str_phase,
                                                                                                 total_uncertainty),
                                  **config.title_font_medium)
                    plt.axis('off')

                if not errors_only:
                    # plot (2) MEAN STD (over classes) next to BALD heatmap, so we can compare the two measures
                    # get the stddev value for the first 4 or last 4 classes (ES/ED) and average over classes (dim0)
                    if phase == 0:
                        mean_slice_stddev = np.mean(slice_stddev[:half_classes], axis=0)
                        # mean_slice_stddev = filtered_add_std_map_es[:, :, img_slice]
                    else:
                        mean_slice_stddev = np.mean(slice_stddev[half_classes:], axis=0)
                        # mean_slice_stddev = filtered_add_std_map_ed[:, :, img_slice]

                    total_uncertainty = np.count_nonzero(mean_slice_stddev)
                    max_mean_stddev = np.max(mean_slice_stddev)
                    # print("Phase {} row {} - MEAN STDDEV heatmap".format(phase, row))
                    ax4a = plt.subplot2grid((rows, columns), (row, 2), rowspan=2, colspan=2)
                    ax4aplot = ax4a.imshow(mean_slice_stddev, cmap=plt.get_cmap('jet'),
                                           vmin=0., vmax=max_mean_stddev)
                    ax4a.set_aspect('auto')
                    fig.colorbar(ax4aplot, ax=ax4a, fraction=0.046, pad=0.04)
                    ax4a.set_title("Slice {} {}: MEAN stddev-values (#u={})".format(img_slice + 1, str_phase,
                                                                                    total_uncertainty),
                                   **config.title_font_medium)
                    plt.axis('off')

                if use_bald and plot_detailed_hists:
                    # last 2 rows: left-> histogram of bald uncertainties or softmax probabilities
                    #              right-> 4 histograms for each class, showing stddev uncertainties or softmax-probs
                    # create histogram
                    if phase == 0:
                        bald_corr = img_slice_probs["es_cor_bald"]
                        bald_err = img_slice_probs["es_err_bald"]
                        # Currently disabled because we're not calculating any p-value statistics
                        # p_value_ttest = img_slice_probs["es_pvalue_ttest_bald"]
                        # p_value_mannwhitu = img_slice_probs["es_pvalue_mwhitu_bald"]
                    else:
                        bald_corr = img_slice_probs["ed_cor_bald"]
                        bald_err = img_slice_probs["ed_err_bald"]
                        # Currently disabled because we're not calculating any p-value statistics
                        # p_value_ttest = img_slice_probs["ed_pvalue_ttest_bald"]
                        # p_value_mannwhitu = img_slice_probs["ed_pvalue_mwhitu_bald"]
                        # print("p-values ttest/Mann-Withney-U {:.2E}/{:.2E} ".format(p_value_ttest, p_value_mannwhitu))
                    # if p_value_ttest >= 0.001 or p_value_mannwhitu >= 0.001:

                    #    str_p_value = "p={:.3f}/{:.3f}".format(p_value_ttest, p_value_mannwhitu)
                    # else:
                    #    str_p_value = None

                    xs = np.linspace(0, bald_max, 20)
                    ax5 = plt.subplot2grid((rows, columns), (row + 3, 0), rowspan=2, colspan=2)
                    # print("Phase {} row {} - BALD histogram".format(phase, row + 2))
                    if bald_err is not None:

                        ax5.hist(bald_err, bins=xs,
                                 label=r"$bald_{{pred(fp+fn)}}({})$".format(bald_err.shape[0])
                                 , color='b', alpha=0.2, histtype='stepfilled')
                        ax5.legend(loc="best", prop={'size': 12})
                        ax5.grid(False)
                    if bald_corr is not None:
                        ax5b = ax5.twinx()
                        ax5b.hist(bald_corr, bins=xs,
                                  label=r"$bald_{{pred(tp)}}({})$".format(bald_corr.shape[0]),
                                  color='g', alpha=0.4, histtype='stepfilled')
                        ax5b.legend(loc=2, prop={'size': 12})
                        ax5b.grid(False)
                    ax5.set_xlabel("BALD value", **config.axis_font)
                    # Currently disabled because we're not calculating any p-value statistics
                    # if str_p_value is not None:
                    #    title_suffix = "(" + str_p_value + ")"
                    #else:
                    #    title_suffix = ""
                    # ax5.set_title("Slice {} {}: Distribution of BALD values ".format(img_slice + 1, str_phase)
                    #              + title_suffix, ** config.title_font_medium)
                    ax5.set_title("Slice {} {}: Distribution of BALD values ".format(img_slice + 1, str_phase)
                                  , ** config.title_font_medium)

                # In case we're only showing the error segmentation map we skip the next part (histograms and
                # stddev uncertainty maps per class. If we only skip the histograms, we visualize the uncertainty
                # maps for each class (at least for the stddev maps.
                if not errors_only:
                    row += 2
                    row_offset = 1
                    col_offset = 0
                    counter = 0
                    if phase == 0:
                        max_stdddev_over_classes = np.max(slice_stddev[:half_classes])

                    else:
                        max_stdddev_over_classes = np.max(slice_stddev[half_classes:])

                    for cls in np.arange(half_classes):
                        if plot_detailed_hists:
                            if phase == 0:
                                if info_type == "uncertainty":
                                    p_err_std = np.array(img_slice_probs["es_err_std"][cls])
                                    p_corr_std = np.array(img_slice_probs["es_cor_std"][cls])
                                else:
                                    p_err_std = np.array(img_slice_probs["es_err_p"][cls])
                                    p_corr_std = np.array(img_slice_probs["es_cor_p"][cls])
                            else:
                                if info_type == "uncertainty":
                                    p_err_std = np.array(img_slice_probs["ed_err_std"][cls])
                                    p_corr_std = np.array(img_slice_probs["ed_cor_std"][cls])
                                else:
                                    p_err_std = np.array(img_slice_probs["ed_err_p"][cls])
                                    p_corr_std = np.array(img_slice_probs["ed_cor_p"][cls])

                        # in the next subplot row we visualize the uncertainties per class
                        # print("phase {} row {} counter {}".format(phase, row, counter))
                        ax3 = plt.subplot2grid((rows, columns), (row, counter), colspan=1)
                        if std_threshold != 0.:
                            std_map_cls = filtered_slice_stddev[cls + cls_offset]
                        else:
                            std_map_cls = slice_stddev[cls + cls_offset]
                        # cmap = plt.get_cmap('jet')
                        # std_rgba_img = cmap(std_map_cls)
                        # std_rgb_img = np.delete(std_rgba_img, 3, 2)
                        # std_map_cls[std_map_cls < std_threshold] = 0
                        ax3plot = ax3.imshow(std_map_cls, vmin=0.,
                                             vmax=max_stdddev_over_classes, cmap=plt.get_cmap('jet'))
                        ax3.set_aspect('auto')
                        if cls == half_classes - 1:
                            fig.colorbar(ax3plot, ax=ax3, fraction=0.046, pad=0.04)
                        ax3.set_title("{} stddev: {} ".format(str_phase, column_lbls[cls]),
                                      **config.title_font_medium)
                        plt.axis("off")
                        # finally in the next row we plot the uncertainty densities per class
                        # print("cls {} col_offset {}".format(cls, col_offset))
                        if plot_detailed_hists:
                            ax2 = plt.subplot2grid((rows, columns), (row + row_offset, 2 + col_offset), colspan=1)
                            col_offset += 1
                            std_max = max(p_err_std.max() if p_err_std.shape[0] > 0 else 0,
                                          p_corr_std.max() if p_corr_std.shape[0] > 0 else 0.)

                            if info_type == "uncertainty":
                                xs = np.linspace(0, std_max, 20)
                            else:
                                xs = np.linspace(0, std_max, 10)
                            if p_err_std is not None:
                                ax2b = ax2.twinx()
                                # p_err_std = p_err_std[np.where((p_err_std>0.1) & (p_err_std< 0.9) )]
                                # if info_type == "uncertainty":
                                #    p_err_std = p_err_std[p_err_std >= std_threshold]

                                ax2b.hist(p_err_std, bins=xs,
                                         label=r"$\sigma_{{pred(fp+fn)}}({})$".format(cls_errors[cls])
                                         ,color="b", alpha=0.2)
                                ax2b.legend(loc=2, prop={'size': 9})

                            if p_corr_std is not None:
                                # if info_type == "uncertainty":
                                #    p_corr_std = p_corr_std[p_corr_std >= std_threshold]
                                # p_corr_std = p_corr_std[np.where((p_corr_std > 0.1) & (p_corr_std < 0.9))]
                                ax2.hist(p_corr_std, bins=xs,
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
                elif not plot_detailed_hists:
                    row += 1
                else:
                    row += 3
            # fig.tight_layout()
            if not errors_only and plot_detailed_hists:
                fig.tight_layout(rect=[0, 0.03, 1, 0.97])
            if do_save:

                fig_img_dir = self._create_figure_dir(image_name)
                fig_name = "analysis_seg_err_slice{}".format(img_slice+1) \
                            + "_mc" + str(mc_samples)
                if std_threshold > 0.:
                    tr_string = "_tr" + str(std_threshold).replace(".", "_")
                    fig_name += tr_string
                    if ref_positives_only:
                        fig_name += "_pos_only"
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
        stddev_maps = self.stddev_maps
        bald_maps = self.bald_maps
        mc_pred_probs = self.mc_pred_probs
        self.images = []
        self.labels = []
        self.pred_labels = []
        self.image_probs_categorized = []
        self.stddev_maps = []
        self.bald_maps = []
        self.mc_pred_probs = []

        if outfile is None:
            num_of_images = len(self.image_names)

            if self.use_dropout:
                outfile = "test_results_{}imgs_mc{}".format(num_of_images, self.mc_samples)
            else:
                outfile = "test_results_{}imgs".format(num_of_images)
            if fold_ids is not None:
                str_fold_ids = "_fold" + "_".join([str(i) for i in fold_ids])
                outfile += str_fold_ids
            if epoch_id is not None:
                str_epoch = "_ep" + str(epoch_id)
                outfile += str_epoch

            if self.referral_threshold > 0.:
                u_thre = "_utr{:.2f}".format(self.referral_threshold).replace(".", "_")
                outfile += u_thre

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
        self.stddev_maps = stddev_maps
        self.bald_maps = bald_maps
        self.mc_pred_probs = mc_pred_probs
        del images
        del labels
        del pred_labels
        del image_probs_categorized

    @staticmethod
    def load_results(path_to_exp, generate_stats=False, verbose=True):
        if verbose:
            print("INFO - Loading results from file {}".format(path_to_exp))
        try:
            with open(path_to_exp, 'rb') as f:
                test_results = dill.load(f)

        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
            print("ERROR - Can't open file {}".format(path_to_exp))
            raise IOError
        if generate_stats:
            if verbose:
                print("INFO - Generating slice statistics for all {} images.".format(test_results.N))
            for image_num in tqdm(np.arange(len(test_results.N))):
                test_results.generate_slice_statistics(image_num)
        if verbose:
            print("INFO - Successfully loaded TestResult object.")
        return test_results

    def show_results(self):

        if self.dice_results is not None:
            mean_dice = self.dice_results[0]
            stddev = self.dice_results[1]
            print("------------------------------------------------------------------------------------------------")
            print("Overall:\t"
                  "dice(RV/Myo/LV): ES {:.2f} ({:.2f})/{:.2f} ({:.2f})/{:.2f} ({:.2f})\t"
                  "ED {:.2f} ({:.2f})/{:.2f} ({:.2f})/{:.2f} ({:.2f})".format(mean_dice[1], stddev[1], mean_dice[2],
                                                                              stddev[2], mean_dice[3], stddev[3],
                                                                              mean_dice[5], stddev[5], mean_dice[6],
                                                                              stddev[6], mean_dice[7], stddev[7]))
            if self.referral:
                mean_dice = self.dice_referral_results[0]
                stddev = self.dice_referral_results[1]
                print("After referral:\t\t\t ES {:.2f} ({:.2f})/{:.2f} ({:.2f})/{:.2f} ({:.2f})\t"
                      "ED {:.2f} ({:.2f})/{:.2f} ({:.2f})/{:.2f} ({:.2f})".format(mean_dice[1], stddev[1], mean_dice[2],
                                                                                  stddev[2], mean_dice[3], stddev[3],
                                                                                  mean_dice[5], stddev[5], mean_dice[6],
                                                                                  stddev[6], mean_dice[7], stddev[7]))
        if self.hd_results is not None:
            mean_hd = self.hd_results[0]
            stddev = self.hd_results[1]
            print("Hausdorff(RV/Myo/LV):\tES {:.2f} ({:.2f})/{:.2f} ({:.2f})/{:.2f} ({:.2f})\t"
                  "ED {:.2f} ({:.2f})/{:.2f} ({:.2f})/{:.2f} ({:.2f})".format(mean_hd[1], stddev[1], mean_hd[2],
                                                                              stddev[2], mean_hd[3], stddev[3],
                                                                              mean_hd[5], stddev[5], mean_hd[6],
                                                                              stddev[6], mean_hd[7], stddev[7]))
            if self.referral:
                mean_hd = self.hd_referral_results[0]
                stddev = self.hd_referral_results[1]
                print("After referral:\t\t"
                      "ES {:.2f} ({:.2f})/{:.2f} ({:.2f})/{:.2f} ({:.2f})\t"
                      "ED {:.2f} ({:.2f})/{:.2f} ({:.2f})/{:.2f} ({:.2f})".format(mean_hd[1], stddev[1], mean_hd[2],
                                                                                  stddev[2], mean_hd[3], stddev[3],
                                                                                  mean_hd[5], stddev[5], mean_hd[6],
                                                                                  stddev[6], mean_hd[7], stddev[7]))
        if self.referral:
            ref_perc_es = self.referral_stats_results[0, 0]
            ref_reduc_es = self.referral_stats_results[0, 1]
            ref_pc_es = self.referral_stats_results[0, 2]
            ref_rc_es = self.referral_stats_results[0, 3]
            # measure indices 0=mean; 1=std; 2=median
            print("Ref ES %: {:.2f} (std={:.2f},med={:.2f})\t"
                  "Red%: {:.2f} (std={:.2f},med={:.2f})\t"
                  "PR/RC {:.2f} ({:.2f})/{:.2f} ({:.2f})".format(ref_perc_es[0], ref_perc_es[1],
                                                                           ref_perc_es[2], ref_reduc_es[0],
                                                                           ref_reduc_es[1],
                                                                           ref_reduc_es[2],
                                               ref_pc_es[0], ref_pc_es[1], ref_rc_es[0], ref_rc_es[1]))
            ref_perc_ed = self.referral_stats_results[1, 0]
            ref_reduc_ed = self.referral_stats_results[1, 1]
            ref_pc_ed = self.referral_stats_results[1, 2]
            ref_rc_ed = self.referral_stats_results[1, 3]
            print("Ref ED %: {:.2f} (std={:.2f},med={:.2f})\t"
                  "Red%: {:.2f} (std={:.2f},med={:.2f})\t"
                  "PR/RC {:.2f} ({:.2f})/{:.2f} ({:.2f})".format(ref_perc_ed[0], ref_perc_ed[1],
                                                                           ref_perc_ed[2], ref_reduc_ed[0],
                                                                           ref_reduc_ed[1],
                                                                           ref_reduc_ed[2],
                                                                 ref_pc_ed[0], ref_pc_ed[1], ref_rc_ed[0],
                                                                 ref_rc_ed[1]))

    def _create_figure_dir(self, image_name):

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


def load_all_results(exper_dict, search_prefix="test_results_25imgs*", print_latex_string=True):
    # LOG_DIR is global variable from outside scope
    overall_dice, std_dice = np.zeros(8), np.zeros(8)
    overall_hd, std_hd = np.zeros(8), np.zeros(8)
    results = []
    for fold_id, exper_id in exper_dict.iteritems():
        input_dir = os.path.join(LOG_DIR, os.path.join(exper_id, config.stats_path))
        search_path = os.path.join(input_dir, search_prefix + ".dll")
        filenames = glob.glob(search_path)
        if len(filenames) != 1:
            raise ValueError("ERROR - Found {} result files for this {} experiment."
                             "Must be 1.".format(len(filenames), exper_id))
        res = TestResults.load_results(filenames[0], verbose=False)
        if res.dice_results is None:
            print("WARNING - Compute mean results for experiment {}".format(search_path))
            res.compute_mean_stats()
        results.append(res)

    if len(results) != 4:
        raise ValueError("ERROR - Loaded {} instead of 4 result files.".format(len(results)))
    for idx, res in enumerate(results):
        overall_dice += res.dice_results[0]
        std_dice += res.dice_results[1]
        overall_hd += res.hd_results[0]
        std_hd += res.hd_results[1]
    overall_dice *= 1. / 4
    std_dice *= 1. / 4
    overall_hd *= 1. / 4
    std_hd *= 1. / 4

    print("Overall:\t"
          "dice(RV/Myo/LV): ES {:.2f} ({:.2f})/{:.2f} ({:.2f})/{:.2f} ({:.2f})\t"
          "ED {:.2f} ({:.2f})/{:.2f} ({:.2f})/{:.2f} ({:.2f})".format(overall_dice[1], std_dice[1], overall_dice[2],
                                                                      std_dice[2], overall_dice[3], std_dice[3],
                                                                      overall_dice[5], std_dice[5], overall_dice[6],
                                                                      std_dice[6], overall_dice[7], std_dice[7]))
    print("Hausdorff(RV/Myo/LV):\tES {:.2f} ({:.2f})/{:.2f} ({:.2f})/{:.2f} ({:.2f})\t"
          "ED {:.2f} ({:.2f})/{:.2f} ({:.2f})/{:.2f} ({:.2f})".format(overall_hd[1], std_hd[1], overall_hd[2],
                                                                      std_hd[2], overall_hd[3], std_hd[3],
                                                                      overall_hd[5], std_hd[5], overall_hd[6],
                                                                      std_hd[6], overall_hd[7], std_hd[7]))

    if print_latex_string:
        latex_line = " & {:.2f} $\pm$ {:.2f} & {:.2f}  $\pm$ {:.2f} & {:.2f} $\pm$ {:.2f} &" \
                     "{:.2f} $\pm$ {:.2f} & {:.2f} $\pm$ {:.2f} & {:.2f} $\pm$ {:.2f} "
        # print Latex strings
        print("----------------------------------------------------------------------------------------------")
        print("INFO - Latex strings")
        print("Dice coefficients")
        print(latex_line.format(overall_dice[1], std_dice[1], overall_dice[2], std_dice[2],
                                overall_dice[3], std_dice[3],
                                overall_dice[5], std_dice[5], overall_dice[6], std_dice[6],
                                overall_dice[7], std_dice[7]))
        print("Hausdorff distance")
        print(latex_line.format(overall_hd[1], std_hd[1], overall_hd[2], std_hd[2],
                                overall_hd[3],
                                std_hd[3],
                                overall_hd[5], std_hd[5], overall_hd[6], std_hd[6],
                                overall_hd[7],
                                std_hd[7]))

    return np.array([overall_dice, std_dice]), np.array([overall_hd, std_hd])


ROOT_DIR = "/home/jorg/repository/dcnn_acdc"
LOG_DIR = os.path.join(ROOT_DIR, "logs")
# overall_dice, overall_hd = load_all_results(exp_mc01_brier)
