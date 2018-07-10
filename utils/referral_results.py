import os
import numpy as np
import glob
import dill
from scipy.ndimage import zoom
from collections import OrderedDict
from config.config import config
from in_out.patient_classification import Patients
from common.common import get_dice_diffs


def rescale_slice_ref_improvement_histograms(es_mean_slice_improvements, ed_mean_slice_improvements,
                                             max_scale=100., do_normalize=True):
    """

    :param es_mean_slice_improvements: dictionary (key=#slices) and
            values = numpy array (shape=#slices) with mean (over classes) dice improvement per slice (in %)
            (e.g. volumes with 10 slices=key=10: np.array with shape [10] and mean % improvements per slice index
    :param ed_mean_slice_improvements: see es_mean_slice_improvements

    :param max_scale
    :param do_normalize: mean centered and divided by range
    :return:
    """
    scaled_hist = np.zeros((2, int(max_scale)))
    for num_of_slices, es_dice_imp in es_mean_slice_improvements.iteritems():
        # org_hist is np.array with shape [2, #slices] ES/ED
        ed_dice_imp = ed_mean_slice_improvements[num_of_slices]
        zoom_factor = float(max_scale) / float(num_of_slices)
        scaled_hist[0] += zoom(es_dice_imp, zoom=zoom_factor, order=1)
        scaled_hist[1] += zoom(ed_dice_imp, zoom=zoom_factor, order=1)

    # scaled_hist = np.multiply(scaled_hist, np.expand_dims(1./np.sum(scaled_hist, axis=1), axis=1))
    if do_normalize:
        s_min = np.min(scaled_hist, axis=1)
        s_max = np.max(scaled_hist, axis=1)
        denominator = 1./(s_max - s_min)
        denominator = np.expand_dims(denominator, axis=1)
        numerator = (scaled_hist - np.expand_dims(s_min, axis=1))
        scaled_hist = numerator * denominator

    return scaled_hist


def rescale_slice_referral_histograms(freq_of_slice_referrals, max_scale=100., do_normalize=True):
    """

    :param freq_of_slice_referrals: dictionary (key=#slices) and
            values = numpy array (shape=#slices) with frequency of referral per slice-index
            (e.g. slice3: 4 times referred)

    :param max_scale
    :param do_normalize: normalize/scale y-values
    :return:
    """
    scaled_hist = np.zeros((2, int(max_scale)))
    for num_of_slices, org_hist in freq_of_slice_referrals.iteritems():
        # org_hist is np.array with shape [2, #slices] ES/ED
        zoom_factor = float(max_scale) / float(num_of_slices)
        scaled_hist[0] += zoom(org_hist[0], zoom=zoom_factor, order=1)
        scaled_hist[1] += zoom(org_hist[1], zoom=zoom_factor, order=1)

    # scaled_hist = np.multiply(scaled_hist, np.expand_dims(1./np.sum(scaled_hist, axis=1), axis=1))
    if do_normalize:
        s_min = np.min(scaled_hist, axis=1)
        s_max = np.max(scaled_hist, axis=1)
        denominator = 1./(s_max - s_min)
        denominator = np.expand_dims(denominator, axis=1)
        numerator = (scaled_hist - np.expand_dims(s_min, axis=1))
        scaled_hist = numerator * denominator

    return scaled_hist


def load_referral_handler_results(abs_path_file, verbose=False):
    if verbose:
        print("INFO - Loading results from file {}".format(abs_path_file))
    try:
        data = np.load(file=abs_path_file)
        ref_dice_results = [data["ref_dice_mean"], data["ref_dice_std"]]
        ref_hd_results = [data["ref_hd_mean"], data["ref_hd_std"]]
        dice_results = [data["ref_dice"], data["ref_hd"]]
        org_dice = data["dice"]
        org_hd = data["hd"]
    except IOError as e:
        print "I/O error({0}): {1}".format(e.errno, e.strerror)
        print("ERROR - Can't open file {}".format(abs_path_file))
        raise IOError

    if verbose:
        print("INFO - Successfully loaded referral results.")
    return ref_dice_results, ref_hd_results, dice_results, org_dice, org_hd


class ReferralResults(object):
    """
        These are currently the DICE results of the MC-0.1 model with Brier loss functional

        NOTE: below we keep the results of the DCNN-MC-0.1 model without referral.
              we use this in result_plots.py plot_referral_results method

    """
    results_dice_wo_referral = np.array([[0, 0.85, 0.88, 0.91], [0, 0.92, 0.86, 0.96]])

    def __init__(self, exper_dict, referral_thresholds, print_latex_string=False,
                 print_results=True, fold=None, slice_filter_type=None, use_entropy_maps=False):
        """

        :param exper_dict:
        :param referral_thresholds:
        :param pos_only:
        :param print_latex_string:
        :param print_results:
        :param fold: to load only a specific fold, testing purposes
        :param use_entropy_maps
        :param slice_filter_type: M=mean; MD=median; MS=mean+stddev
                IMPORTANT: if slice_filter_type is NONE we load the referral results in which WE REFER
                           ALL SLICES!

        """
        self.save_output_dir = config.data_dir
        self.referral_thresholds = referral_thresholds
        self.slice_filter_type = slice_filter_type
        self.use_entropy_maps = use_entropy_maps
        if fold is not None:
            self.exper_dict = {fold: exper_dict[fold]}
            print("WARNING - only loading results for fold {}".format(fold))
        else:
            self.exper_dict = exper_dict
        self.num_of_folds = float(len(self.exper_dict))
        self.fold = fold
        self.patients = None
        self.search_prefix = "ref_test_results_25imgs*"
        self.print_results = print_results
        self.print_latex_string = print_latex_string
        self.pos_only = False
        self.root_dir = config.root_dir
        self.log_dir = os.path.join(self.root_dir, "logs")
        # for all dictionaries the referral_threshold is used as key

        # self.ref_dice_stats and self.ref_hd_stats hold the overall referral results. SO NOT PER PATIENT_ID!
        # same is applicable for org_dice stats
        # key of dictionaries is referral threshold. Values is numpy array of shape [2, 2, 4]
        # where dim0=2 :: index0=ES, index1=stddev
        # where dim1=2 :: index0=mean per class, index1=std per class (BG, RV, MYO, LV)
        self.ref_dice_stats = OrderedDict()
        self.ref_hd_stats = OrderedDict()
        self.org_dice_stats = OrderedDict()
        self.org_hd_stats = OrderedDict()
        # dictionary (referral threshold) of dictionaries (patient_id). SO EVERYTHING SAVED BASE ON PATIENT_ID!
        self.ref_dice = OrderedDict()
        self.ref_hd = OrderedDict()
        self.dice_slices = OrderedDict()
        self.hd_slices = OrderedDict()
        self.slice_blobs = OrderedDict()
        self.referral_stats = OrderedDict()
        self.org_dice_slices = OrderedDict()
        self.org_hd_slices = OrderedDict()
        self.org_dice_img = OrderedDict()
        self.org_hd_img = OrderedDict()
        self.img_slice_improvements = OrderedDict()
        self.img_slice_seg_error_improvements = OrderedDict()
        self.img_slice_improvements_per_dcat = OrderedDict()
        self.es_mean_slice_improvements = OrderedDict()
        self.ed_mean_slice_improvements = OrderedDict()
        self.patient_slices_referred = OrderedDict()
        # slice frequencies e.g. #slice=10 with freq=13
        self.es_slice_freqs = OrderedDict()
        self.ed_slice_freqs = OrderedDict()
        # dictionaries key referral_threshold for holding per disease group statistics
        self.mean_blob_uvalue_per_slice = OrderedDict()
        self.summed_blob_uvalue_per_slice = OrderedDict()
        self.total_num_of_slices = OrderedDict()
        self.num_of_slices_referred = OrderedDict()
        self.patient_slices_referred_per_dcat = OrderedDict()
        self.patient_slice_blobs_filtered = OrderedDict()
        # results per disease category
        self.org_dice_per_dcat = OrderedDict()
        self.org_hd_per_dcat = OrderedDict()
        self.ref_dice_per_dcat = OrderedDict()
        self.ref_hd_per_dcat = OrderedDict()
        self.num_per_category = OrderedDict()
        self.perc_slices_referred = OrderedDict()
        self.perc_slices_referred_per_dcat = OrderedDict()
        self.total_num_of_slices_referred = OrderedDict()
        self.total_num_of_slices_per_dcat = OrderedDict()
        # only used temporally, will be reset after all detailed results have been loaded and processed
        self.detailed_results = []
        self.load_all_ref_results()
        self._compute_improvement_per_img_slice()

    def load_all_ref_results(self):

        print("INFO - Loading referral results for thresholds"
              " {}".format(self.referral_thresholds))
        print("WARNING - referral positives-only={}".format(self.pos_only))
        # IMPORTANT first sort thresholds
        self.referral_thresholds.sort()
        for referral_threshold in self.referral_thresholds:
            str_referral_threshold = str(referral_threshold).replace(".", "_")
            overall_dice, std_dice = np.zeros((2, 4)), np.zeros((2, 4))
            overall_hd, std_hd = np.zeros((2, 4)), np.zeros((2, 4))
            results = []
            # Initialize all dictionaries for this referral threshold. We collect the data for these
            # dictionaries over all FOLDS.
            self.mean_blob_uvalue_per_slice[referral_threshold] = dict(
                [(disease_cat, np.zeros((2, 3))) for disease_cat in config.disease_categories.keys()])
            self.summed_blob_uvalue_per_slice[referral_threshold] = dict(
                [(disease_cat, [[], []]) for disease_cat in config.disease_categories.keys()])
            self.total_num_of_slices_per_dcat[referral_threshold] = dict(
                [(disease_cat, 0) for disease_cat in config.disease_categories.keys()])
            self.num_of_slices_referred[referral_threshold] = dict(
                [(disease_cat, np.zeros(2)) for disease_cat in config.disease_categories.keys()])
            # dictionary of dictionaries (per disease cat) that contains a third dictionary that holds
            # an np.array.dtype(np.bool) with key #slices. We're using this object to construct histograms
            # that represent the distribution over slice indices that are referred.
            self.patient_slices_referred[referral_threshold] = OrderedDict()
            self.patient_slices_referred_per_dcat[referral_threshold] = dict(
                [(disease_cat, OrderedDict()) for disease_cat in config.disease_categories.keys()])
            self.patient_slice_blobs_filtered[referral_threshold] = []
            # results per disease category
            self.org_dice_per_dcat[referral_threshold] = dict(
                [(disease_cat, np.zeros((2, 4))) for disease_cat in config.disease_categories.keys()])
            self.org_hd_per_dcat[referral_threshold] = dict(
                [(disease_cat, np.zeros((2, 4))) for disease_cat in config.disease_categories.keys()])
            self.ref_dice_per_dcat[referral_threshold] = dict(
                [(disease_cat, np.zeros((2, 4))) for disease_cat in config.disease_categories.keys()])
            self.ref_hd_per_dcat[referral_threshold] = dict(
                [(disease_cat, np.zeros((2, 4))) for disease_cat in config.disease_categories.keys()])
            self.num_per_category[referral_threshold] = dict(
                [(disease_cat, 0) for disease_cat in config.disease_categories.keys()])
            self.perc_slices_referred_per_dcat[referral_threshold] = dict(
                [(disease_cat, np.zeros(2)) for disease_cat in config.disease_categories.keys()])
            self.total_num_of_slices_referred[referral_threshold] = np.zeros(2)
            self.total_num_of_slices[referral_threshold] = np.zeros(2)
            # ok, start looping over FOLDS aka experiments
            for fold_id, exper_id in self.exper_dict.iteritems():
                # the patient disease classification if we have not done so already
                # print("Main loop Fold-{}".format(fold_id))
                if self.patients is None:
                    self._get_patient_classification()
                input_dir = os.path.join(self.log_dir, os.path.join(exper_id, config.stats_path))
                if self.slice_filter_type is not None:
                    file_name = self.search_prefix + "utr" + str_referral_threshold + "_" + self.slice_filter_type
                else:
                    # loading referral results when ALL SLICES ARE REFERRED!
                    file_name = self.search_prefix + "utr" + str_referral_threshold
                    if self.use_entropy_maps:
                        file_name += "_entropy_maps"

                search_path = os.path.join(input_dir, file_name + ".npz")
                filenames = glob.glob(search_path)
                # print(search_path)
                if len(filenames) != 1:
                    raise ValueError("ERROR - Found {} result files for this {} experiment (pos-only={},"
                                     "slice-filter={}). Must be 1.".format(len(filenames), exper_id,
                                                                           self.pos_only, self.slice_filter_type))
                # ref_dice_results is a list with 2 objects
                # 1st object: ref_dice_mean has shape [2 (ES/ED), 4 classes],
                # 2nd object: ref_dice_std has shape [2 (ES/ED), 4 classes]
                # the same is applicable to object ref_hd_results
                ref_dice_results, ref_hd_results, dice_results, org_dice, org_hd = \
                    load_referral_handler_results(filenames[0], verbose=False)
                results.append([ref_dice_results, ref_hd_results])
                # load ReferralDetailedResults
                # we also load the ReferralDetailedResults objects, filename is identical to previous only extension
                # is .dll instead of .npz
                det_res_path = filenames[0].replace(".npz", ".dll")
                self.detailed_results.append(ReferralDetailedResults.load_results(det_res_path, verbose=False))

            if len(results) != 4 and self.num_of_folds == 4:
                raise ValueError("ERROR - Loaded {} instead of 4 result files.".format(len(results)))
            for ref_dice_results, ref_hd_results in results:
                ref_dice_mean = ref_dice_results[0]
                overall_dice += ref_dice_mean
                ref_dice_std = ref_dice_results[1]
                std_dice += ref_dice_std
                ref_hd_mean = ref_hd_results[0]
                overall_hd += ref_hd_mean
                ref_hd_std = ref_hd_results[1]
                std_hd += ref_hd_std
            # first process the detailed result objects (merge the four objects, one for each fold)
            self._process_detailed_results(referral_threshold)
            # compute the statistics (dice, hd, mean u-value/slice, % referred slices etc) for each disease category
            # for THIS referral threshold
            self._compute_statistics_per_disease_category(referral_threshold)

            if self.print_results:
                self._print_results(referral_threshold)

    def _process_detailed_results(self, referral_threshold):
        """
        self.detailed_results (list) contains for a specific referral threshold the 4 ReferralDetailedResults objects.
        We add the results per patient_id (key) to a couple of OrderedDicts, properties of the ReferralResults
        object, so we can access results per referral threshold per patient.

        The result are a couple of Dictionaries (e.g. self.dice_slices) with key referral_threshold, that contain
        dictionaries with key patient_id.

        :param referral_threshold:
        :return:
        """
        if len(self.detailed_results) != 4 and self.num_of_folds == 4:
            raise ValueError("ERROR - loaded less than four detailed result objects "
                             "(actually {})".format(len(self.detailed_results)))

        referral_stats = OrderedDict()
        dice_slices = OrderedDict()
        hd_slices = OrderedDict()
        slice_blobs = OrderedDict()
        org_dice_slices = OrderedDict()
        org_hd_slices = OrderedDict()
        org_acc = OrderedDict()
        org_hd = OrderedDict()
        ref_dice = OrderedDict()
        ref_hd = OrderedDict()
        # print("--------------------- {:.2f} --------------".format(referral_threshold))
        for det_result_obj in self.detailed_results:
            # print("_process_detailed_results - len(self.summed_blob_uvalue_per_slice[referral_threshold][ARV][0]")
            # print(len(self.summed_blob_uvalue_per_slice[referral_threshold]["ARV"][0]))
            self._process_disease_categories(det_result_obj, referral_threshold)
            # print(len(self.summed_blob_uvalue_per_slice[referral_threshold]["ARV"][0]))
            for idx, patient_id in enumerate(det_result_obj.patient_ids):
                referral_stats[patient_id] = det_result_obj.referral_stats[idx]
                ref_dice[patient_id] = det_result_obj.dices[idx]
                ref_hd[patient_id] = det_result_obj.hds[idx]
                dice_slices[patient_id] = det_result_obj.acc_slices[idx]
                hd_slices[patient_id] = det_result_obj.hd_slices[idx]
                slice_blobs[patient_id] = det_result_obj.umap_blobs_per_slice[idx]
                org_dice_slices[patient_id] = det_result_obj.org_dice_slices[idx]
                org_hd_slices[patient_id] = det_result_obj.org_hd_slices[idx]
                org_acc[patient_id] = np.reshape(det_result_obj.org_dice[idx], (2, 4))
                org_hd[patient_id] = np.reshape(det_result_obj.org_hd[idx], (2, 4))
                p_disease_cat = self.patients[patient_id]
                if self.slice_filter_type is not None:
                    # if we refer specific slices than we want to collect the statistics. we do that
                    # for each referral-threshold and per disease category
                    self._process_referred_slices(p_disease_cat, det_result_obj.patient_slices_referred[idx],
                                                  referral_threshold)

        # finally store the results for this referral threshold and compute statistics if necessary
        self.referral_stats[referral_threshold] = referral_stats
        self.ref_dice[referral_threshold] = ref_dice
        self.ref_hd[referral_threshold] = ref_hd
        self.dice_slices[referral_threshold] = dice_slices
        self.hd_slices[referral_threshold] = hd_slices
        self.slice_blobs[referral_threshold] = slice_blobs
        self.org_dice_slices[referral_threshold] = org_dice_slices
        self.org_hd_slices[referral_threshold] = org_hd_slices
        self.org_dice_img[referral_threshold] = org_acc
        self.org_hd_img[referral_threshold] = org_hd
        self._compute_dice_hd_statistics(referral_threshold)
        self._compute_slice_referral_hists(referral_threshold)
        # reset temporary object
        self.detailed_results = []

    def _process_referred_slices(self, p_disease_cat, patient_slices_referred, referral_threshold):
        """
        here we stack the boolean arrays of referred slices, separately for each disease category (dict key),
        per #slices (dict key)
        :param p_disease_cat:
        :param patient_slices_referred: np.bool array of shape [2, #slices] dim0=ES/ED
        :param referral_threshold:
        :return:
        """
        num_of_slices = patient_slices_referred.shape[1]
        slices_es = np.expand_dims(patient_slices_referred[0], axis=0)
        slices_ed = np.expand_dims(patient_slices_referred[1], axis=0)
        if num_of_slices in self.patient_slices_referred_per_dcat[referral_threshold][p_disease_cat].keys():
            # add ES
            self.patient_slices_referred_per_dcat[referral_threshold][p_disease_cat][num_of_slices][0].append(slices_es)
            # add ED
            self.patient_slices_referred_per_dcat[referral_threshold][p_disease_cat][num_of_slices][1].append(slices_ed)
        else:
            # create new numpy array for this key
            self.patient_slices_referred_per_dcat[referral_threshold][p_disease_cat][num_of_slices] = \
                [[slices_es], [slices_ed]]

    def _compute_slice_referral_hists(self, referral_threshold):

        for disease_cat, dict_of_arr_bool in self.patient_slices_referred_per_dcat[referral_threshold].iteritems():
            for num_of_slices, arr_bool in dict_of_arr_bool.iteritems():
                es_ref_slice_stats = np.sum(np.array(arr_bool[0]), axis=0, keepdims=False)
                ed_ref_slice_stats = np.sum(np.array(arr_bool[1]), axis=0, keepdims=False)
                ref_slice_stats = np.concatenate((es_ref_slice_stats, ed_ref_slice_stats))

                self.patient_slices_referred_per_dcat[referral_threshold][disease_cat][num_of_slices] = ref_slice_stats
                if num_of_slices in self.patient_slices_referred[referral_threshold].keys():
                    self.patient_slices_referred[referral_threshold][num_of_slices] += ref_slice_stats
                else:
                    self.patient_slices_referred[referral_threshold][num_of_slices] = ref_slice_stats

    def _compute_dice_hd_statistics(self, referral_threshold):

        arr_ref_dices_es, arr_ref_dices_ed = [], []
        arr_ref_hds_es, arr_ref_hds_ed = [], []
        arr_org_dices_es, arr_org_dices_ed = [], []
        arr_org_hds_es, arr_org_hds_ed = [], []
        for patient_id, ref_dice in self.ref_dice[referral_threshold].iteritems():
            arr_ref_dices_es.extend(np.expand_dims(ref_dice[0], axis=0))
            arr_ref_dices_ed.extend(np.expand_dims(ref_dice[1], axis=0))
            arr_ref_hds_es.extend(np.expand_dims(self.ref_hd[referral_threshold][patient_id][0], axis=0))
            arr_ref_hds_ed.extend(np.expand_dims(self.ref_hd[referral_threshold][patient_id][1], axis=0))
            arr_org_dices_es.extend(np.expand_dims(self.org_dice_img[referral_threshold][patient_id][0], axis=0))
            arr_org_dices_ed.extend(np.expand_dims(self.org_dice_img[referral_threshold][patient_id][1], axis=0))
            arr_org_hds_es.extend(np.expand_dims(self.org_hd_img[referral_threshold][patient_id][0], axis=0))
            arr_org_hds_ed.extend(np.expand_dims(self.org_hd_img[referral_threshold][patient_id][1], axis=0))

        arr_ref_dices_es, arr_ref_dices_ed = np.vstack(arr_ref_dices_es), np.vstack(arr_ref_dices_ed)
        arr_ref_hds_es, arr_ref_hds_ed = np.vstack(arr_ref_hds_es), np.vstack(arr_ref_hds_ed)
        arr_org_dices_es, arr_org_dices_ed = np.vstack(arr_org_dices_es), np.vstack(arr_org_dices_ed)
        arr_org_hds_es, arr_org_hds_ed = np.vstack(arr_org_hds_es), np.vstack(arr_org_hds_ed)
        ref_dice_stats_es = np.array([np.mean(arr_ref_dices_es, axis=0), np.std(arr_ref_dices_es, axis=0)])
        ref_dice_stats_ed = np.array([np.mean(arr_ref_dices_ed, axis=0), np.std(arr_ref_dices_ed, axis=0)])
        self.ref_dice_stats[referral_threshold] = np.concatenate((np.expand_dims(ref_dice_stats_es, axis=0),
                                                                  np.expand_dims(ref_dice_stats_ed, axis=0)))
        org_dice_stats_es = np.array([np.mean(arr_org_dices_es, axis=0), np.std(arr_org_dices_es, axis=0)])
        org_dice_stats_ed = np.array([np.mean(arr_org_dices_ed, axis=0), np.std(arr_org_dices_ed, axis=0)])
        self.org_dice_stats[referral_threshold] = np.concatenate((np.expand_dims(org_dice_stats_es, axis=0),
                                                                  np.expand_dims(org_dice_stats_ed, axis=0)))
        ref_hd_stats_es = np.array([np.mean(arr_ref_hds_es, axis=0), np.std(arr_ref_hds_es, axis=0)])
        ref_hd_stats_ed = np.array([np.mean(arr_ref_hds_ed, axis=0), np.std(arr_ref_hds_ed, axis=0)])
        self.ref_hd_stats[referral_threshold] = np.concatenate((np.expand_dims(ref_hd_stats_es, axis=0),
                                                                np.expand_dims(ref_hd_stats_ed, axis=0)))
        org_hd_stats_es = np.array([np.mean(arr_org_hds_es, axis=0), np.std(arr_org_hds_es, axis=0)])
        org_hd_stats_ed = np.array([np.mean(arr_org_hds_ed, axis=0), np.std(arr_org_hds_ed, axis=0)])
        self.org_hd_stats[referral_threshold] = np.concatenate((np.expand_dims(org_hd_stats_es, axis=0),
                                                                np.expand_dims(org_hd_stats_ed, axis=0)))

    def _process_disease_categories(self, det_result_obj, referral_threshold):

        for disease_cat in det_result_obj.num_per_category.keys():
            self.ref_dice_per_dcat[referral_threshold][disease_cat] += det_result_obj.ref_dice_per_dcat[disease_cat]
            self.ref_hd_per_dcat[referral_threshold][disease_cat] += det_result_obj.ref_hd_per_dcat[disease_cat]
            self.org_dice_per_dcat[referral_threshold][disease_cat] += det_result_obj.org_dice_per_dcat[disease_cat]
            self.org_hd_per_dcat[referral_threshold][disease_cat] += det_result_obj.org_hd_per_dcat[disease_cat]
            self.num_per_category[referral_threshold][disease_cat] += det_result_obj.num_per_category[disease_cat]
            self.total_num_of_slices_per_dcat[referral_threshold][disease_cat] += det_result_obj.total_num_of_slices[disease_cat]
            self.num_of_slices_referred[referral_threshold][disease_cat] += det_result_obj.num_of_slices_referred[disease_cat]
            self.summed_blob_uvalue_per_slice[referral_threshold][disease_cat][0].extend(
                det_result_obj.summed_blob_uvalue_per_slice[disease_cat][0])
            self.summed_blob_uvalue_per_slice[referral_threshold][disease_cat][1].extend(
                det_result_obj.summed_blob_uvalue_per_slice[disease_cat][1])
            self.total_num_of_slices_referred[referral_threshold] += det_result_obj.num_of_slices_referred[disease_cat]
            self.total_num_of_slices[referral_threshold] += det_result_obj.total_num_of_slices[disease_cat]

    def _compute_statistics_per_disease_category(self, referral_threshold):
        # print("----------------- referral_threshold {:.2f} -------------------".format(referral_threshold))
        for disease_cat, num_of_cases in self.num_per_category[referral_threshold].iteritems():
            # compute statistics for each group
            if self.slice_filter_type is not None:
                # print(disease_cat)
                # print(np.mean(self.summed_blob_uvalue_per_slice[referral_threshold][disease_cat][0]))
                self.mean_blob_uvalue_per_slice[referral_threshold][disease_cat][0] = \
                    [np.mean(self.summed_blob_uvalue_per_slice[referral_threshold][disease_cat][0]),
                     np.median(self.summed_blob_uvalue_per_slice[referral_threshold][disease_cat][0]),
                     np.std(self.summed_blob_uvalue_per_slice[referral_threshold][disease_cat][0])]
                self.mean_blob_uvalue_per_slice[referral_threshold][disease_cat][1] = \
                    [np.mean(self.summed_blob_uvalue_per_slice[referral_threshold][disease_cat][1]),
                     np.median(self.summed_blob_uvalue_per_slice[referral_threshold][disease_cat][1]),
                     np.std(self.summed_blob_uvalue_per_slice[referral_threshold][disease_cat][1])]

                self.perc_slices_referred_per_dcat[referral_threshold][disease_cat] = \
                    np.round(np.nan_to_num(self.num_of_slices_referred[referral_threshold][disease_cat] *
                                  1. / float(self.total_num_of_slices_per_dcat[referral_threshold][disease_cat])), 2)

            self.ref_dice_per_dcat[referral_threshold][disease_cat] = self.ref_dice_per_dcat[referral_threshold][disease_cat] * 1. / float(num_of_cases)
            self.ref_hd_per_dcat[referral_threshold][disease_cat] = self.ref_hd_per_dcat[referral_threshold][disease_cat] * 1. / float(num_of_cases)
            self.org_dice_per_dcat[referral_threshold][disease_cat] = self.org_dice_per_dcat[referral_threshold][disease_cat] * 1. / float(num_of_cases)
            self.org_hd_per_dcat[referral_threshold][disease_cat] = self.org_hd_per_dcat[referral_threshold][disease_cat] * 1. / float(num_of_cases)

        # compute % referred over all CLASSES
        self.perc_slices_referred[referral_threshold] = \
            np.round(np.nan_to_num(self.total_num_of_slices_referred[referral_threshold] *
                                   1. / self.total_num_of_slices[referral_threshold]), decimals=2)

    def _compute_improvement_per_img_slice(self):

        for referral_threshold in self.referral_thresholds:
            mean_slice_stats_es = OrderedDict()
            mean_slice_stats_ed = OrderedDict()
            # note: slice_stats with key patient_id, will contain numpy array of shape [2, #slices] 0=ES; 1=ED
            img_slice_improvements = OrderedDict()
            img_slice_seg_error_improvements = OrderedDict()
            num_of_slices_es = OrderedDict()
            num_of_slices_ed = OrderedDict()
            dict_ref_dice_slices = self.dice_slices[referral_threshold]
            dict_org_dice_slices = self.org_dice_slices[referral_threshold]
            dict_referral_stats = self.referral_stats[referral_threshold]
            for patient_id, dice_slices in dict_ref_dice_slices.iteritems():
                org_dice_slices = dict_org_dice_slices[patient_id]
                phase = 0  # ES
                # axis=0 is over classes, because we want overall improvements per slice
                diffs_es = np.sum(dice_slices[phase, 1:] - org_dice_slices[phase, 1:], axis=0)
                num_of_slices_es, slice_stats_es = get_dice_diffs(diffs_es, num_of_slices_es,
                                                                  mean_slice_stats_es, phase=phase)  # ES
                phase = 1
                diffs_ed = np.sum(dice_slices[phase, 1:] - org_dice_slices[phase, 1:], axis=0)
                num_of_slices_ed, slice_stats_ed = get_dice_diffs(diffs_ed, num_of_slices_ed,
                                                                  mean_slice_stats_ed, phase=1)  # ES
                img_slice_improvements[patient_id] = np.concatenate((np.expand_dims(diffs_es, axis=0),
                                                                    np.expand_dims(diffs_ed, axis=0)))
                # compute segmentation error reductions
                patient_referral_stats = dict_referral_stats[patient_id]
                errors_es_before = np.sum(patient_referral_stats[0, :, 4, :], axis=0)
                errors_es_after = np.sum(patient_referral_stats[0, :, 5, :], axis=0)
                errors_ed_before = np.sum(patient_referral_stats[1, :, 4, :], axis=0)
                errors_ed_after = np.sum(patient_referral_stats[1, :, 5, :], axis=0)
                seg_errors_diffs_es = errors_es_before - errors_es_after
                seg_errors_diffs_ed = errors_ed_before - errors_ed_after
                img_slice_seg_error_improvements[patient_id] = \
                    np.concatenate((np.expand_dims(seg_errors_diffs_es, axis=0),
                                    np.expand_dims(seg_errors_diffs_ed, axis=0)))

            # average improvements, by dividing through frequency
            for num_slices in slice_stats_es.keys():
                mean_slice_stats_es[num_slices] = mean_slice_stats_es[num_slices] * 1. / num_of_slices_es[num_slices]
                mean_slice_stats_ed[num_slices] = mean_slice_stats_ed[num_slices] * 1. / num_of_slices_ed[num_slices]
            self.es_mean_slice_improvements[referral_threshold] = mean_slice_stats_es
            self.ed_mean_slice_improvements[referral_threshold] = mean_slice_stats_ed
            self.es_slice_freqs[referral_threshold] = num_of_slices_es
            self.ed_slice_freqs[referral_threshold] = num_of_slices_ed
            self.img_slice_improvements[referral_threshold] = img_slice_improvements
            self.img_slice_seg_error_improvements[referral_threshold] = img_slice_seg_error_improvements
            self._compute_improvements_per_disease_cat_slice(referral_threshold)

    def _compute_improvements_per_disease_cat_slice(self, referral_threshold):

        self.img_slice_improvements_per_dcat[referral_threshold] = dict(
            [(disease_cat, OrderedDict()) for disease_cat in config.disease_categories.keys()])
        num_of_slices_per_dcat = dict(
            [(disease_cat, OrderedDict()) for disease_cat in config.disease_categories.keys()])

        for patient_id, slice_dice_improve in self.img_slice_improvements[referral_threshold].iteritems():
            # slice_dice_improve is numpy array with shape [2, #slices]
            disease_cat = self.patients[patient_id]
            num_of_slices = slice_dice_improve.shape[1]
            if num_of_slices in self.img_slice_improvements_per_dcat[referral_threshold][disease_cat]:
                self.img_slice_improvements_per_dcat[referral_threshold][disease_cat][num_of_slices] += \
                    slice_dice_improve
                num_of_slices_per_dcat[disease_cat][num_of_slices] += 1
            else:
                num_of_slices_per_dcat[disease_cat][num_of_slices] = 1
                self.img_slice_improvements_per_dcat[referral_threshold][disease_cat][num_of_slices] = \
                                                        slice_dice_improve

        # compute means
        for disease_cat, dict_slices in num_of_slices_per_dcat.iteritems():
            for num_of_slices, freq_slices in dict_slices.iteritems():
                self.img_slice_improvements_per_dcat[referral_threshold][disease_cat][num_of_slices] *= \
                        1./freq_slices

    def get_dice_referral_dict(self):
        self.referral_thresholds.sort()
        ref_dice = OrderedDict()
        for referral_threshold in self.referral_thresholds:
            ref_dice[referral_threshold] = np.concatenate((np.expand_dims(self.ref_dice_stats[referral_threshold][0][0], axis=0),
                                                           np.expand_dims(self.ref_dice_stats[referral_threshold][1][0], axis=0)))
        return ref_dice

    def _get_patient_classification(self):
        p = Patients()
        p.load(config.data_dir)
        self.patients = p.category

    def _print_results(self, referral_threshold):
        # ref_dice_stats has shape [2, 2, 4] dim0=ES/ED; dim1=4means/4stddevs
        overall_dice = np.concatenate((np.expand_dims(self.ref_dice_stats[referral_threshold][0][0], axis=0),
                                       np.expand_dims(self.ref_dice_stats[referral_threshold][1][0], axis=0)))
        std_dice = np.concatenate((np.expand_dims(self.ref_dice_stats[referral_threshold][0][1], axis=0),
                                   np.expand_dims(self.ref_dice_stats[referral_threshold][1][1], axis=0)))
        overall_hd = np.concatenate((np.expand_dims(self.ref_hd_stats[referral_threshold][0][0], axis=0),
                                     np.expand_dims(self.ref_hd_stats[referral_threshold][1][0], axis=0)))
        std_hd = np.concatenate((np.expand_dims(self.ref_hd_stats[referral_threshold][0][1], axis=0),
                                 np.expand_dims(self.ref_hd_stats[referral_threshold][1][1], axis=0)))
        print("Evaluation with referral threshold {:.2f}".format(referral_threshold))
        print("Overall:\t"
              "dice(RV/Myo/LV): ES {:.2f} ({:.2f})/{:.2f} ({:.2f})/{:.2f} ({:.2f})\t"
              "ED {:.2f} ({:.2f})/{:.2f} ({:.2f})/{:.2f} ({:.2f})".format(overall_dice[0, 1], std_dice[0, 1],
                                                                          overall_dice[0, 2],
                                                                          std_dice[0, 2], overall_dice[0, 3],
                                                                          std_dice[0, 3],
                                                                          overall_dice[1, 1], std_dice[1, 1],
                                                                          overall_dice[1, 2],
                                                                          std_dice[1, 2], overall_dice[1, 3],
                                                                          std_dice[1, 3]))
        print("Hausdorff(RV/Myo/LV):\t\tES {:.2f} ({:.2f})/{:.2f} ({:.2f})/{:.2f} ({:.2f})\t"
              "ED {:.2f} ({:.2f})/{:.2f} ({:.2f})/{:.2f} ({:.2f})".format(overall_hd[0, 1], std_hd[0, 1],
                                                                          overall_hd[0, 2],
                                                                          std_hd[0, 2], overall_hd[0, 3],
                                                                          std_hd[0, 3],
                                                                          overall_hd[1, 1], std_hd[1, 1],
                                                                          overall_hd[1, 2],
                                                                          std_hd[1, 2], overall_hd[1, 3],
                                                                          std_hd[1, 3]))

        if self.print_latex_string:
            latex_line = " & {:.2f} $\pm$ {:.2f} & {:.2f}  $\pm$ {:.2f} & {:.2f} $\pm$ {:.2f} &" \
                         "{:.2f} $\pm$ {:.2f} & {:.2f} $\pm$ {:.2f} & {:.2f} $\pm$ {:.2f} "
            # print Latex strings
            print("----------------------------------------------------------------------------------------------")
            print("INFO - Latex strings")
            print("Dice coefficients")
            print(latex_line.format(overall_dice[0, 1], std_dice[0, 1], overall_dice[0, 2], std_dice[0, 2],
                                    overall_dice[0, 3], std_dice[0, 3],
                                    overall_dice[1, 1], std_dice[1, 1], overall_dice[1, 2], std_dice[1, 2],
                                    overall_dice[1, 3], std_dice[1, 3]))
            print("Hausdorff distance")
            print(latex_line.format(overall_hd[0, 1], std_hd[0, 1], overall_hd[0, 2], std_hd[0, 2],
                                    overall_hd[0, 3],
                                    std_hd[0, 3],
                                    overall_hd[1, 1], std_hd[1, 1], overall_hd[1, 2], std_hd[1, 2],
                                    overall_hd[1, 3],
                                    std_hd[1, 3]))

    def save(self, filename):

        outfile = os.path.join(self.save_output_dir, filename)

        try:
            with open(outfile, 'wb') as f:
                dill.dump(self, f)
            print("INFO - Saved results to {}".format(outfile))
        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
            print("ERROR - can't save results to {}".format(outfile))

    @staticmethod
    def load_results(path_to_exp, verbose=True):
        """
        Loading ReferralResult object
        :param path_to_exp:
        :param verbose:
        :return:
        """
        if verbose:
            print("INFO - Loading ReferralResults object from file {}".format(path_to_exp))
        try:
            with open(path_to_exp, 'rb') as f:
                referral_results = dill.load(f)

        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
            print("ERROR - Can't open file {}".format(path_to_exp))
            raise IOError

        if verbose:
            print("INFO - Successfully loaded ReferralResults object.")
        return referral_results


class ReferralDetailedResults(object):

    """
        We save one of these objects for each referral threshold
    """

    def __init__(self, exper_handler, do_filter_slices=False, debug=False):
        self.debug = debug
        self.acc_slices = []
        self.hd_slices = []
        self.org_dice_slices = []
        self.org_hd_slices = []
        # these are the original blobs per slice not filtered by  ublob_stats.min_area_size[referral_threshold]
        # or config.min_size_blob_area. The obj self.patient_slice_blobs_filtered contains the filtered blob area sizes
        # which can never be more than 5 at the moment (see config.py)
        self.umap_blobs_per_slice = []
        # original dice & hd per patient
        self.org_dice = []
        self.org_hd = []
        """ 
            Remember: referral_stats has shape [2, 3classes, 13values, #slices]
            1) for positive-only: % referred pixels; 2) % errors reduced; 3) #true labels 4) #pixels above threshold;
            5) #errors before referral 6) #errors after referral 7) F1-value; 8) Precision; 9) Recall; 
            10) #true positives; 11) #false positives; 12) #false negatives 13) #referred pixels (same as (4))
        """
        self.referral_stats = []
        self.patient_ids = []
        self.fold_id = exper_handler.exper.run_args.fold_ids[0]
        # referral results dice and hd per patient
        self.dices = []
        self.hds = []
        # we first initialize mean_blob_uvalue_per_slice with two empty lists to collect the sum of u-values per slice.
        # later we compute per disease category the following statistics [mean, median, stddev] and hence for each key
        # mean_blob_uvalue_per_slice will contain a numpy array of shape [2, 3values]
        self.mean_blob_uvalue_per_slice = dict([(disease_cat, np.zeros((2, 3))) for disease_cat in config.disease_categories.keys()])
        self.summed_blob_uvalue_per_slice = dict(
            [(disease_cat, [[], []]) for disease_cat in config.disease_categories.keys()])
        self.total_num_of_slices = dict([(disease_cat, 0) for disease_cat in config.disease_categories.keys()])
        self.num_of_slices_referred = dict([(disease_cat, np.zeros(2)) for disease_cat in config.disease_categories.keys()])
        # contains per patient np.array of bool with shape [2, #slices]
        self.patient_slices_referred = []
        self.patient_slice_blobs_filtered = []
        # results per disease category
        self.org_dice_per_dcat = dict([(disease_cat, np.zeros((2, 4))) for disease_cat in config.disease_categories.keys()])
        self.org_hd_per_dcat = dict([(disease_cat, np.zeros((2, 4))) for disease_cat in config.disease_categories.keys()])
        self.ref_dice_per_dcat = dict([(disease_cat, np.zeros((2, 4))) for disease_cat in config.disease_categories.keys()])
        self.ref_hd_per_dcat = dict([(disease_cat, np.zeros((2, 4))) for disease_cat in config.disease_categories.keys()])
        # mean values per category
        self.mean_org_dice_per_dcat = dict(
            [(disease_cat, np.zeros((2, 4))) for disease_cat in config.disease_categories.keys()])
        self.mean_org_hd_per_dcat = dict(
            [(disease_cat, np.zeros((2, 4))) for disease_cat in config.disease_categories.keys()])
        self.mean_ref_dice_per_dcat = dict(
            [(disease_cat, np.zeros((2, 4))) for disease_cat in config.disease_categories.keys()])
        self.mean_ref_hd_per_dcat = dict(
            [(disease_cat, np.zeros((2, 4))) for disease_cat in config.disease_categories.keys()])
        self.num_per_category = dict([(disease_cat, 0) for disease_cat in config.disease_categories.keys()])
        self.ublob_stats = None  # UncertaintyBlobStats object. holds important property min_area_size
        # which is calculated based on property filter_type can be M=mean; MD=median or MS=mean+stddev
        self.do_filter_slices = do_filter_slices
        self.save_output_dir = os.path.join(exper_handler.exper.config.root_dir,
                                            os.path.join(exper_handler.exper.output_dir,
                                                         exper_handler.exper.config.stats_path))

    def process_patient(self, patient_id, arr_idx, patient_disease_cat, do_refer_slices=False):
        if patient_id != self.patient_ids[arr_idx] and not self.debug:
            raise ValueError("ERROR - ReferralDetailedResults.process_patient. Different patient "
                             "IDs {} != {} with arr_idx {}".format(patient_id, self.patient_ids[arr_idx], arr_idx))
        self.ref_dice_per_dcat[patient_disease_cat] += self.dices[arr_idx]
        self.ref_hd_per_dcat[patient_disease_cat] += self.hds[arr_idx]
        self.org_dice_per_dcat[patient_disease_cat] += np.reshape(self.org_dice[arr_idx], (2, 4))
        self.org_hd_per_dcat[patient_disease_cat] += np.reshape(self.org_hd[arr_idx], (2, 4))
        self.num_per_category[patient_disease_cat] += 1
        self.total_num_of_slices[patient_disease_cat] += self.acc_slices[arr_idx].shape[2]  # number of slices
        if do_refer_slices:
            self.num_of_slices_referred[patient_disease_cat] += np.count_nonzero(self.patient_slices_referred[arr_idx], axis=1)
            # umap_blobs_per_slice has shape: [2, #slices, config.num_of_umap_blobs (5)]
            # first sum over num_of_umap_blobs dim, we'll average later over the number of slices per category
            slice_blobs_es = self.umap_blobs_per_slice[arr_idx][0]
            slice_blobs_ed = self.umap_blobs_per_slice[arr_idx][1]
            self.summed_blob_uvalue_per_slice[patient_disease_cat][0].extend(np.sum(slice_blobs_es, axis=1))
            self.summed_blob_uvalue_per_slice[patient_disease_cat][1].extend(np.sum(slice_blobs_ed, axis=1))

    def compute_results_per_group(self, compute_blob_values=True):
        for disease_cat, num_of_cases in self.num_per_category.iteritems():
            if num_of_cases != 5 and not self.debug:
                raise ValueError("ERROR - Number of patients in group {} should be 5 not {}".format(disease_cat,
                                                                                                     num_of_cases))
            if num_of_cases != 0:
                self.mean_ref_dice_per_dcat[disease_cat] = self.ref_dice_per_dcat[disease_cat] * 1. / float(num_of_cases)
                self.mean_ref_hd_per_dcat[disease_cat] = self.ref_hd_per_dcat[disease_cat] * 1. / float(num_of_cases)
                self.mean_org_dice_per_dcat[disease_cat] = self.org_dice_per_dcat[disease_cat] * 1. / float(num_of_cases)
                self.mean_org_hd_per_dcat[disease_cat] = self.org_hd_per_dcat[disease_cat] * 1. / float(num_of_cases)
                if compute_blob_values:
                    self.mean_blob_uvalue_per_slice[disease_cat][0] = \
                        [np.mean(self.summed_blob_uvalue_per_slice[disease_cat][0]),
                         np.median(self.summed_blob_uvalue_per_slice[disease_cat][0]),
                         np.std(self.summed_blob_uvalue_per_slice[disease_cat][0])]
                    self.mean_blob_uvalue_per_slice[disease_cat][1] = \
                        [np.mean(self.summed_blob_uvalue_per_slice[disease_cat][1]),
                         np.median(self.summed_blob_uvalue_per_slice[disease_cat][1]),
                         np.std(self.summed_blob_uvalue_per_slice[disease_cat][1])]
                else:
                    self.mean_blob_uvalue_per_slice[disease_cat][0] = [0, 0, 0]
                    self.mean_blob_uvalue_per_slice[disease_cat][1] = [0, 0, 0]
            else:
                self.mean_blob_uvalue_per_slice[disease_cat][0] = np.zeros(3)
                self.mean_blob_uvalue_per_slice[disease_cat][1] = np.zeros(3)

    def save(self, filename):

        outfile = os.path.join(self.save_output_dir, filename + ".dll")

        try:
            with open(outfile, 'wb') as f:
                dill.dump(self, f)
            print("INFO - Saved results to {}".format(outfile))
        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
            print("ERROR - can't save results to {}".format(outfile))

    @staticmethod
    def load_results(path_to_exp, verbose=True):
        if verbose:
            print("INFO - Loading detailed referral results from file {}".format(path_to_exp))
        try:
            with open(path_to_exp, 'rb') as f:
                detailed_referral_results = dill.load(f)

        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
            print("ERROR - Can't open file {}".format(path_to_exp))
            raise IOError

        if verbose:
            print("INFO - Successfully loaded ReferralDetailedResults object.")
        return detailed_referral_results

    def show_results_per_disease_category(self):

        for disease_cat in self.num_per_category.keys():
            org_dice = self.mean_org_dice_per_dcat[disease_cat]
            print("------------------------------ Results for class {} -----------------"
                  "-----------------".format(disease_cat))

            perc_slices_referred = np.nan_to_num(self.num_of_slices_referred[disease_cat] *
                                   100./float(self.total_num_of_slices[disease_cat])).astype(np.int)

            print("ES & ED Mean/median u-value {:.2f}/{:.2f} & {:.2f}/{:.2f}"
                  "\t % slices referred {:.2f} & {:.2f}".format(self.mean_blob_uvalue_per_slice[disease_cat][0][0],
                                                                self.mean_blob_uvalue_per_slice[disease_cat][0][1],
                                                                self.mean_blob_uvalue_per_slice[disease_cat][1][0],
                                                                self.mean_blob_uvalue_per_slice[disease_cat][1][1],
                                                                perc_slices_referred[0],
                                                                perc_slices_referred[1]))
            print("without referral - "
                  "dice(RV/Myo/LV):\tES {:.2f}/{:.2f}/{:.2f}\t"
                  "ED {:.2f}/{:.2f}/{:.2f}".format(org_dice[0, 1], org_dice[0, 2],
                                                   org_dice[0, 3], org_dice[1, 1],
                                                   org_dice[1, 2], org_dice[1, 3]))
            ref_dice = self.mean_ref_dice_per_dcat[disease_cat]
            print("   with referral - "
                  "dice(RV/Myo/LV):\tES {:.2f}/{:.2f}/{:.2f}\t"
                  "ED {:.2f}/{:.2f}/{:.2f}".format(ref_dice[0, 1], ref_dice[0, 2],
                                                   ref_dice[0, 3], ref_dice[1, 1],
                                                   ref_dice[1, 2], ref_dice[1, 3]))
            org_hd = self.mean_org_hd_per_dcat[disease_cat]
            print("without referral - HD (RV/Myo/LV):\tES {:.2f}/{:.2f}/{:.2f}\t"
                  "ED {:.2f}/{:.2f}/{:.2f}".format(org_hd[0, 1], org_hd[0, 2],
                                                   org_hd[0, 3], org_hd[1, 1],
                                                   org_hd[1, 2], org_hd[1, 3]))

            ref_hd = self.mean_ref_hd_per_dcat[disease_cat]
            print("   with referral - HD (RV/Myo/LV):\tES {:.2f}/{:.2f}/{:.2f}\t"
                  "ED {:.2f}/{:.2f}/{:.2f}".format(ref_hd[0, 1], ref_hd[0, 2],
                                                   ref_hd[0, 3], ref_hd[1, 1],
                                                   ref_hd[1, 2], ref_hd[1, 3]))
            print(" ")


class SliceReferral(object):
    """
        Quick and dirty solution to store the slice referral percentages that we obtained empirically
        when running referral experiments. Can be improved...
    """

    perc_referred = {0.08: np.array([0.37, 0.32]),
                     0.1: np.array([0.35, 0.31]),
                     0.12: np.array([0.32, 0.29]),
                     0.14: np.array([0.31, 0.28]),
                     0.16: np.array([0.3, 0.26]),
                     0.18: np.array([0.27, 0.23]),
                     0.2: np.array([0.23, 0.2]),
                     0.22: np.array([0.2, 0.18])}
