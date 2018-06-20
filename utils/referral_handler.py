import os
import numpy as np
import glob
import copy
import dill
from collections import OrderedDict
from common.common import load_pred_labels
from utils.test_handler import ACDC2017TestHandler
from config.config import config
from in_out.patient_classification import Patients
from utils.uncertainty_blobs import UncertaintyBlobStats


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


def determine_referral_slices(u_map_blobs, referral_threshold, ublob_stats):
    """

    :param u_map_blobs: has shape [2, #slices, config.num_of_umap_blobs]
    :param referral_threshold
    :param ublob_stats: is an UncertaintyBlobStats object that holds u-statistics for each referral threshold
    :return:
    """
    slice_blobs_es = u_map_blobs[0]  # ES
    slice_blobs_ed = u_map_blobs[1]  # ED

    # filter the blob area values, get rid off really tiny ones > 10 and then sum areas of all blobs in a slice
    slice_blobs_es = (slice_blobs_es * (slice_blobs_es > config.min_size_blob_area)).sum(axis=1)
    slice_blobs_ed = (slice_blobs_ed * (slice_blobs_ed > config.min_size_blob_area)).sum(axis=1)

    ref_slices_idx_es = np.argwhere(slice_blobs_es >= ublob_stats.min_area_size[referral_threshold][0])[:, 0]
    ref_slices_idx_ed = np.argwhere(slice_blobs_ed >= ublob_stats.min_area_size[referral_threshold][1])[:, 0]

    return slice_blobs_es, slice_blobs_ed, ref_slices_idx_es, ref_slices_idx_ed


def create_slice_referral_matrix(es_idx, ed_idx, num_of_slices):
    """
    based on the 2 input arrays es_idx and ed_idx which contain the indices of the image slices that were
    or will be referred, we construct 2 boolean vectors of length #slices indicating which slices were referred.
    Later we'll concatenate those vectors and save them to disk in order to use them during evaluation.
    :param es_idx:
    :param ed_idx:
    :param num_of_slices:
    :return:
    """
    es_referred_slices = np.zeros(num_of_slices).astype(np.bool)
    ed_referred_slices = np.zeros(num_of_slices).astype(np.bool)
    if es_idx is not None and len(es_idx) != 0:
        es_referred_slices[es_idx] = True
    if ed_idx is not None and len(ed_idx) != 0:
        ed_referred_slices[ed_idx] = True
    return es_referred_slices, ed_referred_slices


def compute_ref_results_per_pgroup(ref_dice, ref_hd, org_dice, org_hd, patient_cats):
    """
    Compute referral hd and dice increase for different disease groups.
    ref and org objects are dicts with key patient_id
    value np.array with shape [2, 4 classes].
    :param ref_dice:
    :param ref_hd:
    :param org_dice:
    :param org_hd:
    :param patient_cats: dictionary key patient_id, value disease category
    :return:
    """
    # create return dicts
    org_dice_per_dcat = dict([(disease_cat, np.zeros((2, 4))) for disease_cat in config.disease_categories.keys()])
    org_hd_per_dcat = dict([(disease_cat, np.zeros((2, 4))) for disease_cat in config.disease_categories.keys()])
    ref_dice_per_dcat = dict([(disease_cat, np.zeros((2, 4))) for disease_cat in config.disease_categories.keys()])
    ref_hd_per_dcat = dict([(disease_cat, np.zeros((2, 4))) for disease_cat in config.disease_categories.keys()])
    num_per_category = dict([(disease_cat, 0) for disease_cat in config.disease_categories.keys()])
    for patient_id, disease_cat in patient_cats.iteritems():
        ref_dice_per_dcat[disease_cat] += ref_dice[patient_id]
        ref_hd_per_dcat[disease_cat] += ref_hd[patient_id]
        org_dice_per_dcat[disease_cat] += org_dice[patient_id]
        org_hd_per_dcat[disease_cat] += org_hd[patient_id]
        num_per_category[disease_cat] += 1

    for disease_cat, num_of_cases in num_per_category.keys():
        if num_of_cases != 20:
            raise ValueError("ERROR - Number of patients in group {} should be 20 not {}".format(disease_cat,
                                                                                                 num_of_cases))
        ref_dice_per_dcat[disease_cat] = ref_dice_per_dcat[disease_cat] * 1./float(num_of_cases)
        ref_hd_per_dcat[disease_cat] = ref_hd_per_dcat[disease_cat] * 1./float(num_of_cases)
        org_dice_per_dcat[disease_cat] = org_dice_per_dcat[disease_cat] * 1./float(num_of_cases)
        org_hd_per_dcat[disease_cat] = org_hd_per_dcat[disease_cat] * 1./float(num_of_cases)

    return ref_dice_per_dcat, ref_hd_per_dcat, org_dice_per_dcat, org_hd_per_dcat


class ReferralHandler(object):

    def __init__(self, exper_handler, referral_thresholds=None, test_set=None, verbose=False, do_save=False,
                 num_of_images=None, aggregate_func="max", patients=None):
        """

        :param exper_handler:
        :param referral_thresholds:
        :param test_set:
        :param verbose:
        :param do_save:
        :param num_of_images:
        :param aggregate_func:
        :param patients:

        """
        # Overrule!
        if patients is not None:
            num_of_images = None

        # we will set do_filter_slices and referral_only and slice_filter_type later in the test method
        self.do_filter_slices = False
        self.slice_filter_type = None
        self.referral_only = False

        self.exper_handler = exper_handler
        self.aggregate_func = aggregate_func
        self.referral_threshold = None
        self.str_referral_threshold = None
        if referral_thresholds is None:
            self.referral_thresholds = [0.14, 0.15, 0.16, 0.18, 0.2, 0.22, 0.24]
        else:
            self.referral_thresholds = referral_thresholds
        self.verbose = verbose
        self.do_save = do_save
        self.test_set = test_set
        self.pos_only = False  # currently not in use!
        if self.test_set is None:
            self.test_set = ACDC2017TestHandler.get_testset_instance(exper_handler.exper.config,
                                                                     exper_handler.exper.run_args.fold_ids,
                                                                     load_train=False, load_val=True,
                                                                     batch_size=None, use_cuda=True)
        self.fold_id = exper_handler.exper.run_args.fold_ids[0]
        if num_of_images is not None:
            self.num_of_images = num_of_images
            self.image_range = np.arange(num_of_images)
        else:
            if patients is not None:
                self.image_range = [self.test_set.trans_dict[p_id] for p_id in patients]
                self.num_of_images = len(self.image_range)
                print("INFO - Running for {} only".format(patients))
            else:
                self.num_of_images = len(self.test_set.images)
                self.image_range = np.arange(self.num_of_images)
        self.pred_labels_input_dir = os.path.join(exper_handler.exper.config.root_dir,
                                                  os.path.join(exper_handler.exper.output_dir,
                                                               config.pred_lbl_dir))
        self.save_output_dir = os.path.join(exper_handler.exper.config.root_dir,
                                            os.path.join(exper_handler.exper.output_dir,
                                                         exper_handler.exper.config.stats_path))
        self.dice = None
        self.hd = None
        self.ref_dice = np.zeros((self.num_of_images, 2, 4))
        self.ref_hd = np.zeros((self.num_of_images, 2, 4))
        self.ref_dice_mean = None
        self.ref_dice_std = None
        self.dice_mean = None
        self.dice_std = None
        self.outfile = None
        self.det_results = None

    def __load_pred_labels(self, patient_id):

        search_path = os.path.join(self.pred_labels_input_dir,
                                   patient_id + "_pred_labels_mc.npz")
        pred_labels = load_pred_labels(search_path)

        return pred_labels  # , ref_pred_labels

    def test(self, referral_threshold=None, referral_only=False, do_filter_slices=False,
             slice_filter_type=None, verbose=False):
        """

        :param slice_filter_type: M=mean; MD=median; MS=mean+stddev
        :param do_filter_slices: if True we only refer certain slices of an 3D image (mimick clinical workflow)
                                 if False we refer ALL image slices to medical expert (only for comparison)
        :param verbose:
        :param referral_threshold:
        :param referral_only: If True => then we load filtered u-maps from disk otherwise we create them on the fly
        :return:
        """
        self.do_filter_slices = do_filter_slices
        if self.do_filter_slices:
            self.slice_filter_type = slice_filter_type
        else:
            self.slice_filter_type = None
        if referral_threshold is not None:
            self.referral_thresholds = [referral_threshold]

        # load patient disease categorization, NOTE: hold ALL patient_ids not just the 25 validation patients
        # seems odd but is the easiest this way
        self.exper_handler.get_patients()
        ublob_stats = None
        for referral_threshold in self.referral_thresholds:
            # initialize numpy arrays for computation of dice and hd for results WITH and WITHOUT referral
            self.dice = np.zeros((self.num_of_images, 2, 4))
            self.hd = np.zeros((self.num_of_images, 2, 4))
            self.ref_dice = np.zeros((self.num_of_images, 2, 4))
            self.ref_hd = np.zeros((self.num_of_images, 2, 4))
            self.det_results = ReferralDetailedResults(self.exper_handler, do_filter_slices=self.do_filter_slices)
            # get UncertaintyBlobStats object that is essential for slice referral (contains min_area_size) used in
            # procedure "determine_referral_slices"
            if self.do_filter_slices:
                if ublob_stats is None:
                    path_to_root_fold = os.path.join(self.exper_handler.exper.config.root_dir, config.data_dir)
                    ublob_stats = UncertaintyBlobStats.load(path_to_root_fold)
                    ublob_stats.set_min_area_size(filter_type=self.slice_filter_type)
                    self.det_results.ublob_stats = ublob_stats
                # print("Min are size ", ublob_stats.min_area_size)
            else:
                ublob_stats = None
            self.referral_threshold = referral_threshold
            if referral_only:
                # load the filtered u-maps from disk
                self.exper_handler.get_referral_maps(referral_threshold, per_class=False,
                                                     aggregate_func=self.aggregate_func)
            self.str_referral_threshold = str(referral_threshold).replace(".", "_")
            print("INFO - Running evaluation with referral for threshold {} (referral-only={}/"
                  "do-filter-slices={}/filter_type={})".format(self.str_referral_threshold, referral_only,
                                                               self.do_filter_slices,
                                                               self.slice_filter_type))
            for idx, image_num in enumerate(self.image_range):
                patient_id = self.test_set.img_file_names[image_num]
                # disease classification NOR, ARV...
                patient_cat = self.exper_handler.patients[patient_id]
                self.det_results.patient_ids.append(patient_id)
                pred_labels = self.__load_pred_labels(patient_id)
                # when we call method test_set.filter_referrals which will alter the b_labels object
                # it is important that we make a deepcopy of test_set.labels because the original numpy array will be
                # altered as well
                self.test_set.b_labels = copy.deepcopy(self.test_set.labels[image_num])
                self.test_set.b_image = self.test_set.images[image_num]
                num_of_slices = self.test_set.b_image.shape[3]
                self.test_set.b_image_name = patient_id
                self.test_set.b_orig_spacing = self.test_set.spacings[image_num]
                self.test_set.b_new_spacing = tuple((ACDC2017TestHandler.new_voxel_spacing,
                                                     ACDC2017TestHandler.new_voxel_spacing,
                                                     self.test_set.b_orig_spacing[2]))
                # store the segmentation errors. shape [#slices, #classes]
                self.test_set.b_seg_errors = np.zeros((self.test_set.b_image.shape[3],
                                                       self.test_set.num_of_classes * 2)).astype(np.int)
                self.test_set.b_pred_labels = pred_labels
                # --------------- get dice and hd for patient without referral (for comparison) ---------------
                test_accuracy, test_hd, seg_errors = \
                    self.test_set.get_accuracy(compute_hd=True, compute_seg_errors=True, do_filter=False,
                                               compute_slice_metrics=False)
                self.det_results.org_dice.append(test_accuracy)
                self.det_results.org_hd.append(test_hd)
                self.dice[idx] = np.reshape(test_accuracy, (2, -1))
                self.hd[idx] = np.reshape(test_hd, (2, -1))
                if verbose:
                    print("ES/ED without: {:.2f} {:.2f} {:.2f} / "
                          "{:.2f} {:.2f} {:.2f} ".format(test_accuracy[1], test_accuracy[2], test_accuracy[3],
                                                         test_accuracy[5], test_accuracy[6], test_accuracy[7]))
                if verbose:
                    self._show_results(test_accuracy, image_num, msg="wo referral\t")

                # filter original b_labels based on the u-map with this specific referral threshold
                if not referral_only:
                    # we need to create the filtered u-maps first
                    self.exper_handler.create_filtered_umaps(u_threshold=referral_threshold,
                                                             patient_id=patient_id,
                                                             aggregate_func=self.aggregate_func)
                # generate prediction with referral OF UNCERTAIN, POSITIVES ONLY
                ref_u_map = self.exper_handler.referral_umaps[patient_id]
                # get original uncertainty blob values for all slices ES/ED. These will be essential for referral
                # in case we do this based on slice filtering...
                ref_u_map_blobs = self.exper_handler.ref_map_blobs[patient_id]
                # filter uncertainty blobs in slices
                if self.do_filter_slices:
                    slice_blobs_es, slice_blobs_ed, ref_slices_idx_es, ref_slices_idx_ed = \
                        determine_referral_slices(ref_u_map_blobs, referral_threshold, ublob_stats)
                    print(ref_slices_idx_es + 1)
                    print(ref_slices_idx_ed + 1)
                    arr_slice_referrals = [ref_slices_idx_es, ref_slices_idx_ed]
                    referred_slices_es, referred_slices_ed = \
                        create_slice_referral_matrix(ref_slices_idx_es, ref_slices_idx_ed, num_of_slices)
                    referred_slices = np.concatenate((np.expand_dims(referred_slices_es, axis=0),
                                                      np.expand_dims(referred_slices_ed, axis=0)))
                    self.det_results.patient_slices_referred.append(referred_slices)
                    self.det_results.patient_slice_blobs_filtered.append([slice_blobs_es, slice_blobs_ed])
                else:
                    # we set the slice indices array to None, but in the method filter_referrals we make
                    # sure we refer ALL SLICES then.
                    arr_slice_referrals = None
                    referred_slices = None
                self.test_set.b_pred_labels = copy.deepcopy(pred_labels)
                self.test_set.filter_referrals(u_maps=ref_u_map, ref_positives_only=self.pos_only,
                                               referral_threshold=referral_threshold,
                                               arr_slice_referrals=arr_slice_referrals)
                # collect original accuracy/hd on slices for predictions without referral
                # Note: because the self.test_set.property alters each iteration (image) but the object (test_set)
                # is always the same, we need to make a deepcopy of the numpy array, otherwise we end up with the
                # same slice results for all images (error I already ran into).
                self.det_results.org_dice_slices.append(copy.deepcopy(self.test_set.b_acc_slices))
                self.det_results.org_hd_slices.append(copy.deepcopy(self.test_set.b_hd_slices))
                self.det_results.referral_stats.append(copy.deepcopy(self.test_set.referral_stats))
                # ref_u_map_blobs has shape [2, #slices, config.num_of_umap_blobs (5)]
                self.det_results.umap_blobs_per_slice.append(ref_u_map_blobs)
                # print(self.det_results.umap_blobs_per_slice[-1].astype(np.int))
                # get referral accuracy & hausdorff
                # IMPORTANT: we don't filter the referred labels a 2nd time, because we encountered undesirable
                # problems doing that (somehow it happened that we lost connectivity to upper/lower slices
                # which resulted in removal of all myocardium labels in a slice
                test_accuracy_ref, test_hd_ref, seg_errors_ref, acc_slices_ref, hd_slices_ref = \
                    self.test_set.get_accuracy(compute_hd=True, compute_seg_errors=True, do_filter=False,
                                               compute_slice_metrics=True)
                self.det_results.dices.append(np.reshape(test_accuracy_ref, (2, 4)))
                self.det_results.hds.append(np.reshape(test_hd_ref, (2, 4)))
                self.det_results.acc_slices.append(np.reshape(acc_slices_ref, (2, 4, -1)))
                # self.det_results.acc_slices.append(acc_slices_ref)
                # print("After ES-RV ", acc_slices_ref[1, :])
                self.det_results.hd_slices.append(np.reshape(hd_slices_ref, (2, 4, -1)))
                b_ref_acc_slices = self.det_results.acc_slices[-1]
                b_acc_slices = self.det_results.org_dice_slices[-1]
                # add the patient ref and org results to the (not per slice but per patient/image) to the
                # statistics per disease category
                self.det_results.process_patient(patient_id, idx, patient_cat, do_refer_slices=self.do_filter_slices)
                # b_ref_acc_slices = acc_slices_ref, (2, 4, -1)
                diffs_es = np.sum(b_ref_acc_slices[0, 1:] - b_acc_slices[0, 1:], axis=0)
                diffs_ed = np.sum(b_ref_acc_slices[1, 1:] - b_acc_slices[1, 1:], axis=0)
                check = np.any(diffs_es < 0.)
                if check:
                    print("Ref dice phase={} check={}".format(1, check))
                    print(b_ref_acc_slices[0, 1:])
                    print("Original")
                    print(b_acc_slices[0, 1:])
                    print("Diffs")
                    print(diffs_es)
                check = np.any(diffs_ed < 0.)
                if check:
                    print("Ref dice phase={} check={}".format(1, check))
                    print(b_ref_acc_slices[1, 1:])
                    print("Original")
                    print(b_acc_slices[1, 1:])
                    print("Diffs")
                    print(diffs_ed)
                # save the referred labels
                if self.do_filter_slices:
                    self.test_set.save_referred_slices(self.exper_handler.exper.output_dir,
                                                       referral_threshold=referral_threshold,
                                                       slice_filter_type=self.slice_filter_type,
                                                       referred_slices=referred_slices)
                else:
                    self.test_set.save_pred_labels(self.exper_handler.exper.output_dir, u_threshold=referral_threshold,
                                                   mc_dropout=True)
                if verbose:
                    print("INFO - {} with referral (pos-only={}) using "
                          "threshold {:.2f}".format(patient_id, self.pos_only, referral_threshold))
                    print("ES/ED referral: {:.2f} {:.2f} {:.2f} / {:.2f} {:.2f} {:.2f} ".format(test_accuracy_ref[1],
                                                                                                test_accuracy_ref[2],
                                                                                                test_accuracy_ref[3],
                                                                                                test_accuracy_ref[5],
                                                                                                test_accuracy_ref[6],
                                                                                                test_accuracy_ref[7]))
                    # ref_u_map_blobs has shape [2, #slices, num_of_blobs(5)]
                    errors_es_before = np.sum(self.test_set.referral_stats[0, :, 4, :], axis=0)
                    errors_es_after = np.sum(self.test_set.referral_stats[0, :, 5, :], axis=0)
                    err_diffs = errors_es_before - errors_es_after
                    print("Error diff ES")
                    print(err_diffs)
                    print("Dice diffs ES")
                    print(diffs_es)
                    print("Largest blob per slice ES")
                    print(np.max(ref_u_map_blobs[0], axis=1))
                    # print(ref_u_map_blobs[0, 0])
                    print(ref_u_map_blobs[0, 6])
                    print(b_acc_slices[0, 1:, 6])

                self.ref_dice[idx] = np.reshape(test_accuracy_ref, (2, -1))
                self.ref_hd[idx] = np.reshape(test_hd_ref, (2, -1))
                if verbose:
                    self._show_results(test_accuracy_ref, image_num, msg="with referral")
            print("_compute_result")
            self._compute_result()
            print("det_results.compute_results_per_group")
            self.det_results.compute_results_per_group()
            self._show_results(per_disease_cat=True)
            if self.do_save:
                self.save_results()
                self.det_results.save(self.outfile)
        print("INFO - Done")

    def _show_results(self, test_accuracy=None, image_num=None, msg="", per_disease_cat=False):
        if image_num is not None:
            print("Image {} ({}) - "
                  " dice(RV/Myo/LV):\tES {:.2f}/{:.2f}/{:.2f}\t"
                  "ED {:.2f}/{:.2f}/{:.2f}".format(str(image_num + 1) + "-" + self.test_set.b_image_name, msg,
                                                   test_accuracy[1], test_accuracy[2],
                                                   test_accuracy[3], test_accuracy[5],
                                                   test_accuracy[6], test_accuracy[7]))
        else:
            if self.dice_mean is not None:
                print("Overall result wo referral - "
                      " dice(RV/Myo/LV):\tES {:.2f}/{:.2f}/{:.2f}\t"
                      "ED {:.2f}/{:.2f}/{:.2f}".format(self.dice_mean[0, 1], self.dice_mean[0, 2],
                                                       self.dice_mean[0, 3], self.dice_mean[1, 1],
                                                       self.dice_mean[1, 2], self.dice_mean[1, 3]))

            print("Overall result referral - "
                  " dice(RV/Myo/LV):\tES {:.2f}/{:.2f}/{:.2f}\t"
                  "ED {:.2f}/{:.2f}/{:.2f}".format(self.ref_dice_mean[0, 1], self.ref_dice_mean[0, 2],
                                                   self.ref_dice_mean[0, 3], self.ref_dice_mean[1, 1],
                                                   self.ref_dice_mean[1, 2], self.ref_dice_mean[1, 3]))
            if per_disease_cat:
                self.det_results.show_results_per_disease_category()

    def _compute_result(self):
        self.ref_dice_mean = np.mean(self.ref_dice, axis=0)
        self.ref_dice_std = np.std(self.ref_dice, axis=0)
        self.ref_hd_mean = np.mean(self.ref_hd, axis=0)
        self.ref_hd_std = np.std(self.ref_hd, axis=0)
        if self.dice is not None:
            self.dice_mean = np.mean(self.dice, axis=0)
            self.dice_std = np.std(self.dice, axis=0)
            self.hd_mean = np.mean(self.hd, axis=0)
            self.hd_std = np.std(self.hd, axis=0)

    def save_results(self):

        self.outfile = "ref_test_results_{}imgs_fold{}".format(self.num_of_images, self.fold_id)
        if self.slice_filter_type is not None:
            self.outfile += "_utr{}_{}".format(self.str_referral_threshold, self.slice_filter_type)
        else:
            self.outfile += "_utr{}".format(self.str_referral_threshold)
        if self.pos_only:
            self.outfile += "_pos_only"
        outfile = os.path.join(self.save_output_dir, self.outfile)

        try:
            np.savez(outfile, ref_dice_mean=self.ref_dice_mean, ref_dice_std=self.ref_dice_std,
                     ref_hd_mean=self.ref_hd_mean, ref_hd_std=self.ref_hd_std,
                     ref_dice=self.ref_dice, ref_hd=self.ref_hd,
                     dice=self.dice, hd=self.hd)
            print("INFO - Saved results to {}".format(outfile))
        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
            print("ERROR - can't save results to {}".format(outfile))

    @staticmethod
    def load_results(abs_path_file, verbose=False):
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
                 print_results=True, fold=None, slice_filter_type=None):
        """

        :param exper_dict:
        :param referral_thresholds:
        :param pos_only:
        :param print_latex_string:
        :param print_results:
        :param fold: to load only a specific fold, testing purposes
        :param slice_filter_type: M=mean; MD=median; MS=mean+stddev
                IMPORTANT: if slice_filter_type is NONE we load the referral results in which WE REFER
                           ALL SLICES!

        """
        self.referral_thresholds = referral_thresholds
        self.slice_filter_type = slice_filter_type
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
        self.es_mean_slice_improvements = OrderedDict()
        self.ed_mean_slice_improvements = OrderedDict()
        # slice frequencies e.g. #slice=10 with freq=13
        self.es_slice_freqs = OrderedDict()
        self.ed_slice_freqs = OrderedDict()
        # dictionaries key referral_threshold for holding per disease group statistics
        self.mean_blob_uvalue_per_slice = {}
        self.summed_blob_uvalue_per_slice = {}
        self.total_num_of_slices = {}
        self.num_of_slices_referred = {}
        self.patient_slices_referred = {}
        self.patient_slice_blobs_filtered = {}
        # results per disease category
        self.org_dice_per_dcat = {}
        self.org_hd_per_dcat = {}
        self.ref_dice_per_dcat = {}
        self.ref_hd_per_dcat = {}
        self.num_per_category = {}
        self.perc_slices_referred = {}
        # only used temporally, will be reset after all detailed results have been loaded and processed
        self.detailed_results = []
        self.load_all_ref_results()
        self._compute_improvement_per_img_slice()

    def load_all_ref_results(self):

        print("INFO - Loading referral results for thresholds"
              " {}".format(self.referral_thresholds))
        print("WARNING - referral positives-only={}".format(self.pos_only))
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
            self.total_num_of_slices[referral_threshold] = dict(
                [(disease_cat, 0) for disease_cat in config.disease_categories.keys()])
            self.num_of_slices_referred[referral_threshold] = dict(
                [(disease_cat, np.zeros(2)) for disease_cat in config.disease_categories.keys()])
            # contains per patient np.array of bool with shape [2, #slices]
            self.patient_slices_referred[referral_threshold] = []
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
            self.perc_slices_referred[referral_threshold] = dict(
                [(disease_cat, np.zeros(2)) for disease_cat in config.disease_categories.keys()])
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

                search_path = os.path.join(input_dir, file_name + ".npz")
                filenames = glob.glob(search_path)
                if len(filenames) != 1:
                    raise ValueError("ERROR - Found {} result files for this {} experiment (pos-only={},"
                                     "slice-filter={}). Must be 1.".format(len(filenames), exper_id,
                                                                           self.pos_only, self.slice_filter_type))
                # ref_dice_results is a list with 2 objects
                # 1st object: ref_dice_mean has shape [2 (ES/ED), 4 classes],
                # 2nd object: ref_dice_std has shape [2 (ES/ED), 4 classes]
                # the same is applicable to object ref_hd_results
                ref_dice_results, ref_hd_results, dice_results, org_dice, org_hd = \
                    ReferralHandler.load_results(filenames[0], verbose=False)
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
                print(latex_line.format(overall_hd[0, 1], std_hd[0, 1], overall_hd[0, 2],  std_hd[0, 2],
                                        overall_hd[0, 3],
                                        std_hd[0, 3],
                                        overall_hd[1, 1], std_hd[1, 1], overall_hd[1, 2], std_hd[1, 2],
                                        overall_hd[1, 3],
                                        std_hd[1, 3]))

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
        # reset temporary object
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
        self.detailed_results = []

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
        self.ref_hd_stats[referral_threshold] = np.concatenate((np.expand_dims(org_hd_stats_es, axis=0),
                                                                np.expand_dims(org_hd_stats_ed, axis=0)))

    def _process_disease_categories(self, det_result_obj, referral_threshold):

        for disease_cat in det_result_obj.num_per_category.keys():
            self.ref_dice_per_dcat[referral_threshold][disease_cat] += det_result_obj.ref_dice_per_dcat[disease_cat]
            self.ref_hd_per_dcat[referral_threshold][disease_cat] += det_result_obj.ref_hd_per_dcat[disease_cat]
            self.org_dice_per_dcat[referral_threshold][disease_cat] += det_result_obj.org_dice_per_dcat[disease_cat]
            self.org_hd_per_dcat[referral_threshold][disease_cat] += det_result_obj.org_hd_per_dcat[disease_cat]
            self.num_per_category[referral_threshold][disease_cat] += det_result_obj.num_per_category[disease_cat]
            self.total_num_of_slices[referral_threshold][disease_cat] += det_result_obj.total_num_of_slices[disease_cat]
            self.num_of_slices_referred[referral_threshold][disease_cat] += det_result_obj.num_of_slices_referred[disease_cat]
            self.summed_blob_uvalue_per_slice[referral_threshold][disease_cat][0].extend(
                det_result_obj.summed_blob_uvalue_per_slice[disease_cat][0])
            self.summed_blob_uvalue_per_slice[referral_threshold][disease_cat][1].extend(
                det_result_obj.summed_blob_uvalue_per_slice[disease_cat][1])

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

                self.perc_slices_referred[referral_threshold] = \
                    np.nan_to_num(self.num_of_slices_referred[referral_threshold][disease_cat] *
                                  100. / float(self.total_num_of_slices[referral_threshold][disease_cat])).astype(
                        np.int)

            self.ref_dice_per_dcat[referral_threshold][disease_cat] = self.ref_dice_per_dcat[referral_threshold][disease_cat] * 1. / float(num_of_cases)
            self.ref_hd_per_dcat[referral_threshold][disease_cat] = self.ref_hd_per_dcat[referral_threshold][disease_cat] * 1. / float(num_of_cases)
            self.org_dice_per_dcat[referral_threshold][disease_cat] = self.org_dice_per_dcat[referral_threshold][disease_cat] * 1. / float(num_of_cases)
            self.org_hd_per_dcat[referral_threshold][disease_cat] = self.org_hd_per_dcat[referral_threshold][disease_cat] * 1. / float(num_of_cases)

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


class ReferralDetailedResults(object):

    """
        We save one of these objects for each referral threshold
    """

    def __init__(self, exper_handler, do_filter_slices=False):
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
        if patient_id != self.patient_ids[arr_idx]:
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

    def compute_results_per_group(self):
        for disease_cat, num_of_cases in self.num_per_category.iteritems():
            if num_of_cases != 5:
                raise ValueError("ERROR - Number of patients in group {} should be 20 not {}".format(disease_cat,
                                                                                                     num_of_cases))
            if num_of_cases != 0:
                self.mean_ref_dice_per_dcat[disease_cat] = self.ref_dice_per_dcat[disease_cat] * 1. / float(num_of_cases)
                self.mean_ref_hd_per_dcat[disease_cat] = self.ref_hd_per_dcat[disease_cat] * 1. / float(num_of_cases)
                self.mean_org_dice_per_dcat[disease_cat] = self.org_dice_per_dcat[disease_cat] * 1. / float(num_of_cases)
                self.mean_org_hd_per_dcat[disease_cat] = self.org_hd_per_dcat[disease_cat] * 1. / float(num_of_cases)
                self.mean_blob_uvalue_per_slice[disease_cat][0] = \
                    [np.mean(self.summed_blob_uvalue_per_slice[disease_cat][0]),
                     np.median(self.summed_blob_uvalue_per_slice[disease_cat][0]),
                     np.std(self.summed_blob_uvalue_per_slice[disease_cat][0])]
                self.mean_blob_uvalue_per_slice[disease_cat][1] = \
                    [np.mean(self.summed_blob_uvalue_per_slice[disease_cat][1]),
                     np.median(self.summed_blob_uvalue_per_slice[disease_cat][1]),
                     np.std(self.summed_blob_uvalue_per_slice[disease_cat][1])]
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
