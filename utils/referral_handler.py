import os
import numpy as np
import copy
from collections import OrderedDict
from common.common import load_pred_labels
from utils.test_handler import ACDC2017TestHandler
from config.config import config
from utils.uncertainty_blobs import UncertaintyBlobStats
from utils.referral_results import ReferralDetailedResults, ReferralResults


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
        self.referral_thresholds.sort()
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

