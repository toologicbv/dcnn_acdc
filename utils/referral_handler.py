import os
import numpy as np
import copy
from collections import OrderedDict
from common.common import load_pred_labels, setSeed
from utils.test_handler import ACDC2017TestHandler
from config.config import config
from utils.uncertainty_blobs import UncertaintyBlobStats
from utils.referral_results import ReferralDetailedResults, ReferralResults, SliceReferral


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


def randomly_determine_referral_slices(refer_percs_slices, num_of_slices):
    """

    :param refer_percs_slices: numpy array shape [2] for ES/ED slice referral percentages
    :param num_of_slices:
    :return:
    """
    # get random numbers between 0 and 1 for each slice (ES/ED)
    p_es = np.random.uniform(size=num_of_slices)
    p_ed = np.random.uniform(size=num_of_slices)
    # determine slice indices to be referred
    p_ref_es = (refer_percs_slices[0] * num_of_slices) / float(num_of_slices)
    p_ref_ed = (refer_percs_slices[1] * num_of_slices) / float(num_of_slices)
    ref_slices_idx_es = np.nonzero(p_es < p_ref_es)[0]
    ref_slices_idx_ed = np.nonzero(p_ed < p_ref_ed)[0]
    # create dummy slice_blob objects otherwise other stuff will break
    slice_blobs_es = np.zeros(num_of_slices)
    slice_blobs_ed = np.zeros(num_of_slices)

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
                 num_of_images=None, aggregate_func="max", patients=None, type_of_map=None):
        """

        :param exper_handler:
        :param referral_thresholds:
        :param test_set:
        :param verbose:
        :param do_save:
        :param num_of_images:
        :param aggregate_func:
        :param patients:
        :param type_of_map: determines the maps (u-map/entropy-map) we're using during referral.
                            Possible values: "entropy":
                            "umap": u-maps that are thresholded and post-processed
                            "raw_umap": u-maps that are ONLY thresholded


        """
        # Overrule!
        if patients is not None:
            num_of_images = None

        if type_of_map is None or type_of_map not in ["entropy", "umap", "raw_umap"]:
            raise ValueError("ERROR - type_of_map parameter must be (1) entropy (2) umap or (3) raw_umap"
                             "and cannot be None")

        # we will set do_filter_slices and referral_only and slice_filter_type later in the test method
        self.do_filter_slices = False
        self.slice_filter_type = None
        self.referral_only = False
        self.type_of_map = type_of_map
        # boolean, if true, we use the raw entropy maps to create filtered maps and use these to refer voxels above
        # referral threshold
        if self.type_of_map == "entropy":
            self.use_entropy_maps = True
        else:
            self.use_entropy_maps = False

        if self.type_of_map == "raw_umap":
            self.use_raw_umap = True
        else:
            self.use_raw_umap = False

        if self.use_entropy_maps:
            # in case we use the entropy maps we load pred_labels that we generated with 1 sample/prediction
            self.use_mc_samples = False
        else:
            self.use_mc_samples = True
        self.exper_handler = exper_handler
        # check whether we're dealing with an MC-dropout model, sometimes we use entropy maps to test referral
        if "mc" in exper_handler.exper.run_args.model:
            self.mc_model = True
        else:
            self.mc_model = False
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
                self.debug = True
            else:
                self.debug = False
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

    def __load_pred_labels(self, patient_id, with_mc=True):
        if with_mc:
            search_path = os.path.join(self.pred_labels_input_dir,
                                       patient_id + "_pred_labels_mc.npz")
        else:
            search_path = os.path.join(self.pred_labels_input_dir,
                                       patient_id + "_pred_labels.npz")
        pred_labels = load_pred_labels(search_path)

        return pred_labels  # , ref_pred_labels

    def test(self, referral_threshold=None, referral_only=False, slice_filter_type=None, verbose=False):
        """

        :param slice_filter_type: M=mean; MD=median; MS=mean+stddev; R=Randomly refer slices!
                                  based on empirical % computed by means of other methods

        :param verbose:
        :param referral_threshold:
        :param referral_only: If True => then we load filtered u-maps from disk otherwise we create them on the fly
        :return:
        """

        self.slice_filter_type = slice_filter_type
        if self.slice_filter_type is not None:
            self.do_filter_slices = True
        else:
            self.do_filter_slices = False

        if self.slice_filter_type == "R":
            # set seed
            setSeed(4325)
            # Random referral we need referral percentages
            for ref in self.referral_thresholds:
                if ref not in SliceReferral.perc_referred.keys():
                    raise ValueError("ERROR - slice_filter_type={} but object SliceReferral does not contain"
                                     " referral % for all referral thresholds "
                                     "e.g. {:.2f}".format(self.slice_filter_type, ref))
        if referral_threshold is not None:
            self.referral_thresholds = [referral_threshold]

        # load patient disease categorization, NOTE: hold ALL patient_ids not just the 25 validation patients
        # seems odd but is the easiest this way
        self.exper_handler.get_patients(use_four_digits=self.test_set.generate_flipped_images)
        ublob_stats = None
        self.referral_thresholds.sort()
        for referral_threshold in self.referral_thresholds:
            if self.slice_filter_type == "R":
                # get numpy array shape [2] with ES/ED slice referral percentages
                es_ed_ref_percs = SliceReferral.perc_referred[referral_threshold]
            else:
                es_ed_ref_percs = None
            # initialize numpy arrays for computation of dice and hd for results WITH and WITHOUT referral
            self.dice = np.zeros((self.num_of_images, 2, 4))
            self.hd = np.zeros((self.num_of_images, 2, 4))
            self.ref_dice = np.zeros((self.num_of_images, 2, 4))
            self.ref_hd = np.zeros((self.num_of_images, 2, 4))
            self.det_results = ReferralDetailedResults(self.exper_handler, do_filter_slices=self.do_filter_slices,
                                                       debug=self.debug)
            # get UncertaintyBlobStats object that is essential for slice referral (contains min_area_size) used in
            # procedure "determine_referral_slices"
            if self.do_filter_slices and self.slice_filter_type != "R":
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
                if self.use_entropy_maps:
                    # load entropy maps
                    self.exper_handler.create_entropy_maps(do_save=True)
                    self.exper_handler.get_entropy_maps()
                else:
                    # load the filtered u-maps from disk
                    self.exper_handler.get_referral_maps(referral_threshold, per_class=False,
                                                         aggregate_func=self.aggregate_func,
                                                         use_raw_maps=self.use_raw_umap)
            self.str_referral_threshold = str(referral_threshold).replace(".", "_")
            print("INFO - Running evaluation with referral for threshold {} (referral-only={}/"
                  "do-filter-slices={}/filter_type={}/"
                  "type-of-map={}/use_entropy_maps={}/use_raw_maps={})".format(self.str_referral_threshold,
                                                                               referral_only,
                                                                               self.do_filter_slices,
                                                                               self.slice_filter_type,
                                                                               self.type_of_map,
                                                                               self.use_entropy_maps,
                                                                               self.use_raw_umap))
            for idx, image_num in enumerate(self.image_range):
                patient_id = self.test_set.img_file_names[image_num]
                # disease classification NOR, ARV...
                patient_cat = self.exper_handler.patients[patient_id]
                self.det_results.patient_ids.append(patient_id)
                pred_labels = self.__load_pred_labels(patient_id, self.use_mc_samples)
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
                    if self.use_entropy_maps:
                        # load entropy maps
                        self.exper_handler.create_entropy_maps(do_save=True)
                        self.exper_handler.get_entropy_maps()
                    else:
                        self.exper_handler.create_filtered_umaps(u_threshold=referral_threshold,
                                                                 patient_id=patient_id,
                                                                 aggregate_func=self.aggregate_func)
                # generate prediction with referral OF UNCERTAIN, POSITIVES ONLY
                if self.use_entropy_maps:
                    ref_u_map = self.exper_handler.entropy_maps[patient_id]
                else:
                    ref_u_map = self.exper_handler.referral_umaps[patient_id]

                # filter uncertainty blobs in slices
                if self.do_filter_slices:
                    # get original uncertainty blob values for all slices ES/ED. These will be essential for referral
                    # in case we do this based on slice filtering...
                    ref_u_map_blobs = self.exper_handler.ref_map_blobs[patient_id]
                    if self.slice_filter_type == "R":
                        # we randomly refer slices
                        slice_blobs_es, slice_blobs_ed, ref_slices_idx_es, ref_slices_idx_ed = \
                            randomly_determine_referral_slices(es_ed_ref_percs, num_of_slices)
                    else:
                        # refer slices based on u-value statistics
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
                    ref_u_map_blobs = None
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
                                                   mc_dropout=self.mc_model, used_entropy=self.use_entropy_maps)
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
            self._compute_result()
            self.det_results.compute_results_per_group(compute_blob_values=self.do_filter_slices)
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
        if self.use_entropy_maps:
            self.outfile += "_entropy_maps"
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

