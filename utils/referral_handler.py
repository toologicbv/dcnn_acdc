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


class ReferralHandler(object):

    def __init__(self, exper_handler, referral_thresholds=None, test_set=None, verbose=False, do_save=False,
                 num_of_images=None, pos_only=False, aggregate_func="max", patients=None,
                 do_filter_slices=False):
        """

        :param exper_handler:
        :param referral_thresholds:
        :param test_set:
        :param verbose:
        :param do_save:
        :param num_of_images:
        :param pos_only:
        :param aggregate_func:
        :param patients:
        :param do_filter_slices: if true we only refer certain slices of an 3D image (mimick clinical workflow)
        """
        # Overrule!
        if patients is not None:
            num_of_images = None

        self.do_filter_slices = do_filter_slices
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
        self.pos_only = pos_only
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
        self.det_results = ReferralDetailedResults(exper_handler, do_filter_slices=do_filter_slices)

    def __load_pred_labels(self, patient_id):

        search_path = os.path.join(self.pred_labels_input_dir,
                                   patient_id + "_pred_labels_mc.npz")
        pred_labels = load_pred_labels(search_path)

        return pred_labels  # , ref_pred_labels

    def test(self, without_referral=False, verbose=False, referral_threshold=None):
        if without_referral:
            self.dice = np.zeros((self.num_of_images, 2, 4))
            self.hd = np.zeros((self.num_of_images, 2, 4))
        if referral_threshold is not None:
            self.referral_thresholds = [referral_threshold]
        # load patient disease categorization, NOTE: hold ALL patient_ids not just the 25 validation patients
        # seems odd but is the easiest this way
        self.exper_handler.get_patients()
        # get UncertaintyBlobStats object that is essential for slice referral (contains min_area_size) used in
        # procedure "determine_referral_slices"
        if self.do_filter_slices:
            path_to_root_fold = os.path.join(self.exper_handler.exper.config.root_dir, config.data_dir)
            ublob_stats = UncertaintyBlobStats.load(path_to_root_fold)
            ublob_stats.set_min_area_size(filter_type="M")
            self.det_results.ublob_stats = ublob_stats
            # print("Min are size ", ublob_stats.min_area_size)
        else:
            ublob_stats = None

        for referral_threshold in self.referral_thresholds:
            self.referral_threshold = referral_threshold
            self.str_referral_threshold = str(referral_threshold).replace(".", "_")
            print("INFO - Running evaluation with referral for threshold {}"
                  " (do-filter-slices={})".format(self.str_referral_threshold, self.do_filter_slices))
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
                self.test_set.b_image_name = patient_id
                self.test_set.b_orig_spacing = self.test_set.spacings[image_num]
                self.test_set.b_new_spacing = tuple((ACDC2017TestHandler.new_voxel_spacing,
                                                     ACDC2017TestHandler.new_voxel_spacing,
                                                     self.test_set.b_orig_spacing[2]))
                # store the segmentation errors. shape [#slices, #classes]
                self.test_set.b_seg_errors = np.zeros((self.test_set.b_image.shape[3],
                                                       self.test_set.num_of_classes * 2)).astype(np.int)
                self.test_set.b_pred_labels = pred_labels
                if without_referral:
                    test_accuracy, test_hd, seg_errors = \
                        self.test_set.get_accuracy(compute_hd=True, compute_seg_errors=True, do_filter=False,
                                                   compute_slice_metrics=False)
                    self.det_results.org_acc.append(test_accuracy)
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
                else:
                    arr_slice_referrals = None
                self.test_set.b_pred_labels = copy.deepcopy(pred_labels)
                self.test_set.filter_referrals(u_maps=ref_u_map, ref_positives_only=self.pos_only,
                                               referral_threshold=referral_threshold,
                                               arr_slice_referrals=arr_slice_referrals)
                # collect original accuracy/hd on slices for predictions without referral
                # Note: because the self.test_set.property alters each iteration (image) but the object (test_set)
                # is always the same, we need to make a deepcopy of the numpy array, otherwise we end up with the
                # same slice results for all images (error I already ran into).
                self.det_results.org_acc_slices.append(copy.deepcopy(self.test_set.b_acc_slices))
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
                self.det_results.acc_slices.append(np.reshape(acc_slices_ref, (2, 4, -1)))
                # self.det_results.acc_slices.append(acc_slices_ref)
                # print("After ES-RV ", acc_slices_ref[1, :])
                self.det_results.hd_slices.append(np.reshape(hd_slices_ref, (2, 4, -1)))
                b_ref_acc_slices = self.det_results.acc_slices[-1]
                b_acc_slices = self.det_results.org_acc_slices[-1]
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
                self.test_set.save_pred_labels(self.exper_handler.exper.output_dir, u_threshold=referral_threshold,
                                               ref_positives_only=self.pos_only, mc_dropout=True)
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
            self._show_results()
            if self.do_save:
                self.save_results()
                self.det_results.save(self.outfile)
        print("INFO - Done")

    def _show_results(self, test_accuracy=None, image_num=None, msg=""):
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

    def __init__(self, exper_dict, referral_thresholds, pos_only=False, print_latex_string=False,
                 print_results=True, fold=None):
        """

        :param exper_dict:
        :param referral_thresholds:
        :param pos_only:
        :param print_latex_string:
        :param print_results:
        :param fold: to load only a specific fold, testing purposes
        """
        self.referral_thresholds = referral_thresholds
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
        self.pos_only = pos_only
        self.root_dir = config.root_dir
        self.log_dir = os.path.join(self.root_dir, "logs")
        # for all dictionaries the referral_threshold is used as key
        self.dice = OrderedDict()
        self.hd = OrderedDict()
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
            for fold_id, exper_id in self.exper_dict.iteritems():
                # the patient disease classification if we have not done so already
                if self.patients is None:
                    self._get_patient_classification()
                input_dir = os.path.join(self.log_dir, os.path.join(exper_id, config.stats_path))
                if self.pos_only:
                    search_path = os.path.join(input_dir, self.search_prefix + "utr" + str_referral_threshold +
                                               "_pos_only.npz")

                else:
                    search_path = os.path.join(input_dir, self.search_prefix + "utr" + str_referral_threshold + ".npz")
                filenames = glob.glob(search_path)
                if len(filenames) != 1:
                    raise ValueError("ERROR - Found {} result files for this {} experiment (pos-only={}). "
                                     "Must be 1.".format(len(filenames), exper_id, self.pos_only))
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
            overall_dice *= 1. / self.num_of_folds
            std_dice *= 1. / self.num_of_folds
            overall_hd *= 1. / self.num_of_folds
            std_hd *= 1. / self.num_of_folds
            if self.print_results:
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

            self.dice[referral_threshold] = overall_dice
            self.hd[referral_threshold] = overall_hd

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
        for det_result_obj in self.detailed_results:
            for idx, patient_id in enumerate(det_result_obj.patient_ids):
                referral_stats[patient_id] = det_result_obj.referral_stats[idx]
                dice_slices[patient_id] = det_result_obj.acc_slices[idx]
                hd_slices[patient_id] = det_result_obj.hd_slices[idx]
                slice_blobs[patient_id] = det_result_obj.umap_blobs_per_slice[idx]
                org_dice_slices[patient_id] = det_result_obj.org_acc_slices[idx]
                org_hd_slices[patient_id] = det_result_obj.org_hd_slices[idx]
                org_acc[patient_id] = det_result_obj.org_acc[idx]
                org_hd[patient_id] = det_result_obj.org_hd[idx]
        # reset temporary object
        self.referral_stats[referral_threshold] = referral_stats
        self.dice_slices[referral_threshold] = dice_slices
        self.hd_slices[referral_threshold] = hd_slices
        self.slice_blobs[referral_threshold] = slice_blobs
        self.org_dice_slices[referral_threshold] = org_dice_slices
        self.org_hd_slices[referral_threshold] = org_hd_slices
        self.org_dice_img[referral_threshold] = org_acc
        self.org_hd_img[referral_threshold] = org_hd
        self.detailed_results = []

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
        self.org_acc = []
        self.org_hd = []
        self.org_acc_slices = []
        self.org_hd_slices = []
        self.umap_blobs_per_slice = []
        """ 
            Remember: referral_stats has shape [2, 3classes, 13values, #slices]
            1) for positive-only: % referred pixels; 2) % errors reduced; 3) #true labels 4) #pixels above threshold;
            5) #errors before referral 6) #errors after referral 7) F1-value; 8) Precision; 9) Recall; 
            10) #true positives; 11) #false positives; 12) #false negatives 13) #referred pixels 
        """
        self.referral_stats = []
        self.patient_ids = []
        self.fold_id = exper_handler.exper.run_args.fold_ids[0]
        self.dices = []
        self.hds = []
        self.total_blob_uncertainty_es = {'NOR': [0, 0], 'DCM': [0, 0], 'MINF': [0, 0], 'ARV': [0, 0], 'HCM': [0, 0]}
        self.total_num_of_slices = {'NOR': 0, 'DCM': 0, 'MINF': 0, 'ARV': 0, 'HCM': 0}
        self.num_of_slices_referred = {'NOR': [0, 0], 'DCM': [0, 0], 'MINF': [0, 0], 'ARV': [0, 0], 'HCM': [0, 0]}
        self.patient_slices_referred = {}
        self.ublob_stats = None  # UncertaintyBlobStats object. holds important property min_area_size
        # which is calculated based on property filter_type can be M=mean; MD=median or MS=mean+stddev
        self.do_filter_slices = do_filter_slices
        self.save_output_dir = os.path.join(exper_handler.exper.config.root_dir,
                                            os.path.join(exper_handler.exper.output_dir,
                                                         exper_handler.exper.config.stats_path))

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
