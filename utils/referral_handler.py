import os
import numpy as np
import glob
import copy
import dill
from collections import OrderedDict
from common.common import load_pred_labels
from utils.test_handler import ACDC2017TestHandler
from config.config import config


class ReferralHandler(object):

    def __init__(self, exper_handler, referral_thresholds=None, test_set=None, verbose=False, do_save=False,
                 num_of_images=None, pos_only=False, aggregate_func="max"):

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
        else:
            self.num_of_images = len(self.test_set.images)
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
        self.det_results = ReferralDetailedResults(exper_handler)

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

        for referral_threshold in self.referral_thresholds:
            self.referral_threshold = referral_threshold
            self.str_referral_threshold = str(referral_threshold).replace(".", "_")
            print("INFO - Running evaluation with referral for threshold {}"
                  " (pos-only={})".format(self.str_referral_threshold, self.pos_only))
            for image_num in np.arange(self.num_of_images):
                patient_id = self.test_set.img_file_names[image_num]
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
                    self.dice[image_num] = np.reshape(test_accuracy, (2, -1))
                    self.hd[image_num] = np.reshape(test_hd, (2, -1))
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
                self.test_set.b_pred_labels = copy.deepcopy(pred_labels)
                self.test_set.filter_referrals(u_maps=ref_u_map, ref_positives_only=self.pos_only,
                                               referral_threshold=referral_threshold)
                # collect original accuracy/hd on slices for predictions without referral
                # Note: because the self.test_set.property alters each iteration (image) but the object (test_set)
                # is always the same, we need to make a deepcopy of the numpy array, otherwise we end up with the
                # same slice results for all images (error I already ran into).
                self.det_results.org_acc_slices.append(copy.deepcopy(self.test_set.b_acc_slices))
                self.det_results.org_hd_slices.append(copy.deepcopy(self.test_set.b_hd_slices))
                self.det_results.referral_stats.append(copy.deepcopy(self.test_set.referral_stats))
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

                self.ref_dice[image_num] = np.reshape(test_accuracy_ref, (2, -1))
                self.ref_hd[image_num] = np.reshape(test_hd_ref, (2, -1))
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
        else:
            self.exper_dict = exper_dict
        self.num_of_folds = float(len(self.exper_dict))
        self.fold = fold
        self.search_prefix = "ref_test_results_25imgs*"
        self.print_results = print_results
        self.print_latex_string = print_latex_string
        self.pos_only = pos_only
        self.root_dir = config.root_dir
        self.log_dir = os.path.join(self.root_dir, "logs")
        self.dice = OrderedDict()
        self.hd = OrderedDict()
        self.dice_slices = OrderedDict()
        self.hd_slices = OrderedDict()
        self.referral_stats = OrderedDict()
        self.org_dice_slices = OrderedDict()
        self.org_hd_slices = OrderedDict()
        self.org_dice_img = OrderedDict()
        self.org_hd_img = OrderedDict()
        # only used temporally, will be reset after all detailed results have been loaded and processed
        self.detailed_results = []
        self.load_all_ref_results()

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
        org_dice_slices = OrderedDict()
        org_hd_slices = OrderedDict()
        org_acc = OrderedDict()
        org_hd = OrderedDict()
        for det_result_obj in self.detailed_results:
            for idx, patient_id in enumerate(det_result_obj.patient_ids):
                referral_stats[patient_id] = det_result_obj.referral_stats[idx]
                dice_slices[patient_id] = det_result_obj.acc_slices[idx]
                hd_slices[patient_id] = det_result_obj.hd_slices[idx]
                org_dice_slices[patient_id] = det_result_obj.org_acc_slices[idx]
                org_hd_slices[patient_id] = det_result_obj.org_hd_slices[idx]
                org_acc[patient_id] = det_result_obj.org_acc[idx]
                org_hd[patient_id] = det_result_obj.org_hd[idx]
        # reset temporary object
        self.referral_stats[referral_threshold] = referral_stats
        self.dice_slices[referral_threshold] = dice_slices
        self.hd_slices[referral_threshold] = hd_slices
        self.org_dice_slices[referral_threshold] = org_dice_slices
        self.org_hd_slices[referral_threshold] = org_hd_slices
        self.org_dice_img[referral_threshold] = org_acc
        self.org_hd_img[referral_threshold] = org_hd
        self.detailed_results = []


class ReferralDetailedResults(object):

    def __init__(self, exper_handler):
        self.acc_slices = []
        self.hd_slices = []
        self.org_acc = []
        self.org_hd = []
        self.org_acc_slices = []
        self.org_hd_slices = []
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
