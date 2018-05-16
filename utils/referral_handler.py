import os
import numpy as np
import glob

from common.common import load_pred_labels
from utils.test_handler import ACDC2017TestHandler
from config.config import config


class ReferralHandler(object):

    def __init__(self, exper_handler, referral_thresholds=None, test_set=None, verbose=False, do_save=False,
                 num_of_images=None):

        self.exper_handler = exper_handler
        self.referral_threshold = None
        self.str_referral_threshold = None
        if referral_thresholds is None:
            self.referral_thresholds = [0.16, 0.18, 0.2, 0.22, 0.24]
        else:
            self.referral_thresholds = referral_thresholds
        self.verbose = verbose
        self.do_save = do_save
        self.test_set = test_set
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

    def __load_pred_labels(self, patient_id):

        search_path = os.path.join(self.pred_labels_input_dir,
                                   patient_id + "_pred_labels_mc.npz")
        pred_labels = load_pred_labels(search_path)
        search_path = os.path.join(self.pred_labels_input_dir,
                                   patient_id + "_filtered_pred_labels_mc" + self.str_referral_threshold + ".npz")
        ref_pred_labels = load_pred_labels(search_path)
        return pred_labels, ref_pred_labels

    def test(self, non_referral=False, verbose=False, referral_threshold=None):
        if non_referral:
            self.dice = np.zeros((self.num_of_images, 2, 4))
            self.hd = np.zeros((self.num_of_images, 2, 4))
        if referral_threshold is not None:
            self.referral_thresholds = [referral_threshold]

        for referral_threshold in self.referral_thresholds:
            self.referral_threshold = referral_threshold
            self.str_referral_threshold = str(referral_threshold).replace(".", "_")
            for image_num in np.arange(self.num_of_images):
                patient_id = self.test_set.img_file_names[image_num]
                pred_labels, ref_pred_labels = self.__load_pred_labels(patient_id)
                self.test_set.b_labels = self.test_set.labels[image_num]
                self.test_set.b_image = self.test_set.images[image_num]
                self.test_set.b_image_name = patient_id
                self.test_set.b_orig_spacing = self.test_set.spacings[image_num]
                self.test_set.b_new_spacing = tuple((ACDC2017TestHandler.new_voxel_spacing,
                                                     ACDC2017TestHandler.new_voxel_spacing,
                                                     self.test_set.b_orig_spacing[2]))
                # store the segmentation errors. shape [#slices, #classes]
                self.test_set.b_seg_errors = np.zeros((self.test_set.b_image.shape[3],
                                                       self.test_set.num_of_classes * 2)).astype(np.int)
                if non_referral:
                    self.test_set.b_pred_labels = pred_labels
                    test_accuracy, test_hd, seg_errors = \
                        self.test_set.get_accuracy(compute_hd=True, compute_seg_errors=True, do_filter=False)
                    self.dice[image_num] = np.reshape(test_accuracy, (2, -1))
                    self.hd[image_num] = np.reshape(test_hd, (2, -1))
                    if verbose:
                        self._show_results(test_accuracy, image_num, msg="wo referral\t")

                self.test_set.b_pred_labels = ref_pred_labels
                test_accuracy_ref, test_hd_ref, seg_errors_ref = \
                    self.test_set.get_accuracy(compute_hd=True, compute_seg_errors=True, do_filter=False)
                self.ref_dice[image_num] = np.reshape(test_accuracy_ref, (2, -1))
                self.ref_hd[image_num] = np.reshape(test_hd_ref, (2, -1))
                if verbose:
                    self._show_results(test_accuracy_ref, image_num, msg="with referral")
            self._compute_result()
            self._show_results()
            if self.do_save:
                self.save_results()

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
        outfile = "ref_test_results_{}imgs_fold{}".format(self.num_of_images, self.fold_id)
        outfile += "_utr{}".format(self.str_referral_threshold)
        outfile = os.path.join(self.save_output_dir, outfile)
        try:
            np.savez(outfile, ref_dice_mean=self.ref_dice_mean, ref_dice_std=self.ref_dice_std,
                     ref_hd_mean=self.ref_hd_mean, ref_hd_std=self.ref_hd_std,
                     ref_dice=self.ref_dice, ref_hd=self.ref_hd)
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
        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
            print("ERROR - Can't open file {}".format(abs_path_file))
            raise IOError

        if verbose:
            print("INFO - Successfully loaded referral results.")
        return ref_dice_results, ref_hd_results, dice_results


def load_all_ref_results(exper_dict, referral_threshold, search_prefix="ref_test_results_25imgs*",
                         root_dir="/home/jorg/repository/dcnn_acdc"):

    log_dir = os.path.join(root_dir, "logs")
    str_referral_threshold = str(referral_threshold).replace(".", "_")
    overall_dice, std_dice = np.zeros((2, 4)), np.zeros((2, 4))
    overall_hd, std_hd = np.zeros((2, 4)), np.zeros((2, 4))
    results = []
    for fold_id, exper_id in exper_dict.iteritems():
        input_dir = os.path.join(log_dir, os.path.join(exper_id, config.stats_path))
        search_path = os.path.join(input_dir, search_prefix + "utr" + str_referral_threshold + ".npz")
        filenames = glob.glob(search_path)
        if len(filenames) != 1:
            raise ValueError("ERROR - Found {} result files for this {} experiment."
                             "Must be 1.".format(len(filenames), exper_id))
        # ref_dice_results is a list with 2 objects
        # 1st object: ref_dice_mean has shape [2 (ES/ED), 4 classes],
        # 2nd object: ref_dice_std has shape [2 (ES/ED), 4 classes]
        # the same is applicable to object ref_hd_results
        ref_dice_results, ref_hd_results, _ = ReferralHandler.load_results(filenames[0], verbose=False)
        results.append([ref_dice_results, ref_hd_results])

    if len(results) != 4:
        raise ValueError("ERROR - Loaded {} instead of 4 result files.".format(len(results)))
    for _ in results:
        overall_dice += results[0][0]
        std_dice += results[0, 1]
        overall_hd += results[1, 0]
        std_hd += results[1, 1]
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
