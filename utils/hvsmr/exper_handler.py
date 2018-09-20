from collections import OrderedDict
import glob
import os

import numpy as np
import torch
from utils.experiment import ExperimentHandler
from common.common import setSeed
from common.hvsmr.config import config_hvsmr
from utils.hvsmr.test_handler import HVSMRTesthandler


class HVSMRExperimentHandler(ExperimentHandler):

    def __init__(self):
        super(HVSMRExperimentHandler, self).__init__()

    def test(self, checkpoints, test_handler, image_num=0, mc_samples=1, sample_weights=False, compute_hd=False,
             use_seed=False, verbose=False, store_details=False,
             do_filter=True, save_pred_labels=False,
             store_test_results=True):
        """

        :param model:
        :param test_handler:
        :param image_num:
        :param mc_samples:
        :param sample_weights:
        :param compute_hd:

        :param use_seed:
        :param verbose:
        :param do_filter: use post processing 6-connected components on predicted labels (per class)
        :param store_details: if TRUE TestResult object will hold all images/labels/predicted labels, mc-stats etc.
        which basically results in a very large object. Should not be used for evaluation of many test images,
               evaluate the performance on the test set. Variable is used in order to indicate this situation
        most certainly only for 1-3 images in order to generate some figures.
        :param save_pred_labels: save probability maps and predicted segmentation maps to file
        :param store_test_results:
        :param checkpoints: list of model checkpoints that we use for the ensemble test
        :return:
        """

        # if we're lazy and just pass checkpoints as a single number, we convert this here to a list
        if not isinstance(checkpoints, list):
            checkpoints = [checkpoints]

        # -------------------------------------- local procedures END -----------------------------------------
        if use_seed:
            setSeed(4325, self.exper.run_args.cuda)
        if self.test_results is None:
            # TODO
            pass
            # self.test_results = TestResults(self.exper, use_dropout=sample_weights, mc_samples=mc_samples)

        num_of_checkpoints = len(checkpoints)
        # correct the divisor for calculation of stddev when low number of samples (biased), used in np.std
        if num_of_checkpoints * mc_samples <= 25 and mc_samples != 1:
            ddof = 1
        else:
            ddof = 0
        # b_predictions has shape [num_of_checkpoints * mc_samples, classes, width, height, slices]
        b_predictions = np.zeros(tuple([num_of_checkpoints * mc_samples] + [test_handler.num_of_classes] +
                                       list(test_handler.labels[image_num].shape)))
        slice_idx = 0
        # IMPORTANT boolean indicator: if we sample weights during testing we switch of BATCH normalization.
        # We added a new mode in pytorch module.eval method to enable this (see file dilated_cnn.py)
        mc_dropout = sample_weights
        means_test_loss = []
        # batch_generator iterates over the slices of a particular image
        for image_slice, labels_per_class_slice, labels_multiclass_slice in test_handler.batch_generator(image_num):

            # IMPORTANT: if we want to sample weights from the "posterior" over model weights, we
            # need to "tell" pytorch that we use "train" mode even during testing, otherwise dropout is disabled
            b_test_losses = np.zeros(num_of_checkpoints * mc_samples)
            # loop over model checkpoints
            for run_id, checkpoint in enumerate(checkpoints):
                if checkpoint not in self.ensemble_models.keys():
                    self.ensemble_models[checkpoint] = self.load_checkpoint(verbose=False,
                                                                            drop_prob=self.exper.run_args.drop_prob,
                                                                            checkpoint=checkpoint)
                    model = self.ensemble_models[checkpoint]
                else:
                    model = self.ensemble_models[checkpoint]
                sample_offset = run_id * mc_samples
                # generate samples for this checkpoint
                for s in np.arange(mc_samples):
                    # New in pytorch 0.4.0, use local context manager to turn off history tracking
                    with torch.set_grad_enabled(False):
                        test_loss, test_pred = model.do_test(image_slice, labels_per_class_slice,
                                                             voxel_spacing=test_handler.new_voxel_spacing,
                                                             compute_hd=False,
                                                             mc_dropout=mc_dropout)
                    means_test_loss.append(test_loss)
                    b_predictions[s + sample_offset, :, :, :, test_handler.slice_counter] = test_pred.data.cpu().numpy()

                    b_test_losses[s + sample_offset] = test_loss.data.cpu().numpy()

            # mean/std for each pixel for each class
            mc_probs = b_predictions[:, :, :, :, test_handler.slice_counter]
            mean_test_pred, std_test_pred = np.mean(mc_probs, axis=0, keepdims=True), \
                                                np.std(mc_probs, axis=0, ddof=ddof)

            means_test_loss.append(np.mean(b_test_losses))
            test_handler.set_stddev_map(std_test_pred)
            test_handler.set_pred_labels(mean_test_pred, verbose=verbose, do_filter=False)

            slice_idx += 1

        test_accuracy, test_hd, seg_errors = test_handler.get_accuracy(compute_hd=compute_hd, compute_seg_errors=True,
                                                                       do_filter=do_filter)
        means_test_loss = np.mean(np.array(means_test_loss))
        # we only want to save predicted labels when we're not SAMPLING weights
        if save_pred_labels:
            test_handler.save_pred_labels(self.exper.output_dir, mc_dropout=mc_dropout)
            # save probability maps (mean softmax)
            test_handler.save_pred_probs(self.exper.output_dir, mc_dropout=mc_dropout)

        print("Image {} - test loss {:.3f} "
              " dice(RV/Myo/LV):\t {:.2f}/{:.2f}\t"
              "".format(str(image_num + 1) + "-" + test_handler.b_image_name, means_test_loss,
                                               test_accuracy[1], test_accuracy[2]))
        if compute_hd:
            print("\t\t\t\t\t"
                  "Hausdorff(Myo/LV):\t {:.2f}/{:.2f}"
                  "".format(test_hd[1], test_hd[2]))
        if store_test_results:
            pass
            # self.test_results.add_results(test_set.b_image, test_set.b_labels, test_set.b_image_id,
            #                               test_set.b_pred_labels, b_predictions, test_set.b_stddev_map,
            #                               test_accuracy, test_hd, seg_errors=seg_errors,
            #                               store_all=store_details,
            #                               bald_maps=test_set.b_bald_map,
            #                               uncertainty_stats=test_set.b_uncertainty_stats,
            #                               test_accuracy_slices=test_set.b_acc_slices,
            #                               test_hd_slices=test_set.b_hd_slices,
            #                               image_name=test_set.b_image_name,
            #                               referral_accuracy=None, referral_hd=None,
            #                               referral_stats=test_set.referral_stats)

    def create_entropy_maps(self, do_save=False):
        eps = 1e-7
        input_dir = os.path.join(self.exper.config.root_dir,
                                           os.path.join(self.exper.output_dir, config_hvsmr.pred_lbl_dir))
        output_dir = os.path.join(self.exper.config.root_dir,
                                 os.path.join(self.exper.output_dir, config_hvsmr.u_map_dir))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        search_path = os.path.join(input_dir, "*" + "_pred_probs.npz")
        files = glob.glob(search_path)
        if len(files) == 0:
            raise ImportError("ERROR - no predicted probs found in {}".format(search_path))
        self.entropy_maps = OrderedDict()
        min_ent, max_ent = 0, 0
        for fname in glob.glob(search_path):
            try:
                pred_data = np.load(fname)
                pred_probs = pred_data["pred_probs"]
            except IOError:
                print("ERROR - Can't open file {}".format(fname))
            except KeyError:
                print("ERROR - pred_probs is not an existing archive")
            # pred_probs has shape [3, height, width, #slices]: next step compute two entropy maps ES/ED
            # for numerical stability we add eps (tiny) to softmax probability to prevent np.log2 on zero
            # probability values, which result in nan-values. This actually only happens when we trained the model
            # with the soft-dice.
            entropy = (-pred_probs * np.log2(pred_probs + eps)).sum(axis=0)
            entropy = np.nan_to_num(entropy)

            p_min, p_max = np.min(entropy), np.max(entropy)
            if p_min < min_ent:
                min_ent = p_min
            if p_max > max_ent:
                max_ent = p_max
            file_basename = os.path.splitext(os.path.basename(fname))[0]
            patient_id = file_basename[:file_basename.find("_")]
            self.entropy_maps[patient_id] = entropy
        # print("Final min/max values {:.2f}/{:.2f}".format(min_ent, max_ent))
        for patient_id, entropy_map in self.entropy_maps.iteritems():
            # normalize to values between 0 and 0.5, same scale as stddev values
            # p_min, p_max = np.min(self.entropy_maps[patient_id]), np.max(self.entropy_maps[patient_id])
            # print("Before normalize {:.2f}/{:.2f}".format(p_min, p_max))
            self.entropy_maps[patient_id] = ((entropy_map - min_ent) * 1./(max_ent - min_ent)) * 0.4
            # p_min, p_max = np.min(self.entropy_maps[patient_id]), np.max(self.entropy_maps[patient_id])
            # print("After normalize {:.2f}/{:.2f}".format(p_min, p_max))
            if do_save:
                out_fname = os.path.join(output_dir, patient_id + "_entropy_map.npz")
                try:
                    np.savez(out_fname, entropy_map=self.entropy_maps[patient_id])
                except IOError:
                    print("ERROR - Unable to save entropy maps to {}".format(out_fname))
        if do_save:
            print("INFO - Saved all entropy maps to {}".format(output_dir))

    def get_entropy_maps(self, patient_id=None):
        self.entropy_maps = OrderedDict()
        input_dir = os.path.join(self.exper.config.root_dir,
                                  os.path.join(self.exper.output_dir, config_hvsmr.u_map_dir))
        search_suffix = "_entropy_map.npz"
        if patient_id is None:
            search_mask = "*" + search_suffix
        else:
            search_mask = patient_id + search_suffix

        search_path = os.path.join(input_dir, search_mask)
        if len(glob.glob(search_path)) == 0:
            self.create_entropy_maps(do_save=True)
        for fname in glob.glob(search_path):
            try:
                entropy_data = np.load(fname)
                entropy_map = entropy_data["entropy_map"]
            except IOError:
                print("Unable to load entropy maps from {}".format(fname))
            file_basename = os.path.splitext(os.path.basename(fname))[0]
            patient_id = file_basename[:file_basename.find("_")]
            self.entropy_maps[patient_id] = entropy_map

    def get_test_set(self):
        if self.test_set is None:
            fold_id = self.exper.run_args.fold_ids[0]
            self.test_set = HVSMRTesthandler.get_testset_instance(self.exper.config, fold_id, use_cuda=False)

