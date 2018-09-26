from collections import OrderedDict
import glob
import os

import numpy as np
import torch
from utils.experiment import ExperimentHandler
from common.common import setSeed
from common.hvsmr.config import config_hvsmr
from utils.hvsmr.test_handler import HVSMRTesthandler
from in_out.hvsmr.load_data import HVSMR2016DataSet


class HVSMRExperimentHandler(ExperimentHandler):

    def __init__(self):
        super(HVSMRExperimentHandler, self).__init__()

    def test(self, checkpoints, patient_id, mc_samples=1, sample_weights=False, compute_hd=False,
             use_seed=False, verbose=False, store_details=False,
             do_filter=True, save_pred_labels=False,
             store_test_results=False, save_umaps=False):
        """

        :param model:
        :param patient_id:
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
        :param save_umaps: save Bayesian uncertainty maps
        :param checkpoints: list of model checkpoints that we use for the ensemble test
        :return:
        """

        # if we're lazy and just pass checkpoints as a single number, we convert this here to a list
        if not isinstance(checkpoints, list):
            checkpoints = [checkpoints]
        if self.test_set is None:
            self.get_test_set(use_cuda=self.exper.run_args.cuda)
        image_num = self.test_set.trans_dict[patient_id]

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
        b_predictions = np.zeros(tuple([num_of_checkpoints * mc_samples] + [self.test_set.num_of_classes] +
                                       list(self.test_set.labels[image_num].shape)))
        slice_idx = 0
        # IMPORTANT boolean indicator: if we sample weights during testing we switch of BATCH normalization.
        # We added a new mode in pytorch module.eval method to enable this (see file dilated_cnn.py)
        mc_dropout = sample_weights
        means_test_loss = []
        # batch_generator iterates over the slices of a particular image
        for image_slice, labels_per_class_slice, labels_multiclass_slice in self.test_set.batch_generator(image_num):

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
                                                             voxel_spacing=self.test_set.new_voxel_spacing,
                                                             compute_hd=False,
                                                             mc_dropout=mc_dropout)
                    means_test_loss.append(test_loss)
                    b_predictions[s + sample_offset, :, :, :, self.test_set.slice_counter] = test_pred.data.cpu().numpy()

                    b_test_losses[s + sample_offset] = test_loss.data.cpu().numpy()

            # mean/std for each pixel for each class
            mc_probs = b_predictions[:, :, :, :, self.test_set.slice_counter]
            mean_test_pred, std_test_pred = np.mean(mc_probs, axis=0, keepdims=True), \
                                                np.std(mc_probs, axis=0, ddof=ddof)

            means_test_loss.append(np.mean(b_test_losses))
            if sample_weights:
                self.test_set.set_stddev_map(std_test_pred)
            self.test_set.set_pred_labels(mean_test_pred, verbose=verbose, do_filter=False)

            slice_idx += 1

        test_accuracy, test_hd, seg_errors = self.test_set.get_accuracy(compute_hd=compute_hd, compute_seg_errors=True,
                                                                        do_filter=do_filter)
        means_test_loss = np.mean(np.array(means_test_loss))
        # we only want to save predicted labels when we're not SAMPLING weights
        if save_pred_labels:
            self.test_set.save_pred_labels(self.exper.output_dir, mc_dropout=mc_dropout)
            # save probability maps (mean softmax)
            self.test_set.save_pred_probs(self.exper.output_dir, mc_dropout=mc_dropout)
        # if necessary save Bayesian uncertainty maps
        if sample_weights:
            if save_umaps:
                self.save_bayes_umap(self.test_set.b_stddev_map,
                                     patient_id=self.test_set.b_image_name)

        print("Image {} - test loss {:.3f} "
              " dice(RV/Myo/LV):\t {:.2f}/{:.2f}\t"
              "".format(str(image_num + 1) + "-" + self.test_set.b_image_name, means_test_loss,
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

    def create_entropy_maps(self, patient_id=None, do_save=False):
        eps = 1e-7
        self._check_dirs(config_hvsmr)
        if patient_id is None:
            search_path = os.path.join(self.pred_output_dir, "*" + "_pred_probs.npz")
        else:
            search_path = os.path.join(self.pred_output_dir, patient_id + "_pred_probs.npz")
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
        for p_id, entropy_map in self.entropy_maps.iteritems():
            # normalize to values between 0 and 0.5, same scale as stddev values
            # p_min, p_max = np.min(self.entropy_maps[patient_id]), np.max(self.entropy_maps[patient_id])
            # print("Before normalize {:.2f}/{:.2f}".format(p_min, p_max))
            self.entropy_maps[p_id] = ((entropy_map - min_ent) * 1./(max_ent - min_ent)) * 0.4
            # p_min, p_max = np.min(self.entropy_maps[patient_id]), np.max(self.entropy_maps[patient_id])
            # print("After normalize {:.2f}/{:.2f}".format(p_min, p_max))
            if do_save:
                out_fname = os.path.join(self.umap_output_dir, p_id + "_entropy_map.npz")
                try:
                    np.savez(out_fname, entropy_map=self.entropy_maps[p_id])
                except IOError:
                    print("ERROR - Unable to save entropy maps to {}".format(out_fname))
        if do_save:
            print("INFO - Saved all entropy maps to {}".format(self.umap_output_dir))

        if patient_id is not None:
            return self.entropy_maps[patient_id]

    def get_entropy_maps(self, patient_id=None):
        if self.umap_output_dir is None:
            self._check_dirs(config_hvsmr)

        if self.entropy_maps is not None and patient_id is not None:
            if patient_id in self.entropy_maps.keys():
                return self.entropy_maps[patient_id]
        search_suffix = "_entropy_map.npz"
        if patient_id is None:
            search_mask = "*" + search_suffix
        else:
            search_mask = patient_id + search_suffix

        search_path = os.path.join(self.umap_output_dir, search_mask)
        if len(glob.glob(search_path)) == 0:
            self.create_entropy_maps(do_save=True)
        for fname in glob.glob(search_path):
            try:
                entropy_data = np.load(fname)
                entropy_map = entropy_data["entropy_map"]
            except IOError:
                self.info("ERROR - Unable to load entropy maps from {}".format(fname))
            file_basename = os.path.splitext(os.path.basename(fname))[0]
            patient_id = file_basename[:file_basename.find("_")]
            self.entropy_maps[patient_id] = entropy_map
        if patient_id is not None:
            return self.entropy_maps[patient_id]

    def save_bayes_umap(self, u_map, patient_id):
        if self.umap_output_dir is None:
            self._check_dirs(config_hvsmr)
        try:
            #
            filename = patient_id + "_raw" + config_hvsmr.bayes_umap_suffix
            filename = os.path.join(self.umap_output_dir, filename)

            np.savez(filename, u_map=u_map, u_threshold=0.)

        except IOError:
            self.info("ERROR - Unable to save uncertainty maps to {}".format(filename))

    def get_bayes_umaps(self, patient_id=None, aggregate_func=None, force_reload=False):
        if self.umap_output_dir is None:
            self._check_dirs(config_hvsmr)

        if aggregate_func is not None:
            u_maps = self.agg_umaps
            map_suffix = "_" + aggregate_func + config_hvsmr.bayes_umap_suffix
            archive_name = "agg_umap"
        else:
            u_maps = self.u_maps
            archive_name = "u_map"
            map_suffix = "_raw" + config_hvsmr.bayes_umap_suffix
        # may be we already stored the umap for this patient
        if u_maps is not None and patient_id is not None and not force_reload:
            if patient_id in u_maps.keys():
                print(u_maps.keys())
                return u_maps[patient_id]

        if patient_id is None:
            search_path = os.path.join(self.umap_output_dir, "*" + map_suffix)
        else:
            filename = patient_id + map_suffix
            search_path = os.path.join(self.umap_output_dir, filename)

        if len(glob.glob(search_path)) == 0:
            raise ValueError("ERROR - No search result for {}".format(search_path))

        for fname in glob.glob(search_path):
            # get base filename first and then extract patient/name/ID (filename is e.g. patient012_raw_umap.npz)
            file_basename = os.path.splitext(os.path.basename(fname))[0]
            patient_id = file_basename[:file_basename.find("_")]
            try:
                data = np.load(fname)
                u_maps[patient_id] = data[archive_name]
                del data
            except IOError:
                self.info("ERROR - Unable to load uncertainty maps from {}".format(fname))

        if patient_id is not None:
            return u_maps[patient_id]

    def create_agg_bayes_umap(self, patient_id=None, aggregate_func="max"):
        if self.umap_output_dir is None:
            self._check_dirs(config_hvsmr)
        if patient_id is not None:
            u_maps = {patient_id: self.get_bayes_umaps(patient_id=patient_id, aggregate_func=None, force_reload=True)}
        else:
            u_maps = self.get_bayes_umaps(aggregate_func=None)
        for p_id, raw_u_map in u_maps.iteritems():
            # raw_u_map has shape [num_of_classes, w, h, #slices]
            if aggregate_func == "max":
                agg_umap = np.max(raw_u_map, axis=0)
            else:
                agg_umap = np.mean(raw_u_map, axis=0)

            file_name = p_id + "_" + aggregate_func + config_hvsmr.bayes_umap_suffix
            file_name = os.path.join(self.umap_output_dir, file_name)
            try:
                np.savez(file_name, agg_umap=agg_umap)
            except IOError:
                self.info("ERROR - Unable to save aggregated umap file {}".format(file_name))

    def set_root_dir(self, root_dir):
        self.exper.config.root_dir = root_dir
        self.exper.config.data_dir = os.path.join(self.exper.config.root_dir, "data/HVSMR/Folds")

    def get_test_set(self, use_cuda=False):
        if self.test_set is None:
            fold_id = self.exper.run_args.fold_ids[0]
            self.test_set = HVSMRTesthandler.get_testset_instance(self.exper.config, fold_id, use_cuda=use_cuda)

    def get_testset_ids(self):
        val_path = os.path.join(config_hvsmr.data_dir + "fold" + str(self.exper.run_args.fold_ids[0]),
                                os.path.join(HVSMR2016DataSet.val_path, HVSMR2016DataSet.image_path))
        crawl_mask = os.path.join(val_path, "*.nii")
        for test_file in glob.glob(crawl_mask):
            file_name = os.path.splitext(os.path.basename(test_file))[0]
            patient_id = file_name[:file_name.find("_")]
            self.test_set_ids[patient_id] = int(patient_id.strip("patient"))



