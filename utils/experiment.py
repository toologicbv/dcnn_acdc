import sys
import time
import torch
import numpy as np
import os
import glob
import shutil
import copy

from collections import OrderedDict
from datetime import datetime
from pytz import timezone
import dill
from common.parsing import create_def_argparser, run_dict

from common.common import create_logger, create_exper_label, setSeed, load_pred_labels
from config.config import config, DEFAULT_DCNN_MC_2D
from utils.batch_handlers import TwoDimBatchHandler, BatchStatistics
from utils.generate_uncertainty_maps import InferenceGenerator, ImageUncertainties, OutOfDistributionSlices
from utils.post_processing import filter_connected_components, detect_largest_umap_areas
import models.dilated_cnn
import models.hvsmr.dilated_cnn
from utils.test_results import TestResults
from common.acquisition_functions import bald_function
from plotting.uncertainty_plots import analyze_slices
from plotting.main_seg_results import plot_seg_erros_uncertainties
from in_out.load_data import ACDC2017DataSet
from utils.hvsmr.batch_handler import HVSMRTwoDimBatchHandler
from utils.test_handler import ACDC2017TestHandler
from in_out.patient_classification import Patients
from utils.detector.generate_dt_maps import generate_dt_maps, determine_target_voxels


class ExperimentHandler(object):

    exp_filename = "exper_stats"

    def __init__(self):

        self.exper = None
        self.u_maps = None
        self.pred_prob_maps = None
        self.pred_labels = None
        self.referral_umaps = None
        self.entropy_maps = None
        self.agg_umaps = None
        self.dt_maps = None
        # for ROI learning of areas in automatic segmentations that we need to inspect after inference
        self.target_roi_maps = None
        self.ref_map_blobs = None
        self.test_results = None
        self.test_set = None
        self.logger = None
        self.model_name = None
        self.num_val_runs = 0
        self.saved_model_state_dict = None
        self.ensemble_models = OrderedDict()
        self.test_set_ids = {}
        self.patients = None
        self.referred_slices = None
        # for u-maps/e-maps
        self.umap_output_dir = None
        # for Hausdorff distance maps
        self.dt_map_dir = None
        # dir for predicted probs and labels
        self.pred_output_dir = None
        # ROI areas of the automatic references generated for the test set. We use this for ROI learning of areas
        # to be inspected after segmentation
        self.troi_map_dir = None
        self._check_maps()

    def _check_maps(self):
        if self.u_maps is None:
            self.u_maps = OrderedDict()
        if self.agg_umaps is None:
            self.agg_umaps = OrderedDict()
        if self.entropy_maps is None:
            self.entropy_maps = OrderedDict()
        if self.dt_maps is None:
            self.dt_maps = OrderedDict()
        if self.target_roi_maps is None:
            self.target_roi_maps = OrderedDict()

    def _check_dirs(self, config_env):
        if self.umap_output_dir is None:
            self.umap_output_dir = os.path.join(self.exper.config.root_dir,
                                                os.path.join(self.exper.output_dir, config_env.u_map_dir))
        if not os.path.isdir(self.umap_output_dir):
            os.mkdir(self.umap_output_dir)

        # predicted labels/masks
        if self.pred_output_dir is None:
            self.pred_output_dir = os.path.join(self.exper.config.root_dir, os.path.join(self.exper.output_dir,
                                                                                         config_env.pred_lbl_dir))
        if not os.path.isdir(self.pred_output_dir):
            os.mkdir(self.pred_output_dir)

        # Hausdorff Distance maps (used in detector application)
        if self.dt_map_dir is None:
            self.dt_map_dir = os.path.join(self.exper.config.root_dir, os.path.join(self.exper.output_dir,
                                                                                     config_env.dt_map_dir))
        if not os.path.isdir(self.dt_map_dir):
            os.mkdir(self.dt_map_dir)

        if self.troi_map_dir is None:
            self.troi_map_dir = os.path.join(self.exper.config.root_dir, os.path.join(self.exper.output_dir,
                                                                                    config_env.troi_map_dir))
        if not os.path.isdir(self.troi_map_dir):
            os.mkdir(self.troi_map_dir)

    def set_exper(self, exper, use_logfile=False):
        self.exper = exper
        if use_logfile:
            self.logger = create_logger(self.exper, file_handler=use_logfile)
        else:
            self.logger = None

    def set_root_dir(self, root_dir):
        self.exper.config.root_dir = root_dir
        self.exper.config.data_dir = os.path.join(self.exper.config.root_dir, "data/Folds/")

    def next_epoch(self):
        self.exper.epoch_id += 1

    def next_val_run(self):
        self.exper.val_run_id += 1
        self.num_val_runs += 1
        self.exper.val_stats["epoch_ids"][self.exper.val_run_id] = self.exper.epoch_id

    def init_batch_statistics(self, trans_img_name):
        # trans_img_name is a dictionary (created in load_data.py) that we use to translate patientID to imageID
        # so key of dict is "patien040" and value is e.g. 10 (imageID)
        # create a batch statistics object
        self.exper.batch_stats = BatchStatistics(trans_img_name)

    def save_experiment(self, file_name=None, final_run=False):

        if file_name is None:
            if final_run:
                file_name = ExperimentHandler.exp_filename + ".dll"
            else:
                file_name = ExperimentHandler.exp_filename + "@{}".format(self.exper.epoch_id) + ".dll"

        exper_out_dir = os.path.join(self.exper.config.root_dir, self.exper.stats_path)

        outfile = os.path.join(exper_out_dir, file_name)
        with open(outfile, 'wb') as f:
            dill.dump(self.exper, f)

        if self.logger is not None:
            self.logger.info("Epoch: {} - Saving experimental details to {}".format(self.exper.epoch_id, outfile))
        else:
            print("Epoch: {} - Saving experimental details to {}".format(self.exper.epoch_id, outfile))

    def print_flags(self):
        """
        Prints all entries in argument parser.
        """
        for key, value in vars(self.exper.run_args).items():
            self.logger.info(key + ' : ' + str(value))

        if self.exper.run_args.cuda:
            self.logger.info(" *** RUNNING ON GPU *** ")

    def load_checkpoint(self, exper_dir=None, checkpoint=None, verbose=False, drop_prob=0., retrain=False):

        if exper_dir is None:
            # chkpnt_dir should be /home/jorg/repository/dcnn_acdc/logs/<experiment dir>/checkpoints/
            chkpnt_dir = os.path.join(self.exper.config.root_dir, self.exper.chkpnt_dir)
            # will be concatenated with "<checkpoint dir> further below
        elif retrain:
            # if we retrain a model exper_dir should be a relative path of the experiment:
            # e.g. "20180330_09_56_01_dcnn_mcv1_150000E_lr2e02".
            # first we concatenate root_dir (/home/jorg/repo/dcnn_acdc/) with self.log_root_path (e.g. "logs/")
            chkpnt_dir = os.path.join(self.exper.config.root_dir, self.exper.config.log_root_path)
            # then concatenate with the exper_dir (name of experiment abbreviation)
            chkpnt_dir = os.path.join(chkpnt_dir, exper_dir)
            # and then finally with the checkpoint_path e.g. "checkpoints/"
            chkpnt_dir = os.path.join(chkpnt_dir, self.exper.config.checkpoint_path)

        else:
            chkpnt_dir = os.path.join(self.exper.config.root_dir, self.exper.chkpnt_dir)

        if checkpoint is None:
            checkpoint = self.exper.epoch_id
        if "hvsmr" in self.exper.run_args.model:
            str_classname = "HVSMRDilated2DCNN"
            model_class = getattr(models.hvsmr.dilated_cnn, str_classname)
            # for ACDC dataset we use 2 times 4 classes, because we feed the network with 2 input channels
            # for HVSMR dataset we only use 3 classes and 1 input channel, hence use_dua_head is off
            use_dual_head = False
        else:
            str_classname = "BaseDilated2DCNN"
            model_class = getattr(models.dilated_cnn, str_classname)
            use_dual_head = True

        checkpoint_file = str_classname + "checkpoint" + str(checkpoint).zfill(5) + ".pth.tar"

        if not hasattr(self.exper.run_args, "drop_prob"):
            print("WARNING - exper.run_args has not attribute drop_prob, using prob {:.2}".format(drop_prob))
            drop_prob = 0.1
        else:
            drop_prob = self.exper.run_args.drop_prob
        if hasattr(self.exper.config, "get_architecture"):
            architecture = self.exper.config.get_architecture(model=self.exper.run_args.model,
                                                              drop_prob=drop_prob)
        else:
            architecture = DEFAULT_DCNN_MC_2D

        model = model_class(architecture=architecture, optimizer=self.exper.config.optimizer,
                            lr=self.exper.run_args.lr,
                            weight_decay=self.exper.run_args.weight_decay,
                            use_cuda=self.exper.run_args.cuda,
                            cycle_length=self.exper.run_args.cycle_length,
                            verbose=verbose, use_reg_loss=self.exper.run_args.use_reg_loss,
                            loss_function=self.exper.run_args.loss_function,
                            use_dual_head=use_dual_head)
        abs_checkpoint_dir = os.path.join(chkpnt_dir, checkpoint_file)
        if os.path.exists(abs_checkpoint_dir):
            model_state_dict = torch.load(abs_checkpoint_dir)
            model.load_state_dict(model_state_dict["state_dict"])
            if self.exper.run_args.cuda:
                model.cuda()
            if verbose and not retrain:
                self.info("INFO - loaded existing model with checkpoint {} from dir {}".format(checkpoint,
                                                                                               abs_checkpoint_dir))
            else:
                self.info("Loading existing model with checkpoint {} from dir {}".format(checkpoint, chkpnt_dir))
        else:
            raise IOError("Path to checkpoint not found {}".format(abs_checkpoint_dir))

        return model

    def set_config_object(self, new_config):
        self.exper.config = new_config

    def eval(self, dataset, model, val_set_size=None):
        start_time = time.time()
        # validate model
        if val_set_size is None:
            val_set_size = self.exper.config.val_set_size
        if val_set_size > dataset.get_num_of_slices(train=False):
            val_set_size = dataset.get_num_of_slices(train=False)

        self.next_val_run()
        if val_set_size <= self.exper.config.val_batch_size:
            # we don't need to chunk, if the number of patches for validation is smaller than the batch_size
            num_of_chunks = 1
            val_batch_size = val_set_size
        else:
            num_of_chunks = val_set_size // self.exper.config.val_batch_size
            val_batch_size = self.exper.config.val_batch_size
        if dataset.name == "ACDC":
            val_batch = TwoDimBatchHandler(self.exper, batch_size=val_batch_size)
            arr_val_acc = np.zeros(6)  # array of 6 values for accuracy of ES/ED RV/MYO/LV
            arr_val_dice = np.zeros(2)  # array of 2 values, loss for ES and ED
        elif dataset.name == "HVSMR":
            val_batch = HVSMRTwoDimBatchHandler(self.exper, batch_size=val_batch_size)
            arr_val_acc = np.zeros(2)  # array of 2 values for accuracy of MYO/LV
            arr_val_dice = np.zeros(2)  # actually we don't use this, same as above
        arr_val_loss = np.zeros(num_of_chunks)

        s_offset = 0
        for chunk in np.arange(num_of_chunks):
            slice_range = np.arange(s_offset, s_offset + val_batch_size)
            val_batch.generate_batch_2d(dataset.images(train=False), dataset.labels(train=False),
                                        slice_range=slice_range)
            # New in pytorch 0.4.0, use local context manager to turn off history tracking
            with torch.set_grad_enabled(False):
                val_loss, _ = model.do_test(val_batch.get_images(), val_batch.get_labels(),
                                            num_of_labels_per_class=val_batch.get_num_labels_per_class(),
                                            multi_labels=val_batch.get_labels_multiclass())

            arr_val_loss[chunk] = val_loss.data.cpu().numpy()
            # returns array of 6 values
            arr_val_acc += model.get_accuracy()
            arr_val_dice += model.get_dice_losses(average=True)
            s_offset += val_batch_size
        val_loss = np.mean(arr_val_loss)
        arr_val_acc *= 1./float(num_of_chunks)
        arr_val_dice *= 1./float(num_of_chunks)
        self.exper.val_stats["mean_loss"][self.num_val_runs - 1] = val_loss
        self.exper.val_stats["dice_coeff"][self.num_val_runs - 1] = arr_val_acc
        self.set_accuracy(arr_val_acc, val_run_id=self.num_val_runs)

        duration = time.time() - start_time
        if dataset.name == "ACDC":
            self.set_dice_losses(arr_val_dice, val_run_id=self.num_val_runs)
            self.logger.info("---> VALIDATION epoch {} (#patches={}): current loss {:.3f}\t "
                             "dice-coeff:: ES {:.3f}/{:.3f}/{:.3f} --- "
                             "ED {:.3f}/{:.3f}/{:.3f}  (time={:.2f} sec)".format(self.exper.epoch_id,
                                                                                 val_set_size, val_loss,
                                                                                 arr_val_acc[0], arr_val_acc[1],
                                                                                 arr_val_acc[2], arr_val_acc[3],
                                                                                 arr_val_acc[4], arr_val_acc[5],
                                                                                 duration))
        elif dataset.name == "HVSMR":
            self.logger.info("---> VALIDATION epoch {} (#patches={}): current loss {:.3f} (time={:.2f} sec)"
                             " dice (Myo/LV): {:.3f}/{:.3f}".format(self.exper.epoch_id, val_set_size,
                              val_loss, duration, arr_val_acc[0], arr_val_acc[1]))
        del val_batch

    def test(self, checkpoints, test_set, image_num=0, mc_samples=1, sample_weights=False, compute_hd=False,
             use_seed=False, verbose=False, store_details=False,
             do_filter=True, u_threshold=0., save_pred_labels=False,
             store_test_results=True):
        """

        :param model:
        :param test_set:
        :param image_num:
        :param mc_samples:
        :param sample_weights:
        :param compute_hd:

        :param referral_threshold: see explanation "use_uncertainty". This is the threshold parameter.

        label (for a particular class). We're discarding all the negatives that we predicted because there're so
        many of them due to the vast background

        :param u_threshold: Another threshold! used to compute the uncertainty statistics. We're currently using
        a default value of 0. => NO FILTERING OF INITIAL RAW U-MAPS!
        image slices.
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
            checkpoints = list(checkpoints)

        # -------------------------------------- local procedures END -----------------------------------------
        if use_seed:
            setSeed(4325, self.exper.run_args.cuda)
        if self.test_results is None:
            self.test_results = TestResults(self.exper, use_dropout=sample_weights, mc_samples=mc_samples)
        # if we use referral then we need to load the u-maps
        u_maps_image = None
        num_of_checkpoints = len(checkpoints)
        # correct the divisor for calculation of stddev when low number of samples (biased), used in np.std
        if num_of_checkpoints * mc_samples <= 25 and mc_samples != 1:
            ddof = 1
        else:
            ddof = 0
        # b_predictions has shape [num_of_checkpoints * mc_samples, classes, width, height, slices]
        b_predictions = np.zeros(tuple([num_of_checkpoints * mc_samples] + list(test_set.labels[image_num].shape)))
        slice_idx = 0
        # IMPORTANT boolean indicator: if we sample weights during testing we switch of BATCH normalization.
        # We added a new mode in pytorch module.eval method to enable this (see file dilated_cnn.py)
        mc_dropout = sample_weights
        # batch_generator iterates over the slices of a particular image
        for batch_image, batch_labels, b_num_labels_per_class in test_set.batch_generator(image_num):

            # NOTE: batch image shape (autograd.Variable): [1, 2, width, height] 1=batch size, 2=ES/ED image
            #       batch labels shape (autograd.Variable): [1, #classes, width, height] 8 classes, 4ES, 4ED
            # IMPORTANT: if we want to sample weights from the "posterior" over model weights, we
            # need to "tell" pytorch that we use "train" mode even during testing, otherwise dropout is disabled
            bald_values = np.zeros((2, batch_labels.shape[2], batch_labels.shape[3]))
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
                        test_loss, test_pred = model.do_test(batch_image, batch_labels,
                                                             voxel_spacing=test_set.new_voxel_spacing,
                                                             compute_hd=True,
                                                             num_of_labels_per_class=b_num_labels_per_class,
                                                             mc_dropout=mc_dropout)
                    b_predictions[s + sample_offset, :, :, :, test_set.slice_counter] = test_pred.data.cpu().numpy()

                    b_test_losses[s + sample_offset] = test_loss.data.cpu().numpy()

            # mean/std for each pixel for each class
            mc_probs = b_predictions[:, :, :, :, test_set.slice_counter]
            mean_test_pred, std_test_pred = np.mean(mc_probs, axis=0, keepdims=True), \
                                                np.std(mc_probs, axis=0, ddof=ddof)

            bald_values[0] = bald_function(b_predictions[:, 0:4, :, :, test_set.slice_counter])
            bald_values[1] = bald_function(b_predictions[:, 4:, :, :, test_set.slice_counter])
            means_test_loss = np.mean(b_test_losses)
            test_set.set_stddev_map(std_test_pred, u_threshold=u_threshold)
            test_set.set_bald_map(bald_values, u_threshold=u_threshold)
            # THIS IS A PIECE OF RESIDUAL CODING WHICH SHOULD BE NOT USED. THE FUNCTIONALITY IS STILL IN TACT
            # BUT WE ARE CURRENTLY USING THE Uncertainty MAPS which we generate to refer image slices-phase-class
            # to an UNKNOWN expert, we mark the dice values as OUTLIERS.
            test_set.set_pred_labels(mean_test_pred, verbose=verbose, do_filter=False)
            slice_acc, slice_hd = test_set.compute_slice_accuracy(compute_hd=compute_hd)

            if sample_weights:
                # NOTE: currently only displaying the MEAN STDDEV uncertainty stats but we also capture the stddev stats
                # b_uncertainty_stats["stddev"] has shape [2, 4cls, 4measures, #slices]
                es_total_uncert, es_num_of_pixel_uncert, es_num_pixel_uncert_above_tre, num_of_conn_commponents = \
                    np.mean(test_set.b_uncertainty_stats["stddev"][0, :, :, slice_idx], axis=0)  # ES
                ed_total_uncert, ed_num_of_pixel_uncert, ed_num_pixel_uncert_above_tre, num_of_conn_commponents = \
                    np.mean(test_set.b_uncertainty_stats["stddev"][1, :, :, slice_idx], axis=0) # ED
                es_seg_errors = np.sum(test_set.b_seg_errors[slice_idx, :4])
                ed_seg_errors = np.sum(test_set.b_seg_errors[slice_idx, 4:])
                # CURRENTLY DISABLED
                if False:
                    if slice_hd is None:
                        slice_hd = np.zeros(8)
                    print("Test img/slice {}/{}".format(image_num, slice_idx))
                    print("ES: Total U-value/seg-errors/#pixel/#pixel(tre) \tDice (RV/Myo/LV)\tHD (RV/Myo/LV)")
                    print("  \t{:.2f}/{}/{}/{} \t\t\t{:.2f}/{:.2f}/{:.2f}"
                          "\t\t{:.2f}/{:.2f}/{:.2f}".format(es_total_uncert, es_seg_errors, es_num_of_pixel_uncert,
                                                                       es_num_pixel_uncert_above_tre,
                                                                       slice_acc[1], slice_acc[2], slice_acc[3],
                                                                       slice_hd[1], slice_hd[2], slice_hd[3]))
                    # print(np.array_str(np.array(es_region_mean_uncert), precision=3))
                    print("ED: Total U-value/seg-errors/#pixel/#pixel(tre)\tDice (RV/Myo/LV)\tHD (RV/Myo/LV)")
                    print("  \t{:.2f}/{}/{}/{} \t\t\t{:.2f}/{:.2f}/{:.2f}"
                          "\t\t{:.2f}/{:.2f}/{:.2f}".format(ed_total_uncert, ed_seg_errors, ed_num_of_pixel_uncert,
                                                          ed_num_pixel_uncert_above_tre,
                                                          slice_acc[5], slice_acc[6], slice_acc[7],
                                                          slice_hd[5], slice_hd[6], slice_hd[7]))

                    print("------------------------------------------------------------------------")
            slice_idx += 1

        test_accuracy, test_hd, seg_errors = test_set.get_accuracy(compute_hd=compute_hd, compute_seg_errors=True,
                                                                   do_filter=do_filter)
        if verbose:
            for slice_id in np.arange(seg_errors.shape[2]):
                slice_seg_error = seg_errors[:, :, slice_id]
                print("\t Segmentation errors - \tES {}/{}/{}\t\t"
                      "ED {}/{}/{}".format(slice_seg_error[0, 1], slice_seg_error[0, 2], slice_seg_error[0, 3],
                                           slice_seg_error[1, 1], slice_seg_error[1, 2], slice_seg_error[1, 3]))
        # we only want to save predicted labels when we're not SAMPLING weights
        if save_pred_labels:
            test_set.save_pred_labels(self.exper.output_dir, u_threshold=0., mc_dropout=mc_dropout)
            # save probability maps (mean softmax)
            test_set.save_pred_probs(self.exper.output_dir, mc_dropout=mc_dropout)

        print("Image {} - test loss {:.3f} "
              " dice(RV/Myo/LV):\tES {:.2f}/{:.2f}/{:.2f}\t"
              "ED {:.2f}/{:.2f}/{:.2f}".format(str(image_num+1) + "-" + test_set.b_image_name, means_test_loss,
                                               test_accuracy[1], test_accuracy[2],
                                               test_accuracy[3], test_accuracy[5],
                                               test_accuracy[6], test_accuracy[7]))
        if compute_hd:
            print("\t\t\t\t\t"
                  "Hausdorff(RV/Myo/LV):\tES {:.2f}/{:.2f}/{:.2f}\t"
                  "ED {:.2f}/{:.2f}/{:.2f}".format(test_hd[1], test_hd[2],
                                                   test_hd[3], test_hd[5],
                                                   test_hd[6], test_hd[7]))
        if store_test_results:
            self.test_results.add_results(test_set.b_image, test_set.b_labels, test_set.b_image_id,
                                          test_set.b_pred_labels, b_predictions, test_set.b_stddev_map,
                                          test_accuracy, test_hd, seg_errors=seg_errors,
                                          store_all=store_details,
                                          bald_maps=test_set.b_bald_map,
                                          uncertainty_stats=test_set.b_uncertainty_stats,
                                          test_accuracy_slices=test_set.b_acc_slices,
                                          test_hd_slices=test_set.b_hd_slices,
                                          image_name=test_set.b_image_name,
                                          referral_accuracy=None, referral_hd=None,
                                          referral_stats=test_set.referral_stats)

    def create_u_maps(self, model=None, checkpoints=None, mc_samples=10, u_threshold=0.,
                      save_actual_maps=False, test_set=None, generate_figures=False, verbose=False,
                      aggregate_func="max", store_test_results=False,
                      patient_ids=None):

        if model is None and checkpoints is None:
            raise ValueError("When model parameter is None, you need to specify the checkpoint model "
                             "that needs to be loaded.")

        maps_generator = InferenceGenerator(self, test_set=test_set, verbose=verbose,
                                                  mc_samples=mc_samples, u_threshold=u_threshold,
                                                  checkpoints=checkpoints, store_test_results=store_test_results,
                                                  aggregate_func=aggregate_func)
        maps_generator(clean_up=False, save_actual_maps=save_actual_maps,
                       generate_figures=generate_figures, patient_ids=patient_ids)

    def get_u_maps(self):
        # returns a dictionary key patientID with the uncertainty maps for each patient/image of shape
        # [2, 4classes, width, height, #slices]
        self.u_maps = ImageUncertainties.load_uncertainty_maps(self, u_maps_only=True)

    def create_outlier_dataset(self, dataset, model=None, checkpoint=None, mc_samples=10, u_threshold=0.1,
                               do_save_u_stats=False, use_high_threshold=False, use_train_set=True,
                               do_save_outlier_stats=False, test_set=None, use_existing_umaps=False,
                               do_analyze_slices=False):
        """

        :param model: if not specified
        :param dataset: we need the original training/val dataset in order to get hold of the image slices
        and augmented slices and use the same imgID's as the original training set
        :param mc_samples:
        :param do_save_u_stats: save the uncertainty statistics and raw values. See object InferenceGenerator
        method save_maps for more details (will be stored in "/u_maps/" directory of the experiment.
        :param do_save_outlier_stats: saves 2 dictionaries of the object "OutOfDistributionSlices", enables us
        to trace the img/slice outlier statistics later.
        :param u_threshold: we only consider pixel uncertainties above this value. VERY IMPORTANT!
        :param use_high_threshold: use a threshold of MEAN + STDDEV. If False only use MEAN
        :param use_train_set: use the current training set of the dataset for evaluation aka generation
        of the uncertainty values which will determine the slice outliers per image. If False we use the validation set
        :param use_existing_umaps: If True we load existing U-maps from the u_maps directory, so skipping the step
        of generating the U-maps.
        :param test_set: the test_set we're operating on i.e. we're determining the outliers. If not specified
        InferenceGenerator will load the test_set, assuming, we want to load the validation set of the FOLD
        we trained on (info comes from exper_handler.exper.run_args.fold_ids
        :param do_analyze_slices: generate plots for each image, which can be used to visually inspect which of
        the slices have "bizar" uncertainties.
        :return:

        Note: we don't specify the test_set. But, InferenceGenerator will by default, load all training
        images from the FOLD the current model is trained on!
        """
        if model is None and checkpoint is None:
            raise ValueError("When model parameter is None, you need to specify the checkpoint model"
                             "that needs to be loaded.")

        if not use_existing_umaps:
            maps_generator = InferenceGenerator(self, model=model, test_set=test_set, verbose=False,
                                                      mc_samples=mc_samples, u_threshold=u_threshold,
                                                      checkpoint=checkpoint, store_test_results=True)
            maps_generator(do_save=do_save_u_stats, clean_up=True)
            image_uncertainties = ImageUncertainties.create_from_testresult(maps_generator.test_results)
        else:
            # load existing u_maps from umaps directory of current experiment
            image_uncertainties = ImageUncertainties.load_uncertainty_maps(self)
            fold_id = self.exper.run_args.fold_ids[0]
            test_res_file = "test_results_25imgs_mc" + str(mc_samples) + "_fold" + str(fold_id) + "_ep150000.dll"
            input_dir = os.path.join(self.exper.config.root_dir,
                                     os.path.join(self.exper.output_dir,
                                                  self.exper.config.stats_path))
            search_path = os.path.join(input_dir, test_res_file)
            filenames = glob.glob(search_path)
            if len(filenames) != 1:
                raise ValueError("ERROR - found no OR too many test result files with"
                                 "search mask {}".format(search_path))
            self.test_results = TestResults.load_results(filenames[0])
        # detect outliers and return object OutOfDistributionSlices
        img_outliers = image_uncertainties.get_outlier_obj(use_high_threshold)
        img_outliers.create_dataset(dataset, train=use_train_set)

        dta_set_num_of_slices = dataset.get_num_of_slices(train=use_train_set)
        referral_perc = len(img_outliers.images) / float(dta_set_num_of_slices) * 100
        self.info("INFO - Successfully created outlier-dataset (use_train_set={}/use_high_threshold={}). "
                  "Number of outlier slices {}. "
                  "Dataset contains {} slices in total "
                  "(referral {:.2f})%.".format(use_train_set, use_high_threshold, len(img_outliers.outlier_slices),
                                               len(img_outliers.images),
                                               referral_perc))
        if do_save_outlier_stats:
            save_output_dir = os.path.join(self.exper.config.root_dir,
                                           os.path.join(self.exper.output_dir, self.exper.config.stats_path))
            if checkpoint is None:
                checkpoint = self.exper.epoch_id
            out_filename = os.path.join(save_output_dir, "image_outliers_fold{}_".format(self.exper.run_args.fold_ids[0])
                                        + str(checkpoint) + ".dll")
            img_outliers.save(out_filename)

        if do_analyze_slices:
            if self.test_results is None:
                # if test_results object of experiment handler is None (because we generated u-maps) we point to the
                # test_result object of the map generator. If we re-use maps we load the object, see above
                self.test_results = maps_generator.test_results

            if len(self.test_results.images) == 0:
                if use_train_set:
                    self.test_results.images = dataset.train_images
                    self.test_results.labels = dataset.train_labels
                else:
                    self.test_results.images = dataset.val_images
                    self.test_results.labels = dataset.val_labels

            analyze_slices(self, image_range=None, do_save=True, do_show=False, u_type="stddev",
                           use_saved_umaps=False)

        try:
            del maps_generator
            del image_uncertainties
        except:
            pass

        return img_outliers

    def get_outlier_stats(self, verbose=False):
        load_dir_stats = os.path.join(self.exper.config.root_dir,
                                      os.path.join(self.exper.output_dir, self.exper.config.stats_path))
        search_path = os.path.join(load_dir_stats, "image_outliers*.dll")
        self.exper.outliers_per_epoch = OrderedDict()
        self.exper.test_results_per_epoch = OrderedDict()
        files = glob.glob(search_path)
        if len(files) == 0:
            raise ValueError("ERROR - There are no outlier statistics in {}".format(search_path))
        c = 0
        for filename in glob.glob(search_path):
            # outlier_slices, dict of [2 (es/ed), 4 classes] per (patientxxx, sliceid), see specifications
            # in method load for details.
            # outliers_per_img_class_es: key (patientxxx, sliceid, phase)
            outliers_per_img, outlier_slices, outliers_per_img_es, outliers_per_img_ed, outliers_per_img_class_es, \
                            outliers_per_img_class_ed = OutOfDistributionSlices.load(filename)

            start = filename.rfind("_") + 1
            end = filename.find(".")
            epoch = int(filename[start:end])
            c += 1
            search_test_result = os.path.join(load_dir_stats, "test_results*{}.dll".format(epoch))
            filename_test_result = glob.glob(search_test_result)[0]
            test_result = TestResults.load_results(path_to_exp=filename_test_result, verbose=False)
            # see method load_results for further details about TestResults properties
            self.exper.outliers_per_epoch[epoch] = tuple((outliers_per_img, outlier_slices,
                                                          outliers_per_img_es, outliers_per_img_ed,
                                                          outliers_per_img_class_es,
                                                          outliers_per_img_class_ed))
            # test_accuracy_slices, test_hd_slices, seg_errors: [phase, class, #slices]
            self.exper.test_results_per_epoch[epoch] = tuple((test_result.test_accuracy_slices,
                                                              test_result.test_hd_slices,
                                                              test_result.seg_errors,
                                                              test_result.trans_dict))
        if verbose:
            print("Loaded outlier stats for {} training epochs"
                  " (property=outliers_per_epoch, dictionary (key=epoch) with tuple (6 elements)"
                  " and property=test_results_per_epoch, dict (key=epoch) with tuple (4))".format(c))

    def get_referral_maps(self, u_threshold, per_class=True, aggregate_func="max", use_raw_maps=False,
                          patient_id=None, load_ref_map_blobs=True):
        """

        :param u_threshold: also called referral_threshold in other context.
        :param per_class:
        :param aggregate_func:
        :param patient_id:
        :param load_ref_map_blobs: We only need this object when we referred specific slices
        :param use_raw_maps: if true, we use the thresholded/filtered u-maps with NO POST-PROCESSING steps
                             ONLY applicable for PER_CLASS = FALSE.
                             We use these maps also during referral to compare with entropy maps
        :return:
        """
        if self.referral_umaps is None:
            self.referral_umaps = OrderedDict()
        if load_ref_map_blobs and self.ref_map_blobs is None:
            self.ref_map_blobs = OrderedDict()
        if patient_id is not None:
            if patient_id in self.referral_umaps.keys():
                return

        input_dir = os.path.join(self.exper.config.root_dir,
                                           os.path.join(self.exper.output_dir, config.u_map_dir))
        u_threshold = str(u_threshold).replace(".", "_")
        if patient_id is None:
            search_prefix = "*"
        else:
            search_prefix = patient_id
        if per_class:
            search_path = os.path.join(input_dir, search_prefix + "_filtered_cls_umaps_" + aggregate_func +
                                       u_threshold + ".npz")
        else:
            search_path = os.path.join(input_dir, search_prefix + "_filtered_umaps_" + aggregate_func + u_threshold
                                       + ".npz")
        files = glob.glob(search_path)
        if len(files) == 0:
            raise ImportError("ERROR - no referral u-maps found in {}".format(search_path))

        for fname in glob.glob(search_path):
            file_basename = os.path.splitext(os.path.basename(fname))[0]
            patientID = file_basename[:file_basename.find("_")]
            try:
                data = np.load(fname)
            except IOError:
                print("Unable to load uncertainty maps from {}".format(fname))
            if per_class:
                self.referral_umaps[patientID] = data["filtered_cls_umap"]
            else:
                # this is the one we use during referral at the moment
                if use_raw_maps:
                    # raw means no post-processing applied (largest connected components per class)
                    self.referral_umaps[patientID] = data["filtered_raw_umap"]
                else:
                    self.referral_umaps[patientID] = data["filtered_umap"]
            if load_ref_map_blobs:
                try:
                    self.ref_map_blobs[patientID] = data["filtered_stddev_blobs"]
                except KeyError:
                    print("WARNING - ExperimentHandler.get_referral_maps - filtered_stddev_blobs "
                          "object does not exist for {}".format(patientID))

    def get_referred_slices(self, referral_threshold, slice_filter_type=None, patient_id=None):

        input_dir = os.path.join(self.exper.config.root_dir,
                                 os.path.join(self.exper.output_dir, config.pred_lbl_dir))

        if patient_id is None:
            search_mask = "patient_id*_referred_slices_mc"
        else:
            search_mask = patient_id + "_referred_slices_mc"
            print("get_referred_slices - ", search_mask)

        search_mask = search_mask + str(referral_threshold).replace(".", "_")
        if slice_filter_type is not None:
            search_mask += "_" + slice_filter_type
        search_mask = search_mask + ".npz"
        file_name = os.path.join(input_dir, search_mask)
        if self.referred_slices is None:
            self.referred_slices = OrderedDict()

        for fname in glob.glob(file_name):
            file_basename = os.path.splitext(os.path.basename(fname))[0]
            f_patient_id = file_basename[:file_basename.find("_")]
            try:
                np_archive = np.load(file_name)
                referred_slices = np_archive["referred_slices"]
                self.referred_slices[f_patient_id] = referred_slices
            except IOError:
                print("ERROR - Unable to load data from numpy file {}".format(file_name))
            except KeyError:
                print("ERROR - Archive referred_slices does not exist.")

        if patient_id is not None:
            return referred_slices

    def get_pred_prob_maps(self, patient_id=None, mc_dropout=True):

        input_dir = os.path.join(self.exper.config.root_dir,
                                 os.path.join(self.exper.output_dir, config.pred_lbl_dir))

        if mc_dropout:
            search_suffix = "_pred_probs_mc"
        else:
            search_suffix = "_pred_probs"
        if patient_id is None:
            search_mask = "*" + search_suffix
        else:
            search_mask = patient_id + search_suffix

        search_mask = search_mask + ".npz"
        file_name = os.path.join(input_dir, search_mask)
        if self.pred_prob_maps is None:
            self.pred_prob_maps = OrderedDict()

        if len(glob.glob(file_name)) == 0:
            raise ValueError("ERROR - no file found with search mask {}".format(search_mask))
        for fname in glob.glob(file_name):
            file_basename = os.path.splitext(os.path.basename(fname))[0]
            f_patient_id = file_basename[:file_basename.find("_")]
            try:
                np_archive = np.load(fname)
                pred_probs = np_archive["pred_probs"]
                self.pred_prob_maps[f_patient_id] = pred_probs
            except IOError as e:
                print("ERROR - Unable to load predicted prob-maps from numpy file {}".format(fname))
                print(e)
            except KeyError:
                print("ERROR - Archive pred_probs does not exist in {}.".format(fname))

        if patient_id is not None:
            return pred_probs

    def create_filtered_umaps(self, u_threshold, verbose=False, patient_id=None, aggregate_func="max",
                              filter_per_slice=False):
        if u_threshold < 0.:
            raise ValueError("ERROR - u_threshold must be greater than zero. ({:.2f})".format(u_threshold))
        # returns an OrderedDict with key "patient_id" and value=tensor of shape [2, 4classes, width, height, #slices]
        # which is stored in self.u_maps
        self.get_u_maps()
        if patient_id is not None:
            u_maps = {patient_id: self.u_maps.get(patient_id)}
        else:
            u_maps = self.u_maps
        for patient_id, raw_u_map in u_maps.iteritems():
            # IMPORTANT: raw_u_map has shape [2, 4, width, height, #slices]
            num_of_phases = raw_u_map.shape[0]
            num_of_classes = raw_u_map.shape[1]
            num_of_slices = raw_u_map.shape[4]
            raw_u_map_copy = copy.deepcopy(raw_u_map)
            # Yes I know, completely inconsistent the output is [8classes, width, height, #slices] instead of [2, 4...]
            filtered_cls_stddev_map = np.zeros((num_of_phases * num_of_classes, raw_u_map.shape[2], raw_u_map.shape[3],
                                                raw_u_map.shape[4]))
            # here we store the maps per phase ES/ED taking the max uncertainty over 4 classes after we've filtered
            # the u-maps per class by means of 6 connectivity components.
            filtered_stddev_map = np.zeros((num_of_phases, raw_u_map.shape[2], raw_u_map.shape[3], raw_u_map.shape[4]))
            # the raw_stddev_map is NOT filtered with post-processing steps but only filtered/thresholded w.r.t.
            # uncertainties (above certain threshold). Also, we take the max or mean over all CLASSES. So dim1
            # of the original u-maps is squeezed out.
            raw_stddev_map = np.zeros((num_of_phases, raw_u_map.shape[2], raw_u_map.shape[3], raw_u_map.shape[4]))
            # anything_there = np.count_nonzero(raw_u_map)
            for phase in np.arange(num_of_phases):
                cls_offset = phase * num_of_classes
                for cls in np.arange(0, num_of_classes):
                    # if cls != 0 and cls != 4:
                    u_3dmaps_cls = raw_u_map[phase, cls]

                    # set all uncertainties below threshold to zero
                    u_3dmaps_cls[u_3dmaps_cls < u_threshold] = 0
                    # do not yet filter 6 largest connected components here when we average over the stddev values
                    # we do that after we've averaged, for max-aggregate this wouldn't have an effect
                    if aggregate_func == "max":
                        if filter_per_slice:
                            for slice_id in np.arange(num_of_slices):
                                slice_cls_u_map = u_3dmaps_cls[:, :, slice_id]
                                filtered_cls_stddev_map[cls + cls_offset, :, :, slice_id] = \
                                    filter_connected_components(slice_cls_u_map, threshold=u_threshold)
                        else:

                            filtered_cls_stddev_map[cls + cls_offset] = filter_connected_components(u_3dmaps_cls,
                                                                                                    threshold=u_threshold)
                            # anything_left = np.count_nonzero(filtered_cls_stddev_map[cls + cls_offset])
                            # print("Before {} and what's left {}".format(anything_there, anything_left))
                    else:
                        filtered_cls_stddev_map[cls + cls_offset] = u_3dmaps_cls

                raw_map_phase = raw_u_map_copy[phase]
                raw_map_phase[raw_map_phase < u_threshold] = 0
                if phase == 0:
                    if aggregate_func == "max":
                        filtered_stddev_map[phase] = np.max(filtered_cls_stddev_map[:num_of_classes], axis=0)
                        raw_stddev_map[phase] = np.max(raw_map_phase, axis=0)
                    else:
                        filtered_stddev_map[phase] = np.mean(filtered_cls_stddev_map[:num_of_classes], axis=0)
                        raw_stddev_map[phase] = np.mean(raw_map_phase, axis=0)
                else:
                    if aggregate_func == "max":
                        filtered_stddev_map[phase] = np.max(filtered_cls_stddev_map[num_of_classes:], axis=0)
                        raw_stddev_map[phase] = np.max(raw_map_phase, axis=0)
                    else:
                        filtered_stddev_map[phase] = np.mean(filtered_cls_stddev_map[num_of_classes:], axis=0)
                        raw_stddev_map[phase] = np.mean(raw_map_phase, axis=0)
                if aggregate_func == "mean":
                    filtered_stddev_map[phase] = filter_connected_components(filtered_stddev_map[phase],
                                                                             threshold=u_threshold)

            del raw_map_phase
            u_map_c_areas, _ = detect_largest_umap_areas(filtered_stddev_map,
                                                         rank_structure=config.erosion_rank_structure,
                                                         max_objects=config.num_of_umap_blobs)
            # save map
            umap_output_dir = os.path.join(self.exper.config.root_dir, os.path.join(self.exper.output_dir,
                                                                                    config.u_map_dir))
            if not os.path.isdir(umap_output_dir):
                os.makedirs(umap_output_dir)
            str_u_threshold = str(u_threshold).replace(".", "_")
            file_name_cls = patient_id + "_filtered_cls_umaps_" + aggregate_func + str_u_threshold + ".npz"
            file_name_cls = os.path.join(umap_output_dir, file_name_cls)
            file_name = patient_id + "_filtered_umaps_" + aggregate_func + str_u_threshold + ".npz"
            file_name = os.path.join(umap_output_dir, file_name)
            try:
                np.savez(file_name_cls, filtered_cls_umap=filtered_cls_stddev_map, filtered_stddev_blobs=u_map_c_areas)
            except IOError:
                print("ERROR - Unable to save filtered cls-umaps file {}".format(file_name_cls))
            try:
                # 3 objects to save: (1) the filtered u-map with aggregate over classes with post-processing
                #                    (2) the raw filtered/thresholded u-map with aggregate over classes (no post-pro)
                #                        (filtered_raw_umap)
                #                    (3) a list containing integers specifying a u-value for each of the remaining
                #                        blob areas in the filtered u-map with post processing
                np.savez(file_name, filtered_umap=filtered_stddev_map, filtered_raw_umap=raw_stddev_map,
                         filtered_stddev_blobs=u_map_c_areas)
            except IOError:
                print("ERROR - Unable to save filtered umaps file {}".format(file_name))
        if verbose:
            print("INFO - Creating {} filtered u-map-{:.2f} in {}".format(len(self.u_maps.keys()),
                                                                          u_threshold, umap_output_dir))
        if patient_id is not None:
            # let's add this to the self.referral_umaps object, in most cases we use it in the next step to
            # make referral predictions
            if self.referral_umaps is None:
                self.referral_umaps = OrderedDict()
                self.ref_map_blobs = OrderedDict()
            self.referral_umaps[patient_id] = filtered_stddev_map
            self.ref_map_blobs[patient_id] = u_map_c_areas
        # we don't need this object, to big to keep
        self.u_maps = None
        del filtered_cls_stddev_map
        del filtered_stddev_map

    def get_testset_ids(self):
        val_path = os.path.join(config.data_dir + "fold" + str(self.exper.run_args.fold_ids[0]),
                                os.path.join(ACDC2017DataSet.val_path, ACDC2017DataSet.image_path))
        crawl_mask = os.path.join(val_path, "*.mhd")
        for test_file in glob.glob(crawl_mask):
            file_name = os.path.splitext(os.path.basename(test_file))[0]
            patient_id = file_name[:file_name.find("_")]
            self.test_set_ids[patient_id] = int(patient_id.strip("patient"))

    def generate_figures(self, test_set, image_range=None, referral_thresholds=[0.], patients=None,
                         slice_type_filter=None):
        if patients is not None:
            image_range = [test_set.trans_dict[p_id] for p_id in patients]

        args = self.exper.run_args
        model_name = args.model + " (p={:.2f})".format(args.drop_prob) + " - {}".format(args.loss_function)
        if image_range is None:
            image_range = np.arange(len(test_set.images))

        for image_num in image_range:
            patient_id = test_set.img_file_names[image_num]
            for referral_threshold in referral_thresholds:
                if isinstance(referral_threshold, str):
                    referral_threshold = float(referral_threshold)
                plot_seg_erros_uncertainties(self, test_set, patient_id=patient_id,
                                             test_results=None, slice_filter_type=slice_type_filter,
                                             referral_threshold=referral_threshold, do_show=False,
                                             model_name=model_name, info_type="uncertainty",
                                             do_save=True, slice_range=None, errors_only=False,
                                             load_base_model_pred_labels=True)

    def get_patients(self, use_four_digits=False):
        """
        loading the complete set of patient_ids with the corresponding disease classification

        :return:
        """
        patients = Patients()
        patients.load(self.exper.config.data_dir, use_four_digits)
        self.patients = patients.category

    def create_entropy_maps(self, do_save=False):
        eps = 1e-7
        input_dir = os.path.join(self.exper.config.root_dir,
                                           os.path.join(self.exper.output_dir, config.pred_lbl_dir))
        output_dir = os.path.join(self.exper.config.root_dir,
                                 os.path.join(self.exper.output_dir, config.u_map_dir))

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
            # pred_probs has shape [8, height, width, #slices]: next step compute two entropy maps ES/ED
            pred_probs_es, pred_probs_ed = pred_probs[:4], pred_probs[4:]
            # for numerical stability we add eps (tiny) to softmax probability to prevent np.log2 on zero
            # probability values, which result in nan-values. This actually only happens when we trained the model
            # with the soft-dice.
            entropy_es = (-pred_probs_es * np.log2(pred_probs_es + eps)).sum(axis=0)
            entropy_ed = (-pred_probs_ed * np.log2(pred_probs_ed + eps)).sum(axis=0)
            entropy_es = np.nan_to_num(entropy_es)
            entropy_ed = np.nan_to_num(entropy_ed)
            entropy = np.concatenate((np.expand_dims(entropy_es, axis=0),
                                      np.expand_dims(entropy_ed, axis=0)))
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
                    print("Unable to load uncertainty maps from {}".format(fname))
        if do_save:
            print("INFO - Saved all entropy maps to {}".format(output_dir))

    def get_entropy_maps(self, patient_id=None, force_reload=False):
        self._check_dirs(config_env=config)
        self._check_maps()
        if self.entropy_maps is not None and patient_id is not None and not force_reload:
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
                print("Unable to load entropy maps from {}".format(fname))
            file_basename = os.path.splitext(os.path.basename(fname))[0]
            patient_id = file_basename[:file_basename.find("_")]
            self.entropy_maps[patient_id] = entropy_map
        if patient_id is not None:
            return self.entropy_maps[patient_id]

    def generate_dt_maps(self, patient_id=None, voxelspacing=ACDC2017DataSet.new_voxel_spacing):
        """
        Generating the distance transform maps for all tissue classes per slice.
        dt_slices (the result) has shape [num_of_classes, w, h, #slices]
        We need the inter-observer error-margins, and the gt references of the tissue labels.

        :param patient_id:
        :param voxelspacing:
        :return:
        """
        self._check_dirs(config_env=config)
        self._check_maps()
        if self.test_set is None:
            self.get_test_set()
        if patient_id is None:
            p_range = self.test_set.trans_dict.keys()
        else:
            p_range = [patient_id]
        for p_id in p_range:
            _, labels = self.test_set.get_test_pair(p_id)
            dt_slices = generate_dt_maps(labels, voxelspacing=voxelspacing)
            file_name = os.path.join(self.dt_map_dir, p_id + "_dt_map.npz")
            try:
                self.dt_maps[p_id] = dt_slices
                np.savez(file_name, dt_map=dt_slices)
            except IOError:
                print("ERROR - cannot save hd map to {}".format(file_name))
                raise

        if patient_id is not None:
            return dt_slices

    def get_dt_maps(self, patient_id=None, force_reload=False):
        self._check_dirs(config_env=config)
        self._check_maps()
        if self.dt_maps is not None and patient_id is not None and not force_reload:
            if patient_id in self.dt_maps.keys():
                return self.dt_maps[patient_id]

        if patient_id is None:
            search_path = os.path.join(self.dt_map_dir, "*dt_map.npz")
        else:
            filename = patient_id + "_dt_map.npz"
            search_path = os.path.join(self.dt_map_dir, filename)
        if len(glob.glob(search_path)) == 0:
            raise ValueError("ERROR - No search result for {}".format(search_path))
        for fname in glob.glob(search_path):
            # get base filename first and then extract patient/name/ID (filename is e.g. patient012_dt_map.npz)
            file_basename = os.path.splitext(os.path.basename(fname))[0]
            patient_id = file_basename[:file_basename.find("_")]
            try:
                data = np.load(fname)
                self.dt_maps[patient_id] = data["dt_map"]
                del data
            except IOError:
                self.info("ERROR - Unable to load distance transform maps from {}".format(fname))
                raise
        if patient_id is not None:
            return self.dt_maps[patient_id]

    def generate_target_rois_for_learning(self, patient_id=None, mc_dropout=False):
        """
        Generation of target areas in automatic segmentations (test set) that we want to inspect after prediction.
        These areas must be detected by our "detection model". So there're just for supervised learning
        For each patient study we produce numpy array with shape [num_of_classes, w, h, #slices]

        IMPORTANT: we have different rois for the automatic segmentations produced by single prediction (dropout=False)
                   or by a Bayesian network using T (we used 10) samples. In the latter case mc_dropout=True.

        :param patient_id:
        :param mc_dropout:
        :return:
        """
        self._check_dirs(config_env=config)
        self._check_maps()
        if self.test_set is None:
            self.get_test_set()

        if patient_id is None:
            p_range = self.test_set.trans_dict.keys()
        else:
            p_range = [patient_id]
        for p_id in p_range:
            auto_pred = self.get_pred_labels(patient_id=p_id, mc_dropout=mc_dropout, force_reload=True)
            _, labels = self.test_set.get_test_pair(p_id)
            dt_slices = self.get_dt_maps(p_id)
            self.target_roi_maps[p_id] = determine_target_voxels(auto_pred, labels, dt_slices)
            if mc_dropout:
                file_suffix = p_id + "_troi_map_mc.npz"
            else:
                file_suffix = p_id + "_troi_map.npz"
            file_name = os.path.join(self.troi_map_dir, file_suffix)
            try:
                np.savez(file_name, target_roi=self.target_roi_maps[p_id])
            except IOError:
                print("ERROR - cannot save target-roi map to {}".format(file_name))
                raise

    def get_target_roi_maps(self, patient_id=None, mc_dropout=False, force_reload=False):
        self._check_dirs(config_env=config)
        self._check_maps()
        if self.target_roi_maps is not None and patient_id is not None and not force_reload:
            if patient_id in self.target_roi_maps.keys():
                return self.target_roi_maps[patient_id]
        # we didn't find target_roi_map in dictionary, or patient_id was empty, load from disk
        if mc_dropout:
            file_suffix = "_troi_map_mc.npz"
        else:
            file_suffix = "_troi_map.npz"
        if patient_id is None:
            search_path = os.path.join(self.troi_map_dir, "*" + file_suffix)
        else:
            filename = patient_id + file_suffix
            search_path = os.path.join(self.troi_map_dir, filename)

        if len(glob.glob(search_path)) == 0:
            raise ValueError("ERROR - No search result for {}".format(search_path))
        for fname in glob.glob(search_path):
            # get base filename first and then extract patient/name/ID (filename is e.g. patient012_dt_map.npz)
            file_basename = os.path.splitext(os.path.basename(fname))[0]
            patient_id = file_basename[:file_basename.find("_")]
            try:
                data = np.load(fname)
                self.target_roi_maps[patient_id] = data["target_roi"]
                del data
            except IOError:
                self.info("ERROR - Unable to load target roi maps from {}".format(fname))
                raise
        if patient_id is not None:
            return self.target_roi_maps[patient_id]

    def info(self, message):
        if self.logger is None:
            print(message)
        else:
            self.logger.info(message)

    def set_lr(self, lr):
        self.exper.epoch_stats["lr"][self.exper.epoch_id - 1] = lr

    def set_batch_loss(self, loss, used_outliers=False, reg_loss=None):
        if isinstance(loss, torch.FloatTensor):
            loss = loss.data.cpu().squeeze().numpy()
        if not used_outliers:
            self.exper.epoch_stats["mean_loss"][self.exper.epoch_id-1] = loss
        else:
            self.exper.epoch_stats["mean_loss_outliers"][self.exper.epoch_id - 1] = loss
            self.exper.epoch_stats["num_of_outlier_batches"] += 1

        if reg_loss is not None:
            self.exper.epoch_stats["mean_reg_loss"][self.exper.epoch_id - 1] = reg_loss

    def set_dice_losses(self, dice_losses, val_run_id=None):

        if val_run_id is None:
            self.exper.epoch_stats["soft_dice_loss"][self.exper.epoch_id - 1] = dice_losses
        else:
            self.exper.val_stats["soft_dice_loss"][val_run_id-1] = dice_losses

    def set_accuracy(self, accuracy, val_run_id=None, used_outliers=False):

        if val_run_id is None:
            if not used_outliers:
                self.exper.epoch_stats["dice_coeff"][self.exper.epoch_id - 1] = accuracy
            else:
                self.exper.epoch_stats["dice_coeff_outliers"][self.exper.epoch_id - 1] = accuracy
        else:
            # want to store the
            self.exper.val_stats["dice_coeff"][val_run_id-1] = accuracy

    def get_epoch_loss(self):
        return self.exper.epoch_stats["mean_loss"][self.exper.epoch_id - 1]

    def get_epoch_dice_losses(self):
        return self.exper.epoch_stats["soft_dice_loss"][self.exper.epoch_id - 1]

    def get_epoch_dice_coeffients(self):
        return self.exper.epoch_stats["dice_coeff"][self.exper.epoch_id - 1]

    def reset_results(self, use_dropout=False, mc_samples=1):
        self.test_results = TestResults(self.exper, use_dropout=use_dropout, mc_samples=mc_samples)

    def set_model_name(self, model_name):
        self.exper.model_name = model_name

    def get_test_set(self, load_train=False, load_val=True, use_cuda=True):
        if self.test_set is None:
            fold_id = self.exper.run_args.fold_ids[0]
            self.test_set = ACDC2017TestHandler.get_testset_instance(self.exper.config,
                                                                     fold_id,
                                                                     load_train=load_train, load_val=load_val,
                                                                     batch_size=None, use_cuda=use_cuda)

    def get_pred_labels(self, patient_id=None, mc_dropout=True, force_reload=False):
        self._check_dirs(config_env=config)
        if self.pred_labels is None:
            self.pred_labels = OrderedDict()
        if patient_id is not None:
            if patient_id in self.pred_labels.keys() and not force_reload:
                return self.pred_labels[patient_id]

        if mc_dropout:
            search_suffix = "_pred_labels_mc.npz"
        else:
            search_suffix = "_pred_labels.npz"

        if patient_id is not None:
            search_path = os.path.join(self.pred_output_dir, patient_id + search_suffix)
        else:
            search_path = os.path.join(self.pred_output_dir, "*" + search_suffix)

        if len(glob.glob(search_path)) == 0:
            raise ValueError("ERROR - no file found with search mask {}".format(search_path))
        for fname in glob.glob(search_path):
            pred_labels = load_pred_labels(fname)
            file_basename = os.path.splitext(os.path.basename(fname))[0]
            patient_id = file_basename[:file_basename.find("_")]
            self.pred_labels[patient_id] = pred_labels

        if patient_id is not None:
            return self.pred_labels[patient_id]

    def change_exper_dirs(self, new_dir, move_dir=False):

        """

        :param new_dir:
        usage: exper_hdl_base_brier.change_exper_dirs("20180628_13_53_01_dcnn_f1_brier_150KE_lr2e02")

        """

        print("Current directory names:")
        old_exper_dir = os.path.join(self.exper.config.root_dir, self.exper.output_dir)
        print("log_dir = {}".format(self.exper.run_args.log_dir))
        print("output_dir = {}".format(self.exper.output_dir))
        print("stats_path = {}".format(self.exper.stats_path))
        print("chkpnt_dir = {}".format(self.exper.chkpnt_dir))
        self.exper.run_args.log_dir = new_dir
        self.exper.output_dir = os.path.join(config.log_root_path, new_dir)
        self.exper.stats_path = os.path.join(self.exper.output_dir, config.stats_path)
        self.exper.chkpnt_dir = os.path.join(self.exper.output_dir, config.checkpoint_path)
        # create new directories if they don't exist
        exper_new_out_dir = os.path.join(self.exper.config.root_dir, self.exper.output_dir)
        # create_dir_if_not_exist(os.path.join(self.exper.config.root_dir, self.exper.output_dir))
        # create_dir_if_not_exist(os.path.join(self.exper.config.root_dir, self.exper.stats_path))
        # create_dir_if_not_exist(os.path.join(self.exper.config.root_dir, self.exper.chkpnt_dir))
        # create_dir_if_not_exist(os.path.join(exper_new_out_dir, config.figure_path))
        if move_dir:
            print("WARNING - Copying dir {} to {}".format(old_exper_dir, exper_new_out_dir))
            shutil.move(old_exper_dir, exper_new_out_dir)

        print("New directory names:")
        print("log_dir = {}".format(self.exper.run_args.log_dir))
        print("output_dir = {}".format(self.exper.output_dir))
        print("stats_path = {}".format(self.exper.stats_path))
        print("chkpnt_dir = {}".format(self.exper.chkpnt_dir))

    @staticmethod
    def check_compatibility(exper):
        arg_dict = vars(exper.run_args)
        # add use_reg_loss argument to run arguments, because we added this arg later
        if "use_reg_loss" not in arg_dict.keys():
            arg_dict["use_reg_loss"] = False

        return exper

    def load_experiment(self, path_to_exp, full_path=False, epoch=None, use_logfile=True, verbose=True):

        path_to_exp = os.path.join(path_to_exp, config.stats_path)

        if epoch is None:
            path_to_exp = os.path.join(path_to_exp, ExperimentHandler.exp_filename + ".dll")
        else:
            exp_filename = ExperimentHandler.exp_filename + "@{}".format(epoch) + ".dll"
            path_to_exp = os.path.join(path_to_exp, exp_filename)
        if not full_path:
            path_to_exp = os.path.join(config.root_dir, os.path.join(config.log_root_path, path_to_exp))
        if verbose:
            print("Load experiment from {}".format(path_to_exp))
        try:
            with open(path_to_exp, 'rb') as f:
                experiment = dill.load(f)

        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
            print("Can't open file {}".format(path_to_exp))
            raise IOError
        except:
            print "Unexpected error:", sys.exc_info()[0]
            raise

        self.exper = ExperimentHandler.check_compatibility(experiment)
        self.model_name = "{} p={:.2f} fold={} loss={}".format(self.exper.run_args.model,
                                                               self.exper.run_args.drop_prob,
                                                               self.exper.run_args.fold_ids[0],
                                                               self.exper.run_args.loss_function)
        if use_logfile:
            self.logger = create_logger(self.exper, file_handler=use_logfile)
        else:
            self.logger = None


class Experiment(object):

    def __init__(self, config, run_args=None):

        # logging
        self.epoch_id = 0
        self.chkpnt_dir = None
        self.logger = None
        # will be set to a relative path w.r.t. root directory of python project
        self.output_dir = None
        self.optimizer = None
        # set this later
        self.batches_per_epoch = 0
        if run_args is None:

            self.run_args = create_def_argparser(**run_dict)
        else:
            self.run_args = run_args

        self.config = config
        self.model_name = ""
        self.epoch_stats = None
        self.val_stats = None
        self.stats_path = None
        self._set_path()
        self.num_val_runs = 0
        self.val_run_id = -1
        self.init_statistics()
        self.batch_statistics = None
        self.outliers_per_epoch = None
        self.test_results_per_epoch = None

    def init_statistics(self):
        if self.run_args.val_freq != 0:
            self.num_val_runs = (self.run_args.epochs // self.run_args.val_freq)

            if self.run_args.epochs % self.run_args.val_freq == 0:
                pass
            else:
                # one extra run because max epoch is not divided by val_freq
                self.num_val_runs += 1
        self.epoch_stats = {'lr': np.zeros(self.run_args.epochs),
                            'mean_loss': np.zeros(self.run_args.epochs),
                            'mean_reg_loss': np.zeros(self.run_args.epochs),
                            'mean_loss_outliers': np.zeros(self.run_args.epochs),
                            'num_of_outlier_batches': 0,
                            # storing the mean dice loss for ES and ED separately
                            'soft_dice_loss': np.zeros((self.run_args.epochs, 2)),
                            # storing dice coefficients for LV, RV and myocardium classes for ES and ED = six values
                            'dice_coeff': np.zeros((self.run_args.epochs, 6)),
                            'dice_coeff_outliers': np.zeros((self.run_args.epochs, 6))}
        self.val_stats = {'epoch_ids': np.zeros(self.num_val_runs),
                          'mean_loss': np.zeros(self.num_val_runs),
                          'mean_reg_loss': np.zeros(self.run_args.epochs),
                          'soft_dice_loss': np.zeros((self.num_val_runs, 2)),
                          'dice_coeff': np.zeros((self.num_val_runs, 6))}

    def get_loss(self, validation=False):

        if not validation:
            return self.epoch_stats["mean_loss"]
        else:
            return self.val_stats["mean_loss"]

    def get_reg_loss(self, validation=False):
        if not validation:
            return self.epoch_stats["mean_reg_loss"]
        else:
            return self.val_stats["mean_reg_loss"]

    @property
    def validation_epoch_ids(self):
        return self.val_stats['epoch_ids']

    def _set_path(self):
        if self.run_args.log_dir is None:
            self.run_args.log_dir = str.replace(datetime.now(timezone('Europe/Berlin')).strftime(
                                    '%Y-%m-%d %H:%M:%S.%f')[:-7], ' ', '_') + "_" + create_exper_label(self) + \
                                "_lr" + "{:.0e}".format(self.run_args.lr)
            self.run_args.log_dir = str.replace(str.replace(self.run_args.log_dir, ':', '_'), '-', '')

        else:
            # custom log dir
            self.run_args.log_dir = str.replace(self.run_args.log_dir, ' ', '_')
            self.run_args.log_dir = str.replace(str.replace(self.run_args.log_dir, ':', '_'), '-', '')
        log_dir = os.path.join(self.config.log_root_path, self.run_args.log_dir)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
            fig_path = os.path.join(log_dir, self.config.figure_path)
            os.makedirs(fig_path)
            # make directory for exper statistics
            self.stats_path = os.path.join(log_dir, self.config.stats_path)
            os.makedirs(self.stats_path)
            if self.run_args.chkpnt:
                self.chkpnt_dir = os.path.join(log_dir, self.config.checkpoint_path)
                os.makedirs(self.chkpnt_dir)
        self.output_dir = log_dir

    @property
    def log_directory(self):
        return self.output_dir

    @property
    def root_directory(self):
        return self.config.root_dir

    def set_new_config(self, new_config):
        self.config = new_config

    def copy_from_object(self, obj):

        for key, value in obj.__dict__.iteritems():
            self.__dict__[key] = value



