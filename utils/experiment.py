import sys
import time
import torch
from torch.autograd import Variable
import numpy as np
import os
import glob
import copy
from collections import OrderedDict
from datetime import datetime
from pytz import timezone
import dill
from common.parsing import create_def_argparser, run_dict

from common.common import create_logger, create_exper_label, setSeed
from config.config import config, DEFAULT_DCNN_MC_2D, DEFAULT_DCNN_2D
from utils.batch_handlers import TwoDimBatchHandler, BatchStatistics
from utils.generate_uncertainty_maps import UncertaintyMapsGenerator, ImageUncertainties, OutOfDistributionSlices
from utils.generate_uncertainty_maps import ImageOutliers
from utils.post_processing import filter_connected_components, detect_largest_umap_areas
import models.dilated_cnn
from utils.test_results import TestResults
from common.acquisition_functions import bald_function
from plotting.uncertainty_plots import analyze_slices
from plotting.main_seg_results import plot_seg_erros_uncertainties
from in_out.load_data import ACDC2017DataSet
from in_out.patient_classification import Patients


class ExperimentHandler(object):

    exp_filename = "exper_stats"

    def __init__(self, exper, logger=None, use_logfile=True):

        self.exper = exper
        self.u_maps = None
        self.pred_prob_maps = None
        self.referral_umaps = None
        self.ref_map_blobs = None
        self.test_results = None
        self.logger = None
        self.num_val_runs = 0
        self.saved_model_state_dict = None
        self.ensemble_models = OrderedDict()
        self.test_set_ids = {}
        self.patients = None
        self.referred_slices = None
        if logger is None:
            self.logger = create_logger(self.exper, file_handler=use_logfile)
        else:
            self.logger = logger

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

        outfile = os.path.join(self.exper.stats_path, file_name)
        with open(outfile, 'wb') as f:
            dill.dump(self.exper, f)

        if self.logger is not None:
            self.logger.info("Epoch: {} - Saving experimental details to {}".format(self.exper.epoch_id, outfile))

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

        str_classname = "BaseDilated2DCNN"
        checkpoint_file = str_classname + "checkpoint" + str(checkpoint).zfill(5) + ".pth.tar"
        act_class = getattr(models.dilated_cnn, str_classname)
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

        model = act_class(architecture=architecture, optimizer=self.exper.config.optimizer,
                          lr=self.exper.run_args.lr,
                          weight_decay=self.exper.run_args.weight_decay,
                          use_cuda=self.exper.run_args.cuda,
                          cycle_length=self.exper.run_args.cycle_length,
                          verbose=verbose, use_reg_loss=self.exper.run_args.use_reg_loss)
        abs_checkpoint_dir = os.path.join(chkpnt_dir, checkpoint_file)
        if os.path.exists(abs_checkpoint_dir):
            model_state_dict = torch.load(abs_checkpoint_dir)
            model.load_state_dict(model_state_dict["state_dict"])
            if self.exper.run_args.cuda:
                model.cuda()
            if verbose and not retrain:
                self.info("INFO - loaded existing model with checkpoint {} from dir {}".format(abs_checkpoint_dir,
                                                                                               checkpoint))
            else:
                self.info("Loading existing model with checkpoint {} from dir {}".format(chkpnt_dir, checkpoint))
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

        arr_val_loss = np.zeros(num_of_chunks)
        arr_val_acc = np.zeros(6)  # array of 6 values for accuracy of ES/ED RV/MYO/LV
        arr_val_dice = np.zeros(2)  # array of 2 values, loss for ES and ED
        s_offset = 0
        for chunk in np.arange(num_of_chunks):
            slice_range = np.arange(s_offset, s_offset + val_batch_size)
            val_batch = TwoDimBatchHandler(self.exper, batch_size=val_batch_size,
                                           test_run=True)
            val_batch.generate_batch_2d(dataset.images(train=False), dataset.labels(train=False),
                                        slice_range=slice_range)
            val_loss, _ = model.do_test(val_batch.get_images(), val_batch.get_labels(),
                                        num_of_labels_per_class=val_batch.get_num_labels_per_class())
            arr_val_loss[chunk] = val_loss.data.cpu().numpy()[0]
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
        self.set_dice_losses(arr_val_dice, val_run_id=self.num_val_runs)
        duration = time.time() - start_time
        self.logger.info("---> VALIDATION epoch {} (#patches={}): current loss {:.3f}\t "
                         "dice-coeff:: ES {:.3f}/{:.3f}/{:.3f} --- "
                         "ED {:.3f}/{:.3f}/{:.3f}  (time={:.2f} sec)".format(self.exper.epoch_id, val_set_size, val_loss,
                                                                             arr_val_acc[0], arr_val_acc[1],
                                                                             arr_val_acc[2], arr_val_acc[3],
                                                                             arr_val_acc[4], arr_val_acc[5], duration))
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

        :param store_test_results:
        :param checkpoints: list of model checkpoints that we use for the ensemble test
        :return:
        """
        half_classes = 4
        ref_dice_es = [[], [], []]
        ref_dice_ed = [[], [], []]
        dice_es = [[], [], []]
        dice_ed = [[], [], []]
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
            test_set.save_pred_labels(self.exper.output_dir, u_threshold=0., mc_dropout=mc_dropout,
                                      forced_save=True)
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

    def create_u_maps(self, model=None, checkpoints=None, mc_samples=10, u_threshold=0., do_save_u_stats=False,
                      save_actual_maps=False, test_set=None, generate_figures=False, verbose=False,
                      aggregate_func="max", referral_thresholds=None, store_test_results=False):

        if model is None and checkpoints is None:
            raise ValueError("When model parameter is None, you need to specify the checkpoint model "
                             "that needs to be loaded.")

        if save_actual_maps and not do_save_u_stats:
            print("WARNING - Setting do_save_u_stats=True because you specified that you want to "
                  "save the actual uncertainty maps!")
            do_save_u_stats = True

        maps_generator = UncertaintyMapsGenerator(self, test_set=test_set, verbose=verbose,
                                                  mc_samples=mc_samples, u_threshold=u_threshold,
                                                  checkpoints=checkpoints, store_test_results=store_test_results,
                                                  aggregate_func=aggregate_func,
                                                  referral_thresholds=referral_thresholds)
        maps_generator(do_save=do_save_u_stats, clean_up=False, save_actual_maps=save_actual_maps,
                       generate_figures=generate_figures)

        # if generate_figures:
        #     if self.test_results is None:
        #         # if test_results object of experiment handler is None (because we generated u-maps) we point to the
        #         # test_result object of the map generator. If we re-use maps we load the object, see above
        #         self.test_results = maps_generator.test_results
        #
        #     analyze_slices(self, image_range=None, do_save=True, do_show=False, u_type="stddev",
        #                    use_saved_umaps=False)

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
        :param do_save_u_stats: save the uncertainty statistics and raw values. See object UncertaintyMapsGenerator
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
        UncertaintyMapsGenerator will load the test_set, assuming, we want to load the validation set of the FOLD
        we trained on (info comes from exper_handler.exper.run_args.fold_ids
        :param do_analyze_slices: generate plots for each image, which can be used to visually inspect which of
        the slices have "bizar" uncertainties.
        :return:

        Note: we don't specify the test_set. But, UncertaintyMapsGenerator will by default, load all training
        images from the FOLD the current model is trained on!
        """
        if model is None and checkpoint is None:
            raise ValueError("When model parameter is None, you need to specify the checkpoint model"
                             "that needs to be loaded.")

        if not use_existing_umaps:
            maps_generator = UncertaintyMapsGenerator(self, model=model, test_set=test_set, verbose=False,
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

    def get_referral_maps(self, u_threshold, per_class=True, aggregate_func="max"):
        input_dir = os.path.join(self.exper.config.root_dir,
                                           os.path.join(self.exper.output_dir, config.u_map_dir))
        u_threshold = str(u_threshold).replace(".", "_")
        if per_class:
            search_path = os.path.join(input_dir, "*" + "_filtered_cls_umaps_" + aggregate_func +
                                       u_threshold + ".npz")
        else:
            search_path = os.path.join(input_dir, "*" + "_filtered_umaps_" + aggregate_func + u_threshold + ".npz")
        files = glob.glob(search_path)
        if len(files) == 0:
            raise ImportError("ERROR - no referral u-maps found in {}".format(search_path))
        self.referral_umaps = OrderedDict()
        self.ref_map_blobs = OrderedDict()
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
                self.referral_umaps[patientID] = data["filtered_umap"]
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

    def get_pred_prob_maps(self, patient_id=None):

        input_dir = os.path.join(self.exper.config.root_dir,
                                 os.path.join(self.exper.output_dir, config.pred_lbl_dir))

        if patient_id is None:
            search_mask = "patient_id*_pred_probs_mc"
        else:
            search_mask = patient_id + "_pred_probs_mc"

        search_mask = search_mask + ".npz"
        file_name = os.path.join(input_dir, search_mask)
        if self.pred_prob_maps is None:
            self.pred_prob_maps = OrderedDict()

        for fname in glob.glob(file_name):
            file_basename = os.path.splitext(os.path.basename(fname))[0]
            f_patient_id = file_basename[:file_basename.find("_")]
            try:
                np_archive = np.load(file_name)
                pred_probs = np_archive["pred_probs"]
                self.pred_prob_maps[f_patient_id] = pred_probs
            except IOError:
                print("ERROR - Unable to load predicted prob-maps from numpy file {}".format(file_name))
            except KeyError:
                print("ERROR - Archive pred_probs does not exist in {}.".format(file_name))

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
            # Yes I know, completely inconsistent the output is [8classes, width, height, #slices] instead of [2, 4...]
            filtered_cls_stddev_map = np.zeros((num_of_phases * num_of_classes, raw_u_map.shape[2], raw_u_map.shape[3],
                                                raw_u_map.shape[4]))
            # here we store the maps per phase ES/ED taking the max uncertainty over 4 classes after we've filtered
            # the u-maps per class by means of 6 connectivity components.
            filtered_stddev_map = np.zeros((num_of_phases, raw_u_map.shape[2], raw_u_map.shape[3], raw_u_map.shape[4]))
            for phase in np.arange(num_of_phases):
                cls_offset = phase * num_of_classes
                for cls in np.arange(0, num_of_classes):
                    # if cls != 0 and cls != 4:
                    u_3dmaps_cls = raw_u_map[phase, cls]
                    # set all uncertainties below threshold to zero
                    u_3dmaps_cls[u_3dmaps_cls < u_threshold] = 0
                    # do not yet filter 6 largest connected comonents here when we average over the stddev values
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
                    else:
                        filtered_cls_stddev_map[cls + cls_offset] = u_3dmaps_cls

                if phase == 0:
                    if aggregate_func == "max":
                        filtered_stddev_map[phase] = np.max(filtered_cls_stddev_map[:num_of_classes], axis=0)
                    else:
                        filtered_stddev_map[phase] = np.mean(filtered_cls_stddev_map[:num_of_classes], axis=0)
                else:
                    if aggregate_func == "max":
                        filtered_stddev_map[phase] = np.max(filtered_cls_stddev_map[num_of_classes:], axis=0)
                    else:
                        filtered_stddev_map[phase] = np.mean(filtered_cls_stddev_map[num_of_classes:], axis=0)
                if aggregate_func == "mean":
                    filtered_stddev_map[phase] = filter_connected_components(filtered_stddev_map[phase],
                                                                             threshold=u_threshold)
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
                np.savez(file_name, filtered_umap=filtered_stddev_map, filtered_stddev_blobs=u_map_c_areas)
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

    def get_patients(self):
        """
        loading the complete set of patient_ids with the corresponding disease classification

        :return:
        """
        patients = Patients()
        patients.load(self.exper.config.data_dir)
        self.patients = patients.category

    def info(self, message):
        if self.logger is None:
            print(message)
        else:
            self.logger.info(message)

    def set_lr(self, lr):
        self.exper.epoch_stats["lr"][self.exper.epoch_id - 1] = lr

    def set_batch_loss(self, loss, used_outliers=False, reg_loss=None):
        if isinstance(loss, Variable) or isinstance(loss, torch.FloatTensor):
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

    @staticmethod
    def check_compatibility(exper):
        arg_dict = vars(exper.run_args)
        # add use_reg_loss argument to run arguments, because we added this arg later
        if "use_reg_loss" not in arg_dict.keys():
            arg_dict["use_reg_loss"] = False

        return exper

    @staticmethod
    def load_experiment(path_to_exp, full_path=False, epoch=None):

        path_to_exp = os.path.join(path_to_exp, config.stats_path)

        if epoch is None:
            path_to_exp = os.path.join(path_to_exp, ExperimentHandler.exp_filename + ".dll")
        else:
            exp_filename = ExperimentHandler.exp_filename + "@{}".format(epoch) + ".dll"
            path_to_exp = os.path.join(path_to_exp, exp_filename)
        if not full_path:
            path_to_exp = os.path.join(config.root_dir, os.path.join(config.log_root_path, path_to_exp))

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

        experiment = ExperimentHandler.check_compatibility(experiment)
        return experiment


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



