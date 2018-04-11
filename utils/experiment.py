import sys
import time
import torch
from torch.autograd import Variable
import numpy as np
import os
from datetime import datetime
from pytz import timezone
import dill
from common.parsing import create_def_argparser, run_dict

from common.common import create_logger, create_exper_label, setSeed
from config.config import config, DEFAULT_DCNN_MC_2D, DEFAULT_DCNN_2D
from utils.batch_handlers import TwoDimBatchHandler, BatchStatistics
import models.dilated_cnn
from utils.test_results import TestResults
from common.acquisition_functions import bald_function


class ExperimentHandler(object):

    exp_filename = "exper_stats"

    def __init__(self, exper, logger=None, use_logfile=True):

        self.exper = exper
        self.test_results = None
        self.logger = None
        self.num_val_runs = 0
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

    def load_checkpoint(self, root_dir=None, checkpoint=None, verbose=False, drop_prob=0.):

        if root_dir is None:
            root_dir = self.exper.config.root_dir

        if checkpoint is None:
            checkpoint = self.exper.epoch_id

        if self.logger is None:
            use_logger = False
        else:
            use_logger = True

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
                          verbose=verbose)
        abs_checkpoint_dir = os.path.join(root_dir,
                                          os.path.join(self.exper.chkpnt_dir, checkpoint_file))
        if os.path.exists(abs_checkpoint_dir):
            checkpoint = torch.load(abs_checkpoint_dir)
            model.load_state_dict(checkpoint["state_dict"])
            if self.exper.run_args.cuda:
                model.cuda()
            if verbose:
                if use_logger:
                    self.logger.info("INFO - loaded existing model from checkpoint {}".format(abs_checkpoint_dir))
                else:
                    print("INFO - loaded existing model from checkpoint {}".format(abs_checkpoint_dir))
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
        self.next_val_run()

        if val_set_size <= self.exper.config.val_batch_size:
            # we don't need to chunk, the number of patches for validation is smaller than the batch_size
            num_of_chunks = 1
        else:
            num_of_chunks = val_set_size // self.exper.config.val_batch_size

        arr_val_loss = np.zeros(num_of_chunks)
        arr_val_acc = np.zeros(6)  # array of 6 values for accuracy of ES/ED RV/MYO/LV
        arr_val_dice = np.zeros(2)  # array of 2 values, loss for ES and ED
        for chunk in np.arange(num_of_chunks):
            val_batch = TwoDimBatchHandler(self.exper, batch_size=self.exper.config.val_batch_size,
                                           test_run=True)
            val_batch.generate_batch_2d(dataset.images(train=False), dataset.labels(train=False))
            val_loss, _ = model.do_test(val_batch.get_images(), val_batch.get_labels())
            arr_val_loss[chunk] = val_loss.data.cpu().numpy()[0]
            # returns array of 6 values
            arr_val_acc += model.get_accuracy()
            arr_val_dice += model.get_dice_losses(average=True)

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

    def test(self, model, test_set, image_num=0, mc_samples=1, sample_weights=False, compute_hd=False,
             use_uncertainty=False, referral_threshold=None, use_seed=False, verbose=False, store_details=False,
             do_filter=True, repeated_run=False, u_threshold=0.01):
        """

        :param model:
        :param test_set:
        :param image_num:
        :param mc_samples:
        :param sample_weights:
        :param compute_hd:
        :param use_uncertainty: Boolean indicating whether we use uncertainties (bald/stddev) to set pixel labels
        to most common value (binary) if uncertainty of pixel is above a certain threshold (parameter
        referral_threshold)
        :param referral_threshold: see explanation "use_uncertainty". This is the threshold parameter.
        :param u_threshold: Another threshold! used to compute the uncertainty statistics. We're currently using
        a default value of 0.01 to filter out very tiny values but could be used to investigate the effect of
        filtering out lower uncertainty values because may be this helps distinguishing levels of uncertainties in
        image slices.
        :param use_seed:
        :param verbose:
        :param do_filter: use post processing 6-connected components on predicted labels (per class)
        :param store_details: if TRUE TestResult object will hold all images/labels/predicted labels, mc-stats etc.
        which basically results in a very large object. Should not be used for evaluation of many test images,
        most certainly only for 1-3 images in order to generate some figures.
        :param repeated_run: used during ensemble testing. so we use the same model, different checkpoint, to
               evaluate the performance on the test set. Variable is used in order to indicate this situation
        :return:
        """

        if use_seed:
            setSeed(1234, self.exper.run_args.cuda)
        if self.test_results is None:
            self.test_results = TestResults(self.exper, use_dropout=sample_weights, mc_samples=mc_samples)

        # correct the divisor for calculation of stdev when low number of samples (biased), used in np.std
        if mc_samples <= 25 and mc_samples != 1:
            ddof = 1
        else:
            ddof = 0
        # b_predictions has shape [mc_samples, classes, width, height, slices]
        b_predictions = np.zeros(tuple([mc_samples] + list(test_set.labels[image_num].shape)))
        slice_idx = 0
        # batch_generator iterates over the slices of a particular image
        for batch_image, batch_labels in test_set.batch_generator(image_num):

            # NOTE: batch image shape (autograd.Variable): [1, 2, width, height] 1=batch size, 2=ES/ED image
            #       batch labels shape (autograd.Variable): [1, #classes, width, height] 8 classes, 4ES, 4ED
            # IMPORTANT: if we want to sample weights from the "posterior" over model weights, we
            # need to "tell" pytorch that we use "train" mode even during testing, otherwise dropout is disabled
            pytorch_test_mode = not sample_weights
            b_test_losses = np.zeros(mc_samples)
            bald_values = np.zeros((2, batch_labels.shape[2], batch_labels.shape[3]))
            for s in np.arange(mc_samples):
                test_loss, test_pred = model.do_test(batch_image, batch_labels,
                                                     voxel_spacing=test_set.new_voxel_spacing,
                                                     compute_hd=True, test_mode=pytorch_test_mode)
                b_predictions[s, :, :, :, test_set.slice_counter] = test_pred.data.cpu().numpy()

                b_test_losses[s] = test_loss.data.cpu().numpy()
                # dice_loss_es, dice_loss_ed = model.get_dice_losses(average=True)
            # mean/std for each pixel for each class
            mc_probs = b_predictions[:, :, :, :, test_set.slice_counter]
            mean_test_pred, std_test_pred = np.mean(mc_probs, axis=0, keepdims=True), \
                                                np.std(mc_probs, axis=0, ddof=ddof)

            bald_values[0] = bald_function(b_predictions[:, 0:4, :, :, test_set.slice_counter])
            bald_values[1] = bald_function(b_predictions[:, 4:, :, :, test_set.slice_counter])

            means_test_loss = np.mean(b_test_losses)
            if use_uncertainty:
                test_set.set_pred_labels(mean_test_pred, pred_stddev=std_test_pred,
                                         referral_threshold=referral_threshold,
                                         verbose=verbose, do_filter=do_filter)
            else:
                # NOTE: we set do_filter (connected components post-processing) to FALSE here because we will do
                # this at the end for the complete 3D label object, but not for the individual slices
                # NOTE: we set referral_threshold to 0. because we don't want to use uncertainties!
                test_set.set_pred_labels(mean_test_pred, referral_threshold=0.,
                                         verbose=verbose, do_filter=False)

            test_set.set_stddev_map(std_test_pred, u_threshold=u_threshold)
            test_set.set_bald_map(bald_values, u_threshold=u_threshold)
            slice_acc, slice_hd = test_set.compute_slice_accuracy(compute_hd=compute_hd)
            if sample_weights:
                # NOTE: currently only displaying the BALD uncertainty stats but we also capture the stddev stats
                es_total_uncert, es_num_of_pixel_uncert, es_num_pixel_uncert_above_tre, num_of_conn_commponents = \
                    test_set.b_uncertainty_stats["bald"][0, :, slice_idx]  # ES
                ed_total_uncert, ed_num_of_pixel_uncert, ed_num_pixel_uncert_above_tre, num_of_conn_commponents = \
                    test_set.b_uncertainty_stats["bald"][1, :, slice_idx]  # ED
                es_seg_errors = np.sum(test_set.b_seg_errors[slice_idx, :4])
                ed_seg_errors = np.sum(test_set.b_seg_errors[slice_idx, 4:])
                if verbose:
                    print("Test img/slice {}/{}".format(image_num, slice_idx))
                    print("ES: Total BALD/seg-errors/#pixel/#pixel(tre) \tDice (RV/Myo/LV)\tHD (RV/Myo/LV)")
                    print("  \t{:.2f}/{}/{}/{} \t\t\t{:.2f}/{:.2f}/{:.2f}"
                          "\t\t{:.2f}/{:.2f}/{:.2f}".format(es_total_uncert, es_seg_errors, es_num_of_pixel_uncert,
                                                                       es_num_pixel_uncert_above_tre,
                                                                       slice_acc[1], slice_acc[2], slice_acc[3],
                                                                       slice_hd[1], slice_hd[2], slice_hd[3]))
                    # print(np.array_str(np.array(es_region_mean_uncert), precision=3))
                    print("ED: Total BALD/seg-errors/#pixel/#pixel(tre)\tDice (RV/Myo/LV)\tHD (RV/Myo/LV)")
                    print("  \t{:.2f}/{}/{}/{} \t\t\t{:.2f}/{:.2f}/{:.2f}"
                          "\t\t{:.2f}/{:.2f}/{:.2f}".format(ed_total_uncert, ed_seg_errors, ed_num_of_pixel_uncert,
                                                          ed_num_pixel_uncert_above_tre,
                                                          slice_acc[5], slice_acc[6], slice_acc[7],
                                                          slice_hd[5], slice_hd[6], slice_hd[7]))
                    # print(np.array_str(np.array(ed_region_mean_uncert), precision=3))
                    print("------------------------------------------------------------------------")
            slice_idx += 1

        test_accuracy, test_hd = test_set.get_accuracy(compute_hd=compute_hd, do_filter=do_filter)
        self.test_results.add_results(test_set.b_image, test_set.b_labels, test_set.b_image_id,
                                      test_set.b_pred_labels, b_predictions, test_set.b_stddev_map,
                                      test_accuracy, test_hd, seg_errors=test_set.b_seg_errors,
                                      store_all=store_details,
                                      bald_maps=test_set.b_bald_map,
                                      uncertainty_stats=test_set.b_uncertainty_stats,
                                      test_accuracy_slices=test_set.b_acc_slices,
                                      test_hd_slices=test_set.b_hd_slices,
                                      image_name=test_set.b_image_name, repeated_run=repeated_run)
        print("Image {} - Test accuracy: test loss {:.3f}\t "
              "dice(RV/Myo/LV): ES {:.2f}/{:.2f}/{:.2f} --- "
              "ED {:.2f}/{:.2f}/{:.2f}".format(image_num+1, means_test_loss,
                                               test_accuracy[1], test_accuracy[2],
                                               test_accuracy[3], test_accuracy[5],
                                               test_accuracy[6], test_accuracy[7]))
        if compute_hd:
            print("Image {} - Test accuracy: test loss {:.3f}\t "

                  "Hausdorff(RV/Myo/LV): ES {:.2f}/{:.2f}/{:.2f} --- "
                  "ED {:.2f}/{:.2f}/{:.2f}".format(image_num+1, means_test_loss,
                                                   test_hd[1], test_hd[2],
                                                   test_hd[3], test_hd[5],
                                                   test_hd[6], test_hd[7]))

    def set_lr(self, lr):
        self.exper.epoch_stats["lr"][self.exper.epoch_id - 1] = lr

    def set_batch_loss(self, loss):
        if isinstance(loss, Variable) or isinstance(loss, torch.FloatTensor):
            loss = loss.data.cpu().squeeze().numpy()
        self.exper.epoch_stats["mean_loss"][self.exper.epoch_id-1] = loss

    def set_dice_losses(self, dice_losses, val_run_id=None):

        if val_run_id is None:
            self.exper.epoch_stats["soft_dice_loss"][self.exper.epoch_id - 1] = dice_losses
        else:
            self.exper.val_stats["soft_dice_loss"][val_run_id-1] = dice_losses

    def set_accuracy(self, accuracy, val_run_id=None):

        if val_run_id is None:
            self.exper.epoch_stats["dice_coeff"][self.exper.epoch_id - 1] = accuracy
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
                            # storing the mean dice loss for ES and ED separately
                            'soft_dice_loss': np.zeros((self.run_args.epochs, 2)),
                            # storing dice coefficients for LV, RV and myocardium classes for ES and ED = six values
                            'dice_coeff': np.zeros((self.run_args.epochs, 6))}
        self.val_stats = {'epoch_ids': np.zeros(self.num_val_runs),
                          'mean_loss': np.zeros(self.num_val_runs),
                          'soft_dice_loss': np.zeros((self.num_val_runs, 2)),
                          'dice_coeff': np.zeros((self.num_val_runs, 6))}

    def get_loss(self, validation=False):

        if not validation:
            return self.epoch_stats["mean_loss"]
        else:
            return self.val_stats["mean_loss"]

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



