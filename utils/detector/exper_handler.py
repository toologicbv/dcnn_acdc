import sys
import dill
import os
import time
import shutil
import torch
from collections import OrderedDict
import numpy as np
from common.common import create_logger
from sklearn.metrics import auc

from scipy.signal import convolve2d

from models.model_handler import load_region_detector_model
from common.detector.config import config_detector

from utils.detector.batch_handler import BatchHandler
from utils.dslices.accuracies import compute_eval_metrics


class ExperimentHandler(object):

    exp_filename = "exper_stats"

    def __init__(self):

        self.exper = None
        self.u_maps = None
        self.pred_prob_maps = None
        self.pred_labels = None
        self.referral_umaps = None
        self.entropy_maps = None
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
        self.device = None
        # store reference to object ExperHandlerEnsemble
        self.seg_exper_ensemble = None
        # ID for different test runs (after training)
        self.last_test_id = 0
        # not consequent, but we'll add a dictionary for the test runs to the handler instead of the experiment object
        self.test_stats = {}
        # objects to compute ROC-AUC and ROC-PR
        self.mean_x_values = np.linspace(0, 1, 100)  # used for interpolating values in order to compute mean
        self.mean_tpos_rate = []
        self.mean_prec_rate = []
        self.aucs_roc = []
        self.aucs_pr = []
        self.eval_loss = []
        self.arr_eval_metrics = []
        self.stats_auc_roc = None
        # number of test/validation slices that contained at least one true positive in the GT labels. The
        # evaluation metrics (f1 auc_pr etc) are based on this number
        self.num_of_pos_eval_slices = 0
        self.num_of_neg_eval_slices = 0
        # total number of test/validation slices that we processed
        self.num_of_eval_slices = 0

    def set_seg_ensemble(self, seg_ensemble):
        self.seg_exper_ensemble = seg_ensemble

    def set_loss(self, loss, val_run_id=None):

        if val_run_id is None:
            self.exper.epoch_stats["loss"][self.exper.epoch_id - 1] = loss
        else:
            self.exper.val_stats["loss"][val_run_id-1] = loss

    def set_exper(self, exper, use_logfile=False):
        self.exper = exper
        if use_logfile:
            self.logger = create_logger(self.exper, file_handler=use_logfile)
        else:
            self.logger = None

    def set_root_dir(self, root_dir):
        self.exper.config.root_dir = root_dir
        self.exper.config.data_dir = os.path.join(self.exper.config.root_dir, "data/Folds/")

    def set_checkpoint_dir(self):
        log_dir = os.path.join(config_detector.log_root_path, self.exper.run_args.log_dir)
        self.exper.chkpnt_dir = os.path.join(log_dir, config_detector.checkpoint_path)

    def next_epoch(self):
        self.exper.epoch_id += 1

    def next_val_run(self):
        self.exper.val_run_id += 1
        self.num_val_runs += 1
        self.exper.val_stats["epoch_ids"][self.exper.val_run_id] = self.exper.epoch_id
        self.reset_eval_metrics()

    def next_test_id(self):
        self.last_test_id += 1
        return self.last_test_id

    def compute_mean_aucs(self):

        self.mean_tpos_rate = np.mean(self.mean_tpos_rate, axis=0)
        self.mean_tpos_rate[-1] = 1.0
        mean_auc_roc = auc(self.mean_x_values, self.mean_tpos_rate)
        std_auc_roc = np.std(self.aucs_roc)
        self.stats_auc_roc = tuple((mean_auc_roc, std_auc_roc))

    def run_eval(self, data_set, model, verbose=False, eval_size=None, keep_batch=False, disrupt_chnnl=None):
        self.reset_eval_metrics()
        not_one_tp = True
        eval_set_size = data_set.get_size(is_train=False)
        # Currently not using the complete test set
        if eval_size is not None and eval_set_size > eval_size:
            eval_set_size = eval_size

        eval_batch = BatchHandler(data_set=data_set, is_train=False, cuda=self.exper.run_args.cuda, backward_freq=1,
                                  num_of_max_pool_layers=self.exper.config.num_of_max_pool)
        model.eval()
        # we use the following 3 arrays to collect the predicted (and gt) results in order to calculate the
        # final performance measures (f1, roc_auc, pr_auc, rec, precision
        np_gt_labels = np.empty(0)
        np_pred_labels = np.empty(0)
        np_pred_probs = np.empty(0)
        """
            Parameter: disrupt_chnnl {None, 0, 1, 2} 
            Input channels for network: 0=mri image 1=uncertainty map 2=auto seg-mask
        """
        for x_input, y_labels_dict in eval_batch(batch_size=eval_set_size, keep_batch=keep_batch,
                                                 disrupt_chnnl=disrupt_chnnl):
            # New in pytorch 0.4.0, use local context manager to turn off history tracking
            with torch.set_grad_enabled(False):
                # batch_size=None and do_balance=True => number of degenerate slices/per patient determines
                #                                        the batch_size.
                # batch_size=None and do_balance=False => classify all slices for a particular patient
                """
                    NOTE: we can evaluate the model with grid-spacing 8 or 4 (default former)
                    grid-spacing 8: y_pred_lbls = y_lbl_grid8 and pred_probs = pred_probs_list[0]
                    grid-spacing 4: y_pred_lbls = y_lbl_grid4 and pred_probs = pred_probs_list[1]
                """
                y_lbl_max_grid = y_labels_dict[self.exper.config.max_grid_spacing]
                y_gt_lbl_grid4 = y_labels_dict[4]
                y_gt_lbl_slice = y_labels_dict[1]
                y_gt_lbls = y_lbl_max_grid
                eval_loss, pred_probs_list = model.do_forward_pass(x_input, y_lbl_max_grid, y_labels_extra=y_gt_lbl_grid4)
                # indices of pred_probs_list:
                # 0 index = 8 grid-spacing hence 9x9 predictions
                # 1 index = 4 grid-spacing hence 18x18 predictions
                pred_probs = pred_probs_list[0]
                pred_probs = pred_probs.data.cpu().numpy()
                # IMPORTANT: Enable this when you want to TEST THE NAIVE APPROACH!!!
                # pred_probs[:, 1] = 1.
                pred_labels = np.argmax(pred_probs, axis=1)
            # NOTE: THIS IS EXPERIMENTAL STUFF: we convolve the pred_labels i.o.t. smooth them. Increases recall
            #                                   and reduce precision.
            # pred_labels = convolve2d(pred_labels.squeeze(), np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]), mode="same",
            #                         boundary="wrap")
            # pred_labels[pred_labels != 0] = 1
            # pred_labels = pred_labels.astype(np.bool)
            # other than pred_labels and gt_labels we store pred_probs per slice because we want the softmax-probs
            # per slice in the grid form (probably 8x8) in order to produce the probs heat map per slice (batch_handler)
            if keep_batch:
                slice_id = eval_batch.batch_dta_slice_ids[-1]
                eval_batch.add_probs(pred_probs[:, 1], slice_id=slice_id)
                eval_batch.add_gt_labels_slice(y_gt_lbl_slice, slice_id=slice_id)
            np_gt_labels = np.concatenate((np_gt_labels, (y_gt_lbls.data.cpu().numpy()).flatten()))
            np_pred_labels = np.concatenate((np_pred_labels, pred_labels.flatten()))
            # Remember, we're only interested in the softmax-probs for the positive class
            np_pred_probs = np.concatenate((np_pred_probs, pred_probs[:, 1].flatten()))
            self.eval_loss.append([eval_loss.item()])
            self.num_of_eval_slices += 1

        num_of_pos_grids = np.count_nonzero(np_gt_labels)
        self.num_of_pos_eval_slices += num_of_pos_grids
        self.num_of_neg_eval_slices += np_gt_labels.shape[0] - num_of_pos_grids
        f1, roc_auc, pr_auc, prec, rec, fpr, tpr, precision, recall = \
            compute_eval_metrics(np_gt_labels, np_pred_labels, np_pred_probs)
        # keep predicted probs & labels
        if keep_batch:
            eval_batch.add_pred_labels(np_pred_labels)
            eval_batch.add_gt_labels(np_gt_labels)
        if f1 != -1:
            not_one_tp = False
            self.arr_eval_metrics = np.array([f1, roc_auc, pr_auc, prec, rec])
        self.eval_loss = np.array(self.eval_loss)
        if self.eval_loss.shape[0] > 1:
            self.eval_loss = np.mean(self.eval_loss)
        else:
            self.eval_loss = self.eval_loss[0]
        # if we have at least one TP, otherwise...
        if not_one_tp:
            # we only had test slices that don't contain ANY positive voxels that needed to be detected. Hence, we
            # have no statistics
            self.stats_auc_roc = tuple((0, 0))
            self.arr_eval_metrics = np.array([0, 0, 0, 0, 0])

        if verbose:
            self.info("Evaluation - #slices={} (negatives/positives={}/{}) - f1={:.3f} - roc_auc={:.3f} "
                      "- pr_auc={:.3f} - prec={:.3f} - rec={:.3f}".format(self.num_of_eval_slices,
                                                                          self.num_of_neg_eval_slices,
                                                                          self.num_of_pos_eval_slices,
                                                                          f1, roc_auc, pr_auc, prec, rec))
        model.train()
        if keep_batch:
            return eval_batch
        else:
            del eval_batch
            return None

    def _save_eval_features(self, features, labels, np_extra_lbls):
        out_filename = os.path.join(self.exper.config.root_dir, self.exper.stats_path)
        out_filename = os.path.join(out_filename, "eval_feature_arrays")
        try:
            np.savez(out_filename, features=features, labels=labels, extra_labels=np_extra_lbls)
            self.info("INFO - Saved features+labels of eval run to {}".format(out_filename))
        except IOError:
            print("ERROR - Can't save features+labels to {}".format(out_filename))

    def eval(self, data_set, model, eval_size=25, keep_batch=False, verbose=False, is_test=False, disrupt_chnnl=None):
        start_time = time.time()
        if is_test:
            test_id = self.next_test_id()
            self.next_test_id()
        else:
            self.next_val_run()
        batch = self.run_eval(data_set=data_set, model=model, eval_size=eval_size, keep_batch=keep_batch,
                              verbose=verbose, disrupt_chnnl=disrupt_chnnl)
        if is_test:
            test_stats = {}
            test_stats["loss"] = self.eval_loss
            test_stats["f1"] = self.arr_eval_metrics[0]
            test_stats["roc_auc"] = self.arr_eval_metrics[1]
            test_stats["pr_auc"] = self.arr_eval_metrics[2]
            test_stats["prec"] = self.arr_eval_metrics[3]
            test_stats["rec"] = self.arr_eval_metrics[4]
            self.test_stats[test_id] = test_stats
        else:
            self.exper.val_stats["loss"][self.num_val_runs - 1] = self.eval_loss
            self.exper.val_stats["f1"][self.num_val_runs - 1] = self.arr_eval_metrics[0]
            self.exper.val_stats["roc_auc"][self.num_val_runs - 1] = self.arr_eval_metrics[1]
            self.exper.val_stats["pr_auc"][self.num_val_runs - 1] = self.arr_eval_metrics[2]
            self.exper.val_stats["prec"][self.num_val_runs - 1] = self.arr_eval_metrics[3]
            self.exper.val_stats["rec"][self.num_val_runs - 1] = self.arr_eval_metrics[4]
        duration = time.time() - start_time

        self.info("---> END VALIDATION epoch {} #slices={}(skipped {}) (negatives/positives={}/{}) loss {:.3f}: "
                  "f1={:.3f} - roc_auc={:.3f} - pr_auc={:.3f} - prec={:.3f} - rec={:.3f} "
                  "- {:.2f} seconds".format(self.exper.epoch_id, self.num_of_eval_slices,
                                            0 if not keep_batch else batch.num_of_skipped_slices,
                                            self.num_of_neg_eval_slices,
                                            self.num_of_pos_eval_slices, self.eval_loss, self.arr_eval_metrics[0],
                                            self.arr_eval_metrics[1], self.arr_eval_metrics[2],
                                            self.arr_eval_metrics[3], self.arr_eval_metrics[4], duration))
        # self.logger.info("\t Check: roc_auc={:.3f} - pr_auc={:.3f}".format(arr_val_eval[1], arr_val_eval[2]))
        self.reset_eval_metrics()
        if keep_batch:
            return batch

    def reset_eval_metrics(self):
        self.mean_tpos_rate = []
        self.mean_prec_rate = []
        self.aucs_roc = []
        self.aucs_pr = []
        self.eval_loss = []
        self.arr_eval_metrics = []  # store f1, roc_auc, pr_auc, precision, recall scores
        self.num_of_eval_slices = 0
        self.num_of_pos_eval_slices = 0
        self.num_of_neg_eval_slices = 0

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

    def load_checkpoint(self, exper_dir=None, checkpoint=None, verbose=False, retrain=False):
        # we first "re-construct" the self.exper.chkpnt_dir, because could be different on a machine when we
        # run the evaluation.
        self.set_checkpoint_dir()

        if exper_dir is None:
            # chkpnt_dir should be /home/jorg/repository/dcnn_acdc/logs/<experiment dir>/checkpoints/
            chkpnt_dir = os.path.join(self.exper.config.root_dir, self.exper.chkpnt_dir)
            # will be concatenated with "<checkpoint dir> further below
        else:
            chkpnt_dir = os.path.join(self.exper.config.root_dir, self.exper.chkpnt_dir)

        if checkpoint is None:
            checkpoint = self.exper.epoch_id

        str_classname = config_detector.base_class
        checkpoint_file = str_classname + "checkpoint" + str(checkpoint).zfill(5) + ".pth.tar"
        model = load_region_detector_model(self, verbose=verbose)
        abs_checkpoint_dir = os.path.join(chkpnt_dir, checkpoint_file)
        if os.path.exists(abs_checkpoint_dir):
            model_state_dict = torch.load(abs_checkpoint_dir)
            model.load_state_dict(model_state_dict["state_dict"])
            if self.exper.run_args.cuda:
                model.cuda()
            if verbose and not retrain:
                self.info("INFO - loaded existing model with checkpoint {} from dir {}".format(checkpoint,
                                                                                               abs_checkpoint_dir ))
        else:
            raise IOError("ERROR - Path to checkpoint not found {}".format(abs_checkpoint_dir))

        return model

    def info(self, message):
        if self.logger is None:
            print(message)
        else:
            self.logger.info(message)

    def set_config_object(self, new_config):
        self.exper.config = new_config

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
        self.exper.output_dir = os.path.join(config_detector.log_root_path, new_dir)
        self.exper.stats_path = os.path.join(self.exper.output_dir, config_detector.stats_path)
        self.exper.chkpnt_dir = os.path.join(self.exper.output_dir, config_detector.checkpoint_path)
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
        if "use_random_map" not in arg_dict.keys():
            arg_dict["use_random_map"] = False

        return exper

    def load_experiment(self, path_to_exp, full_path=False, epoch=None, use_logfile=True, verbose=True):

        path_to_exp = os.path.join(path_to_exp, config_detector.stats_path)

        if epoch is None:
            path_to_exp = os.path.join(path_to_exp, ExperimentHandler.exp_filename + ".dll")
        else:
            exp_filename = ExperimentHandler.exp_filename + "@{}".format(epoch) + ".dll"
            path_to_exp = os.path.join(path_to_exp, exp_filename)
        if not full_path:
            path_to_exp = os.path.join(config_detector.root_dir,
                                       os.path.join(config_detector.log_root_path, path_to_exp))
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
        self.exper = experiment
        type_of_map = "no-" + self.exper.run_args.type_of_map if self.exper.run_args.use_no_map else \
            self.exper.run_args.type_of_map
        self.model_name = "{}-{}-f{} ".format(self.exper.run_args.model,
                                              type_of_map,
                                              self.exper.run_args.fold_id)
        if use_logfile:
            self.logger = create_logger(self.exper, file_handler=use_logfile)
        else:
            self.logger = None


def create_experiment(exper_id, verbose=False):

    log_dir = os.path.join(config_detector.root_dir, config_detector.log_root_path)
    exp_model_path = os.path.join(log_dir, exper_id)
    exper_handler = ExperimentHandler()
    exper_handler.load_experiment(exp_model_path, use_logfile=False)
    exper_handler.set_root_dir(config_detector.root_dir)
    exper_args = exper_handler.exper.run_args
    if verbose:
        info_str = "{} fold={} loss={}".format(exper_args.model, exper_args.fold_ids,
                                               exper_args.loss_function)
        print("INFO - Experimental details extracted:: " + info_str)
    return exper_handler
