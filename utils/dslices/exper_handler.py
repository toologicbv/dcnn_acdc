import sys
import dill
import os
import time
import shutil
import torch
from collections import OrderedDict
import numpy as np
from common.common import create_logger
from sklearn.metrics import precision_recall_curve, roc_curve

from models.model_handler import load_slice_detector_model
from common.dslices.config import config
from common.dslices.helper import create_experiment
from utils.referral_results import ReferralResults
from utils.dslices.batch_handler import BatchHandler as BatchHandlerSD
from utils.dslices.accuracies import compute_eval_metrics
from in_out.dataset_slice_detector import create_dataset


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

    def next_epoch(self):
        self.exper.epoch_id += 1

    def next_val_run(self):
        self.exper.val_run_id += 1
        self.num_val_runs += 1
        self.exper.val_stats["epoch_ids"][self.exper.val_run_id] = self.exper.epoch_id

    def next_test_id(self):
        self.last_test_id += 1
        return self.last_test_id

    def eval(self, data_set, model, val_set_size=None, verbose=False):
        start_time = time.time()
        # set the size of the validation set. If not passed as argument, use config setting (depends on machine)
        # make sure it's never bigger than the actual size of the validation set
        if val_set_size is None or val_set_size > data_set.get_size(is_train=False):
            val_set_size = len(data_set.get_patient_ids(is_train=False))
        self.next_val_run()

        arr_val_loss = []
        arr_val_eval = []  # store f1, roc_auc, pr_auc, precision, recall scores
        # create batch object
        val_batch = BatchHandlerSD(data_set=data_set, is_train=False, cuda=self.exper.run_args.cuda)
        self.logger.info("---> BEGIN VALIDATION epoch {}".format(self.exper.epoch_id))
        all_labels = []
        all_pred_lbls = []
        all_pred_probs = []

        for _ in np.arange(val_set_size):
            # New in pytorch 0.4.0, use local context manager to turn off history tracking
            with torch.set_grad_enabled(False):
                # batch_size=None and do_balance=True => number of degenerate slices/per patient determines
                #                                        the batch_size.
                x_input, y_labels, _ = val_batch(batch_size=None, backward_freq=1, do_balance=True)
                val_loss, pred_probs = model.do_forward_pass(x_input, y_labels)
            pred_labels = np.argmax(pred_probs.data.cpu().numpy(), axis=1)
            np_pred_probs = pred_probs.data.cpu().numpy()
            f1, roc_auc, pr_auc, acc, prec, rec = compute_eval_metrics(y_labels.data.cpu().numpy(), pred_labels,
                                                                       np_pred_probs[:, 1])

            if f1 != -1:
                arr_val_loss.append([val_loss.item()])
                arr_val_eval.append([np.array([f1, roc_auc, pr_auc, prec, rec])])
                all_labels.append(y_labels.data.cpu().numpy())
                all_pred_lbls.append(pred_labels)
                all_pred_probs.append(np_pred_probs)
            else:
                self.logger.info("***WARNING*** - OMITTING validation example due to no TP")
            str_slice_ids = ", ".join([str(i) for i in val_batch.current_slice_ids])
            if verbose:
                self.logger.info("Patient {}: current non-deg slice ids {}".format(val_batch.current_patient_id,
                                                                                   str_slice_ids))
            # self.logger.info("GT labels")
            # self.logger.info(y_labels.data.cpu().numpy())
            # self.logger.info("Predicted labels")
            # self.logger.info(pred_labels)
            # self.logger.info(np_pred_probs[:, 1])
            # self.logger.info("VALIDATION - patient {} - f1={:.3f} - roc_auc={:.3f} "
            #                 "- pr_auc={:.3f} - prec={:.3f} - rec={:.3f}".format(val_batch.current_patient_id,
            #                                                                     f1, roc_auc, pr_auc, prec, rec))
        arr_val_loss = np.concatenate(arr_val_loss)
        arr_val_eval = np.concatenate(arr_val_eval)
        all_labels = np.concatenate(all_labels)
        all_pred_lbls = np.concatenate(all_pred_lbls)
        all_pred_probs = np.concatenate(all_pred_probs)
        total_num_of_slices = all_pred_probs.shape[0]
        f1, roc_auc, pr_auc, acc, prec, rec = compute_eval_metrics(all_labels, all_pred_lbls, all_pred_probs[:, 1])
        if arr_val_loss.shape[0] > 1:
            val_loss = np.mean(arr_val_loss)
        else:
            val_loss = arr_val_loss[0]

        self.exper.val_stats["loss"][self.num_val_runs - 1] = val_loss
        self.exper.val_stats["f1"][self.num_val_runs - 1] = f1
        self.exper.val_stats["roc_auc"][self.num_val_runs - 1] = roc_auc
        self.exper.val_stats["pr_auc"][self.num_val_runs - 1] = pr_auc
        self.exper.val_stats["prec"][self.num_val_runs - 1] = prec
        self.exper.val_stats["rec"][self.num_val_runs - 1] = rec
        duration = time.time() - start_time

        self.logger.info("---> END VALIDATION epoch {} #slices={}: f1={:.3f} - roc_auc={:.3f} "
                         "- pr_auc={:.3f} - prec={:.3f} - rec={:.3f} "
                         "- {:.2f} seconds".format(self.exper.epoch_id, total_num_of_slices,
                                                                             f1, roc_auc, pr_auc, prec, rec, duration))
        del val_batch

    def test(self, data_set=None, model=None, test_id=None, verbose=False):
        if model is None:
            # get model. 1st arg=experiment label "20180824_13_06_44_sdvgg11_bn_f1p01_brier_umap_6KE_lr1e05"
            #            2nd arg=last epoch id aka checkpoint
            model = self.load_checkpoint(self.exper.run_args.log_dir, checkpoint=self.exper.epoch_id)
        if data_set is None:
            data_set = self.load_dataset()
        if test_id is None:
            test_id = self.next_test_id()
        start_time = time.time()
        test_set_size = len(data_set.get_patient_ids(is_train=False))
        # create batch object
        test_batch = BatchHandlerSD(data_set=data_set, is_train=False, cuda=self.exper.run_args.cuda)
        self.info("INFO - Begin test run {}".format(test_id))
        arr_test_loss = []
        all_labels = []
        all_pred_lbls = []
        all_pred_probs = []
        test_stats = {}

        for _ in np.arange(test_set_size):
            # New in pytorch 0.4.0, use local context manager to turn off history tracking
            with torch.set_grad_enabled(False):
                # batch_size=None and do_balance=True => number of degenerate slices/per patient determines
                #                                        the batch_size.
                x_input, y_labels, _ = test_batch(batch_size=None, backward_freq=1, do_balance=True)
                test_loss, pred_probs = model.do_forward_pass(x_input, y_labels)
            pred_labels = np.argmax(pred_probs.data.cpu().numpy(), axis=1)
            np_pred_probs = pred_probs.data.cpu().numpy()
            f1, roc_auc, pr_auc, acc, prec, rec = compute_eval_metrics(y_labels.data.cpu().numpy(), pred_labels,
                                                                       np_pred_probs[:, 1])

            if f1 != -1:
                arr_test_loss.append([test_loss.item()])
                all_labels.append(y_labels.data.cpu().numpy())
                all_pred_lbls.append(pred_labels)
                all_pred_probs.append(np_pred_probs)
            else:
                self.info("***WARNING*** - OMITTING validation example due to no TP")
            str_slice_ids = ", ".join([str(i) for i in test_batch.current_slice_ids])
            if verbose:
                self.info("Patient {}: current non-deg slice ids {}".format(test_batch.current_patient_id,
                                                                                   str_slice_ids))

        arr_test_loss = np.concatenate(arr_test_loss)
        all_labels = np.concatenate(all_labels)
        all_pred_lbls = np.concatenate(all_pred_lbls)
        all_pred_probs = np.concatenate(all_pred_probs)
        total_num_of_slices = all_pred_probs.shape[0]
        precision, recall, thresholds = precision_recall_curve(all_labels, all_pred_probs[:, 1])
        fpr, tpr, thresholds = roc_curve(all_labels, all_pred_probs[:, 1])

        f1, roc_auc, pr_auc, acc, prec, rec = compute_eval_metrics(all_labels, all_pred_lbls, all_pred_probs[:, 1])
        if arr_test_loss.shape[0] > 1:
            val_loss = np.mean(arr_test_loss)
        else:
            val_loss = arr_test_loss[0]

        test_stats["loss"] = val_loss
        test_stats["f1"] = f1
        test_stats["roc_auc"] = roc_auc
        test_stats["pr_auc"] = pr_auc
        test_stats["prec"] = prec
        test_stats["rec"] = rec
        test_stats["pr_curve"] = tuple((precision, recall, thresholds))
        test_stats["roc_curve"] = tuple((fpr, tpr, thresholds))
        self.test_stats[test_id] = test_stats
        duration = time.time() - start_time
        self.info("INFO - End test run {} #slices={}: f1={:.3f} - roc_auc={:.3f} "
                         "- pr_auc={:.3f} - prec={:.3f} - rec={:.3f} "
                         "- {:.2f} seconds".format(test_id, total_num_of_slices,
                                                   f1, roc_auc, pr_auc, prec, rec, duration))
        del test_batch
        del data_set
        del model

    def load_dataset(self):
        if self.seg_exper_ensemble is None:
            raise ValueError("ERROR - parameter data_set is None and I don't have a reference to the"
                             " segmentation handler ensemble. Hence, can't load the dataset.")
        data_set = create_dataset(self.exper.run_args.fold_id, self.seg_exper_ensemble,
                                  type_of_map=self.exper.run_args.type_of_map,
                                  degenerate_type="mean", pos_label=1, logger=self.logger)
        return data_set

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
        else:
            chkpnt_dir = os.path.join(self.exper.config.root_dir, self.exper.chkpnt_dir)

        if checkpoint is None:
            checkpoint = self.exper.epoch_id

        str_classname = config.base_class
        checkpoint_file = str_classname + "checkpoint" + str(checkpoint).zfill(5) + ".pth.tar"
        model = load_slice_detector_model(self)
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
                self.info("Loading existing model with checkpoint {} from dir {}".format(checkpoint, chkpnt_dir))
        else:
            raise IOError("Path to checkpoint not found {}".format(abs_checkpoint_dir))

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

        # self.exper = ExperimentHandler.check_compatibility(experiment)
        self.exper = experiment
        self.model_name = "{}-{}-f{} ".format(self.exper.run_args.model,
                                                    self.exper.run_args.type_of_map,
                                                    self.exper.run_args.fold_id)
        if use_logfile:
            self.logger = create_logger(self.exper, file_handler=use_logfile)
        else:
            self.logger = None


class ExperHandlerEnsemble(object):

    def __init__(self, exper_dict):
        self.seg_exper_handlers = {}
        self.exper_dict = exper_dict
        self.patient_fold = {}
        # we will load the corresponding object from the ReferralResults because they also contain the
        # non-referral dice scores per slice per patient. We use a "dummy" referral_threshold of 0.001
        # but any other available will also do the job. Object is assigned in load_dice_without_referral method
        # below.
        self.dice_score_slices = None
        for exper_id in exper_dict.values():
            exp_handler = create_experiment(exper_id)
            exp_handler.get_testset_ids()
            fold_id = int(exp_handler.exper.run_args.fold_ids[0])
            self.seg_exper_handlers[fold_id] = exp_handler
            for patient_id in exp_handler.test_set_ids.keys():
                self.patient_fold[patient_id] = fold_id

    def get_patient_fold_id(self, patient_id):
        return self.patient_fold[patient_id]

    def load_dice_without_referral(self, type_of_map="u_map", referral_threshold=0.001):
        """
        :param type_of_map:
        :param referral_threshold:
        :return:
        """
        if type_of_map == "e_map":
            use_entropy_maps = True
        else:
            use_entropy_maps = False

        ref_result_obj = ReferralResults(self.exper_dict, [referral_threshold], print_results=False,
                                         fold=None, slice_filter_type=None, use_entropy_maps=use_entropy_maps)
        self.dice_score_slices = ref_result_obj.org_dice_slices[referral_threshold]
