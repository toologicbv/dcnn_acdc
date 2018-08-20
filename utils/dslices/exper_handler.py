import sys
import dill
import os
import shutil
import torch
from collections import OrderedDict

from common.common import create_logger
import models.slice_detector
from common.dslices.config import config
from common.dslices.helper import create_experiment
from utils.referral_results import ReferralResults


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

        str_classname = config.base_class
        checkpoint_file = str_classname + "checkpoint" + str(checkpoint).zfill(5) + ".pth.tar"
        act_class = getattr(models.slice_detector, str_classname)

        model = act_class()
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
        if type_of_map == "entropy":
            use_entropy_maps = True
        else:
            use_entropy_maps = False

        ref_result_obj = ReferralResults(self.exper_dict, [referral_threshold], print_results=False,
                                         fold=None, slice_filter_type=None, use_entropy_maps=use_entropy_maps)
        self.dice_score_slices = ref_result_obj.org_dice_slices[referral_threshold]
