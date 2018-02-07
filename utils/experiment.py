import sys
import torch
from torch.autograd import Variable
import numpy as np
import os
from datetime import datetime
from pytz import timezone
import dill
from common.parsing import create_def_argparser, run_dict

from common.common import create_logger, create_exper_label
from utils.config import config
from utils.batch_handlers import TwoDimBatchHandler, TestHandler
from in_out.load_data import HVSMR2016CardiacMRI
import models.dilated_cnn


class ExperimentHandler(object):

    def __init__(self, exper, logger=None):

        self.exper = exper
        self.logger = None
        if logger is None:
            self.logger = create_logger(self.exper, file_handler=False)
        else:
            self.logger = logger

    def set_root_dir(self, root_dir):
        self.exper.config.root_dir = root_dir

    def next_epoch(self):
        self.exper.epoch_id += 1

    def save_experiment(self, file_name=None):

        if file_name is None:
            file_name = "exper_stats@{}".format(self.exper.epoch_id) + ".dll"

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

    def load_checkpoint(self, root_dir=None, checkpoint=None):

        if root_dir is None:
            root_dir = self.exper.config.root_dir

        str_classname = "BaseDilated2DCNN"
        checkpoint_file = str_classname + "checkpoint" + str(checkpoint).zfill(5) + ".pth.tar"
        act_class = getattr(models.dilated_cnn, str_classname)
        model = act_class(use_cuda=self.exper.run_args.cuda)
        abs_checkpoint_dir = os.path.join(root_dir,
                                          os.path.join( self.exper.chkpnt_dir, checkpoint_file))
        if os.path.exists(abs_checkpoint_dir):
            checkpoint = torch.load(abs_checkpoint_dir)
            model.load_state_dict(checkpoint["state_dict"])
            if self.exper.run_args.cuda:
                model.cuda()

            self.logger.info("INFO - loaded existing model from checkpoint {}".format(abs_checkpoint_dir))
        else:
            raise IOError("Path to checkpoint not found {}".format(abs_checkpoint_dir))

        return model

    def set_config_object(self, new_config):
        self.exper.config = new_config

    def eval(self, images, labels, model, batch_size=None, val_run_id=None):

        # validate model
        val_batch = TwoDimBatchHandler(self.exper, batch_size=batch_size)
        val_batch.generate_batch_2d(images, labels)
        model.eval()
        b_predictions = model(val_batch.b_images)
        val_loss = model.get_loss(b_predictions, val_batch.b_labels)
        val_loss = val_loss.data.cpu().squeeze().numpy()[0]
        # compute dice score for both classes (myocardium and bloodpool)
        dice = HVSMR2016CardiacMRI.compute_accuracy(b_predictions, val_batch.b_labels)
        # store epochID and validation loss
        if val_run_id is not None:
            self.exper.val_stats["mean_loss"][val_run_id - 1] = np.array([self.exper.epoch_id, val_loss])
            self.exper.val_stats["dice_coeff"][val_run_id - 1] = np.array([self.exper.epoch_id, dice[0], dice[1]])
        self.logger.info("Model validation in epoch {}: current loss {:.3f}\t "
                         "dice-coeff(myo/blood) {:.3f}/{:.3f}".format(self.exper.epoch_id, val_loss,
                                                                      dice[0], dice[1]))
        model.train()
        del val_batch

    def test_full_image(self, model, images, labels=None, batch_size=None, views=None,
                        spacing=None, gen_overlays=True):
        if views is None:
            views = ['axial', 'coronal', 'saggital']

        model.eval()
        test_hdl = TestHandler(images, labels=labels, use_cuda=self.exper.run_args.cuda, batch_size=batch_size,
                               spacing=spacing)
        for view in views:
            test_hdl.generate_3D_batches(s_axis=view)
            test_hdl(model, self)

        if gen_overlays:
            test_hdl.generate_overlays(self)

        model.train()

    @staticmethod
    def load_experiment(path_to_exp, full_path=False):

        if not full_path:
            path_to_exp = os.path.join(config.root_dir, os.path.join(config.log_root_path , path_to_exp))

        print("Load from {}".format(path_to_exp))
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

    def __init__(self, config, run_args=None, set_seed=True):

        # logging
        self.epoch_id = 0
        self.chkpnt_dir = None
        self.logger = None
        self.output_dir = None
        self.optimizer = None
        # set this later
        self.batches_per_epoch = 0
        if run_args is None:
            self.run_args = create_def_argparser(**run_dict)
        else:
            self.run_args = run_args
        self.config = config
        self.epoch_stats = None
        self.val_stats = None
        self.stats_path = None
        self._set_path()
        self.num_val_runs = 0
        self.init_statistics()

        if set_seed:
            SEED = 2345
            torch.manual_seed(SEED)
            if run_args.cuda:
                torch.cuda.manual_seed(SEED)
            np.random.seed(SEED)

    def init_statistics(self):
        self.num_val_runs = (self.run_args.epochs // self.run_args.val_freq)
        if self.run_args.epochs % self.run_args.val_freq == 0:
            pass
        else:
            # one extra run because max epoch is not divided by val_freq
            self.num_val_runs += 1
        self.epoch_stats = {'mean_loss': np.zeros(self.run_args.epochs)}
        self.val_stats = {'mean_loss': np.zeros((self.num_val_runs, 2)),
                          'dice_coeff': np.zeros((self.num_val_runs, 3))}

    def start(self, exper_logger=None):

        pass

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

    def set_new_config(self, new_config):
        self.config = new_config

    def copy_from_object(self, obj):

        for key, value in obj.__dict__.iteritems():
            self.__dict__[key] = value



