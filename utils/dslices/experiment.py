import numpy as np
import os

from datetime import datetime
from pytz import timezone


class Experiment(object):

    def __init__(self, config, seg_exper, run_args=None):

        # logging
        self.epoch_id = 0
        self.fold_id = None
        self.chkpnt_dir = None
        self.logger = None
        # will be set to a relative path w.r.t. root directory of python project
        self.output_dir = None
        self.optimizer = None
        # set this later
        self.batches_per_epoch = 0
        self.run_args = run_args
        self.exper_label = None
        self.fold_id = None
        self.config = config
        self.model_name = ""
        self.epoch_stats = None
        self.val_stats = None
        self.test_stats = None
        self.stats_path = None
        self.num_val_runs = 0
        self.val_run_id = -1
        self.init_statistics()
        self.test_results_per_epoch = None
        self.exper_label = self.create_exper_label(seg_exper)
        self.fold_id = seg_exper.run_args.fold_ids[0]
        self._set_path()

    def init_statistics(self):
        if self.run_args.val_freq != 0:
            self.num_val_runs = (self.run_args.epochs // self.run_args.val_freq)

            if self.run_args.epochs % self.run_args.val_freq == 0:
                pass
            else:
                # one extra run because max epoch is not divided by val_freq
                self.num_val_runs += 1
        self.epoch_stats = {'lr': np.zeros(self.run_args.epochs),
                            'loss': np.zeros(self.run_args.epochs),
                            'f1': np.zeros(self.num_val_runs),
                            'roc_auc': np.zeros(self.num_val_runs),
                            'pr_auc': np.zeros(self.num_val_runs),
                            'prec': np.zeros(self.num_val_runs),
                            'rec': np.zeros(self.num_val_runs)
                            }
        self.val_stats = {'epoch_ids': np.zeros(self.num_val_runs),
                          'loss': np.zeros(self.num_val_runs),
                          'f1': np.zeros(self.num_val_runs),
                          'roc_auc': np.zeros(self.num_val_runs),
                          'pr_auc': np.zeros(self.num_val_runs),
                          'prec': np.zeros(self.num_val_runs),
                          'rec': np.zeros(self.num_val_runs)}
        self.test_stats = {'test_id': None,
                           'loss': None,
                           'f1': None,
                           'roc_auc': None,
                           'pr_auc': None,
                           'prec': None,
                           'rec': None}

    def get_loss(self, validation=False):

        if not validation:
            return self.epoch_stats["loss"]
        else:
            return self.val_stats["loss"]

    @property
    def validation_epoch_ids(self):
        return self.val_stats['epoch_ids']

    def _set_path(self):

        if self.run_args.log_dir is None:
            self.run_args.log_dir = str.replace(datetime.now(timezone('Europe/Berlin')).strftime(
                                    '%Y-%m-%d %H:%M:%S.%f')[:-7], ' ', '_') + "_" + self.exper_label + \
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

    def create_exper_label(self, seg_exper):
        if seg_exper.run_args.loss_function == "softdice":
            loss_func_name = "sdice"
        elif seg_exper.run_args.loss_function == "brier":
            loss_func_name = "brier"
        elif seg_exper.run_args.loss_function == "cross-entropy":
            loss_func_name = "entrpy"
        else:
            raise ValueError("ERROR - {} as loss functional is not supported!".format(seg_exper.run_args.loss_function))

        # prepare type of map indication, indicate whether the maps is used in prediction or only to indicate
        # the degenerate slices based on their dice score (we need the test predictions for that from the former
        # experiments
        map_name = "no" + self.run_args.type_of_map.replace("_", "") if self.run_args.use_no_map else \
            self.run_args.type_of_map.replace("_", "")
        prob = "p" + str(seg_exper.run_args.drop_prob).replace(".", "")
        prob += "_" + loss_func_name + "_" + map_name
        exper_label = self.run_args.model + "_f" + str(seg_exper.run_args.fold_ids[0]) + \
                      prob + "_" + str(self.run_args.epochs / 1000) + "KE"

        return exper_label
