import numpy as np
from utils.experiment import Experiment


class HVSMRExperiment(Experiment):

    def __init__(self, config, run_args=None):
        super(HVSMRExperiment, self).__init__(config, run_args)

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
                            # storing dice coefficients for LV, myocardium classes
                            'dice_coeff': np.zeros((self.run_args.epochs, 2))}
        self.val_stats = {'epoch_ids': np.zeros(self.num_val_runs),
                          'mean_loss': np.zeros(self.num_val_runs),
                          'dice_coeff': np.zeros((self.num_val_runs, 2))}