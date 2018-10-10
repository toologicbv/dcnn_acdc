import torch.nn as nn
import torch
import numpy as np

OPTIMIZER_DICT = {'sgd': torch.optim.SGD,  # Gradient Descent
                  'adadelta': torch.optim.Adadelta,  # Adadelta
                  'adagrad': torch.optim.Adagrad,  # Adagrad
                  'adam': torch.optim.Adam,  # Adam
                  'sparse_adam': torch.optim.SparseAdam, # SparseAdam
                  'rmsprop': torch.optim.RMSprop  # RMSprop
                  }


class BaseConfig(object):

    def __init__(self):

        # default data directory
        # remember to ADD env variable REPO_PATH on machine. REPO_PATH=<absolute path to repository >
        self.dt_map_dir = "dt_maps"
        # ES: 1 = RV; 2 = MYO; 3 = LV; tuple(non-apex-basal inter-observ-var, apex-basal inter-observ-var)
        # ED: 5 = RV; 6 = MYO; 7 = LV
        self.acdc_inter_observ_var = {1: [14.05, 9.05], 2: [7.8, 5.8], 3: [8.3, 5.65],  # ES
                                      5: [12.35, 8.15], 6: [6.95, 5.25], 7: [5.9, 4.65]}  # ED
        self.acdc_background_classes = [0, 4]
        # TODO We don't know size of padding yet. Depends on model architecture!
        self.acdc_pad_size = 0
        self.quick_run_size = 10

        # Settings for determining the target areas/pixels to be inspected. used in procedure
        # find_multiple_connected_rois (file box_utils.py)
        self.min_roi_area = 2  # minimum number of 2D 4-connected component. Otherwise target pixels are discarded
        # if the number of 2D 4-connected components in a structure is smaller than this amount than we fill the
        # total ROI area (make it rectangular). We do this in order to prevent target structures to be too small
        self.max_roi_area_for_fill = 10

        # patch size during training
        self.patch_size = np.array([72, 72])
        self.optimizer = "adam"
        self.detector_cfg = {
            'base': [16, 'M', 32, 'M', 32, 'M', 64],
            'num_of_input_channels': 2,
            'num_of_classes': 2,
            'use_batch_norm': True,
            'classification_loss': nn.NLLLoss2d,
            'optimizer': "adam",
            'weight_decay': 0.001
        }
        self.num_of_max_pool = len([i for i, s in enumerate(self.detector_cfg["base"]) if s == 'M'])
        # assuming patch size is quadratic
        self.output_stride = int(self.patch_size[0] / 2**self.num_of_max_pool)


config_detector = BaseConfig()
