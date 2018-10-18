import os
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
        self.root_dir = BaseConfig.get_rootpath()
        self.data_dir = os.path.join(self.root_dir, "data/Folds/")
        self.log_root_path = "logs/RD"
        self.figure_path = "figures"
        self.stats_path = "stats"
        self.dt_map_dir = "dt_maps"
        self.checkpoint_path = "checkpoints"
        self.logger_filename = "run_out.log"
        # ES: 1 = RV; 2 = MYO; 3 = LV; tuple(non-apex-basal inter-observ-var, apex-basal inter-observ-var)
        # ED: 5 = RV; 6 = MYO; 7 = LV
        self.acdc_inter_observ_var = {1: [14.05, 9.05], 2: [7.8, 5.8], 3: [8.3, 5.65],  # ES
                                      5: [12.35, 8.15], 6: [6.95, 5.25], 7: [5.9, 4.65]}  # ED
        self.acdc_background_classes = [0, 4]
        # TODO We don't know size of padding yet. Depends on model architecture!
        self.acdc_pad_size = 0
        self.quick_run_size = 6

        # Settings for determining the target areas/pixels to be inspected. used in procedure
        # find_multiple_connected_rois (file box_utils.py)
        self.min_roi_area = 2  # minimum number of 2D 4-connected component. Otherwise target pixels are discarded
        # if the number of 2D 4-connected components in a structure is smaller than this amount than we fill the
        # total ROI area (make it rectangular). We do this in order to prevent target structures to be too small
        self.max_roi_area_for_fill = 10

        # patch size during training
        self.patch_size = np.array([72, 72])
        # the spacing of grids after the last maxPool layer.
        self.max_grid_spacing = 8
        self.architecture = None
        self.num_of_max_pool = None
        self.output_stride = None
        self.detector_cfg = {"rd1":
                                 {'model_id': "rd1",
                                  'base': [16, 'M', 32, 'M', 32, 'M', 64],
                                  'num_of_input_channels': 3,
                                  'num_of_classes': 2,
                                  'use_batch_norm': True,
                                  'classification_loss': nn.NLLLoss(),
                                  'optimizer': "adam",
                                  'weight_decay': 0.001,
                                  'drop_prob': 0.5,
                                  "backward_freq": 1,
                                  "use_extra_classifier": False,
                                  "description": "rd1-detector"},
                             "rd2":
                                 {'model_id': "rd2",
                                  'base': [16, 'M', 32, 'M'],
                                  'num_of_input_channels': 3,
                                  'num_of_classes': 2,
                                  'drop_prob': 0.5,
                                  'use_batch_norm': True,
                                  'classification_loss': nn.NLLLoss(),
                                  'optimizer': "adam",
                                  'weight_decay': 0.001,
                                  "backward_freq": 1,
                                  "use_extra_classifier": True,
                                  "description": "rd2-detector"}
        }
        self.base_class = "RegionDetector"
        # dictionary of experiment ids
        self.exper_dict_brier = {3: "20180426_14_14_57_dcnn_mc_f3p01_brier_150KE_lr2e02",
                                 2: "20180426_14_14_39_dcnn_mc_f2p01_brier_150KE_lr2e02",
                                 1: "20180426_14_13_46_dcnn_mc_f1p01_brier_150KE_lr2e02",
                                 0: "20180418_15_02_05_dcnn_mcv1_150000E_lr2e02"}

        self.exper_dict_softdice = {3: "20180630_10_26_32_dcnn_mc_f3p01_150KE_lr2e02",
                                    2: "20180630_10_27_07_dcnn_mc_f2p01_150KE_lr2e02",
                                    1: "20180629_11_28_29_dcnn_mc_f1p01_150KE_lr2e02",
                                    0: "20180629_10_33_08_dcnn_mc_f0p01_150KE_lr2e02"}

        self.exper_dict_centropy = {3: "20180703_18_15_22_dcnn_mc_f3p01_entrpy_150KE_lr2e02",
                                    2: "20180703_18_11_10_dcnn_mc_f2p01_entrpy_150KE_lr2e02",
                                    1: "20180703_18_13_51_dcnn_mc_f1p01_entrpy_150KE_lr2e02",
                                    0: "20180703_18_09_33_dcnn_mc_f0p01_entrpy_150KE_lr2e02"}

    def get_architecture(self, model_name):
        if model_name == "rd1":
            self.architecture = self.detector_cfg[model_name]
            self.num_of_max_pool = len([i for i, s in enumerate(self.architecture["base"]) if s == 'M'])
            # assuming patch size is quadratic
            self.output_stride = int(self.patch_size[0] / 2 ** self.num_of_max_pool)
        elif model_name == "rd2":
            self.architecture = self.detector_cfg[model_name]
            # TODO: in the beginning added all maxPool2D layers to 'base' configuration but then
            # started to define different network paths in the RegionDetector object. Hence, started to add
            # maxPool layers in addition here (+1)
            self.num_of_max_pool = len([i for i, s in enumerate(self.architecture["base"]) if s == 'M']) + 1
            # assuming patch size is quadratic
            self.output_stride = int(self.patch_size[0] / 2 ** self.num_of_max_pool)
        else:
            raise NotImplementedError("ERROR - {} is not a valid model name".format(model_name))

    @staticmethod
    def get_rootpath():
        return os.environ.get("REPO_PATH", os.environ.get('HOME'))


config_detector = BaseConfig()
