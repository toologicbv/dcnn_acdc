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
        # ES: 1 = RV; 2 = MYO; 3 = LV; tuple(apex-base inter-observer-var, NON-apex-base inter-observer-var)
        # ED: 5 = RV; 6 = MYO; 7 = LV
        self.acdc_inter_observ_var = {1: [14.05, 9.05], 2: [7.8, 5.8], 3: [8.3, 5.65],  # ES
                                      5: [12.35, 8.15], 6: [6.95, 5.25], 7: [5.9, 4.65]}  # ED
        # range of area (target tissue structure per slice) from 5 to 95 percentile (we can compute this by means of
        # TestHandler.generate_bbox_target_roi method).
        self.label_area_num_bins = 10
        self.range_label_area = [500, 5500]
        # used in np.digitize(array, bins) to determine
        self.label_area_bins = np.array([0, 550, 1100, 1650, 2200, 2750, 3300, 3850, 4400, 4950])
        self.acdc_background_classes = [0, 4]
        # IMPORTANT: we don't use PADDING because during batch training and testing we crop out a region that fits
        # the max-pool operations & convolutions. Due toe fully-conv NN we can process different sizes of images.
        self.acdc_pad_size = 0
        self.quick_run_size = 10

        # Settings for determining the target areas/pixels to be inspected. used in procedure
        # find_multiple_connected_rois (file box_utils.py)
        self.min_roi_area = 2  # minimum number of 2D 4-connected component. Otherwise target pixels are discarded
        self.fraction_negatives = 0.67
        # patch size during training
        self.patch_size = np.array([80, 80])
        # the spacing of grids after the last maxPool layer.
        self.max_grid_spacing = 8
        self.architecture = None
        self.num_of_max_pool = None
        self.output_stride = None
        self.detector_cfg = {"rd1":
                                 {'model_id': "rd1",
                                  # 'base': [16, 'M', 32, 'M', 32, 'M', 64],
                                  'base': [16, 'M', 16, 'M', 32, 'M', 32],
                                  'num_of_input_channels': 3,
                                  'num_of_classes': 2,
                                  'use_batch_norm': True,
                                  'classification_loss': nn.NLLLoss(),
                                  'optimizer': "adam",
                                  'weight_decay': 0.001,
                                  'drop_prob': 0.5,
                                  "backward_freq": 1,
                                  "use_extra_classifier": False,
                                  "use_fn_loss": True,
                                  "fn_penalty_weight": 0.1,
                                  "fp_penalty_weight": 0.5,
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
                                  "use_fn_loss": True,
                                  "fn_penalty_weight": 0.1,
                                  "fp_penalty_weight": 0.15,
                                  "description": "rd2-detector"},
                             "rd3":
                                 {'model_id': "rd3",
                                  # 'base': [16, 'M', 32, 'M', 32, 'M', 64, 'M', 64],  # first version
                                  'base': [16, 'M', 16, 'M', 32, 'M', 32, 'M', 32],  # 2nd version
                                  'num_of_input_channels': 3,
                                  'num_of_classes': 2,
                                  'drop_prob': 0.5,
                                  'use_batch_norm': True,
                                  'classification_loss': nn.NLLLoss(),
                                  'optimizer': "adam",
                                  'weight_decay': 0.001,
                                  "backward_freq": 1,
                                  "use_extra_classifier": False,
                                  "use_fn_loss": True,
                                  "fn_penalty_weight": 0.1,
                                  "fp_penalty_weight": 0.5,
                                  "description": "rd3-detector"},
                             "rd3L":
                                 {'model_id': "rd3",
                                  'base': [16, 'M', 32, 'M', 32, 'M', 64, 'M', 64],  # first version
                                  # 'base': [16, 'M', 32, 'M', 64, 'M', 128, 'M', 256],
                                  'num_of_input_channels': 3,
                                  'num_of_classes': 2,
                                  'drop_prob': 0.5,
                                  'use_batch_norm': True,
                                  'classification_loss': nn.NLLLoss(),
                                  'optimizer': "adam",
                                  'weight_decay': 0.001,
                                  "backward_freq": 1,
                                  "use_extra_classifier": False,
                                  "use_fn_loss": True,
                                  "fn_penalty_weight": 0.35,
                                  "fp_penalty_weight": 0.15,
                                  "description": "rd3-detector"}
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

        # plotting
        self.title_font_large = {'fontname': 'Monospace', 'size': '36', 'color': 'black', 'weight': 'normal'}
        self.title_font_medium = {'fontname': 'Monospace', 'size': '20', 'color': 'black', 'weight': 'normal'}
        self.title_font_small = {'fontname': 'Monospace', 'size': '16', 'color': 'black', 'weight': 'normal'}
        self.axis_font = {'fontname': 'Monospace', 'size': '16', 'color': 'black', 'weight': 'normal'}
        self.axis_font18 = {'fontname': 'Monospace', 'size': '18', 'color': 'black', 'weight': 'normal'}
        self.axis_font20 = {'fontname': 'Monospace', 'size': '20', 'color': 'black', 'weight': 'normal'}
        self.axis_font22 = {'fontname': 'Monospace', 'size': '22', 'color': 'black', 'weight': 'normal'}
        self.axis_font24 = {'fontname': 'Monospace', 'size': '24', 'color': 'black', 'weight': 'normal'}

    def get_architecture(self, model_name):
        if model_name[:3] == "rd1":
            self.architecture = self.detector_cfg[model_name]
            self.num_of_max_pool = len([i for i, s in enumerate(self.architecture["base"]) if s == 'M'])
            # assuming patch size is quadratic
            self.output_stride = int(self.patch_size[0] / 2 ** self.num_of_max_pool)
            self.max_grid_spacing = 8
        elif model_name[:3] == "rd2":
            self.architecture = self.detector_cfg[model_name]
            # TODO: in the beginning added all maxPool2D layers to 'base' configuration but then
            # started to define different network paths in the RegionDetector object. Hence, started to add
            # maxPool layers in addition here (+1)
            self.num_of_max_pool = len([i for i, s in enumerate(self.architecture["base"]) if s == 'M']) + 1
            # assuming patch size is quadratic
            self.output_stride = int(self.patch_size[0] / 2 ** self.num_of_max_pool)
            self.max_grid_spacing = 8
        elif model_name[:3] == "rd3":
            self.architecture = self.detector_cfg[model_name]
            self.num_of_max_pool = len([i for i, s in enumerate(self.architecture["base"]) if s == 'M'])
            # assuming patch size is quadratic
            self.output_stride = int(self.patch_size[0] / 2 ** self.num_of_max_pool)
            # patch size during training, we need a slightly bigger patch size 80 instead of 72
            self.patch_size = np.array([80, 80])
            self.max_grid_spacing = self.num_of_max_pool ** 2
        else:
            raise NotImplementedError("ERROR - {} is not a valid model name".format(model_name))

    @staticmethod
    def get_rootpath():
        return os.environ.get("REPO_PATH", os.environ.get('HOME'))


config_detector = BaseConfig()
