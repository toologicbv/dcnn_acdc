import os
import torch
import torch.nn as nn
import socket
from models.nn_functions import CReLU


DEFAULT_DCNN_2D = {'num_of_layers': 10,
                   'input_channels': 2,
                   'kernels': [3, 3, 3, 3, 3, 3, 3, 3, 1, 1],
                   'channels': [32, 32, 32, 32, 32, 32, 64, 128, 128, 4],  # NOTE: last channel is num_of_classes
                   'dilation': [1, 1, 2, 4, 8, 16, 32, 1, 1, 1],
                   'stride': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   'batch_norm': [False, True, True, True, True, True, True, True, True, False],
                   'non_linearity': [nn.ELU, nn.ELU, nn.ReLU, nn.ReLU, nn.ReLU, nn.ReLU, nn.ReLU, nn.ELU, nn.ELU,
                                     nn.Softmax],
                   'dropout': [0., 0., 0., 0., 0., 0., 0., 0.5, 0.5, 0.],
                   'loss_function': nn.NLLLoss,
                   'description': 'DEFAULT_DCNN_2D'
                   }

DEFAULT_DCNN_MC_2D = {'num_of_layers': 10,
                      'input_channels': 2,
                      'kernels': [3, 3, 3, 3, 3, 3, 3, 3, 1, 1],
                      'channels': [32, 32, 32, 32, 32, 32, 64, 128, 128, 4],  # NOTE: last channel is num_of_classes
                      'dilation': [1, 1, 2, 4, 8, 16, 32, 1, 1, 1],
                      'stride': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      'batch_norm': [False, True, True, True, True, True, True, True, True, False],
                      'non_linearity': [nn.ELU, nn.ELU, nn.ReLU, nn.ReLU, nn.ReLU, nn.ReLU, nn.ReLU, nn.ELU, nn.ELU,
                                        nn.Softmax],
                      'dropout': [0.1] * 10,
                      'loss_function': nn.NLLLoss,
                      'description': 'DEFAULT_DCNN_MC_2D'
                      }

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
        self.root_dir = self.get_rootpath()
        self.data_dir = os.path.join(self.root_dir, "data/Folds/")
        self.log_root_path = "logs/ACDC"
        self.figure_path = "figures"
        self.stats_path = "stats"
        self.u_map_dir = "u_maps"
        self.dt_map_dir = "dt_maps"
        self.pred_lbl_dir = "pred_lbls"
        self.troi_map_dir = "t_roi_maps"
        self.checkpoint_path = "checkpoints"
        self.logger_filename = "run_out.log"
        # standard image name
        self.dflt_image_name = "*patient*"
        self.dflt_label_name = "*label*"
        # used as filename for the logger
        self.logger_filename = "dcnn_run.log"
        # directory for models
        self.model_path = "models"
        self.numpy_save_filename = "aug_"
        self.bayes_umap_suffix = "_umaps.npz"

        # optimizer
        self.optimizer = "adam"
        self.cycle_length = 100

        # normalization method "normalize or rescaling"
        self.norm_method = "normalize"

        # padding to left and right of the image in order to reach the final image size for classification
        self.pad_size = 65

        # config values for uncertainty map blob filtering
        self.erosion_rank_structure = 3
        self.num_of_umap_blobs = 5
        self.min_size_blob_area = 10

        # disease categories, often used as keys in dictionaries for result evaluation
        self.disease_categories = {'NOR': "Normal",
                                   'DCM': "Dilated Cardiomyopathy",
                                   'MINF': "Systolic heart failure with infarction",
                                   'ARV': "Abnormal right ventricle",
                                   'HCM': "Hypertrophic Cardiomyopathy"}

        # class labels
        self.class_lbl_background = 0
        self.class_lbl_RV = 1
        self.class_lbl_myo = 2
        self.class_lbl_LV = 3

        # noise threshold
        self.noise_threshold = 0.01

        # validation settings, of running on GPU with low RAM we choose a smaller val set size
        if socket.gethostname() == "qiaubuntu" or socket.gethostname() == "toologic-ubuntu2":
            self.val_set_size = 128
            self.val_batch_size = 16
        else:
            self.val_set_size = 640
            self.val_batch_size = 128

        # plotting
        self.title_font_large = {'fontname': 'Monospace', 'size': '36', 'color': 'black', 'weight': 'normal'}
        self.title_font_medium = {'fontname': 'Monospace', 'size': '20', 'color': 'black', 'weight': 'normal'}
        self.title_font_small = {'fontname': 'Monospace', 'size': '16', 'color': 'black', 'weight': 'normal'}
        self.axis_font = {'fontname': 'Monospace', 'size': '16', 'color': 'black', 'weight': 'normal'}
        self.axis_font18 = {'fontname': 'Monospace', 'size': '18', 'color': 'black', 'weight': 'normal'}
        self.axis_font20 = {'fontname': 'Monospace', 'size': '20', 'color': 'black', 'weight': 'normal'}
        self.axis_font22 = {'fontname': 'Monospace', 'size': '22', 'color': 'black', 'weight': 'normal'}
        self.axis_font24 = {'fontname': 'Monospace', 'size': '24', 'color': 'black', 'weight': 'normal'}
        # self.axis_font28 = {'fontname': 'Monospace', 'size': '28', 'color': 'black', 'weight': 'normal'}
        self.axis_font28 = {'fontname': 'Monospace', 'size': '40', 'color': 'black', 'weight': 'normal'}
        self.axis_ticks_font_size = 12

        self.figure_ext = ".png"

    def copy_from_object(self, obj):

        for key, value in obj.__dict__.iteritems():
            self.__dict__[key] = value

    def get_rootpath(self):
        return os.environ.get("REPO_PATH", os.environ.get('HOME'))

    def datapath(self, dataset=None):
        return self.get_datapath(dataset)

    def get_datapath(self, dataset=None):
        if dataset is None:
            return os.environ.get("PYTHON_DATA_FOLDER", "data")
        env_variable = "PYTHON_DATA_FOLDER_%s" % dataset.upper()
        return os.environ.get(env_variable, "data")

    @staticmethod
    def get_architecture(model="dcnn", **kwargs):

        if kwargs is not None:
            if "drop_prob" not in kwargs:
                kwargs["drop_prob"] = 0.5
        else:
            kwargs["drop_prob"] = 0.5

        # print("WARNING - using drop-prob {:.3f}".format(kwargs["drop_prob"]))
        if model == "dcnn":
            architecture = {'num_of_layers': 10,
                            'input_channels': 2,
                            'kernels': [3, 3, 3, 3, 3, 3, 3, 3, 1, 1],
                            'channels': [32, 32, 32, 32, 32, 32, 64, 128, 128, 4],
                            # NOTE: last channel is num_of_classes
                            'dilation': [1, 1, 2, 4, 8, 16, 32, 1, 1, 1],
                            'stride': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            'batch_norm': [False, True, True, True, True, True, True, True, True, False],
                            'non_linearity': [nn.ELU, nn.ELU, nn.ReLU, nn.ReLU, nn.ReLU, nn.ReLU, nn.ReLU, nn.ELU,
                                              nn.ELU, nn.Softmax],
                            'dropout': [0., 0., 0., 0., 0., 0., 0., kwargs["drop_prob"], kwargs["drop_prob"], 0.],
                            'loss_function': nn.NLLLoss,
                            'description': 'DEFAULT_DCNN_2D'
                            }
        elif model == "dcnn_mc":
            num_of_layers = 10
            architecture = {'num_of_layers': num_of_layers,
                            'input_channels': 2,
                            'kernels': [3, 3, 3, 3, 3, 3, 3, 3, 1, 1],
                            'channels': [32, 32, 32, 32, 32, 32, 64, 128, 128, 4],
                            # NOTE: last channel is num_of_classes
                            'dilation': [1, 1, 2, 4, 8, 16, 32, 1, 1, 1],
                            'stride': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            'batch_norm': [False, True, True, True, True, True, True, True, True, False],
                            'non_linearity': [nn.ELU, nn.ELU, nn.ReLU, nn.ReLU, nn.ReLU, nn.ReLU, nn.ReLU, nn.ELU,
                                              nn.ELU, nn.Softmax],
                            # 'dropout': [kwargs["drop_prob"]] * num_of_layers,
                            'dropout': [kwargs["drop_prob"], kwargs["drop_prob"], kwargs["drop_prob"],
                                        kwargs["drop_prob"], kwargs["drop_prob"], kwargs["drop_prob"],
                                        kwargs["drop_prob"], kwargs["drop_prob"], kwargs["drop_prob"], 0.],
                            'loss_function': nn.NLLLoss2d,
                            'description': 'DCNN_2D_MC_DROPOUT_{}'.format(kwargs["drop_prob"])
                            }

        return architecture


config = BaseConfig()
