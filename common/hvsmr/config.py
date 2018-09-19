import os
import socket
import torch
import torch.nn as nn


OPTIMIZER_DICT = {'sgd': torch.optim.SGD,  # Gradient Descent
                  'adadelta': torch.optim.Adadelta,  # Adadelta
                  'adagrad': torch.optim.Adagrad,  # Adagrad
                  'adam': torch.optim.Adam,  # Adam
                  'sparse_adam': torch.optim.SparseAdam, # SparseAdam
                  'rmsprop': torch.optim.RMSprop  # RMSprop
                  }


class BaseConfig(object):

    def __init__(self):
        self.root_dir = self.get_rootpath()
        self.data_dir = os.path.join(self.root_dir, "data/HVSMR/Folds/")
        self.log_root_path = "logs/HVSMR"
        self.figure_path = "figures/HVSMR"
        self.stats_path = "stats/HVSMR"
        self.u_map_dir = "u_maps/HVSMR"
        self.pred_lbl_dir = "pred_lbls/HVSMR"
        self.checkpoint_path = "checkpoints/HVSMR"
        self.logger_filename = "run_out.log"
        # standard image name
        self.dflt_image_name = "*patient*"
        self.dflt_label_name = "*lbl*"
        # used as filename for the logger
        self.logger_filename = "dcnn_hvsmr_run.log"
        self.model_path = "models/HVSMR"
        self.numpy_save_filename = None
        # HVSMR voxelspacing after resampling
        self.voxelspacing = tuple((0.65, 0.65))

        # padding to left and right of the image in order to reach the final image size for classification
        self.pad_size = 65

        # class labels
        self.class_lbl_background = 0
        self.class_lbl_myocardium = 1
        self.class_lbl_bloodpool = 2

        # optimizer
        self.optimizer = "adam"
        self.cycle_length = 100

        # validation settings, of running on GPU with low RAM we choose a smaller val set size
        if socket.gethostname() == "toologic-ubuntu2":
            self.val_set_size = 128
            self.val_batch_size = 16
        else:
            self.val_set_size = 640
            self.val_batch_size = 128

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
        if model == "dcnn_hvsmr_mc":
            num_of_layers = 10
            architecture = {'num_of_layers': num_of_layers,
                            'input_channels': 1,
                            'kernels': [3, 3, 3, 3, 3, 3, 3, 3, 1, 1],
                            'channels': [32, 32, 32, 32, 32, 32, 32, 32, 192, 3],
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
                            'description': 'DCNN_HVSMR_2D_MC{}'.format(kwargs["drop_prob"])
                            }

        return architecture


config_hvsmr = BaseConfig()
