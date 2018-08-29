import os
import socket


class BaseConfig(object):

    def __init__(self):

        # default data directory
        self.architecture = None
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
        # remember to ADD env variable REPO_PATH on machine. REPO_PATH=<absolute path to repository >
        self.root_dir = self.get_rootpath()
        self.data_dir = os.path.join(self.root_dir, "data/Folds/")
        self.log_root_path = "logs"
        self.figure_path = "figures"
        self.stats_path = "stats"
        self.u_map_dir = "u_maps"
        self.pred_lbl_dir = "pred_lbls"
        self.checkpoint_path = "checkpoints"
        self.logger_filename = "run_out.log"
        # standard image name
        self.dflt_image_name = "*patient*"
        self.dflt_label_name = "*label*"
        # used as filename for the logger
        self.logger_filename = "sdvgg_run.log"
        # directory for models
        self.model_path = "models"
        self.numpy_save_filename = "aug_"

        # optimizer
        self.optimizer = "adam"
        self.cycle_length = 100

        # normalization method "normalize or rescaling"
        self.norm_method = "normalize"

        # base-class name
        self.base_class = "DegenerateSliceDetector"

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

        self.class_lbl_background = 0
        self.class_lbl_myocardium = 1
        self.class_lbl_bloodpool = 2

        # dice score threshold to identify "degenerate" slices (bad segmentations)
        self.dice_threshold = 0.7
        # noise threshold
        self.noise_threshold = 0.01

        # validation settings, of running on GPU with low RAM we choose a smaller val set size
        # Number of iterations over the validation/test set. Because we balance the test/val set between classes
        # and we've much more negatives than TP, we sample non-degenerate slices for each test-batch (equal to
        # num-of degenerate slices for that patient). Hence we want each non-degenerate slice once in the test batch.
        self.val_iterations = 10
        if socket.gethostname() == "toologic-ubuntu2":
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

    def get_architecture(self, base_model="vgg11_bn"):

        if base_model == "sdvgg11_bn":
            self.architecture = {"base_model": "vgg11_bn"}
        elif base_model == "sdvgg11":
            self.architecture = {"base_model": "vgg11"}
        elif base_model == "sdvgg16_bn":
            self.architecture = {"base_model": "vgg16_bn"}
        elif base_model == "sdvgg16":
            self.architecture = {"base_model": "vgg16"}

        else:
            raise ValueError("ERROR - {} is an unknown base model.".format(base_model))

        self.architecture["spp_pyramid"] = [4, 2, 1]
        self.architecture["num_of_classes"] = 2
        self.architecture["drop_percentage"] = 0.5
        self.architecture["weight_decay"] = 5.
        self.architecture["optimizer"] = "adam"
        # None disables additional loss of fp_soft + fn_soft.
        # Should increase precision by penalizing false positives.
        self.architecture["fp_penalty_weight"] = 7.
        # number of batches before performing a pytorch backward operation
        self.architecture["backward_freq"] = 10


config = BaseConfig()
