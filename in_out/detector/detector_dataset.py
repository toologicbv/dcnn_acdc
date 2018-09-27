import numpy as np
from collections import OrderedDict
from common.dslices.config import config
from tqdm import tqdm

from in_out.load_data import ACDC2017DataSet


class RegionDetectorDataSet(object):

    def __init__(self):
        # The key of the dictionaries is patient_id.
        self.train_images = OrderedDict()
        self.train_labels = OrderedDict()
        self.train_extra_labels = OrderedDict()
        # contain numpy array with shape [#slices, 3, w, h]
        self.test_images = OrderedDict()
        # contain numpy array with shape [#slices]
        self.test_labels = OrderedDict()
        # contain numpy array with shape [#slices, 4] 1) phase 2) slice id 3) mean-dice/slice 4) patient id
        self.test_extra_labels = OrderedDict()
        # actually "size" here is relate to number of patients
        self.size_train = 0
        self.size_test = 0
        self.train_patient_ids = None
        self.test_patient_ids = None

    def get(self, patient_id, is_train=True):
        """
        :param patient_id:
        :param is_train:
        :param do_merge_sets: if True, the two numpy arrays [#slices,...] are concatenated along dim0
        :return:
        """
        if is_train:
            image = self.train_images[patient_id]
            label = self.train_labels[patient_id]
            extra_label = self.train_extra_labels[patient_id]
        else:
            image = self.test_images[patient_id]
            label = self.test_labels[patient_id]
            extra_label = self.test_extra_labels[patient_id]

        return image, label, extra_label

    def get_patient_ids(self, is_train=True):
        if is_train:
            return self.train_patient_ids
        else:
            return self.test_patient_ids

    def get_size(self, is_train=True):
        if is_train:
            return self.size_train
        else:
            return self.size_test

    def load_from_file(self):
        raise NotImplementedError()


def create_dataset(exper_hdl, exper_ensemble, type_of_map="u_map", num_of_input_chnls=2,
                   acdc_dataset=None, logger=None, verbose=False, patient_ids=None):
    """

    :param exper_hdl: This is the experiment handler from the detector model
    :param exper_ensemble: This is an ensemble of DCNN segmentation handlers from the previous experiments on ACDC data
    :param type_of_map: u_map: Bayesian uncertainty map; e_map: Entropy maps;
    :param num_of_input_chnls: 2: automatic segmentation masks + uncertainties; 3: + original image
    :param acdc_dataset: If None, we create the ACDC set for the specific fold we're training (see below)
    :param logger:
    :param verbose:
    :param patient_ids: list of patient_ids that we use to restrict our experiment to a smaller size. used during
                        initial proof of concept.
    :return:
    """
    # only works for fold0 (testing purposes, small dataset), we can overwrite the quick_run argument
    # means we're loading a very small dataset (probably 2-3 images)

    if exper_hdl.exper.run_args.fold_id == 0 and exper_hdl.exper.run_args.quick_run:
        quick_run = True
    else:
        quick_run = exper_hdl.exper.run_args.quick_run
    # if acdc_dataset is None:
    fold_id = exper_hdl.exper.run_args.fold_id
    if acdc_dataset is None:
        seg_exper_hdl = exper_ensemble.seg_exper_handlers[fold_id]
        acdc_dataset = ACDC2017DataSet(seg_exper_hdl.exper.config,
                                       search_mask=seg_exper_hdl.exper.config.dflt_image_name + ".mhd",
                                       fold_ids=[fold_id], preprocess=False,
                                       debug=quick_run, do_augment=False,
                                       incomplete_only=False)
    if patient_ids is None:
        patient_ids = acdc_dataset.trans_dict.keys()
    else:
        if not isinstance(patient_ids, list):
            patient_ids = [patient_ids]
    exper_ensemble.prepare_handlers(type_of_map=type_of_map, force_reload=False)
    exper_handlers = exper_ensemble.seg_exper_handlers
    # instead of using class labels 0, 1, 2, 3 for the seg-masks we will use values between [0, 1]
    labels_float = [0., 0.3, 0.6, 0.9]
    # set negative label based on positive label
    dataset = RegionDetectorDataSet()
    # loop over training images
    for patient_id in tqdm(patient_ids):
        # found a degenerate slice
        stripped_patient_id = int(patient_id.replace("patient", ""))
        # IMPORTANT: val_image_names only contains the patient_ids of the test/val set
        #            whereas acdc.dataset.image_names contains all names (confusing I know)
        # HENCE, we check whether the patient ID is in the first list, if not it's a training image
        if patient_id not in acdc_dataset.val_image_names:
            # training set
            stats_idx = 0
            is_train = True
            image_num = acdc_dataset.image_names.index(patient_id)
            image = acdc_dataset.train_images[image_num]
            # print("Train --- patient id/image_num {}/{}".format(patient_id, image_num))
        else:
            stats_idx = 1
            # must be a test image
            is_train = False
            image_num = acdc_dataset.val_image_names.index(patient_id)
            image = acdc_dataset.val_images[image_num]
            # print("Test --- patient id/image_num {}/{}".format(patient_id, image_num))
        # merge the first dimension ES/ED [2, w, h, #slices] to [w, h, 2*#slices]
        image = np.concatenate((image[0], image[1]), axis=2)
        num_of_slices = image.shape[2]
        width = image.shape[0]
        height = image.shape[1]
        half_slices = num_of_slices / 2
        # image has shape [2, w, h, #slices] ES/ED
        # get fold_id, patient belongs to. we need the fold in order to retrieve the predicted seg-mask and the
        # u-map/e-map from the exper_ensemble. The results (dice) of the 75 training cases are distributed over the
        # other three folds, because they used them for test evaluation.
        pfold_id = exper_ensemble.get_patient_fold_id(patient_id)
        exper_handlers[pfold_id].get_pred_labels(patient_id=patient_id)
        # please NOTE that pred_labels has shape [8, w, h, #slices]
        pred_labels = exper_handlers[pfold_id].pred_labels[patient_id]
        # get the distance transform maps, that we'll use "somehow" for supervised training (voxel of region based)
        # regression or classification.
        dt_maps = exper_handlers[pfold_id].dt_maps[patient_id]
        # get Bayesian uncertainty map or entropy map, shape [2, w, h, #slices]
        if type_of_map == "u_map":
            u_maps = exper_handlers[pfold_id].referral_umaps[patient_id]
        elif type_of_map == "e_map":
            u_maps = exper_handlers[pfold_id].entropy_maps[patient_id]

        """
            We construct a numpy array as input to our model with the following shape:
                [#slices, 3channels, w, h]


        """
        # merge the first dimension ES/ED [2, w, h, #slices] to [w, h, 2*#slices] for uncertainty m aps
        u_maps = np.concatenate((u_maps[0], u_maps[1]), axis=2)

        img_3dim = np.zeros((non_deg_num_slices, num_of_input_chnls, width, height))
        img_3dim_deg = np.zeros((deg_num_slices, num_of_input_chnls, width, height))
        # do the same for the labels, but we only need one position per slice
        label_slices = np.zeros(non_deg_num_slices)
        # (1) phase (2) original sliceID (3) mean-dice (4) patient_id (5) apex/base indication
        # (6) continuous slice_id between [0, 1] for t-SNE visualization of apex/base continuum using colors
        extra_label_slices = np.zeros((non_deg_num_slices, 6))
        label_slices_deg = np.zeros(deg_num_slices)
        extra_label_slices_deg = np.zeros((deg_num_slices, 6))
        s_counter = 0
        s_counter_deg = 0
        # loop over slices
        for s in np.arange(num_of_slices):
            if s >= half_slices:
                # ED
                phase = 1
                cls_offset = pred_labels.shape[0] / 2
                phase_slice_counter = s - half_slices
            else:
                # ES
                phase = 0
                cls_offset = 0
                phase_slice_counter = s
            # merge 4 class labels into one dimension: use labels 0, 1, 2, 3
            # please NOTE that pred_labels has shape [8, w, h, #slices]
            p_lbl = np.zeros((pred_labels.shape[1], pred_labels.shape[2]))
            # loop over LV, MY and RV class (omit BG)
            for cls in np.arange(1, pred_labels.shape[0] / 2):
                p_lbl[pred_labels[cls + cls_offset, :, :, phase_slice_counter] == 1] = labels_float[cls]

            if type_of_map is not None:
                x = np.concatenate((np.expand_dims(image[:, :, s], axis=0),
                                    np.expand_dims(p_lbl, axis=0),
                                    np.expand_dims(u_maps[:, :, s], axis=0)), axis=0)
            else:
                # only concatenate original image and segmentation mask (predicted)
                x = np.concatenate((np.expand_dims(image[:, :, s], axis=0),
                                    np.expand_dims(p_lbl, axis=0)), axis=0)

            # if the mean dice score for this slice is equal or below threshold (0.7) then set label to 1
            if is_degenerate:
                pat_with_deg = True
                t_num_deg[phase, stats_idx] += 1
                if phase_slice_counter != 0 and phase_slice_counter != half_slices - 1:
                    t_num_deg_non[phase, stats_idx] += 1
                    apex_base = 0
                    cont_slice_id = float(phase_slice_counter) / float(half_slices)
                else:
                    apex_base = 1
                    cont_slice_id = 0 if phase_slice_counter == 0 else 1
                img_3dim_deg[s_counter_deg, :, :, :] = x
                label_slices_deg[s_counter_deg] = pos_label
                extra_label_slices_deg[s_counter_deg, 0] = stripped_patient_id
                extra_label_slices_deg[s_counter_deg, 1] = phase
                extra_label_slices_deg[s_counter_deg, 2] = phase_slice_counter
                extra_label_slices_deg[s_counter_deg, 3] = slice_dice
                extra_label_slices_deg[s_counter_deg, 4] = apex_base
                extra_label_slices_deg[s_counter_deg, 5] = cont_slice_id
                s_counter_deg += 1
            else:
                if phase_slice_counter != 0 and phase_slice_counter != half_slices - 1:
                    apex_base = 0
                    cont_slice_id = float(phase_slice_counter) / float(half_slices)
                else:
                    apex_base = 1
                    cont_slice_id = 0 if phase_slice_counter == 0 else 1
                img_3dim[s_counter, :, :, :] = x
                label_slices[s_counter] = neg_label
                extra_label_slices[s_counter, 0] = stripped_patient_id
                extra_label_slices[s_counter, 1] = phase
                extra_label_slices[s_counter, 2] = phase_slice_counter
                extra_label_slices[s_counter, 3] = slice_dice
                extra_label_slices[s_counter, 4] = apex_base
                extra_label_slices[s_counter, 5] = cont_slice_id
                s_counter += 1

        if is_train:
            dataset.train_images[patient_id] = {"slices": img_3dim, "deg_slices": img_3dim_deg}
            dataset.train_labels[patient_id] = {"slices": label_slices, "deg_slices": label_slices_deg}
            dataset.train_extra_labels[patient_id] = {"slices": extra_label_slices,
                                                      "deg_slices": extra_label_slices_deg}
        else:
            dataset.test_images[patient_id] = {"slices": img_3dim, "deg_slices": img_3dim_deg}
            dataset.test_labels[patient_id] = {"slices": label_slices, "deg_slices": label_slices_deg}
            dataset.test_extra_labels[patient_id] = {"slices": extra_label_slices,
                                                     "deg_slices": extra_label_slices_deg}

    dataset.size_train = len(dataset.train_images)
    dataset.train_patient_ids = dataset.train_images.keys()
    dataset.size_test = len(dataset.test_images)
    dataset.test_patient_ids = dataset.test_images.keys()

    del acdc_dataset

    return dataset
