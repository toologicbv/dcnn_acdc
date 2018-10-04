import numpy as np
from collections import OrderedDict
from common.detector.config import config_detector
from tqdm import tqdm
from utils.exper_hdl_ensemble import ExperHandlerEnsemble
from common.dslices.config import config
from common.detector.helper import find_bbox_object


class RegionDetectorDataSet(object):

    pixel_dta_type = 'float32'
    pad_size = config_detector.acdc_pad_size
    num_of_augs = 4

    def __init__(self, num_of_channels=2):
        self.num_of_channels = num_of_channels
        # The key of the dictionaries is patient_id.
        self.train_images = []
        self.train_labels = []
        self.train_img_rois = []
        self.train_lbl_rois = []
        self.train_extra_labels = []
        # contain numpy array with shape [#slices, 3, w, h]
        self.test_images = []
        # contain numpy array with shape [#slices]
        self.test_labels = []
        self.test_img_rois = []
        self.test_lbl_rois = []
        # contain numpy array with shape [#slices, 4] 1) phase 2) slice id 3) mean-dice/slice 4) patient id
        self.test_extra_labels = []
        # actually "size" here is relate to number of patients
        self.size_train = 0
        self.size_test = 0
        self.train_patient_ids = None
        self.test_patient_ids = None
        # translation dictionary from patient id to list index numbers. Key=patient_id,
        # value=tuple(is_train, [<indices>])
        self.trans_dict = OrderedDict()

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

    def augment_data(self, input_chnl1, input_chnl2, label_slices, is_train=False):
        """
        Augments image slices by rotating z-axis slices for 90, 180 and 270 degrees

        :param input_chnl1: [w, h, #slices] uncertainty maps
        :param input_chnl2: [w, h, #slices] automatic reference
        :param label_slices: [w, h, #slices] binary values indicating whether voxel should be corrected
        :param is_train: boolean
        :return None (fills train/test arrays of object)

        """

        def rotate_slice(input_chnl1, input_chnl2, label_slice, is_train=False):
            """

            :param input_chnl1: [w, h]
            :param input_chnl2  [w, h]
            :param label_slice: [w, h]
            :param is_train: boolean
            :return: None
            """
            for rots in range(RegionDetectorDataSet.num_of_augs):
                p_slice1 = np.pad(input_chnl1, RegionDetectorDataSet.pad_size, 'constant',
                                  constant_values=(0,)).astype(RegionDetectorDataSet.pixel_dta_type)
                p_slice2 = np.pad(input_chnl2, RegionDetectorDataSet.pad_size, 'constant',
                                  constant_values=(0,)).astype(RegionDetectorDataSet.pixel_dta_type)
                padded_input_slices = np.concatenate((p_slice1[np.newaxis], p_slice2[np.newaxis]))
                # should result again in [2, w+pad_size, h+pad_size]

                if is_train:
                    self.train_images.append(padded_input_slices)
                    self.train_labels.append(label_slice)
                    list_array_indices.append(len(self.train_images) - 1)
                else:
                    self.test_images.append(padded_input_slices)
                    self.test_labels.append(label_slice)
                    list_array_indices.append(len(self.test_images) - 1)
                # rotate for next iteration
                input_chnl1 = np.rot90(input_chnl1)
                input_chnl2 = np.rot90(input_chnl2)
                label_slice = np.rot90(label_slice)

        num_of_slices = input_chnl1.shape[2]
        list_array_indices = []
        # for each image-slice rotate the img four times. We're doing that for all three orientations
        for z in range(num_of_slices):
            input_chnl1_slice = input_chnl1[:, :, z]
            input_chnl2_slice = input_chnl2[:, :, z]
            label_slice = label_slices[:, :, z]
            rotate_slice(input_chnl1_slice, input_chnl2_slice, label_slice, is_train)

        return list_array_indices

    @staticmethod
    def collapse_roi_maps(target_roi_maps):
        """
        our binary roi maps specify the voxels to "inspect" per tissue class, for training this distinction
        doesn't matter, so here, we collapse the classes to 1
        :param target_roi_maps: [#classes, w, h, #slices]
        :return: [w, h, #slices]

        """
        target_roi_maps = np.sum(target_roi_maps, axis=0)
        # kind of logical_or over all classes. if one voxels equal to 1 or more, than set voxel to 1 which means
        # we need to correct/inspect that voxel
        target_roi_maps[target_roi_maps > 1] = 1
        return target_roi_maps.astype(np.int)

    @staticmethod
    def convert_to_multilabel(labels, multiclass_idx):
        """
        Assuming label_slice has shape [#classes, w, h, #slices]

        :param label_slice:
        :param multiclass_idx:
        :return:
        """
        nclasses, w, h, num_slices = labels.shape
        multilabel_slice = np.zeros((w, h, num_slices))
        for slice_id in np.arange(num_slices):
            lbl_slice = np.zeros((w, h))
            for cls_idx in np.arange(nclasses):
                if cls_idx != 0:
                    lbl_slice[labels[cls_idx, :, :, slice_id] == 1] = multiclass_idx[cls_idx]
            multilabel_slice[:, :, slice_id] = lbl_slice
        return multilabel_slice

    @staticmethod
    def remove_padding(image):
        """

        :param image:
        :return:
        """
        if image.ndim == 2:
            # assuming [w, h]
            return image[RegionDetectorDataSet.pad_size:-RegionDetectorDataSet.pad_size,
                         RegionDetectorDataSet.pad_size:-RegionDetectorDataSet.pad_size]
        elif image.ndim == 3:
            # assuming [#channels, w, h]
            return image[:, RegionDetectorDataSet.pad_size:-RegionDetectorDataSet.pad_size,
                         RegionDetectorDataSet.pad_size:-RegionDetectorDataSet.pad_size]
        elif image.ndim == 4:
            # assuming [#channels, w, h, #slices]
            return image[:, RegionDetectorDataSet.pad_size:-RegionDetectorDataSet.pad_size,
                         RegionDetectorDataSet.pad_size:-RegionDetectorDataSet.pad_size, :]


def create_dataset(exper_ensemble, train_fold_id, type_of_map="e_map", num_of_input_chnls=2,
                   logger=None, verbose=False, quick_run=False):
    """

    :param fold_id: This is the fold_id we use for the experiment handler
    :param exper_ensemble: This is an ensemble of DCNN segmentation handlers from the previous experiments on ACDC data
    :param type_of_map: u_map: Bayesian uncertainty map; e_map: Entropy maps;
    :param num_of_input_chnls: 2: automatic segmentation masks + uncertainties; 3: + original image
    :param acdc_dataset: If None, we create the ACDC set for the specific fold we're training (see below)
    :param logger:
    :param verbose:
    :param quick_run: reduce training dataset to small number (currently 10)
    :return:
    """
    # only works for fold0 (testing purposes, small dataset), we can overwrite the quick_run argument
    # means we're loading a very small dataset (probably 2-3 images)
    exper_handlers = exper_ensemble.seg_exper_handlers
    exper_hdl = exper_handlers[train_fold_id]
    # get the patient ids of the test set for this segmentation handler. We use that to distinguish between
    # train and test set patient ids.
    exper_hdl.get_testset_ids()
    # get the list of ALL patient IDs
    patient_ids = exper_ensemble.patient_fold.keys()
    # reduce training set in case we are debugging the model
    ensemble_patients = None
    if quick_run:
        patient_ids = patient_ids[:4]
        ensemble_patients = patient_ids
    print("INFO - Preparing experimental handlers. This may take a while. Be patient...")
    print("INFO - Ready. Loop through patient ids.")
    # REMEMBER: type of map determines whether we're loading mc-dropout predictions or single-predictions
    exper_ensemble.prepare_handlers(type_of_map=type_of_map, force_reload=True, for_detector_dtaset=True,
                                    patient_ids=ensemble_patients)

    # instead of using class labels 0, 1, 2, 3 for the seg-masks we will use values between [0, 1]
    labels_float = [0., 0.3, 0.6, 0.9]
    dataset = RegionDetectorDataSet(num_of_channels=num_of_input_chnls)
    # loop over training images
    for patient_id in tqdm(patient_ids):
        # we store the list indices for this patient in RegionDetector.train_images or test_images
        # this can help us for plotting the dataset or batches (debugging)
        list_data_indices = []
        stripped_patient_id = int(patient_id.replace("patient", ""))
        patient_fold_id = exper_ensemble.patient_fold[patient_id]
        #
        if not exper_hdl.is_in_test_set(patient_id):
            # training set
            is_train = True
            # print("Train --- patient id/image_num {}".format(patient_id))
        else:
            is_train = False
            # print("Test --- patient id/image_num {}".format(patient_id))
        # automatic reference: [#classes, w, h, #slices] - 0:4=ES and 4:8=ED
        pred_labels = exper_handlers[patient_fold_id].pred_labels[patient_id]
        # target_rois (our binary voxel labels)
        target_rois = exper_handlers[patient_fold_id].get_target_roi_maps(patient_id)
        nclasses, w, h, num_of_slices = pred_labels.shape
        pred_labels_multi = np.zeros((2, w, h, num_of_slices))
        pred_labels_multi[0] = RegionDetectorDataSet.convert_to_multilabel(pred_labels[0:4], labels_float)
        pred_labels_multi[1] = RegionDetectorDataSet.convert_to_multilabel(pred_labels[4:], labels_float)
        # uncertainty maps: [2, w, h, #slices] - 0=ES and 1=ED
        if type_of_map == "u_map":
            u_maps = exper_handlers[patient_fold_id].referral_umaps[patient_id]
        elif type_of_map == "e_map":
            u_maps = exper_handlers[patient_fold_id].entropy_maps[patient_id]
        # for ACDC databaset which combines ES/ED the first dimension of the u-maps must be 2
        num_of_phases = u_maps.shape[0]
        for phase in np.arange(num_of_phases):
            phase_offset = phase * (nclasses / 2)
            roi_slice = slice(phase_offset, phase_offset + (nclasses / 2))
            label_slices = RegionDetectorDataSet.collapse_roi_maps(target_rois[roi_slice])
            # print(input_slices.shape, target_rois.shape, label_slices.shape)
            # returns list of indices
            list_data_indices.extend(dataset.augment_data(u_maps[phase], pred_labels_multi[phase], label_slices,
                                                          is_train=is_train))
        dataset.trans_dict[patient_id] = tuple((is_train, list_data_indices))

    dataset.size_train = len(dataset.train_images)
    dataset.size_test = len(dataset.test_images)

    return dataset


if __name__ == '__main__':
    seg_exper_ensemble = ExperHandlerEnsemble(config.exper_dict_brier)
    dataset = create_dataset(seg_exper_ensemble, train_fold_id=0, quick_run=True, num_of_input_chnls=2)

    del dataset