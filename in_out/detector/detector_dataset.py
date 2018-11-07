import numpy as np
from collections import OrderedDict
from common.detector.config import config_detector
from tqdm import tqdm
from utils.exper_hdl_ensemble import ExperHandlerEnsemble
from common.dslices.config import config
from common.detector.box_utils import find_multiple_connected_rois, BoundingBox, find_bbox_object, find_box_four_rois
from common.detector.box_utils import adjust_roi_bounding_box
from in_out.load_data import ACDC2017DataSet


class RegionDetectorDataSet(object):

    pixel_dta_type = 'float32'
    pad_size = config_detector.acdc_pad_size
    num_of_augs = 4

    def __init__(self, num_of_channels=2, model_name=None):
        self.num_of_channels = num_of_channels
        # The key of the dictionaries is patient_id.
        self.train_images = []
        self.train_labels = []
        self.train_labels_extra = []
        # list of numpy [N, 4] arrays that describe the target tissue area of the automatic predictions
        # we use these to sample from when generating batches (in order to stay close to the target tissue area)
        self.train_pred_lbl_rois = []
        self.train_lbl_rois = []
        self.train_extra_labels = []
        # contain numpy array with shape [#slices, 3, w, h]
        self.test_images = []
        # contain numpy array with shape [#slices]
        self.test_labels = []
        self.test_labels_extra = []
        self.test_pred_lbl_rois = []
        self.test_lbl_rois = []
        # stores the padding we applied to the test images. tuple of tuples e.g. ((2, 1), (5, 3))
        self.test_paddings = []
        # contain numpy array with shape [#slices, 4] 1) phase 2) slice id 3) mean-dice/slice 4) patient id
        self.test_extra_labels = []
        # actually "size" here is relate to number of patients
        self.size_train = 0
        self.size_test = 0
        self.train_patient_ids = []
        self.test_patient_ids = []
        self.test_patient_slice_id = []
        # translation dictionary from patient id to list index numbers. Key=patient_id,
        # value=tuple(is_train, [<indices>])
        self.trans_dict = OrderedDict()
        # ROI statistics, indices meaning: 0=approx # of target grids
        #                                  1=total ROIs, 2=ROIS in base/apex slices, 3=ROIS in other slices
        self.roi_stats = {"train": np.zeros(4).astype(np.int), "test": np.zeros(4).astype(np.int)}
        # store ROI areas sizes. index: 0=apex/base, 1=other slices
        self.roi_areas = [[], []]
        if model_name is not None:
            self.model_name = model_name
            config_detector.get_architecture(model_name=model_name)
            self.config = config_detector
        else:
            self.config = config_detector

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

    def collect_roi_data(self):
        roi_areas = []
        roi_aspect_ratio = []

        def loop_over_rois(roi_area_list):
            for b in np.arange(0, len(roi_area_list), 4):
                roi_array = roi_area_list[b]
                # roi_array is Nx4 np array
                for i in np.arange(roi_array.shape[0]):
                    bbox = BoundingBox.create(roi_array[i])
                    roi_areas.append(bbox.area)
                    roi_aspect_ratio.append(bbox.width / float(bbox.height))

        loop_over_rois(self.train_lbl_rois)
        loop_over_rois(self.test_lbl_rois)
        roi_areas = np.array(roi_areas)
        roi_aspect_ratio = np.array(roi_aspect_ratio)
        return roi_areas, roi_aspect_ratio

    def augment_data(self, input_chnl1, input_chnl2, input_chnl3, label_slices, is_train=False,
                     do_rotate=False, patient_id=None, cardiac_phase=None):
        """
        Augments image slices by rotating z-axis slices for 90, 180 and 270 degrees

        :param input_chnl1: [w, h, #slices] original mri
        :param input_chnl2: [w, h, #slices] automatic reference
        :param input_chnl3: [w, h, #slices] uncertainty maps
        :param label_slices: [w, h, #slices] binary values indicating whether voxel should be corrected
        :param is_train: boolean
        :param do_rotate: boolean, if True (currently only for training images) then each slice is rotated
                          three times (90, ..., 270)
        :param patient_id
        :param cardiac_phase
        :return None (fills train/test arrays of object)

        """

        def rotate_slice(mri_img_slice, uncertainty_slice, pred_lbl_slice, label_slice,
                         is_train=False):
            """

            :param mri_img_slice: [w, h] original mri image
            :param uncertainty_slice  [w, h] uncertainty map
            :param pred_lbl_slice: [w, h] predicted seg mask
            :param label_slice: [w, h]

            :param is_train: boolean
            :return: None
            """

            for rots in range(RegionDetectorDataSet.num_of_augs):

                self.add_image_to_set(is_train, mri_img_slice, uncertainty_slice, pred_lbl_slice, label_slice,
                                      list_array_indices)
                # rotate for next iteration
                mri_img_slice = np.rot90(mri_img_slice)
                uncertainty_slice = np.rot90(uncertainty_slice)
                pred_lbl_slice = np.rot90(pred_lbl_slice)
                label_slice = np.rot90(label_slice)

        num_of_slices = input_chnl1.shape[2]
        list_array_indices = []
        # for each image-slice rotate the img four times. We're doing that for all three orientations
        for z in range(num_of_slices):
            pat_slice_id = tuple((patient_id, z+1, cardiac_phase))
            input_chnl1_slice = input_chnl1[:, :, z]
            input_chnl2_slice = input_chnl2[:, :, z]
            input_chnl3_slice = input_chnl3[:, :, z]
            label_slice = label_slices[:, :, z]
            w, h = label_slice.shape
            # NOTE: this is only an approximation of the number of target grids for which we predict the presence of
            # voxels to be inspected. We currently tile the images in grids with spacing 8 voxels (see config_detector)
            num_of_grids = int((w / 8) * (h / 8))
            if 0 != np.count_nonzero(label_slice):
                # label_bbox contains np array [N, 4], coordinates of ROI area around target region we're interested
                # in. label_slice_filtered: filtered by 2D 4-connected structure.
                # Meaning, that potential target voxels that are not part of larger 4-connected components will be
                # discarded. Please also see config_detector.min_roi_area, currently = 2!
                label_bbox, label_slice_filtered, bbox_areas = find_multiple_connected_rois(label_slice, padding=1)
                self.roi_stats["train" if is_train else "test"][0] += num_of_grids
                # increase total #ROIS
                num_of_rois = label_bbox.shape[0]
                self.roi_stats["train" if is_train else "test"][1] += num_of_rois
                if z == 0 or z == (num_of_slices - 1):
                    self.roi_stats["train" if is_train else "test"][2] += num_of_rois
                    self.roi_areas[0].extend(bbox_areas)
                    base_apex_slice = 1
                else:
                    self.roi_stats["train" if is_train else "test"][3] += num_of_rois
                    self.roi_areas[1].extend(bbox_areas)
                    base_apex_slice = 0

            else:
                # determine base/apex or middle slice
                if z == 0 or z == (num_of_slices - 1):
                    base_apex_slice = 1
                else:
                    base_apex_slice = 0
                label_slice_filtered = label_slice
            if is_train:
                self.train_labels_extra.append(base_apex_slice)
            else:
                self.test_labels_extra.append(base_apex_slice)
            if do_rotate:
                rotate_slice(input_chnl1_slice, input_chnl2_slice, input_chnl3_slice, label_slice_filtered,
                             is_train)
            else:
                # Note: for the TEST set we also use the label_slice_filtered object.
                self.add_image_to_set(is_train, input_chnl1_slice, input_chnl2_slice, input_chnl3_slice,
                                      label_slice_filtered, list_array_indices, pat_slice_id)

        return list_array_indices

    def add_image_to_set(self, is_train, input_chnl1_slice, input_chnl2_slice, input_chnl3_slice, label_slice,
                         list_array_indices, pat_slice_id=None):
        if is_train:
            p_slice1 = np.pad(input_chnl1_slice, RegionDetectorDataSet.pad_size, 'constant',
                              constant_values=(0,)).astype(RegionDetectorDataSet.pixel_dta_type)
            p_slice2 = np.pad(input_chnl2_slice, RegionDetectorDataSet.pad_size, 'constant',
                              constant_values=(0,)).astype(RegionDetectorDataSet.pixel_dta_type)
            p_slice3 = np.pad(input_chnl3_slice, RegionDetectorDataSet.pad_size, 'constant',
                              constant_values=(0,)).astype(RegionDetectorDataSet.pixel_dta_type)
            # we get the bounding box for the predicted aka automatic segmentation mask. we use this for batch
            # generation
            pred_lbl_roi = find_bbox_object(p_slice3, padding=0)
            # should result again in [3, w+pad_size, h+pad_size]
            padded_input_slices = np.concatenate((p_slice1[np.newaxis], p_slice2[np.newaxis], p_slice3[np.newaxis]))
        else:
            # find_bbox_object returns BoundingBox object. We're looking for bounding boxes of the automatic
            # segmentation mask. If not None (automatic seg mask) then check the size of the bbox. During
            # validation & testing we're only processing mri slices with an automatic segmentation mask.
            # Because we've prior knowlegde about the dataset, we know that the mask will be always smaller than
            # the original image, hence, we NEVER have to pad the image in order to make sure that the size (w, h)
            # is dividable by max_grid_spacing (currently 8).
            pred_lbl_roi = find_bbox_object(input_chnl3_slice, padding=0)
            # in case we are dealing with slices that do not contain an automatic seg-mask we don't need to find
            # the ROI. We will even not use these slices during testing because we're only interested in slices with
            # an automatic seg-mask. Is slice_idx=len(self.test_images) then procedure will print slices bigger than
            # training patch_size. We use this for debugging purposes.
            if not pred_lbl_roi.empty:
                # if the slice contains an automatic segmentation mask, we determine the ROI around it. We use the
                # ROI during training in order to "crop" the image to the area we're actually considering for
                # detection of the regions we want to inspect.
                pred_lbl_roi = adjust_roi_bounding_box(pred_lbl_roi, slice_idx=None, config=self.config)

            padded_input_slices = np.concatenate((input_chnl1_slice[np.newaxis], input_chnl2_slice[np.newaxis],
                                                  input_chnl3_slice[np.newaxis]))

        # we get the bounding boxes for the different target rois in box_four format (x.start, y.start, ...)
        # we use these when generating the batches, because we want to make sure that for the positive batch items
        # at least ONE target rois is in the FOV i.e. included in the patch that we sample from a slice
        bbox_for_rois = find_box_four_rois(label_slice)
        if is_train:
            self.train_images.append(padded_input_slices)
            self.train_labels.append(label_slice)
            # IMPORTANT: the bbox is not rotated!
            self.train_lbl_rois.append(bbox_for_rois)
            list_array_indices.append(len(self.train_images) - 1)
            self.train_pred_lbl_rois.append(pred_lbl_roi.box_four)
        else:
            self.test_images.append(padded_input_slices)
            self.test_labels.append(label_slice)
            # IMPORTANT: the bbox is not rotated!
            self.test_lbl_rois.append(bbox_for_rois)
            list_array_indices.append(len(self.test_images) - 1)
            self.test_pred_lbl_rois.append(pred_lbl_roi.box_four)
            self.test_patient_slice_id.append(pat_slice_id)
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
        target_roi_maps[target_roi_maps >= 1] = 1
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
        if RegionDetectorDataSet.pad_size > 0:
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
        else:
            return image


def create_dataset(exper_ensemble, train_fold_id, type_of_map="e_map", num_of_input_chnls=2, quick_run=False,
                   model_name=None):
    """

    :param train_fold_id: This is the fold_id we use for the experiment handler
    :param exper_ensemble: This is an ensemble of DCNN segmentation handlers from the previous experiments on ACDC data
    :param type_of_map: u_map: Bayesian uncertainty map; e_map: Entropy maps;
    :param num_of_input_chnls: 2: automatic segmentation masks + uncertainties; 3: + original image
    :param acdc_dataset: If None, we create the ACDC set for the specific fold we're training (see below)
    :param logger:
    :param verbose:
    :param quick_run: reduce training dataset to small number (currently 10)
    :param model_name: please refer to the model parameter in the parsing.py file to determine which argument values
                        are currently valid (rd1, rd2, rd3)
    :return:
    """
    if type_of_map == "u_map":
        mc_dropout = True
    else:
        mc_dropout = False
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
        patient_ids = patient_ids[:config_detector.quick_run_size]
        ensemble_patients = patient_ids
    print("INFO - Preparing experimental handlers. This may take a while. Be patient...")
    # REMEMBER: type of map determines whether we're loading mc-dropout predictions or single-predictions
    exper_ensemble.prepare_handlers(type_of_map=type_of_map, force_reload=True, for_detector_dtaset=True,
                                    patient_ids=ensemble_patients, load_dt_roi_maps=False)
    print("INFO - Ready. Loop through patient ids.")
    # instead of using class labels 0, 1, 2, 3 for the seg-masks we will use values between [0, 1]
    labels_float = [0., 0.3, 0.6, 0.9]
    dataset = RegionDetectorDataSet(num_of_channels=num_of_input_chnls, model_name=model_name)
    # loop over training images
    for patient_id in tqdm(patient_ids):
        # we store the list indices for this patient in RegionDetector.train_images or test_images
        # this can help us for plotting the dataset or batches (debugging)
        list_data_indices = []
        patient_fold_id = exper_ensemble.patient_fold[patient_id]
        #
        if not exper_hdl.is_in_test_set(patient_id):
            # training set
            is_train = True
            dataset.train_patient_ids.append(patient_id)
            # print("Train --- patient id/image_num {}".format(patient_id))
        else:
            is_train = False
            dataset.test_patient_ids.append(patient_id)
            # print("Test --- patient id/image_num {}".format(patient_id))
        # automatic reference: [#classes, w, h, #slices] - 0:4=ES and 4:8=ED
        pred_labels = exper_handlers[patient_fold_id].pred_labels[patient_id]
        # target_rois (our binary voxel labels)
        target_rois = exper_handlers[patient_fold_id].get_target_roi_maps(patient_id, mc_dropout=mc_dropout)
        nclasses, w, h, num_of_slices = pred_labels.shape
        # get the original images, from test set. Return [2, w, h, #slices]
        mri_image = exper_handlers[patient_fold_id].test_images[patient_id]
        # remove padding
        mri_image = ACDC2017DataSet.remove_padding(mri_image)
        pred_labels_multi = np.zeros((2, w, h, num_of_slices))
        pred_labels_multi[0] = RegionDetectorDataSet.convert_to_multilabel(pred_labels[0:(nclasses / 2)], labels_float)
        pred_labels_multi[1] = RegionDetectorDataSet.convert_to_multilabel(pred_labels[(nclasses / 2):], labels_float)
        # uncertainty maps: [2, w, h, #slices] - 0=ES and 1=ED
        if type_of_map == "u_map":
            u_maps = exper_handlers[patient_fold_id].referral_umaps[patient_id]
        elif type_of_map == "e_map":
            u_maps = exper_handlers[patient_fold_id].entropy_maps[patient_id]
        else:
            raise ValueError("ERROR - Unknown type of uncertainty map {}".format(type_of_map))
        # for ACDC databaset which combines ES/ED the first dimension of the u-maps must be 2
        num_of_phases = u_maps.shape[0]
        for phase in np.arange(num_of_phases):
            phase_offset = phase * (nclasses / 2)
            roi_slice = slice(phase_offset, phase_offset + (nclasses / 2))
            # the target rois distinguish between the different tissue classes. We don't need this, we're only
            # interested in the ROIs in general, hence, we convert the ROI to a binary mask
            label_slices = RegionDetectorDataSet.collapse_roi_maps(target_rois[roi_slice])
            # print(input_slices.shape, target_rois.shape, label_slices.shape)
            # returns list of indices.
            # NOTE: we're currently NOT rotating the TEST images
            list_data_indices.extend(dataset.augment_data(mri_image[phase], u_maps[phase], pred_labels_multi[phase],
                                                          label_slices, do_rotate=is_train, is_train=is_train,
                                                          patient_id=patient_id, cardiac_phase=phase))
        dataset.trans_dict[patient_id] = tuple((is_train, list_data_indices))

    dataset.size_train = len(dataset.train_images)
    dataset.size_test = len(dataset.test_images)

    # clean up all exper handlers
    for f_id in exper_ensemble.exper_dict.keys():
        if f_id != train_fold_id:
            del exper_ensemble.seg_exper_handlers[f_id].test_images
            del exper_ensemble.seg_exper_handlers[f_id].pred_labels
            del exper_ensemble.seg_exper_handlers[f_id].target_roi_maps
            if exper_ensemble.seg_exper_handlers[f_id].entropy_maps is not None:
                del exper_ensemble.seg_exper_handlers[f_id].entropy_maps
            if exper_ensemble.seg_exper_handlers[f_id].u_maps is not None:
                del exper_ensemble.seg_exper_handlers[f_id].u_maps
        else:
            exper_ensemble.seg_exper_handlers[f_id].test_images = None
            exper_ensemble.seg_exper_handlers[f_id].pred_labels = None
            exper_ensemble.seg_exper_handlers[f_id].target_roi_maps = None
            exper_ensemble.seg_exper_handlers[f_id].entropy_maps = None
            exper_ensemble.seg_exper_handlers[f_id].u_maps = None
    return dataset


if __name__ == '__main__':
    seg_exper_ensemble = ExperHandlerEnsemble(config.exper_dict_brier)
    dataset = create_dataset(seg_exper_ensemble, train_fold_id=0, quick_run=True, num_of_input_chnls=2)

    del dataset