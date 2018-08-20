import numpy as np
from collections import OrderedDict
from common.dslices.config import config


class SliceDetectorDataSet(object):

    def __init__(self):
        self.train_images = OrderedDict()
        self.train_labels = OrderedDict()
        self.test_images = OrderedDict()
        self.test_labels = OrderedDict()

    def load_from_file(self):
        raise NotImplementedError()


def create_dataset(acdc_dataset, exper_ensemble, type_of_map="u_map", referral_threshold=0.001):

    exper_ensemble.load_dice_without_referral(type_of_map=type_of_map, referral_threshold=0.001)
    exper_handlers = exper_ensemble.seg_exper_handlers
    dataset = SliceDetectorDataSet()
    # loop over training images
    for patient_id in acdc_dataset.trans_dict.keys():
        # IMPORTANT: val_image_names only contains the patient_ids of the test/val set
        #            whereas acdc.dataset.image_names contains all names (confusing I know)
        # HENCE, we check whether the patient ID is in the first list, if not it's a training image
        if patient_id not in acdc_dataset.val_image_names:
            # training set
            is_train = True
            image_num = acdc_dataset.image_names.index(patient_id)
            image = acdc_dataset.train_images[image_num]
            # print("Train --- patient id/image_num {}/{}".format(patient_id, image_num))
        else:
            # must be a test image
            is_train = False
            image_num = acdc_dataset.val_image_names.index(patient_id)
            image = acdc_dataset.val_images[image_num]
            # print("Test --- patient id/image_num {}/{}".format(patient_id, image_num))
        # image has shape [2, w, h, #slices] ES/ED
        # get fold_id, patient belongs to. we need the fold in order to retrieve the predicted seg-mask and the
        # u-map/e-map from the exper_ensemble. The results (dice) of the 75 training cases are distributed over the
        # other three folds, because they used them for test evaluation.
        pfold_id = exper_ensemble.get_patient_fold_id(patient_id)
        exper_handlers[pfold_id].get_pred_labels(patient_id=patient_id)
        # please NOTE that pred_labels has shape [8, w, h, #slices]
        pred_labels = exper_handlers[pfold_id].pred_labels[patient_id]
        # get Bayesian uncertainty map or entropy map, shape [2, w, h, #slices]
        if type_of_map == "u_map":
            exper_handlers[pfold_id].get_referral_maps(u_threshold=referral_threshold, per_class=False,
                                                       patient_id=patient_id,
                                                       aggregate_func="max", use_raw_maps=True,
                                                       load_ref_map_blobs=False)
            u_maps = exper_handlers[pfold_id].referral_umaps[patient_id]

        else:
            # get entropy maps
            exper_handlers[pfold_id].get_entropy_maps(patient_id=patient_id)
            u_maps = exper_handlers[pfold_id].entropy_maps[patient_id]
        # construct np-array with zeros for input image with 3 channels
        img_3dim = np.zeros((2, 3, image.shape[1], image.shape[2], image.shape[3]))
        # do the same for the labels, but we only need one position per slice
        label_slices = np.zeros((2, image.shape[3]))
        # loop over ES and then ED for all slices
        for phase in np.arange(image.shape[0]):
            cls_offset = 4 * phase
            # loop over slices
            for s in np.arange(image.shape[3]):
                # merge 4 class labels into one dimesion: use labels 0, 1, 2, 3
                # please NOTE that pred_labels has shape [8, w, h, #slices]
                p_lbl = np.zeros((pred_labels.shape[1], pred_labels.shape[2]))
                # loop over LV, MY and RV class (omit BG)
                for cls in np.arange(1, pred_labels.shape[0] / 2):
                    p_lbl[pred_labels[cls + cls_offset, :, :, s] == 1] = cls

                x = np.concatenate((np.expand_dims(image[phase, :, :, s], axis=0),
                                          np.expand_dims(p_lbl, axis=0),
                                          np.expand_dims(u_maps[phase, :, :, s], axis=0)), axis=0)
                img_3dim[phase, :, :, :, s] = x
                # get the dice scores for this patient, phase, slide ID
                # we'll get a numpy array with shape [2, 4classes, #slices]
                dice_slice = exper_ensemble.dice_score_slices[patient_id][phase, :, s]
                # so we have a numpy array of 4 values for each tissue including background, we only
                # want to average over the last three classes LV, MYO, RV
                mean_slice_dice = np.mean(dice_slice[1:])
                # if the mean dice score for this slice is equal or below threshold (0.7) then set label to 1
                if mean_slice_dice <= config.dice_threshold:
                    label_slices[phase, s] = 1
                else:
                    label_slices[phase, s] = 0
        if is_train:
            dataset.train_images[patient_id] = img_3dim
            dataset.train_labels[patient_id] = label_slices
        else:
            dataset.test_images[patient_id] = img_3dim
            dataset.test_labels[patient_id] = label_slices

    return dataset

