import numpy as np
from collections import OrderedDict
from common.dslices.config import config


class SliceDetectorDataSet(object):

    def __init__(self):
        self.train_images = OrderedDict()
        self.train_labels = OrderedDict()
        self.train_extra_labels = OrderedDict()
        self.test_images = OrderedDict()
        self.test_labels = OrderedDict()
        self.test_extra_labels = OrderedDict()
        self.size_train = 0
        self.size_test = 0
        self.train_patient_ids = None
        self.test_patient_ids = None

    def get(self, patient_id, is_train=True):
        if is_train:
            image = self.train_images[patient_id]
            label = self.train_labels[patient_id]

        else:
            image = self.test_images[patient_id]
            label = self.test_labels[patient_id]

        return image, label

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


def create_dataset(acdc_dataset, exper_ensemble, type_of_map="u_map", referral_threshold=0.001,
                   extra_augs=3, degenerate_type="mean"):

    exper_ensemble.load_dice_without_referral(type_of_map=type_of_map, referral_threshold=0.001)
    exper_handlers = exper_ensemble.seg_exper_handlers
    dataset = SliceDetectorDataSet()
    # slice stats
    t_num_slices = np.zeros((2, 2, 1))
    t_num_deg = np.zeros((2, 2, 1))
    t_num_deg_pat = np.zeros(2)
    t_num_deg_non = np.zeros((2, 2, 1))
    # loop over training images
    for patient_id in acdc_dataset.trans_dict.keys():
        # found a degenerate slice
        pat_with_deg = False
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
        num_of_slices = image.shape[3]
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
        """
            We construct a numpy array as input to our model with the following shape:
                [#slices, 3channels, w, h]
           
            
        """
        img_dice_scores = exper_ensemble.dice_score_slices[patient_id]
        deg_slices, no_augs = determine_augmentation_factor(img_dice_scores, extra_augs=extra_augs,
                                                            degenerate_type=degenerate_type)
        if not is_train:
            no_augs = np.zeros(2)
        total_augs = np.sum(no_augs)
        # if np.any(deg_slices[0]):
        #    print("WARNING - No augmentations ", no_augs)
        #    print(deg_slices[0])
        #    print(deg_slices[1])
        total_slices = int((2 * num_of_slices) + total_augs)
        img_3dim = np.zeros((total_slices, 3, image.shape[1], image.shape[2]))
        # do the same for the labels, but we only need one position per slice
        label_slices = np.zeros(total_slices)
        extra_label_slices = np.zeros((2, total_slices))
        s_counter = 0
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

                # get indicator whether we're dealing with a degenerate slice (below dice threshold)
                is_degenerate = deg_slices[phase, s]
                # if the mean dice score for this slice is equal or below threshold (0.7) then set label to 1
                if is_degenerate:
                    pat_with_deg = True
                    # add slice/label plus extra augmentations to balance data set
                    # IMPORTANT:
                    if is_train:
                        aug_size = extra_augs + 1
                    else:
                        aug_size = 1
                    for e in np.arange(aug_size):
                        img_3dim[s_counter + e, :, :, :] = x
                        label_slices[s_counter + e] = 1
                        extra_label_slices[0, s_counter + e] = phase
                        t_num_slices[phase, stats_idx] += 1
                        t_num_deg[phase, stats_idx] += 1
                        if s != 0 and s != image.shape[3] - 1:
                            t_num_deg_non[phase, stats_idx] += 1

                    s_counter += (e + 1)
                else:
                    t_num_slices[phase, stats_idx] += 1
                    img_3dim[s_counter, :, :, :] = x
                    label_slices[s_counter] = 0
                    extra_label_slices[0, s_counter] = phase
                    s_counter += 1

        if is_train:
            dataset.train_images[patient_id] = img_3dim
            dataset.train_labels[patient_id] = label_slices
        else:
            dataset.test_images[patient_id] = img_3dim
            dataset.test_labels[patient_id] = label_slices
        if pat_with_deg:
            t_num_deg_pat[stats_idx] += 1

    dataset.size_train = len(dataset.train_images)
    dataset.train_patient_ids = dataset.train_images.keys()
    dataset.size_test = len(dataset.test_images)
    dataset.test_patient_ids = dataset.test_images.keys()
    perc_deg = t_num_deg[0, 0] / float(t_num_slices[0, 0])
    perc_deg_non = t_num_deg_non[0, 0] / float(t_num_slices[0, 0])
    print("INFO - ES - train data-set stats: total {} / degenerate {:.2f} % / {:.2f} %".format(t_num_slices[0, 0, 0],
                                                                                          perc_deg[0] * 100,
                                                                                          perc_deg_non[0] * 100))
    perc_deg = t_num_deg[1, 0] / float(t_num_slices[1, 0])
    perc_deg_non = t_num_deg_non[1, 0] / float(t_num_slices[1, 0])
    print("INFO - ED - train data-set stats: total {} / degenerate {:.2f} % / {:.2f} %".format(t_num_slices[1, 0, 0],
                                                                                          perc_deg[0] * 100,
                                                                                          perc_deg_non[0] * 100))

    perc_deg = t_num_deg[0, 1] / float(t_num_slices[0, 1])
    perc_deg_non = t_num_deg_non[0, 1] / float(t_num_slices[0, 1])
    print("INFO - ES - test data-set stats: total {} / degenerate {:.2f} % / {:.2f} %".format(t_num_slices[0, 1, 0],
                                                                                         perc_deg[0] * 100,
                                                                                         perc_deg_non[0] * 100))
    perc_deg = t_num_deg[1, 1] / float(t_num_slices[1, 1])
    perc_deg_non = t_num_deg_non[1, 1] / float(t_num_slices[1, 1])
    print("INFO - ED - test data-set stats: total {} / degenerate {:.2f} % / {:.2f} %".format(t_num_slices[1, 1, 0],
                                                                                              perc_deg[0] * 100,
                                                                                              perc_deg_non[0] * 100))
    print("INFO - Patients with degenerate slices train/test {} / {}".format(t_num_deg_pat[0], t_num_deg_pat[1]))
    return dataset


def determine_augmentation_factor(img_dices, extra_augs=3, degenerate_type="mean"):
    """

    :param img_dices: shape [2, 4classes, #slices]
    :param extra_augs: num of extra augmentations per degenerate slice
    :param degenerate_type: "mean" dice score over 3 classes <= threshold (e.g. 70%)
                        OR  "any" any of the dice scores of the 3 classes is below threshold
    :return:
    """
    num_of_slices = img_dices.shape[2]
    num_degenerate_augs = np.zeros(2)
    # output matrix of shape [2, #slices]
    deg_slices = np.zeros((img_dices.shape[0], num_of_slices))
    # COMBAT IMBALANCES IN THE DATA SET:
    # loop over ES/ED phases
    for ph in np.arange(img_dices.shape[0]):
        # img_dices has shape [2, 4, #slices], want to average over the last three classes LV, MYO, RV per slice
        if degenerate_type == "mean":
            mean_img_dices = np.mean(img_dices[ph, 1:, :], axis=0)
            deg_slices[ph] = mean_img_dices <= config.dice_threshold
        else:
            # this results in a boolean matrix with shape [3classes, #slices]
            b_slices = img_dices[ph, 1:, :] <= config.dice_threshold
            # perform logical OR on axis=0 <- classes. Should result in tensor with shape [#slices]
            deg_slices[ph] = np.any(b_slices, axis=0)
            # mean_img_dices = np.mean(img_dices[ph, 1:, :], axis=0)
            # compare = mean_img_dices <= config.dice_threshold
            #
            # check = np.count_nonzero(deg_slices[ph]) != np.count_nonzero(compare)
            # if check:
            #     print("-------------------------")
            #     print(np.count_nonzero(deg_slices[ph]) - np.count_nonzero(compare))

        if np.any(deg_slices[ph]):
            no_augs = np.count_nonzero(deg_slices[ph]) * extra_augs
        else:
            no_augs = 0

        num_degenerate_augs[ph] = no_augs
    return deg_slices, num_degenerate_augs
