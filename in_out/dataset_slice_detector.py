import numpy as np
from collections import OrderedDict
from common.dslices.config import config
from tqdm import tqdm

from in_out.load_data import ACDC2017DataSet


class SliceDetectorDataSet(object):

    def __init__(self):
        # The key of the dictionaries is patient_id. Each entry contains another dictionary with 2 keys,
        # that is (1) normal slices key="slices" (2) degenerate slices key="deg_slices"
        # the 2nd dictionary contains 2 numpy arrays of different size (because of #slices
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

    def get(self, patient_id, is_train=True, do_merge_sets=False):
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
        if do_merge_sets:
            image = np.concatenate((image["slices"], image["deg_slices"]), axis=0)
            label = np.concatenate((label["slices"], label["deg_slices"]), axis=0)
            extra_label = np.concatenate((extra_label["slices"], extra_label["deg_slices"]), axis=0)
        else:
            image = tuple((image["slices"], image["deg_slices"]))
            label = tuple((label["slices"], label["deg_slices"]))
            extra_label = tuple((extra_label["slices"], extra_label["deg_slices"]))
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


def create_dataset(exper_hdl, exper_ensemble, type_of_map="u_map", referral_threshold=0.001, num_of_input_chnls=3,
                   degenerate_type="mean", pos_label=1, acdc_dataset=None, logger=None, verbose=False,
                   overwrite_quick_run=False):
    # only works for fold0 (testing purposes, small dataset), we can overwrite the quick_run argument
    # means we're loading a very small dataset (probably 2-3 images)
    random_map = exper_hdl.exper.run_args.use_random_map
    if exper_hdl.exper.run_args.fold_id == 0 and overwrite_quick_run:
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
    exper_ensemble.load_dice_without_referral(type_of_map=type_of_map, referral_threshold=0.001)
    exper_handlers = exper_ensemble.seg_exper_handlers
    # instead of using class labels 0, 1, 2, 3 for the seg-masks we will use values between [0, 1]
    labels_float = [0., 0.3, 0.6, 0.9]
    # set negative label based on positive label
    neg_label = 1 - pos_label
    dataset = SliceDetectorDataSet()
    # slice stats
    t_num_slices = np.zeros((2, 2))
    t_num_deg = np.zeros((2, 2))
    t_num_deg_pat = np.zeros(2)
    t_num_deg_non = np.zeros((2, 2))
    # loop over training images
    for patient_id in tqdm(acdc_dataset.trans_dict.keys()):
        # found a degenerate slice
        pat_with_deg = False
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
        # get Bayesian uncertainty map or entropy map, shape [2, w, h, #slices]
        if type_of_map == "u_map" and num_of_input_chnls == 3 and is_train or \
                (not is_train and not random_map):
            exper_handlers[pfold_id].get_referral_maps(u_threshold=referral_threshold, per_class=False,
                                                       patient_id=patient_id,
                                                       aggregate_func="max", use_raw_maps=True,
                                                       load_ref_map_blobs=False)
            u_maps = exper_handlers[pfold_id].referral_umaps[patient_id]
        elif type_of_map == "e_map" and num_of_input_chnls == 3 and is_train or \
                (not is_train and not random_map):
            # get entropy maps
            exper_handlers[pfold_id].get_entropy_maps(patient_id=patient_id)
            u_maps = exper_handlers[pfold_id].entropy_maps[patient_id]
        elif random_map and not is_train:
            # original: u_maps = np.random.normal(loc=0.1, scale=1., size=image.shape)
            # temporary
            exper_handlers[pfold_id].get_entropy_maps(patient_id=patient_id)
            u_maps = exper_handlers[pfold_id].entropy_maps[patient_id]
            u_maps = np.concatenate((u_maps[0], u_maps[1]), axis=2)
            image = np.random.normal(loc=0.1, scale=1., size=image.shape)
        else:
            # we should only end-up here if we have 2 input channels
            if num_of_input_chnls == 3:
                raise ValueError("ERROR - Something went wrong, you shouldn't end up here.")
            # We don't use any uncertainty map. So dimension 1 is equal to 2 instead of 3
            type_of_map = None
        """
            We construct a numpy array as input to our model with the following shape:
                [#slices, 3channels, w, h]
           
            
        """
        # merge the first dimension ES/ED [2, w, h, #slices] to [w, h, 2*#slices]
        if type_of_map is not None and is_train or \
                (not is_train  and not random_map):
            u_maps = np.concatenate((u_maps[0], u_maps[1]), axis=2)
        img_dice_scores = exper_ensemble.dice_score_slices[patient_id]
        # also merge first dimension of ES/ED img_dice_scores obj
        img_dice_scores = np.concatenate((img_dice_scores[0], img_dice_scores[1]), axis=1)
        deg_slices, class_count, skip_patient = determine_degenerate_slices(img_dice_scores, config.dice_threshold,
                                                                            degenerate_type=degenerate_type)
        if not skip_patient:
            # we compute "again" the mean dice (without BG) for each slice, because we store this as extra label info
            mean_img_dices = np.mean(img_dice_scores[1:, :], axis=0)
            non_deg_num_slices = int(class_count[0])
            deg_num_slices = int(class_count[1])
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
                    #ED
                    phase = 1
                    cls_offset = pred_labels.shape[0] / 2
                    phase_slice_counter = s - half_slices
                else:
                    #ES
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

                # get indicator whether we're dealing with a degenerate slice (below dice threshold)
                is_degenerate = deg_slices[s]
                slice_dice = mean_img_dices[s]
                t_num_slices[phase, stats_idx] += 1
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

        # if one of the slices is degenerate, this is True and we increase the counter for "degenerated patients"
        if pat_with_deg:
            t_num_deg_pat[stats_idx] += 1

    dataset.size_train = len(dataset.train_images)
    dataset.train_patient_ids = dataset.train_images.keys()
    dataset.size_test = len(dataset.test_images)
    dataset.test_patient_ids = dataset.test_images.keys()
    perc_deg = t_num_deg[0, 0] / float(t_num_slices[0, 0])
    perc_deg_non = t_num_deg_non[0, 0] / float(t_num_slices[0, 0])

    total_perc_deg = np.sum(t_num_deg) / float(np.sum(t_num_slices))
    total_perc_deg_non = np.sum(t_num_deg_non) / float(np.sum(t_num_slices))

    if random_map:
        msg0 = "WARNING - Using RANDOM uncertainty maps!"
    msg1 = "INFO - Total degenerate over ES/ED and train/test set {:.2f}% / {:.2f}%".format(total_perc_deg * 100,
                                                                                           total_perc_deg_non * 100)

    msg2 = "INFO - ES - train data-set stats: total {} / degenerate {:.2f} % / {:.2f} %".format(t_num_slices[0, 0],
                                                                                          perc_deg * 100,
                                                                                          perc_deg_non * 100)
    perc_deg = t_num_deg[1, 0] / float(t_num_slices[1, 0])
    perc_deg_non = t_num_deg_non[1, 0] / float(t_num_slices[1, 0])
    msg3 = "INFO - ED - train data-set stats: total {} / degenerate {:.2f} % / {:.2f} %".format(t_num_slices[1, 0],
                                                                                          perc_deg * 100,
                                                                                          perc_deg_non * 100)

    perc_deg = t_num_deg[0, 1] / float(t_num_slices[0, 1])
    perc_deg_non = t_num_deg_non[0, 1] / float(t_num_slices[0, 1])
    msg4 = "INFO - ES - test data-set stats: total {} / degenerate {:.2f} % / {:.2f} %".format(t_num_slices[0, 1],
                                                                                         perc_deg * 100,
                                                                                         perc_deg_non * 100)
    perc_deg = t_num_deg[1, 1] / float(t_num_slices[1, 1])
    perc_deg_non = t_num_deg_non[1, 1] / float(t_num_slices[1, 1])
    msg5 = "INFO - ED - test data-set stats: total {} / degenerate {:.2f} % / {:.2f} %".format(t_num_slices[1, 1],
                                                                                              perc_deg * 100,
                                                                                              perc_deg_non * 100)
    msg6 = "INFO - Patients with degenerate slices train/test {} / {}".format(t_num_deg_pat[0], t_num_deg_pat[1])
    msg7 = "INFO - #Patients in train/test set {}/{}".format(dataset.size_train, dataset.size_test)
    if verbose:
        if logger is None:
            print(msg1)
            print("--------------------------------")
            print(msg2)
            print(msg3)
            print(msg4)
            print(msg5)
        else:
            logger.info(msg1)
            logger.info("--------------------------------")
            logger.info(msg2)
            logger.info(msg3)
            logger.info(msg4)
            logger.info(msg5)

    del acdc_dataset
    if logger is not None:
        if random_map:
            logger.info(msg0)
        logger.info(msg6)
        logger.info(msg7)
    else:
        if random_map:
            print(msg0)
        print(msg6)
        print(msg7)
    return dataset


def determine_degenerate_slices(img_dices, dice_threshold=0.7, degenerate_type="mean"):
    """

    :param img_dices: shape [4classes, 2*#slices] ES/ED concatenated
    :param dice_threshold: decimal between [0, 1.] indicating the dice threshold to be used to determine a
                           so called "degenerate" slice. Should be in config object.
    :param degenerate_type: "mean" dice score over 3 classes <= threshold (e.g. 70%)
                        OR  "any" any of the dice scores of the 3 classes is below threshold
    :return: boolean vector [#slices] indicating the degenerate slices
             class_count: shape 2, 0=#normal slices 1=#degenerate slices
    """

    if degenerate_type == "mean":
        mean_img_dices = np.mean(img_dices[1:, :], axis=0)
        deg_slices = mean_img_dices <= dice_threshold
    else:
        # this results in a boolean matrix with shape [3classes, #slices]
        b_slices = img_dices[1:, :] <= dice_threshold
        # perform logical OR on axis=0 <- classes. Should result in tensor with shape [#slices]
        deg_slices = np.any(b_slices, axis=0)

    # counter for non-degenerate (index 0) and degenerate count
    class_count = np.zeros(2)
    if np.any(deg_slices):
        skip_patient = False
        deg_count = np.count_nonzero(deg_slices)
        class_count[0] = deg_slices.shape[0] - deg_count
        class_count[1] = deg_count
    else:
        class_count[0] = deg_slices.shape[0]
        skip_patient = True

    return deg_slices, class_count, skip_patient
