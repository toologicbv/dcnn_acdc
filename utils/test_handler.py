
import numpy as np
from config.config import config
import os
import glob
import copy
from collections import OrderedDict
from tqdm import tqdm
from in_out.read_save_images import load_mhd_to_numpy, write_numpy_to_image
from utils.dice_metric import dice_coefficient
from utils.img_sampling import resample_image_scipy
from utils.medpy_metrics import hd
from utils.post_processing import filter_connected_components
from common.common import create_mask_uncertainties, convert_to_multiclass
from common.detector.box_utils import find_bbox_object
from common.cardiac_indices import compute_ejection_fraction
from common.detector.box_utils import BoundingBox
import torch
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import generate_binary_structure


def eval_referral(errors_idx, uncertain_idx):
    true_pos_idx = np.logical_and(errors_idx, uncertain_idx)
    true_pos = np.count_nonzero(true_pos_idx)
    false_neg_idx = np.logical_and(errors_idx, ~uncertain_idx)
    false_neg = np.count_nonzero(false_neg_idx)
    false_pos_idx = np.logical_and(~errors_idx, uncertain_idx)
    false_pos = np.count_nonzero(false_pos_idx)
    # compute precision
    if true_pos > 0:
        pr = true_pos / float(true_pos + false_pos)
        rec = true_pos / float(true_pos + false_neg)
    else:
        if false_pos == 0.:
            pr = 1.
        else:
            pr = 0
        if false_neg == 0.:
            rec = 1.
        else:
            rec = 0
    if pr != 0 and rec != 0:
        f1_score = 2 / (1 / float(pr) + 1 / float(rec))
    elif pr == 0 and rec != 0:
        f1_score = 2 / (1 / float(rec))
    elif pr != 0 and rec == 0:
        f1_score = 2 / (1 / float(pr))
    else:
        f1_score = 0

    return f1_score, pr, rec, true_pos, false_pos, false_neg


def test_ensemble(test_set, exper_hdl, mc_samples=10, sample_weights=True, discard_outliers=False,
                  referral_threshold=None, ref_positives_only=False, image_range=None, verbose=False,
                  reset_results=False, save_pred_labels=False,
                  store_details=False, generate_stats=False, save_results=False, use_seed=False,
                  do_filter=False, checkpoints=None, use_uncertainty=False, u_threshold=0.01):
    """

    :param test_set:
    :param exper_hdl:
    :param mc_samples:
    :param sample_weights:
    :param discard_outliers:
    :param use_uncertainty: refer pixels with uncertainty above referral threshold to expert
    :param referral_threshold:
    :param image_range:
    :param verbose:
    :param reset_results:
    :param store_details:
    :param generate_stats:
    :param save_results:
    :param save_filtered_umaps
    :param save_pred_labels
    :param use_seed:
    :param do_filter:
    :param checkpoints:
    :param ref_positives_only: please see experiment.test method for more info
    :param u_threshold: used to filter the pixel uncertainties that we consider when computing slices statistics
    which we later exploit for training guidance
    :return:
    """
    # It should not happen that we're testing on a different fold than we trained on for this model!!!
    if test_set.fold_ids[0] != int(exper_hdl.exper.run_args.fold_ids[0]):
        raise ValueError("Model {} was trained on different fold {} "
                         "than test fold {}".format(exper_hdl.exper.model_name,
                                                    exper_hdl.exper.run_args.fold_ids[0],
                                                    test_set.fold_ids[0]))
    # reset pointers to checkpoint models
    exper_hdl.ensemble_models = OrderedDict()
    if checkpoints is None:
        checkpoints = [100000, 110000, 120000, 130000, 140000, 150000]
    if reset_results:
        exper_hdl.reset_results(mc_samples=mc_samples, use_dropout=sample_weights)
    if use_uncertainty:
        if referral_threshold is None:
            raise ValueError("Error - you need to specify the referral threshold when using uncertainty.")
        print("WARNING - Using uncertainty maps to refer pixels. Threshold {:.2f}".format(referral_threshold))
        exper_hdl.get_referral_maps(u_threshold=referral_threshold, per_class=False)
    if discard_outliers:
        # print("INFO - Using {:.2f} as uncertainty threshold".format(referral_threshold))
        print("WARNING - Using outlier statistics.")
    if image_range is None:
        image_range = np.arange(len(test_set.images))
    num_of_images = len(image_range)
    str_checkpoints = ", ".join([str(c) for c in checkpoints])
    print("INFO - Evaluating model {} on checkpoints {}".format(exper_hdl.exper.model_name, str_checkpoints))
    print("INFO - Running test on {} images".format(num_of_images))
    for image_num in image_range:
        exper_hdl.test(checkpoints, test_set, image_num=image_num, sample_weights=sample_weights,
                       mc_samples=mc_samples, compute_hd=True, discard_outliers=discard_outliers,
                       referral_threshold=referral_threshold, verbose=verbose, store_details=store_details,
                       use_seed=use_seed, do_filter=do_filter,
                       use_uncertainty=use_uncertainty, u_threshold=u_threshold,
                       ref_positives_only=ref_positives_only, save_pred_labels=save_pred_labels)

    exper_hdl.test_results.compute_mean_stats()
    exper_hdl.test_results.show_results()

    if generate_stats:
        if mc_samples == 1:
            print("--->>> WARNING: can't compute statistics because mc_samples=1!! Skipping this step.")
        else:
            print("INFO - generating statistics for {} run(s). May take a while".format(exper_hdl.test_results.N))
            exper_hdl.test_results.generate_all_statistics()
    if save_results:
        exper_hdl.test_results.save_results(fold_ids=test_set.fold_ids)

    try:
        # clean up
        del exper_hdl.exper.outliers_per_epoch
        del exper_hdl.exper.u_maps
    except:
        pass


class ACDC2017TestHandler(object):

    train_path = "train"
    val_path = "validate"
    image_path = "images"
    label_path = "reference"

    pixel_dta_type = 'float32'
    pad_size = config.pad_size
    new_voxel_spacing = 1.4

    def __init__(self, exper_config, search_mask=None, nclass=4, load_func=load_mhd_to_numpy,
                 fold_ids=[1], debug=False, use_cuda=True, batch_size=None, load_train=False,
                 load_val=True, use_iso_path=False, generate_flipped_images=False, verbose=False):

        # IMPORTANT: boolean flag indicating whether to load images from image_iso/reference_iso path or
        # from original directory (not resized). If True we don't resize the image during loading
        self.verbose = verbose
        self.use_iso_path = use_iso_path
        # IMPORTANT flag indicating whether we're loading only the validation set from the specified fold(s)
        self.load_train = load_train
        self.load_val = load_val
        self.generate_flipped_images = generate_flipped_images
        if not load_train and not load_val:
            raise ValueError("You have to load at least train or validation files for this fold.")
        self.config = exper_config
        self.data_dir = os.path.join(self.config.root_dir, exper_config.data_dir)
        self.filtered_umaps_dir = os.path.join(self.config.root_dir, config.u_map_dir)
        self.search_mask = search_mask
        self.num_of_classes = nclass
        self.fold_ids = fold_ids
        self.load_func = load_func
        self.abs_path_fold = os.path.join(self.data_dir, "fold")
        self.images = []
        self.images_flipped = []
        self.cardiac_indices = {}  # will store per patient np.array([2, ESV, EDV, EJ]) for index=0=LV, index=1=RV
        #
        # filename of the figures printed during test evaluation. Contains patient_ids
        self.img_file_names = []
        self.labels = []
        # contains the target region or cropped image size for target region. Contains dictionary of dicts.
        # Outer dice: patient_id with value dictionary, inner dict: 0=ES, 1=ED, values tuples of slices (slice_x, slice_y)
        # where slice_x, slice_y are 2-tuples themselves.
        self.labels_target_roi = {}
        self.labels_flipped = []
        self.num_labels_per_class = []
        self.spacings = []
        self.b_pred_labels = None
        self.b_filtered_pred_labels = None
        self.b_pred_probs = None
        self.b_stddev_map = None
        self.b_bald_map = None  # [2, width, height, #slices] one map for each heart phase ES/ED
        self.b_image = None
        self.b_image_id = None
        self.b_image_name = None  # store the filename from "above" used as identification during testing
        self.b_labels = None
        self.b_num_labels_per_class = None
        self.b_new_spacing = None
        self.b_orig_spacing = None
        self.b_seg_errors = None
        self.b_uncertainty_stats = None
        self.b_filtered_stddev_map = None
        self.b_filtered_umap = False
        self.referral_stats = None
        # accuracy and hausdorff measure for slices (per image)
        self.b_hd_slices = None
        self.b_acc_slices = None
        self.b_ref_hd_slices = None
        self.b_ref_acc_slices = None
        self.debug = debug
        self._set_pathes()
        # dictionary with key is patientID and value is imageID that we assign when loading the stuff
        self.trans_dict = OrderedDict()
        self.num_of_images = 0

        self.slice_counter = 0
        self.num_of_models = 1
        self.use_cuda = use_cuda
        if batch_size is not None:
            self.batch_size = batch_size * 2  # multiply by 2 because each "image" contains an ED and ES part
        else:
            self.batch_size = None
        # self.out_img = np.zeros((self.num_classes, self.image.shape[0], self.image.shape[1], self.image.shape[2]))
        # self.overlay_images = {'ED_RV': np.zeros((self.image.shape[0], self.image.shape[1], self.image.shape[2])),
        #                        'ED_LV': np.zeros((self.image.shape[0], self.image.shape[1], self.image.shape[2]))}

        file_list = self._get_file_lists()
        self._load_file_list(file_list)
        if self.generate_flipped_images:
            self._generate_flipped_test_set()
            print("INFO - Generated >>flipped<< images in objects images_flipped/labels_flipped.")

    def get_test_pair(self, patient_id):
        try:
            image_num = self.trans_dict[patient_id]
        except KeyError:
            print("ERROR - key {} does not exist in translation dict".format(patient_id))
            print("Following keys exist:")
            print(self.trans_dict.keys())
        return self.images[image_num], self.labels[image_num]

    def get_labels_per_class(self, image_num=None):
        if image_num is None:
            return self.labels
        else:
            return self.labels[image_num]

    def _generate_flipped_test_set(self):

        for idx, image in enumerate(self.images):
            # remember image has shape [2, width, height, #slices]
            # we need to ndarray.copy() the flipped tensor, otherwise we run into annoying errors
            # reported here: https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663/4
            self.images_flipped.append(np.flip(image, axis=2).copy())
            # remember label has shape [#classes, width, height, #slices]
            label = self.labels[idx]
            self.labels_flipped.append(np.flip(label, axis=2).copy())
        self.images = self.images_flipped
        self.images_flipped = []
        self.labels = self.labels_flipped
        self.labels_flipped = []

    def generate_bbox_target_roi(self, patient_id=None):
        if patient_id is None:
            label_list = self.labels
        else:
            idx = self.trans_dict[patient_id]
            label_list = [self.labels[idx]]
        for idx, labels in enumerate(label_list):
            if patient_id is None:
                patient_id = self.img_file_names[idx]

            num_of_slices = labels.shape[3]
            pat_slice_rois = OrderedDict()
            for slice_id in np.arange(num_of_slices):
                label_slice = labels[:, :, :, slice_id]
                # label_slice has shape [8, w, h, #slices], hence we separate es and ed and convert slice to multi-label
                label_slice_es = convert_to_multiclass(label_slice[0:4])
                label_slice_ed = convert_to_multiclass(label_slice[4:])
                # returns BoundingBox object for the target roi
                gt_label_roi_es = find_bbox_object(label_slice_es, padding=0)
                gt_label_roi_ed = find_bbox_object(label_slice_ed, padding=0)
                pat_slice_rois[slice_id] = {0: tuple((gt_label_roi_es.slice_x, gt_label_roi_es.slice_y)),
                                            1: tuple((gt_label_roi_ed.slice_x, gt_label_roi_ed.slice_y))}
            self.labels_target_roi[patient_id] = pat_slice_rois
        if patient_id is not None:
            return self.labels_target_roi[patient_id]

    def get_label_areas(self, patient_id):
        """

        :param patient_id:
        :return: numpy array [2, #slice] with tissue label areas per slice (NOT PER CLASS)
        """
        pat_slice_rois = self.labels_target_roi[patient_id]
        num_of_slices = len(pat_slice_rois)
        # 2 = for ES/ED, 0=ES, 1=ED
        pat_slice_areas = np.zeros((2, num_of_slices))
        for slice_id, slice_rois in pat_slice_rois.iteritems():
            roi_box_es = BoundingBox(slice_rois[0][0], slice_rois[0][1])
            roi_box_ed = BoundingBox(slice_rois[1][0], slice_rois[1][1])
            area_es = roi_box_es.width * roi_box_es.height
            area_ed = roi_box_ed.width * roi_box_ed.height
            pat_slice_areas[:, slice_id] = np.array([area_es, area_ed])
        return pat_slice_areas

    def _set_pathes(self):
        # kind of awkward. but because it's a class variable we need to check whether we extended the path
        # earlier (e.g. when running ipython notebook session, otherwise we concatenate it twice).
        if self.use_iso_path:
            if ACDC2017TestHandler.image_path.find("_iso") == -1:
                # load images from a directory that only contains a couple of images
                ACDC2017TestHandler.image_path = ACDC2017TestHandler.image_path + "_iso"
                ACDC2017TestHandler.label_path = ACDC2017TestHandler.label_path + "_iso"

        if self.debug:
            if ACDC2017TestHandler.image_path.find("_test") != -1:
                # load images from a directory that only contains a couple of images
                ACDC2017TestHandler.image_path = ACDC2017TestHandler.image_path + "_test"
                ACDC2017TestHandler.label_path = ACDC2017TestHandler.label_path + "_test"

    def _get_file_lists(self):

        file_list = []

        for fold_id in self.fold_ids:
            self.train_path = os.path.join(self.abs_path_fold + str(fold_id),
                                           os.path.join(ACDC2017TestHandler.train_path, ACDC2017TestHandler.image_path))
            self.val_path = os.path.join(self.abs_path_fold + str(fold_id),
                                         os.path.join(ACDC2017TestHandler.val_path, ACDC2017TestHandler.image_path))

            # get images and labels
            if self.load_train:
                # we're NOT only loading validation images
                search_mask_img = os.path.join(self.train_path, self.search_mask)
                # print("INFO - Testhandler - >>> Search in train-dir for {} <<<".format(search_mask_img))
                for train_file in glob.glob(search_mask_img):
                    ref_file = train_file.replace(ACDC2017TestHandler.image_path, ACDC2017TestHandler.label_path)
                    file_list.append(tuple((train_file, ref_file)))

            # get validation images and labels
            search_mask_img = os.path.join(self.val_path, self.search_mask)
            if self.load_val:
                # print("INFO - Testhandler - >>> Search in val-dir for {} <<<".format(search_mask_img))
                for val_file in glob.glob(search_mask_img):
                    ref_file = val_file.replace(ACDC2017TestHandler.image_path, ACDC2017TestHandler.label_path)
                    file_list.append(tuple((val_file, ref_file)))
        num_of_files = len(file_list)
        if self.verbose:
            print("INFO - File list contains {} files, hence {} patients".format(num_of_files, num_of_files / 2))

        return file_list

    def _load_file_list(self, file_list):

        file_list.sort()
        if self.batch_size is not None:
            batch_file_list = file_list[:self.batch_size]
        else:
            batch_file_list = file_list

        if self.use_iso_path:
            do_zoom = False
        else:
            do_zoom = True

        for idx in tqdm(np.arange(0, len(batch_file_list), 2)):
            # tuple contains [0]=train file name and [1] reference file name
            img_file, ref_file = file_list[idx]
            # first frame is always the end-diastole MRI scan, filename ends with "1"
            mri_scan_ed, origin, spacing = self.load_func(img_file, data_type=ACDC2017TestHandler.pixel_dta_type,
                                                          swap_axis=True)
            # es_abs_file_name = os.path.dirname(img_file)  # without actual filename, just directory
            ed_file_name = os.path.splitext(os.path.basename(img_file))[0]
            # get rid off _frameXX and take only the patient name
            patientID = ed_file_name[:ed_file_name.find("_")]
            if self.generate_flipped_images:
                # we extend the patient IDs with one leading zero for the flipped test set
                str_num = patientID.strip("patient")
                str_num = str_num.zfill(4)
                patientID = "patient" + str_num
            self.img_file_names.append(patientID)
            self.trans_dict[patientID] = self.num_of_images
            # print("{} - {}".format(idx, img_file))
            self.spacings.append(spacing)
            mri_scan_ed = self._preprocess(mri_scan_ed, spacing, poly_order=3, do_pad=True, do_zoom=do_zoom)
            # print("INFO - Loading ES-file {}".format(img_file))
            reference_ed, _, _ = self.load_func(ref_file, data_type=ACDC2017TestHandler.pixel_dta_type,
                                                swap_axis=True)
            reference_ed = self._preprocess(reference_ed, spacing, poly_order=0, do_pad=False, do_zoom=do_zoom)

            # do the same for the End-systolic pair of images
            img_file, ref_file = file_list[idx+1]

            # print("{} - {}".format(idx+1, img_file))
            mri_scan_es, _, _ = self.load_func(img_file, data_type=ACDC2017TestHandler.pixel_dta_type,
                                               swap_axis=True)
            mri_scan_es = self._preprocess(mri_scan_es, spacing, poly_order=3, do_pad=True, do_zoom=do_zoom)

            reference_es, _, _ = self.load_func(ref_file, data_type=ACDC2017TestHandler.pixel_dta_type,
                                                swap_axis=True)
            reference_es = self._preprocess(reference_es, spacing, poly_order=0, do_pad=False, do_zoom=do_zoom)
            # concatenate both images for further processing
            images = np.concatenate((np.expand_dims(mri_scan_es, axis=0),
                                     np.expand_dims(mri_scan_ed, axis=0)))
            # same concatenation for the label files of ED and ES
            labels, num_labels_per_class = self._split_class_labels(reference_ed, reference_es)
            self.images.append(images)
            self.labels.append(labels)
            self.num_labels_per_class.append(num_labels_per_class)
            self.num_of_images += 1
        if self.verbose:
            print("INFO - Successfully loaded {} ED/ES patient pairs".format(len(self.images)))

    def _preprocess(self, image, spacing, poly_order=3, do_zoom=True, do_pad=True):

        if do_zoom:
            zoom_factors = tuple((spacing[0] / ACDC2017TestHandler.new_voxel_spacing,
                                 spacing[1] / ACDC2017TestHandler.new_voxel_spacing, 1))

            image = resample_image_scipy(image, new_spacing=zoom_factors, order=poly_order)
        # we only pad images not the references aka labels...hopefully
        if do_pad:
            image = np.pad(image, ((ACDC2017TestHandler.pad_size, ACDC2017TestHandler.pad_size),
                                   (ACDC2017TestHandler.pad_size, ACDC2017TestHandler.pad_size), (0, 0)),
                           'constant', constant_values=(0,)).astype(ACDC2017TestHandler.pixel_dta_type)

        return image

    def _split_class_labels(self, labels_ed, labels_es):
        """

        :param labels_ed:
        :param labels_es:
        :return: returns tensor [8classes, width, height, #slices] for reference labels (gt) and
        a "count" of the pixels/voxels that belong to that slice per class (simulating volume regression)
        """
        labels_per_class = np.zeros((2 * self.num_of_classes, labels_ed.shape[0], labels_ed.shape[1],
                                     labels_ed.shape[2]))
        b_num_labels_per_class = np.zeros((1, self.num_of_classes * 2, labels_ed.shape[2]))
        for cls_idx in np.arange(self.num_of_classes):
            # store ES class labels in first 4 positions of dim-1
            labels_per_class[cls_idx, :, :, :] = (labels_es == cls_idx).astype('int16')
            # store ED class labels in positions 4-7 of dim-1
            labels_per_class[cls_idx + self.num_of_classes, :, :, :] = (labels_ed == cls_idx).astype('int16')
            if cls_idx != 0:
                b_num_labels_per_class[0, cls_idx] = \
                    np.count_nonzero(labels_per_class[cls_idx, :, :, :], axis=(0, 1))
                b_num_labels_per_class[0, cls_idx + self.num_of_classes] = \
                    np.count_nonzero(labels_per_class[cls_idx + self.num_of_classes, :, :, :], axis=(0, 1))

        return labels_per_class, b_num_labels_per_class

    def batch_generator(self, image_num=0, use_labels=True):
        """
            Remember that self.image is a list, containing images with shape [2, height, width, depth]
            Object self.labels is also a list but with shape [8, height, width, depth]

        """
        self.b_image_id = image_num
        self.b_filtered_umap = False
        b_label = None
        b_num_labels_per_class = None
        self.slice_counter = -1
        self.b_image = self.images[image_num]
        num_of_slices = self.b_image.shape[3]
        self.b_image_name = self.img_file_names[image_num]
        self.b_orig_spacing = self.spacings[image_num]
        self.b_new_spacing = tuple((ACDC2017TestHandler.new_voxel_spacing, ACDC2017TestHandler.new_voxel_spacing,
                                    self.b_orig_spacing[2]))
        # store the segmentation errors. shape [#slices, #classes]
        self.b_seg_errors = np.zeros((self.b_image.shape[3], self.num_of_classes * 2)).astype(np.int)
        # currently measuring 4 values per slice - cardiac phase (2): total uncertainty, number of pixels with
        # uncertainty (>eps), number of pixels with uncertainty above u_threshold, number of connected components in
        # 2D slice (5)
        # Important, for stddev we keep the 8 classes but splitted per phase, hence 2 x self.num_of_classes
        self.b_uncertainty_stats = {'stddev': np.zeros((2, self.num_of_classes, 4, num_of_slices)),
                                    'bald': np.zeros((2, 4, num_of_slices)), "u_threshold": 0.}
        self.b_acc_slices = np.zeros((2, self.num_of_classes, num_of_slices))
        self.b_hd_slices = np.zeros((2, self.num_of_classes, num_of_slices))
        if use_labels:
            self.b_labels = self.labels[image_num]
            self.b_pred_probs = np.zeros_like(self.b_labels)
            self.b_pred_labels = np.zeros_like(self.b_labels)
            # b_num_labels_per_class should have shape [1, 8, #slices]
            self.b_num_labels_per_class = self.num_labels_per_class[image_num]
        # TO DO: actually it could happen now that b_uncertainty has undefined shape (because b_labels is not used
        self.b_stddev_map = np.zeros_like(self.b_labels)
        # b_bald_map shape: [2, width, height, #slices]
        self.b_bald_map = np.zeros((2, self.b_labels.shape[1], self.b_labels.shape[2], self.b_labels.shape[3]))

        for slice in np.arange(num_of_slices):
            b_image = torch.FloatTensor(torch.from_numpy(self.b_image[:, :, :, slice]).float())
            b_image = b_image.unsqueeze(0)
            if self.b_labels is not None:
                b_label = torch.FloatTensor(torch.from_numpy(self.b_labels[:, :, :, slice]).float())
                b_num_labels_per_class = torch.FloatTensor(torch.from_numpy(
                    self.b_num_labels_per_class[:, :, slice]).float())
                b_label = b_label.unsqueeze(0)
            if self.use_cuda:
                b_image = b_image.cuda()
                if self.b_labels is not None:
                    b_label = b_label.cuda()
                    b_num_labels_per_class = b_num_labels_per_class.cuda()
            self.slice_counter += 1
            yield b_image, b_label, b_num_labels_per_class

    def set_pred_labels(self, pred_probs, verbose=False, do_filter=False):
        """

        :param pred_probs: [1, num_of_classes(8), width, height]

        Also note that object self.b_labels has dims [num_classes(8), width, height, slices]

        :param do_filter: filter predicted labels on connected components. This is an important post processing step
                          especially when doing this at the end for the 3D label object
        :param verbose:

        :return:
        """

        if not isinstance(pred_probs, np.ndarray):
            pred_probs = pred_probs.data.cpu().numpy()
        # remember dim0 of pred_probs is 1, so we squeeze it by taking index "0". Hence the argmax is over
        # axis 0, because we lost the original dim0
        pred_labels_es = np.argmax(pred_probs[0, 0:self.num_of_classes, :, :], axis=0)
        pred_labels_ed = np.argmax(pred_probs[0, self.num_of_classes:self.num_of_classes+self.num_of_classes,
                                   :, :], axis=0)

        for cls in np.arange(self.num_of_classes):
            pred_labels_cls_es = (pred_labels_es == cls).astype(np.int)
            pred_labels_cls_ed = (pred_labels_ed == cls).astype(np.int)
            if do_filter:
                # use 6-connected components to filter out blobs that are not relevant for segmentation
                pred_labels_cls_es = filter_connected_components(pred_labels_cls_es, cls, verbose=verbose)
                pred_labels_cls_ed = filter_connected_components(pred_labels_cls_ed, cls, verbose=verbose)

            self.b_pred_labels[cls, :, :, self.slice_counter] = pred_labels_cls_es
            self.b_pred_labels[cls + self.num_of_classes, :, :, self.slice_counter] = pred_labels_cls_ed

        self.b_pred_probs[:, :, :, self.slice_counter] = pred_probs

    def compute_img_slice_uncertainty_stats(self, u_type="bald", connectivity=2, u_threshold=0.):
        """
        In order to get a first idea of how uncertain the model is w.r.t. certain image slices we compute
        same simple overall statistics per slide for stddev and BALD uncertainty measures.
        Note:
            self.b_bald_map     [2, width, height, #slices]
            self.b_stddev_map   [#classes (8), width, height, #slices]
        :param u_type:
        :param u_threshold: Can be used to filter out uncertainties below a certain threshold value. Maybe this
        is helpful during analysis. Currently we set eps and u_threshold to 0.01 and hence the counts
        :param connectivity:
        :param eps: tiny threshold to filter out the smallest uncertainties
        :return:
                total_uncertainty, #num_pixel_uncertain, #num_pixel_uncertain_above_tre (w.r.t u_threshold),
                num_of_objects

                If u_threshold = eps than #pixel measure should be the same

        """
        self.b_uncertainty_stats["u_threshold"] = u_threshold
        if u_type == "bald":
            uncertainty_map = self.b_bald_map[:, np.newaxis, :, :, self.slice_counter]
        elif u_type == "stddev":
            # remember that we store the stddev for each class per pixel, so b_stddev_map has
            # shape [#classes (8), width, height, #slices]. Hence, we average over dim0 (classes) for ES and ED
            uncertainty_map = self.b_stddev_map[:, :, :, self.slice_counter]
            # uncertainty_map = np.concatenate((np.mean(uncertainty_map[:self.num_of_classes], axis=0, keepdims=True),
            #                                  np.mean(uncertainty_map[self.num_of_classes:], axis=0, keepdims=True)))
            # we keep the 8 dimensions for stddev but splitted over 2 x 4 (per phase)
            uncertainty_map = np.concatenate((uncertainty_map[np.newaxis, :self.num_of_classes],
                                              uncertainty_map[np.newaxis, self.num_of_classes:]))
        else:
            raise ValueError("{} is not a supported uncertainty type.".format(u_type))
        # we compute stats for ES and ED phase
        for phase in np.arange(uncertainty_map.shape[0]):
            # this is a loop over classes although for BALD statistics we currently don't split stats per class
            for dim1 in np.arange(uncertainty_map.shape[1]):
                pred_uncertainty_bool = np.zeros(uncertainty_map[phase, dim1].shape).astype(np.bool)
                # index maps for uncertain pixels
                mask_uncertain = uncertainty_map[phase, dim1] >= u_threshold
                # the u_threshold was already set in the method set_pred_labels above
                mask_uncertain_above_tre = uncertainty_map[phase, dim1] > self.b_uncertainty_stats["u_threshold"]
                # create binary mask to determine morphology
                pred_uncertainty_bool[mask_uncertain_above_tre] = True
                # IMPORTANT: we use the MASK_UNCERTAIN_ABOVE_TRE !!! to compute the total uncertainty of a slice
                total_uncertainty = np.sum(uncertainty_map[phase, dim1][mask_uncertain_above_tre])
                t_num_pixel_uncertain = np.count_nonzero(uncertainty_map[phase, dim1][mask_uncertain])
                t_num_pixel_uncertain_above_tre = np.count_nonzero(uncertainty_map[phase, dim1]
                                                                   [mask_uncertain_above_tre])
                # The following two statements are preliminary. We determine the number of connected components.
                # The idea is that more connected components could indicate more spatial diversity in the uncertainty
                # and hence we could use this to compute a ratio between total uncertainty/num_of_objects (components)
                # binary structure
                footprint1 = generate_binary_structure(pred_uncertainty_bool.ndim, connectivity)
                # label distinct binary objects
                _, num_of_objects = label(pred_uncertainty_bool, footprint1)
                if u_type == "bald":
                    self.b_uncertainty_stats["bald"][phase, :, self.slice_counter] = \
                        np.array([total_uncertainty, t_num_pixel_uncertain, t_num_pixel_uncertain_above_tre,
                                  num_of_objects])
                else:
                    self.b_uncertainty_stats["stddev"][phase, dim1, :, self.slice_counter] = \
                        np.array([total_uncertainty, t_num_pixel_uncertain, t_num_pixel_uncertain_above_tre,
                                  num_of_objects])

    def filter_referrals(self, u_maps, referral_threshold=0., ref_positives_only=False,
                         apply_to_cls=[1, 2, 3], arr_slice_referrals=None, verbose=False):
        """
        Based on the uncertainty maps (probably produced before inference time) we "refer" pixels with
        high uncertainty (above referral_threshold) to an expert who corrects the pixels. Hence we compute
        again the dice and HD after correction (simulation).

        :param u_maps: We assume this is a tensor of shape [2 (ES/ED), 4classes, width, height, #slices]
                       NOTE: this is not the original/raw u-map but the filtered u-map for this referral threshold
        :param referral_threshold:
        :param apply_to_cls: array specifying the classes to which referral should be applied
        :param ref_positives_only: we only refer pixels with high uncertainties that we predicted as positive
                i.e. belonging to the specific class
        :param arr_slice_referrals: is an array of two arrays (ES/ED) or NONE if we refer all slices.
               If not NONE, we refer only slices of ES/ED which indices are contained in the arrays.
               Is only applied for pred_labels object, for the rest all statistics are computed for all slices.
        :param verbose:
        :return:
        """
        num_of_slices = self.b_labels.shape[3]  # [#classes, width, height, #slices]

        # referral statistics: see details in referral_handler.py, class ReferralDetailedResults
        # we currently store 14 values
        self.referral_stats = np.zeros((2, self.num_of_classes - 1,  14, num_of_slices))
        # this version assumes that u_maps has shape [2, width, height, #slices]
        if ref_positives_only:
            # unfortunately we need to make a deepcopy of u_maps because we'll alter it here
            u_maps = copy.deepcopy(u_maps)
            # returns mask for ES and ED [2, width, height, #slices]
            ref_positive_mask = create_mask_uncertainties(self.b_pred_labels)
            # set all uncertainties (pixels) to zero where we didn't predict a positive label
            u_maps[0][ref_positive_mask[0]] = 0
            u_maps[1][ref_positive_mask[1]] = 0
        if arr_slice_referrals is not None:
            refer_slices_es = arr_slice_referrals[0]
            refer_slices_ed = arr_slice_referrals[1]
        else:
            slice_range = np.arange(num_of_slices)
            refer_slices_es = slice_range
            refer_slices_ed = slice_range

        total_pixels_positive, total_pixels_referred = 0, 0
        # we call this method during referral procedure i.e. without calling first the batch_generator method
        # so we need to initialize some variables first:
        # if self.b_acc_slices is None:
        self.b_acc_slices = np.zeros((2, self.num_of_classes, num_of_slices))
        self.b_hd_slices = np.zeros((2, self.num_of_classes, num_of_slices))
        self.b_ref_acc_slices = np.zeros((2, self.num_of_classes, num_of_slices))
        self.b_ref_hd_slices = np.zeros((2, self.num_of_classes, num_of_slices))
        pred_labels_es = copy.deepcopy(self.b_pred_labels[:self.num_of_classes, :, :, :])
        pred_labels_ed = copy.deepcopy(self.b_pred_labels[self.num_of_classes:, :, :, :])
        for slice_id in np.arange(num_of_slices):
            slice_uncertainty_idx_es, slice_uncertainty_idx_ed = None, None
            slice_pixels_pos_es, slice_pixels_pos_ed = 0, 0
            slice_total_pixels_ref_es, slice_total_pixels_ref_ed = 0, 0
            total_pos_labels_slice_es, total_pos_labels_slice_ed = 0, 0
            pixel_std_es = u_maps[0, :, :, slice_id]
            pixel_std_ed = u_maps[1, :, :, slice_id]
            # get pixels we're uncertain about
            uncertain_es_idx = pixel_std_es >= referral_threshold
            uncertain_ed_idx = pixel_std_ed >= referral_threshold
            # Note: for "positive-only" mode, the uncertainty map is already filtered i.e. if the count below
            # leverages zero uncertain voxels, this means that we indeed won't refer the slice to an expert
            slice_num_pixel_uncertain_es = np.count_nonzero(uncertain_es_idx)
            slice_num_pixel_uncertain_ed = np.count_nonzero(uncertain_ed_idx)
            # compute the original slice accuracy
            _, _ = self.compute_slice_accuracy(slice_idx=slice_id, compute_hd=True, store_results=True)
            # NOTE: we start with class 1, and skip the background class
            for cls in np.arange(1, self.num_of_classes):
                pred_labels_cls_es = pred_labels_es[cls, :, :, slice_id]
                pred_labels_cls_ed = pred_labels_ed[cls, :, :, slice_id]
                true_labels_cls_es = self.b_labels[cls, :, :, slice_id]
                true_labels_cls_ed = self.b_labels[cls + self.num_of_classes, :, :, slice_id]
                num_true_labels_es = np.count_nonzero(true_labels_cls_es)
                num_true_labels_ed = np.count_nonzero(true_labels_cls_ed)
                # indices of seg-errors before referral
                errors_es_idx = pred_labels_cls_es != true_labels_cls_es
                errors_ed_idx = pred_labels_cls_ed != true_labels_cls_ed
                # number of errors before referral
                errors_es = np.count_nonzero(errors_es_idx)
                errors_ed = np.count_nonzero(errors_ed_idx)

                if cls in apply_to_cls:
                    total_pos_labels_slice_es += num_true_labels_es
                    total_pos_labels_slice_ed += num_true_labels_ed
                    # collect indices of pixels with high uncertainty AND for which we predicted a positive label
                    uncertain_es_idx = pixel_std_es >= referral_threshold
                    if ref_positives_only:
                        pred_labels_pos_idx_es = pred_labels_cls_es == 1
                        c = np.count_nonzero(pred_labels_pos_idx_es)
                        uncertain_es_idx = np.logical_and(uncertain_es_idx, pred_labels_pos_idx_es)
                        total_pixels_positive += c
                        slice_pixels_pos_es += c

                    # which of the uncertain pixels belong to the ones we misclassified? True positives ES
                    f1_es, pr_es, rc_es, true_pos_es, false_pos_es, false_neg_es = eval_referral(errors_es_idx,
                                                                                                 uncertain_es_idx)
                    # number of pixels above threshold
                    num_pixel_above_threshold_es = np.count_nonzero(uncertain_es_idx)
                    slice_total_pixels_ref_es += num_pixel_above_threshold_es
                    # do the same stuff for ED phase
                    uncertain_ed_idx = pixel_std_ed >= referral_threshold
                    # collect indices of pixels with high uncertainty AND for which we predicted a positive label
                    if ref_positives_only:
                        pred_labels_pos_idx_ed = pred_labels_cls_ed == 1
                        c = np.count_nonzero(pred_labels_pos_idx_ed)
                        uncertain_ed_idx = np.logical_and(uncertain_ed_idx, pred_labels_pos_idx_ed)
                        total_pixels_positive += c
                        slice_pixels_pos_ed += c

                    f1_ed, pr_ed, rc_ed, true_pos_ed, false_pos_ed, false_neg_ed = eval_referral(errors_ed_idx,
                                                                                                 uncertain_ed_idx)
                    num_pixel_above_threshold_ed = np.count_nonzero(uncertain_ed_idx)
                    slice_total_pixels_ref_ed += num_pixel_above_threshold_ed
                    # temporary, number of all pixels equal to 0
                    # IMPORTANT: this is the EXPERT, setting the high uncertainty pixels to the ground truth labels
                    if slice_id in refer_slices_es:
                        pred_labels_es[cls, uncertain_es_idx, slice_id] = true_labels_cls_es[uncertain_es_idx]
                    if slice_id in refer_slices_ed:
                        pred_labels_ed[cls, uncertain_ed_idx, slice_id] = true_labels_cls_ed[uncertain_ed_idx]
                    errors_es_filtered = np.count_nonzero(pred_labels_cls_es != true_labels_cls_es)
                    errors_ed_filtered = np.count_nonzero(pred_labels_cls_ed != true_labels_cls_ed)
                    if verbose:
                        print("ES: refer slice {}. #improvements {}".format(slice_id + 1,
                                                                                errors_es-errors_es_filtered))
                    if verbose:
                        print("ED: refer slice {}. #improvements {}".format(slice_id + 1,
                                                                                errors_ed-errors_ed_filtered))
                    # how many seg-errors did we improve (can't be negative)
                    # print("Error before/after {}  {}".format(errors_es, errors_es_filtered))
                    try:
                        if errors_es_filtered == errors_es:
                            error_es_reduction = 0
                        else:
                            error_es_reduction = (1 - (errors_es_filtered / float(errors_es))) * 100
                    except ZeroDivisionError:
                        error_es_reduction = 0.
                    try:
                        if errors_ed_filtered == errors_ed:
                            error_ed_reduction = 0
                        else:
                            error_ed_reduction = (1 - (errors_ed_filtered / float(errors_ed))) * 100
                    except ZeroDivisionError:
                        error_ed_reduction = 0.
                    # ES
                    self.referral_stats[0, cls - 1, :, slice_id] = \
                        np.array([0, error_es_reduction, num_true_labels_es, num_pixel_above_threshold_es,
                                  errors_es, errors_es_filtered,
                                  f1_es, pr_es, rc_es, true_pos_es, false_pos_es, false_neg_es, 0, 0])
                    # ED
                    self.referral_stats[1, cls - 1, :, slice_id] = \
                        np.array([0, error_ed_reduction, num_true_labels_ed, num_pixel_above_threshold_ed,
                                  errors_ed, errors_ed_filtered,
                                  f1_ed, pr_ed, rc_ed, true_pos_ed, false_pos_ed, false_neg_ed, 0, 0])

                    # find the union of all the pixel locations that we're uncertain about for this slice:
                    if slice_uncertainty_idx_es is None:
                        slice_uncertainty_idx_es = uncertain_es_idx
                    else:
                        slice_uncertainty_idx_es = np.logical_or(slice_uncertainty_idx_es, uncertain_es_idx)
                    if slice_uncertainty_idx_ed is None:
                        slice_uncertainty_idx_ed = uncertain_ed_idx
                    else:
                        slice_uncertainty_idx_ed = np.logical_or(slice_uncertainty_idx_ed, uncertain_ed_idx)
                # IMPORTANT: we corrected slice-voxels per class, but we need to correct the voxels over all classes.
                # E.g.: if voxel x was predicted as LV, but must be BG, we need to correct the voxel prediction
                #       for LV (which we did above) BUT also for BG, which we haven't done
                # es_pred_labels = self.b_pred_labels[:self.num_of_classes, :, :, slice_id]
                # es_true_labels = self.b_labels[:self.num_of_classes, :, :, slice_id]
                # es_pred_labels[:, slice_uncertainty_idx_es] = es_true_labels[:, slice_uncertainty_idx_es]
                # self.b_pred_labels[:self.num_of_classes, :, :, slice_id] = es_pred_labels
                # # same for ED
                # ed_pred_labels = self.b_pred_labels[self.num_of_classes:, :, :, slice_id]
                # ed_true_labels = self.b_labels[self.num_of_classes:, :, :, slice_id]
                # ed_pred_labels[:, slice_uncertainty_idx_ed] = ed_true_labels[:, slice_uncertainty_idx_ed]
                # self.b_pred_labels[self.num_of_classes:, :, :, slice_id] = ed_pred_labels

                # end of slice
                es_ref_count = np.count_nonzero(slice_uncertainty_idx_es)
                # we store the total number of referred pixels for this class, which we store redundantly
                # i.e. it's the same for all classes.
                self.referral_stats[0, :, 12, slice_id] = es_ref_count
                self.referral_stats[0, :, 13, slice_id] = slice_num_pixel_uncertain_es
                try:
                    ref_perc_es = (slice_total_pixels_ref_es / float(slice_pixels_pos_es)) * 100
                except ZeroDivisionError:
                    ref_perc_es = 0
                # we didn't set the referral perc. above, so we do it here for all classes at the end of the slice
                # which means, the info is redundant over the classes and should be interpreted as per slice %
                self.referral_stats[0, :, 0, slice_id] = ref_perc_es
                ed_ref_count = np.count_nonzero(slice_uncertainty_idx_ed)
                self.referral_stats[1, :, 12, slice_id] = ed_ref_count
                self.referral_stats[1, :, 13, slice_id] = slice_num_pixel_uncertain_ed
                # increase the total referral counter
                total_pixels_referred += es_ref_count + ed_ref_count
                try:
                    ref_perc_ed = (slice_total_pixels_ref_ed / float(slice_pixels_pos_ed)) * 100
                except ZeroDivisionError:
                    ref_perc_ed = 0
                self.referral_stats[1, :, 0, slice_id] = ref_perc_ed
                # print("Slice {} ES/ED Ref% {:.2f}/{:.2f}".format(slice_id + 1, ref_perc_es, ref_perc_ed))
            # compute slice accuracy/hd after referral
            slice_acc_after, slice_hd_after = self.compute_slice_accuracy(slice_idx=slice_id, compute_hd=True,
                                                                          store_results=False)
            self.b_ref_hd_slices[0, :, slice_id] = slice_hd_after[:self.num_of_classes]  # ES
            self.b_ref_hd_slices[1, :, slice_id] = slice_hd_after[self.num_of_classes:]  # ED
            self.b_ref_acc_slices[0, :, slice_id] = slice_acc_after[:self.num_of_classes]  # ES
            self.b_ref_acc_slices[1, :, slice_id] = slice_acc_after[self.num_of_classes:]  # ED

        # pred_labels_es and ed are deepcopies of self.b_pred_labels. we changed them based on the uncertainty
        # maps. in the final step we "copy" the adjusted pred_labels_es/ed back
        self.b_pred_labels[:self.num_of_classes, :, :, :] = pred_labels_es
        self.b_pred_labels[self.num_of_classes:, :, :, :] = pred_labels_ed
        # referral_stats with shape [2, 4classes, 12, #slices]. We average over slices
        errors_es = np.sum(self.referral_stats[0, :, 4, :])
        errors_es_after = np.sum(self.referral_stats[0, :, 5, :])
        errors_ed = np.sum(self.referral_stats[1, :, 4, :])
        errors_ed_after = np.sum(self.referral_stats[1, :, 5, :])

        # average over classes and slices
        rel_perc_es = np.mean(self.referral_stats[0, :, 0, :])
        rel_perc_ed = np.mean(self.referral_stats[1, :, 0, :])

        print("{}: Ref% ES/ED {:.2f}/{:.2f} - #Errors before/after reduction "
              "ES: {}/{} ED: {}/{}".format(self.b_image_name, rel_perc_es, rel_perc_ed, errors_es, errors_es_after,
                                           errors_ed, errors_ed_after))

    def set_stddev_map(self, slice_std, u_threshold=0.01):
        """
            Important: we assume slice_std is a numpy array with shape [num_classes, width, height]

        """
        self.b_stddev_map[:, :, :, self.slice_counter] = slice_std
        self.compute_img_slice_uncertainty_stats(u_type="stddev", u_threshold=u_threshold)

    def set_bald_map(self, slice_bald, u_threshold=0.01):
        """
        Important: we assume slice_bald is a numpy array with shape [2, width, height].
        First dim has shape two because we need maps for ES and ED phase
        :param slice_bald: contains the BALD values per pixel (for one slice) [2, width, height]
        :return:
        """

        self.b_bald_map[:, :, :, self.slice_counter] = slice_bald
        self.compute_img_slice_uncertainty_stats(u_type="bald", u_threshold=u_threshold)

    def get_accuracy(self, compute_hd=False, compute_seg_errors=False, do_filter=True,
                     compute_slice_metrics=False):
        """

            Compute dice coefficients for the complete 3D volume
            (1) self.b_label contains the ground truth: [num_of_classes, x, y, z]
            (2) self.b_pred_labels contains predicted softmax scores [2, x, y, z]
                                                                    (0=ES, 1=ED)

            compute_hd: Boolean indicating whether or not to compute Hausdorff distance
            compute_seg_errors: boolean indicating whether or not to compute segmentation errors

            :returns: always: 2 numpy arrays dice [8classes], hd [8classes]
                      if we compute seg-errors then in addition: [8classes, #slices] -> errors per class per slice
        """
        dices = np.zeros(2 * self.num_of_classes)
        hausdff = np.zeros(2 * self.num_of_classes)
        num_of_slices = self.b_labels.shape[3]
        if compute_slice_metrics:
            dice_slices = np.zeros((2 * self.num_of_classes, num_of_slices))
            hd_slices = np.zeros((2 * self.num_of_classes, num_of_slices))
        if compute_seg_errors:
            # if we compute seg-errors then shape of results is [2, 4classes, #slices]
            seg_errors = np.zeros((2, self.num_of_classes, num_of_slices))
        for cls in np.arange(self.num_of_classes):
            if do_filter:
                # use 6-connected components to filter out blobs that are not relevant for segmentation
                self.b_pred_labels[cls] = filter_connected_components(self.b_pred_labels[cls], cls, verbose=False)
                self.b_pred_labels[cls + self.num_of_classes] = \
                    filter_connected_components(self.b_pred_labels[cls + self.num_of_classes], cls, verbose=False,
                                                rank=None)

            dices[cls] = dice_coefficient(self.b_labels[cls, :, :, :], self.b_pred_labels[cls, :, :, :])
            dices[cls + self.num_of_classes] = dice_coefficient(self.b_labels[cls + self.num_of_classes, :, :, :],
                                                                self.b_pred_labels[cls + self.num_of_classes, :, :, :])
            if compute_slice_metrics:
                for s_idx in np.arange(num_of_slices):
                    dice_slices[cls, s_idx] = dice_coefficient(self.b_labels[cls, :, :, s_idx],
                                                               self.b_pred_labels[cls, :, :, s_idx])
                    dice_slices[cls + self.num_of_classes, s_idx] = \
                        dice_coefficient(self.b_labels[cls + self.num_of_classes, :, :, s_idx],
                                         self.b_pred_labels[cls + self.num_of_classes, :, :, s_idx])
            if compute_hd:
                # only compute distance if both contours are actually in images
                if 0 != np.count_nonzero(self.b_pred_labels[cls, :, :, :]) and \
                        0 != np.count_nonzero(self.b_labels[cls, :, :, :]):
                    hausdff[cls] = hd(self.b_pred_labels[cls, :, :, :], self.b_labels[cls, :, :, :],
                                      voxelspacing=self.b_new_spacing, connectivity=1)
                else:
                    hausdff[cls] = 0.
                if 0 != np.count_nonzero(self.b_pred_labels[cls + self.num_of_classes, :, :, :]) and \
                        0 != np.count_nonzero(self.b_labels[cls + self.num_of_classes, :, :, :]):
                    hausdff[cls + self.num_of_classes] = \
                        hd(self.b_pred_labels[cls + self.num_of_classes, :, :, :],
                           self.b_labels[cls + self.num_of_classes, :, :, :],
                           voxelspacing=self.b_new_spacing, connectivity=1)
                else:
                    hausdff[cls + self.num_of_classes] = 0.

                if compute_slice_metrics:
                    for s_idx in np.arange(num_of_slices):
                        # only compute distance if both contours are actually in images
                        if 0 != np.count_nonzero(self.b_pred_labels[cls, :, :, s_idx]) and \
                                0 != np.count_nonzero(self.b_labels[cls, :, :, s_idx]):
                            hd_slices[cls, s_idx] = hd(self.b_pred_labels[cls, :, :, s_idx], self.b_labels[cls, :, :, s_idx],
                                                voxelspacing=ACDC2017TestHandler.new_voxel_spacing, connectivity=1)
                        else:
                            hd_slices[cls] = 0
                        if 0 != np.count_nonzero(self.b_pred_labels[cls + self.num_of_classes, :, :, s_idx]) and \
                                0 != np.count_nonzero(self.b_labels[cls + self.num_of_classes, :, :, s_idx]):
                            hd_slices[cls + self.num_of_classes, s_idx] = \
                                hd(self.b_pred_labels[cls + self.num_of_classes, :, :, s_idx],
                                   self.b_labels[cls + self.num_of_classes, :, :, s_idx],
                                   voxelspacing=ACDC2017TestHandler.new_voxel_spacing, connectivity=1)
            if compute_seg_errors:
                true_labels_cls_es = self.b_labels[cls]
                true_labels_cls_ed = self.b_labels[cls + self.num_of_classes]
                pred_labels_cls_es = self.b_pred_labels[cls]
                pred_labels_cls_ed = self.b_pred_labels[cls + self.num_of_classes]
                errors_es = pred_labels_cls_es != true_labels_cls_es
                errors_ed = pred_labels_cls_ed != true_labels_cls_ed
                errors_es = np.count_nonzero(np.reshape(errors_es, (-1, errors_es.shape[2])), axis=0)
                errors_ed = np.count_nonzero(np.reshape(errors_ed, (-1, errors_ed.shape[2])), axis=0)
                seg_errors[0, cls] = errors_es
                seg_errors[1, cls] = errors_ed

        if compute_seg_errors:
            if not compute_slice_metrics:
                return dices, hausdff, seg_errors
            else:
                return dices, hausdff, seg_errors, dice_slices, hd_slices
        else:
            if not compute_slice_metrics:
                return dices, hausdff
            else:
                return dices, hausdff, dice_slices, hd_slices

    def compute_slice_accuracy(self, slice_idx=None, compute_hd=False, store_results=True):
        """

            Compute dice coefficients for the complete 3D volume
            (1) self.b_label contains the ground truth: [num_of_classes, x, y, z]
            (2) self.b_pred_labels contains predicted softmax scores [2, x, y, z]
                                                                    (0=ES, 1=ED)

            compute_hd: Boolean indicating whether or not to compute Hausdorff distance

            store_results:
        """
        if slice_idx is None:
            slice_idx = self.slice_counter
        # Remember self.num_of_classes is 4 in the AC-DC case and not 8 (4-ES and 4-ED)
        dices = np.zeros(2 * self.num_of_classes)
        hausdff = np.zeros(2 * self.num_of_classes)
        for cls in np.arange(self.num_of_classes):

            dices[cls] = dice_coefficient(self.b_labels[cls, :, :, slice_idx],
                                          self.b_pred_labels[cls, :, :, slice_idx])
            dices[cls + self.num_of_classes] = \
                dice_coefficient(self.b_labels[cls + self.num_of_classes, :, :, slice_idx],
                                 self.b_pred_labels[cls + self.num_of_classes, :, :, slice_idx])
            if compute_hd:
                # only compute distance if both contours are actually in images
                if 0 != np.count_nonzero(self.b_pred_labels[cls, :, :, slice_idx]) and \
                        0 != np.count_nonzero(self.b_labels[cls, :, :, slice_idx]):
                    hausdff[cls] = hd(self.b_pred_labels[cls, :, :, slice_idx], self.b_labels[cls, :, :, slice_idx],
                                      voxelspacing=ACDC2017TestHandler.new_voxel_spacing, connectivity=1)
                else:
                    hausdff[cls] = 0
                if 0 != np.count_nonzero(self.b_pred_labels[cls + self.num_of_classes, :, :, slice_idx]) and \
                        0 != np.count_nonzero(self.b_labels[cls + self.num_of_classes, :, :, slice_idx]):
                    hausdff[cls + self.num_of_classes] = \
                        hd(self.b_pred_labels[cls + self.num_of_classes, :, :, slice_idx],
                           self.b_labels[cls + self.num_of_classes, :, :, slice_idx],
                           voxelspacing=ACDC2017TestHandler.new_voxel_spacing, connectivity=1)
                else:
                    hausdff[cls + self.num_of_classes] = 0
        if store_results:
            self.b_hd_slices[0, :, slice_idx] = hausdff[:self.num_of_classes]  # ES
            self.b_hd_slices[1, :, slice_idx] = hausdff[self.num_of_classes:]  # ED
            self.b_acc_slices[0, :, slice_idx] = dices[:self.num_of_classes]  # ES
            self.b_acc_slices[1, :, slice_idx] = dices[self.num_of_classes:]  # ED

        return dices, hausdff

    def filter_u_maps(self, u_threshold=0.):

        self.b_filtered_stddev_map = copy.deepcopy(self.b_stddev_map)
        for cls in np.arange(0, self.num_of_classes * 2):
            if cls != 0 and cls != 4:
                u_3dmaps_cls = self.b_filtered_stddev_map[cls]
                # set all uncertainties below threshold to zero
                u_3dmaps_cls[u_3dmaps_cls < u_threshold] = 0
                self.b_filtered_stddev_map[cls] = filter_connected_components(u_3dmaps_cls, threshold=u_threshold)

        self.b_filtered_umap = True

    def save_filtered_u_maps(self, output_dir, u_threshold):
        """

        :param output_dir: is actually the exper_id e.g. 20180503_13_22_18_dcnn_mc_f2p005....
        :param u_threshold:
        :return:
        """
        if u_threshold <= 0.:
            raise ValueError("ERROR - u_threshold must be greater than zero. ({:.2f})".format(u_threshold))
        umap_output_dir = os.path.join(self.config.root_dir, os.path.join(output_dir, config.u_map_dir))
        if not os.path.isdir(umap_output_dir):
            os.makedirs(umap_output_dir)
        u_threshold = str(u_threshold).replace(".", "_")
        file_name = self.b_image_name + "_filtered_umaps" + u_threshold + ".npz"
        file_name = os.path.join(umap_output_dir, file_name)
        try:
            np.savez(file_name, filtered_umap=self.b_filtered_stddev_map)
        except IOError:
            print("ERROR - Unable to save filtered umaps file {}".format(file_name))

    def save_pred_labels(self, output_dir, u_threshold=0., mc_dropout=False, used_entropy=False):
        """

        :param output_dir: is actually the exper_id e.g. 20180503_13_22_18_dcnn_mc_f2p005....
        :param u_threshold: indicating whether we store a filtered label mask...if <> 0

        :param mc_dropout: boolean indicating whether we obtained the predictions by means of mc-dropout
        :param used_entropy: did we use entropy maps to generate the predicted labels (filtering/referral)?
                if so, we need to distinguish the predictions from the "other" predictions
        saves the predicted label maps for an image. shape [8classes, width, height, #slices]
        """
        pred_lbl_output_dir = os.path.join(self.config.root_dir, os.path.join(output_dir, config.pred_lbl_dir))
        if not os.path.isdir(pred_lbl_output_dir):
            os.makedirs(pred_lbl_output_dir)
        if u_threshold != 0:
            u_threshold = str(u_threshold).replace(".", "_")
            if mc_dropout:
                u_threshold = "_mc" + u_threshold
            else:
                u_threshold = "_" + u_threshold
            if used_entropy:
                u_threshold += "_ep"

            file_name = self.b_image_name + "_filtered_pred_labels" + u_threshold + ".npz"
        else:
            file_name = self.b_image_name + "_pred_labels"
            if mc_dropout:
                file_name += "_mc"
            if used_entropy:
                file_name += "_ep"

            file_name += ".npz"
        file_name = os.path.join(pred_lbl_output_dir, file_name)

        try:
            np.savez(file_name, pred_labels=self.b_pred_labels)
        except IOError:
            print("ERROR - Unable to save predicted labels file {}".format(file_name))

    def save_referred_slices(self, output_dir, referral_threshold=0., slice_filter_type=None,
                             referred_slices=None):
        """

        :param output_dir:
        :param referral_threshold:
        :param slice_filter_type: if not None (but M,MD or MS) this means we're only referring specific slices
        :param referred_slices: vector of shape #slices, boolean indicated whether that slice was referred yes/no
        :return:
        """
        output_dir = os.path.join(self.config.root_dir, os.path.join(output_dir, config.pred_lbl_dir))
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        str_referral_threshold = str(referral_threshold).replace(".", "_")
        str_referral_threshold = "_mc" + str_referral_threshold
        if slice_filter_type is not None:
            str_referral_threshold += "_" + slice_filter_type
        file_name = self.b_image_name + "_referred_slices" + str_referral_threshold + ".npz"
        file_name = os.path.join(output_dir, file_name)
        # we don't overwrite existing predictions we obtained with the base model, so the non-referral predictions!
        try:
            np.savez(file_name, referred_slices=referred_slices)
        except IOError:
            print("ERROR - Unable to save predicted labels file {}".format(file_name))

    def save_pred_probs(self, output_dir, mc_dropout=False):
        """
        Save a 3D volume of the mean softmax probabilities
        :return:
        """
        pred_prob_output_dir = os.path.join(self.config.root_dir, os.path.join(output_dir, config.pred_lbl_dir))
        if not os.path.isdir(pred_prob_output_dir):
            os.makedirs(pred_prob_output_dir)
        if mc_dropout:
            file_name = self.b_image_name + "_pred_probs_mc.npz"
        else:
            file_name = self.b_image_name + "_pred_probs.npz"

        file_name = os.path.join(pred_prob_output_dir, file_name)
        try:
            np.savez(file_name, pred_probs=self.b_pred_probs)
        except IOError:
            print("ERROR - Unable to save softmax probabilities to file {}".format(file_name))

    def save_batch_img_to_files(self, slice_range=None, save_dir=None, wo_padding=True):

        if save_dir is None:
            save_dir = self.config.data_dir
        if slice_range is None:
            # basically save all slices
            slice_range = np.arange(0, self.b_image.shape[3])

        for i in slice_range:
            # each input contains 2 images: 0=ED and 1=ES
            for phase in np.arange(2):
                if phase == 0:
                    suffix = "es"
                else:
                    suffix = "ed"
                filename_img = os.path.join(save_dir, str(i+1).zfill(2) + "_img_ph_"
                                            + suffix + ".nii")
                img = self.b_image[phase, :, :, :]
                if wo_padding:
                    img = img[ACDC2017TestHandler.pad_size:-ACDC2017TestHandler.pad_size,
                              ACDC2017TestHandler.pad_size:-ACDC2017TestHandler.pad_size, :]
                new_spacing = tuple((ACDC2017TestHandler.new_voxel_spacing, ACDC2017TestHandler.new_voxel_spacing,
                                     self.b_orig_spacing[2]))
                # we need to swap the axis because the image is 3D only
                write_numpy_to_image(img, filename=filename_img, swap_axis=True, spacing=new_spacing)
                cls_offset = phase * 4
                for cls in np.arange(self.num_of_classes):
                    if cls != 0 and cls != 4:
                        cls_lbl = self.b_labels[cls_offset + cls]
                        if wo_padding:
                            # do nothing
                            pass
                        else:
                            cls_lbl = np.pad(cls_lbl, ((ACDC2017TestHandler.pad_size, ACDC2017TestHandler.pad_size - 1),
                                                       (ACDC2017TestHandler.pad_size, ACDC2017TestHandler.pad_size - 1),
                                                       (0, 0)),
                                             'constant', constant_values=(0,)).astype(
                                ACDC2017TestHandler.pixel_dta_type)

                        filename_lbl = os.path.join(save_dir, str(i + 1).zfill(2) + "_lbl_ph"
                                                    + suffix + "_cls" + str(cls) + ".nii")
                        write_numpy_to_image(cls_lbl.astype("float32"), filename=filename_lbl, swap_axis=True,
                                             spacing=new_spacing)
                        # if object that store predictions is non None we save them as well for analysis
                        if self.b_pred_labels is not None:
                            pred_cls_lbl = self.b_pred_labels[cls_offset+cls]
                            filename_lbl = os.path.join(save_dir, str(i + 1).zfill(2) + "_pred_lbl_ph"
                                                        + suffix + "_cls" + str(cls) + ".nii")
                            write_numpy_to_image(pred_cls_lbl.astype("float32"), filename=filename_lbl, swap_axis=True,
                                                 spacing=new_spacing)

    def compute_cardiac_indices(self):

        for idx, label_volume in enumerate(self.labels):
            patient_id = self.img_file_names[idx]
            # assuming label_volume has shape [8, w, h, #slices]
            volume_es = label_volume[:4]
            volume_ed = label_volume[4:]
            spacings = self.spacings[idx]
            # LV indices, class-index 3 = LV
            lv_esv, lv_edv, lv_ef = compute_ejection_fraction(volume_es[3], volume_ed[3], spacings)
            # RV indices, class-index 3 = RV
            rv_esv, rv_edv, rv_ef = compute_ejection_fraction(volume_es[1], volume_ed[1], spacings)
            self.cardiac_indices[patient_id] = np.array([[lv_esv, lv_edv, lv_ef], [rv_esv, rv_edv, rv_ef]])

    @staticmethod
    def get_testset_instance(config_obj, fold_id, load_train=False, load_val=True, batch_size=None, use_cuda=True):
        return ACDC2017TestHandler(exper_config=config_obj,
                                   search_mask=config_obj.dflt_image_name + ".mhd",
                                   fold_ids=[fold_id],
                                   debug=False, batch_size=batch_size,
                                   use_cuda=use_cuda,
                                   load_train=load_train, load_val=load_val, use_iso_path=True)


# print("Root path {}".format(os.environ.get("REPO_PATH")))
# dataset = ACDC2017TestHandler(exper_config=config, search_mask=config.dflt_image_name + ".mhd", fold_ids=[0],
#                              debug=False, batch_size=4)

# del dataset
