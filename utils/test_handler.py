import sys
import numpy as np
from config.config import config
import os
import glob
from tqdm import tqdm
from in_out.read_save_images import load_mhd_to_numpy, write_numpy_to_image
from utils.dice_metric import dice_coefficient
from utils.img_sampling import resample_image_scipy
from torch.autograd import Variable
from utils.medpy_metrics import hd
from utils.post_processing import filter_connected_components
import torch
if "/home/jogi/.local/lib/python2.7/site-packages" in sys.path:
    sys.path.remove("/home/jogi/.local/lib/python2.7/site-packages")
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import generate_binary_structure


class ACDC2017TestHandler(object):

    train_path = "train"
    val_path = "validate"
    image_path = "images"
    label_path = "reference"

    pixel_dta_type = 'float32'
    pad_size = config.pad_size
    new_voxel_spacing = 1.4

    def __init__(self, exper_config, search_mask=None, nclass=4, load_func=load_mhd_to_numpy,
                 fold_ids=[1], debug=False, use_cuda=True, batch_size=1):

        self.config = exper_config
        self.data_dir = os.path.join(self.config.root_dir, exper_config.data_dir)
        self.search_mask = search_mask
        self.num_of_classes = nclass
        self.fold_ids = fold_ids
        self.load_func = load_func
        self.abs_path_fold = os.path.join(self.data_dir, "fold")
        self.images = []
        # storing a concatenated filename used as an identification during testing and used for the output
        # filename of the figures printed during test evaluation e.g. patien007_frame001_frame007
        self.img_file_names = []
        self.labels = []
        self.spacings = []
        self.b_pred_labels = None
        self.b_pred_probs = None
        self.b_stddev_map = None
        self.b_bald_map = None  # [2, width, height, #slices] one map for each heart phase ES/ED
        self.b_image = None
        self.b_image_id = None  # store the filename from "above" used as identification during testing
        self.b_labels = None
        self.b_new_spacing = None
        self.b_orig_spacing = None
        self.b_seg_errors = None
        self.b_uncertainty_stats = None
        # accuracy and hausdorff measure for slices (per image)
        self.b_hd_slices = None
        self.b_acc_slices = None
        self.debug = debug
        self._set_pathes()

        self.slice_counter = 0
        self.num_of_models = 1
        self.use_cuda = use_cuda
        self.batch_size = batch_size * 2  # multiply by 2 because each "image" contains an ED and ES part
        # self.out_img = np.zeros((self.num_classes, self.image.shape[0], self.image.shape[1], self.image.shape[2]))
        # self.overlay_images = {'ED_RV': np.zeros((self.image.shape[0], self.image.shape[1], self.image.shape[2])),
        #                        'ED_LV': np.zeros((self.image.shape[0], self.image.shape[1], self.image.shape[2]))}

        file_list = self._get_file_lists()
        self._load_file_list(file_list)

    def _set_pathes(self):

        if self.debug:
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
            search_mask_img = os.path.join(self.train_path, self.search_mask)
            print("INFO - Testhandler - >>> Search for {} <<<".format(search_mask_img))
            for train_file in glob.glob(search_mask_img):
                ref_file = train_file.replace(ACDC2017TestHandler.image_path, ACDC2017TestHandler.label_path)
                file_list.append(tuple((train_file, ref_file)))

            # get validation images and labels
            search_mask_img = os.path.join(self.val_path, self.search_mask)
            for val_file in glob.glob(search_mask_img):
                ref_file = val_file.replace(ACDC2017TestHandler.image_path, ACDC2017TestHandler.label_path)
                file_list.append(tuple((val_file, ref_file)))

        print("INFO - File list contains {} files".format(len(file_list)))

        return file_list

    def _load_file_list(self, file_list):

        file_list.sort()
        batch_file_list = file_list[:self.batch_size]

        for idx in tqdm(np.arange(0, len(batch_file_list), 2)):
            # tuple contains [0]=train file name and [1] reference file name
            img_file, ref_file = file_list[idx]
            # first frame is always the end-systolic MRI scan, filename ends with "1"
            mri_scan_es, origin, spacing = self.load_func(img_file, data_type=ACDC2017TestHandler.pixel_dta_type,
                                                          swap_axis=True)
            es_file_name = os.path.splitext(os.path.basename(img_file))[0]
            print("{} - {}".format(idx, img_file))
            self.spacings.append(spacing)
            mri_scan_es = self._preprocess(mri_scan_es, spacing, poly_order=3, do_pad=True)
            # print("INFO - Loading ES-file {}".format(img_file))
            reference_es, _, _ = self.load_func(ref_file, data_type=ACDC2017TestHandler.pixel_dta_type,
                                                swap_axis=True)
            reference_es = self._preprocess(reference_es, spacing, poly_order=0, do_pad=False)

            # do the same for the End-diastole pair of images
            img_file, ref_file = file_list[idx+1]
            ed_frame_num = (os.path.splitext(os.path.basename(img_file))[0]).split("_")[1]
            self.img_file_names.append(es_file_name + "_" + ed_frame_num)
            print("{} - {}".format(idx+1, img_file))
            mri_scan_ed, _, _ = self.load_func(img_file, data_type=ACDC2017TestHandler.pixel_dta_type,
                                               swap_axis=True)
            mri_scan_ed = self._preprocess(mri_scan_ed, spacing, poly_order=3, do_pad=True)

            reference_ed, _, _ = self.load_func(ref_file, data_type=ACDC2017TestHandler.pixel_dta_type,
                                                swap_axis=True)
            reference_ed = self._preprocess(reference_ed, spacing, poly_order=0, do_pad=False)
            # concatenate both images for further processing
            images = np.concatenate((np.expand_dims(mri_scan_ed, axis=0),
                                     np.expand_dims(mri_scan_es, axis=0)))
            # same concatenation for the label files of ED and ES
            labels = self._split_class_labels(reference_ed, reference_es)
            self.images.append(images)
            self.labels.append(labels)
        print("INFO - Successfully loaded {} ED/ES patient pairs".format(len(self.images)))

    def _preprocess(self, image, spacing, poly_order=3, do_pad=True):

        # print("Spacings ", spacing)
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
        labels_per_class = np.zeros((2 * self.num_of_classes, labels_ed.shape[0], labels_ed.shape[1],
                                     labels_ed.shape[2]))
        for cls_idx in np.arange(self.num_of_classes):
            # store ED class labels in first 4 positions of dim-1
            labels_per_class[cls_idx, :, :, :] = (labels_ed == cls_idx).astype('int16')
            # sotre ES class labels in positions 4-7 of dim-1
            labels_per_class[cls_idx + self.num_of_classes, :, :, :] = (labels_es == cls_idx).astype('int16')

        return labels_per_class

    def batch_generator(self, image_num=0, use_labels=True, use_volatile=True):
        """
            Remember that self.image is a list, containing images with shape [2, height, width, depth]
            Object self.labels is also a list but with shape [8, height, width, depth]

        """
        b_label = None
        self.slice_counter = -1
        self.b_image = self.images[image_num]
        num_of_slices = self.b_image.shape[3]
        self.b_image_id = self.img_file_names[image_num]
        self.b_orig_spacing = self.spacings[image_num]
        self.b_new_spacing = tuple((ACDC2017TestHandler.new_voxel_spacing, ACDC2017TestHandler.new_voxel_spacing,
                                    self.b_orig_spacing[2]))
        # store the segmentation errors. shape [#slices, #classes]
        self.b_seg_errors = np.zeros((self.b_image.shape[3], self.num_of_classes * 2)).astype(np.int)
        # currently measuring four values per slice - cardiac phase (2): total uncertainty, number of pixels with
        # uncertainty (>eps), number of pixels with uncertainty above u_threshold, number of connected components in
        # 2D slice (5)
        self.b_uncertainty_stats = {'stddev': np.zeros((2, 4, num_of_slices)),
                                    'bald': np.zeros((2, 4, num_of_slices)), "u_threshold": 0.}
        self.b_acc_slices = np.zeros((2, self.num_of_classes, num_of_slices))
        self.b_hd_slices = np.zeros((2, self.num_of_classes, num_of_slices))
        if use_labels:
            self.b_labels = self.labels[image_num]
            self.b_pred_probs = np.zeros_like(self.b_labels)
            self.b_pred_labels = np.zeros_like(self.b_labels)
        # TO DO: actually it could happen now that b_uncertainty has undefined shape (because b_labels is not used
        self.b_stddev_map = np.zeros_like(self.b_labels)
        # b_bald_map shape: [2, width, height, #slices]
        self.b_bald_map = np.zeros((2, self.b_labels.shape[1], self.b_labels.shape[2], self.b_labels.shape[3]))

        for slice in np.arange(num_of_slices):
            b_image = Variable(torch.FloatTensor(torch.from_numpy(self.b_image[:, :, :, slice]).float()),
                               volatile=use_volatile)
            b_image = b_image.unsqueeze(0)
            if self.b_labels is not None:
                b_label = Variable(torch.FloatTensor(torch.from_numpy(self.b_labels[:, :, :, slice]).float()),
                                   volatile=use_volatile)
                b_label = b_label.unsqueeze(0)
            if self.use_cuda:
                b_image = b_image.cuda()
                if self.b_labels is not None:
                    b_label = b_label.cuda()
            self.slice_counter += 1
            yield b_image, b_label

    def set_pred_labels(self, pred_probs, pred_stddev=None, u_threshold=0., verbose=False, do_filter=True):
        """

        :param pred_probs: [1, num_of_classes(8), width, height]
        :param pred_stddev: [num_of_classes(8), width, heigth]
        :param u_threshold: only used if pred_stddev object is non None. Used to filter prediction errors based
        on standard deviation (uncertainty)
        Also note that object self.b_labels has dims [num_classes(8), width, height, slices]

        :param do_filter: filter predicted labels on connected components. This is an important post processing step
                          especially when doing this at the end for the 3D label object
        :param verbose:

        :return:
        """
        if isinstance(pred_probs.data, torch.cuda.FloatTensor) or isinstance(pred_probs.data, torch.FloatTensor):
            pred_probs = pred_probs.data.cpu().numpy()

        # we also set the u_threshold here for the computation of the uncertainty stats later
        self.b_uncertainty_stats["u_threshold"] = u_threshold

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
            true_labels_cls_es = self.b_labels[cls, :, :, self.slice_counter]
            true_labels_cls_ed = self.b_labels[cls + self.num_of_classes, :, :, self.slice_counter]
            dice_es_before = dice_coefficient(true_labels_cls_es, pred_labels_cls_es)
            dice_ed_before = dice_coefficient(true_labels_cls_ed, pred_labels_cls_ed)
            errors_es = pred_labels_cls_es != true_labels_cls_es
            errors_ed = pred_labels_cls_ed != true_labels_cls_ed
            if pred_stddev is not None and u_threshold > 0.:
                pixel_std_es = np.copy(pred_stddev[cls])
                pixel_std_ed = np.copy(pred_stddev[cls + self.num_of_classes])
                # IMPORTANT!!! - THE THRESHOLD MUST BE EQUAL TO 0, ONLY IF WE WANT TO USE THE THRESHOLD TO CORRECT
                # LABELS WITH HIGH UNCERTAINTY, THE THRESOLD SHOULD BE ABOVE 0.
                error_es_idx = pixel_std_es > u_threshold
                error_ed_idx = pixel_std_ed > u_threshold

                # little subtlety, if bg class, the most common class label is 1, for all other classes it's 0
                if cls == 0:
                    pred_labels_cls_es[error_es_idx] = 1
                    pred_labels_cls_ed[error_ed_idx] = 1
                else:
                    pred_labels_cls_es[error_es_idx] = 0.
                    pred_labels_cls_ed[error_ed_idx] = 0.
                errors_es_filtered = pred_labels_cls_es != true_labels_cls_es
                errors_ed_filtered = pred_labels_cls_ed != true_labels_cls_ed
                self.b_seg_errors[self.slice_counter, cls] = np.count_nonzero(errors_es_filtered)
                self.b_seg_errors[self.slice_counter, cls + self.num_of_classes] = np.count_nonzero(errors_ed_filtered)
                dice_es_after = dice_coefficient(true_labels_cls_es, pred_labels_cls_es)
                dice_ed_after = dice_coefficient(true_labels_cls_ed, pred_labels_cls_ed)
                if cls != 0 and verbose:
                    print("Slice {} - Class {}: before/after ES: errors: {}/{} dice: {:.2f}/{:.2f} "
                          "ED errors {}/{} dice: {:.2f}/{:.2f}".format(self.slice_counter+1, cls,
                                                                       np.count_nonzero(errors_es),
                                                                       np.count_nonzero(errors_es_filtered),
                                                                       dice_es_before, dice_es_after,
                                                                       np.count_nonzero(errors_ed),
                                                                       np.count_nonzero(errors_ed_filtered),
                                                                       dice_ed_before, dice_ed_after))
            else:
                # not filtering high uncertainty pixels
                # store # of segmentation errors that we made per class and slice
                self.b_seg_errors[self.slice_counter, cls] = np.count_nonzero(errors_es)
                self.b_seg_errors[self.slice_counter, cls + self.num_of_classes] = np.count_nonzero(errors_ed)

            self.b_pred_labels[cls, :, :, self.slice_counter] = pred_labels_cls_es
            self.b_pred_labels[cls + self.num_of_classes, :, :, self.slice_counter] = pred_labels_cls_ed

        self.b_pred_probs[:, :, :, self.slice_counter] = pred_probs

    def compute_img_slice_uncertainty_stats(self, u_type="bald", connectivity=2, eps=0.01):
        """
        In order to get a first idea of how uncertain the model is w.r.t. certain image slices we compute
        same simple overall statistics per slide for stddev and BALD uncertainty measures.
        Note:
            self.b_bald_map     [2, width, height, #slices]
            self.b_stddev_map   [#classes, width, height, #slices]
        :param u_type:
        :param u_threshold:
        :param connectivity:
        :param eps:
        :return:
        """
        if u_type == "bald":
            uncertainty_map = self.b_bald_map[:, :, :, self.slice_counter]
        elif u_type == "stddev":
            # remember that we store the stddev for each class per pixel, so b_stddev_map has
            # shape [#classes, width, height, #slices]. Hence, we average over dim0 (classes) for ES and ED
            uncertainty_map = self.b_stddev_map[:, :, :, self.slice_counter]
            uncertainty_map = np.concatenate((np.mean(uncertainty_map[:self.num_of_classes], axis=0, keepdims=True),
                                              np.mean(uncertainty_map[self.num_of_classes:], axis=0, keepdims=True)))
        # shape uncertainty_map: [2, width, height]

        else:
            raise ValueError("{} is not a supported uncertainty type.".format(u_type))
        # we compute stats for ES and ED phase
        for phase in np.arange(uncertainty_map.shape[0]):
            pred_uncertainty_bool = np.zeros(uncertainty_map[phase].shape).astype(np.bool)
            # index maps for uncertain pixels
            mask_uncertain = uncertainty_map[phase] > eps
            # the u_threshold was already set in the method set_pred_labels above
            mask_uncertain_above_tre = uncertainty_map[phase] > self.b_uncertainty_stats["u_threshold"]
            # create binary mask to determine morphology
            pred_uncertainty_bool[mask_uncertain_above_tre] = True

            # binary structure
            footprint1 = generate_binary_structure(pred_uncertainty_bool.ndim, connectivity)
            # label distinct binary objects
            labelmap1, num_of_objects = label(pred_uncertainty_bool, footprint1)
            total_uncertainty = np.sum(uncertainty_map[phase][mask_uncertain])

            t_num_pixel_uncertain = np.count_nonzero(uncertainty_map[phase][mask_uncertain])
            t_num_pixel_uncertain_above_tre = np.count_nonzero(uncertainty_map[phase][mask_uncertain_above_tre])

            if u_type == "bald":
                self.b_uncertainty_stats["bald"][phase, :, self.slice_counter] = \
                    np.array([total_uncertainty, t_num_pixel_uncertain, t_num_pixel_uncertain_above_tre,
                              num_of_objects])
            else:
                self.b_uncertainty_stats["stddev"][phase, :, self.slice_counter] = \
                    np.array([total_uncertainty, t_num_pixel_uncertain, t_num_pixel_uncertain_above_tre,
                              num_of_objects])

    def set_stddev_map(self, slice_std):
        """
            Important: we assume slice_std is a numpy array with shape [num_classes, width, height]

        """
        self.b_stddev_map[:, :, :, self.slice_counter] = slice_std
        self.compute_img_slice_uncertainty_stats(u_type="stddev")

    def set_bald_map(self, slice_bald):
        """
        Important: we assume slice_bald is a numpy array with shape [2, width, height].
        First dim has shape two because we need maps for ES and ED phase
        :param slice_bald: contains the BALD values per pixel (for one slice) [2, width, height]
        :return:
        """

        self.b_bald_map[:, :, :, self.slice_counter] = slice_bald
        self.compute_img_slice_uncertainty_stats(u_type="bald")

    def get_accuracy(self, compute_hd=False, do_filter=True):
        """

            Compute dice coefficients for the complete 3D volume
            (1) self.b_label contains the ground truth: [num_of_classes, x, y, z]
            (2) self.b_pred_labels contains predicted softmax scores [2, x, y, z]
                                                                    (0=ES, 1=ED)

            compute_hd: Boolean indicating whether or not to compute Hausdorff distance
        """
        dices = np.zeros(2 * self.num_of_classes)
        hausdff = np.zeros(2 * self.num_of_classes)
        for cls in np.arange(self.num_of_classes):
            if do_filter:
                # use 6-connected components to filter out blobs that are not relevant for segmentation
                self.b_pred_labels[cls] = filter_connected_components(self.b_pred_labels[cls], cls, verbose=False)
                self.b_pred_labels[cls + self.num_of_classes] = \
                    filter_connected_components(self.b_pred_labels[cls + self.num_of_classes], cls, verbose=False)

            dices[cls] = dice_coefficient(self.b_labels[cls, :, :, :], self.b_pred_labels[cls, :, :, :])
            dices[cls + self.num_of_classes] = dice_coefficient(self.b_labels[cls + self.num_of_classes, :, :, :],
                                                                self.b_pred_labels[cls + self.num_of_classes, :, :, :])
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

        return dices, hausdff

    def compute_slice_accuracy(self, slice_idx=None, compute_hd=False):
        """

            Compute dice coefficients for the complete 3D volume
            (1) self.b_label contains the ground truth: [num_of_classes, x, y, z]
            (2) self.b_pred_labels contains predicted softmax scores [2, x, y, z]
                                                                    (0=ES, 1=ED)

            compute_hd: Boolean indicating whether or not to compute Hausdorff distance
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
        self.b_hd_slices[0, :, slice_idx] = hausdff[:self.num_of_classes]  # ES
        self.b_hd_slices[1, :, slice_idx] = hausdff[self.num_of_classes:]  # ED
        self.b_acc_slices[0, :, slice_idx] = dices[:self.num_of_classes]  # ES
        self.b_acc_slices[1, :, slice_idx] = dices[self.num_of_classes:]  # ED

        return dices, hausdff

    def visualize_test_slices(self, width=8, height=6, slice_range=None):
        """

        Remember that self.image is a list, containing images with shape [2, height, width, depth]
        NOTE: self.b_pred_labels only contains the image that we just processed and NOT the complete list
        of images as in self.images and self.labels!!!

        NOTE: we only visualize 1 image (given by image_idx)

        """
        column_lbls = ["bg", "RV", "MYO", "LV"]

        if slice_range is None:
            slice_range = np.arange(0, self.b_image.shape[3] // 2)

        _ = plt.figure(figsize=(width, height))
        counter = 1
        columns = self.num_of_classes + 1
        if self.b_pred_labels is not None:
            rows = 4
            plot_preds = True
        else:
            rows = 2
            plot_preds = False

        num_of_subplots = rows * 1 * columns  # +1 because the original image is included
        if len(slice_range) * num_of_subplots > 100:
            print("WARNING: need to limit number of subplots")
            slice_range = slice_range[:5]
        str_slice_range = [str(i) for i in slice_range]
        print("Number of subplots {} columns {} rows {} slices {}".format(num_of_subplots, columns, rows,
                                                                          ",".join(str_slice_range)))
        for idx in slice_range:
            # get the slice and then split ED and ES slices
            img = self.b_image[:, :, :, idx]
            labels = self.b_labels[:, :, :, idx]

            if plot_preds:
                pred_labels = self.b_pred_labels[:, :, :, idx]
            img_ed = img[0]  # INDEX 0 = end-diastole image
            img_es = img[1]  # INDEX 1 = end-systole image

            ax1 = plt.subplot(num_of_subplots, columns, counter)
            ax1.set_title("End-systole image", **config.title_font_medium)
            offx = self.config.pad_size
            offy = self.config.pad_size
            # get rid of the padding that we needed for the image processing
            img_ed = img_ed[offx:-offx, offy:-offy]
            plt.imshow(img_ed, cmap=cm.gray)
            plt.axis('off')
            counter += 1
            for cls1 in np.arange(self.num_of_classes):
                ax2 = plt.subplot(num_of_subplots, columns, counter)
                plt.imshow(labels[cls1], cmap=cm.gray)
                ax2.set_title(column_lbls[cls1] + " (true labels)", **config.title_font_medium)
                plt.axis('off')
                if plot_preds:
                    ax3 = plt.subplot(num_of_subplots, columns, counter + columns)
                    plt.imshow(pred_labels[cls1], cmap=cm.gray)
                    plt.axis('off')
                    ax3.set_title(column_lbls[cls1] + " (pred labels)", **config.title_font_medium)
                counter += 1

            cls1 += 1
            counter += columns
            ax2 = plt.subplot(num_of_subplots, columns, counter)
            ax2.set_title("End-diastole image", **config.title_font_medium)
            img_es = img_es[offx:-offx, offy:-offy]
            plt.imshow(img_es, cmap=cm.gray)
            plt.axis('off')
            counter += 1
            for cls2 in np.arange(self.num_of_classes):
                ax4 = plt.subplot(num_of_subplots, columns, counter)
                plt.imshow(labels[cls1 + cls2], cmap=cm.gray)
                ax4.set_title(column_lbls[cls2] + " (true labels)", **config.title_font_medium)
                plt.axis('off')
                if plot_preds:
                    ax5 = plt.subplot(num_of_subplots, columns, counter + columns)
                    plt.imshow(pred_labels[cls1 + cls2], cmap=cm.gray)
                    ax5.set_title(column_lbls[cls2] + " (pred labels)", **config.title_font_medium)
                    plt.axis('off')
                counter += 1

            counter += columns
        plt.show()

    def visualize_uncertainty(self, width=12, height=12, slice_range=None):

        column_lbls = ["bg", "RV", "MYO", "LV"]
        if slice_range is None:
            slice_range = np.arange(0, self.b_image.shape[3] // 2)

        fig = plt.figure(figsize=(width, height))
        counter = 1
        columns = self.num_of_classes + 1  # currently only original image and uncertainty map
        rows = 2
        num_of_slices = len(slice_range)
        num_of_subplots = rows * num_of_slices * columns  # +1 because the original image is included
        str_slice_range = [str(i) for i in slice_range]
        print("Number of subplots {} columns {} rows {} slices {}".format(num_of_subplots, columns, rows,
                                                                          ",".join(str_slice_range)))
        for idx in slice_range:
            # get the slice and then split ED and ES slices
            image = self.b_image[:, :, :, idx]
            true_labels = self.b_labels[:, :, :, idx]
            pred_labels = self.b_pred_labels[:, :, :, idx]
            pred_probs = self.b_pred_probs[:, :, :, idx]
            uncertainty = self.b_stddev_map[:, :, :, idx]
            for phase in np.arange(2):
                
                img = image[phase]  # INDEX 0 = end-systole image
                ax1 = plt.subplot(num_of_subplots, columns, counter)
                if phase == 0:
                    ax1.set_title("End-systole image", **config.title_font_medium)
                else:
                    ax1.set_title("End-diastole image", **config.title_font_medium)
                offx = self.config.pad_size
                offy = self.config.pad_size
                # get rid of the padding that we needed for the image processing
                img = img[offx:-offx, offy:-offy]
                plt.imshow(img, cmap=cm.gray)
                plt.axis('off')
                counter += 1
                # we use the cls_offset to plot ES and ED images in one loop (phase variable)
                cls_offset = phase * self.num_of_classes
                for cls in np.arange(self.num_of_classes):
                    std = uncertainty[cls + cls_offset]
                    ax2 = plt.subplot(num_of_subplots, columns, counter)
                    plt.imshow(std, cmap=cm.coolwarm)
                    ax2.set_title(r"$\sigma_{{pred}}$ {}".format(column_lbls[cls]), **config.title_font_medium)
                    plt.axis('off')
                    counter += 1
                    true_cls_labels = true_labels[cls + cls_offset]
                    pred_cls_labels = pred_labels[cls + cls_offset]
                    errors = true_cls_labels != pred_cls_labels
                    ax3 = plt.subplot(num_of_subplots, columns, counter + self.num_of_classes)
                    ax3.set_title("Errors {}".format(column_lbls[cls]), **config.title_font_medium)
                    plt.imshow(errors, cmap=cm.gray)
                    plt.axis('off')
                counter += self.num_of_classes + 1  # move counter forward in subplot

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


# print("Root path {}".format(os.environ.get("REPO_PATH")))
# dataset = ACDC2017TestHandler(exper_config=config, search_mask=config.dflt_image_name + ".mhd", fold_ids=[0],
#                              debug=False, batch_size=4)

# del dataset
