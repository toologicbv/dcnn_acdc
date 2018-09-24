import numpy as np
from common.hvsmr.config import config_hvsmr
import os
import torch
import copy
from collections import OrderedDict

from in_out.hvsmr.load_data import HVSMR2016DataSet
from utils.post_processing import filter_connected_components
from utils.medpy_metrics import hd
from utils.dice_metric import dice_coefficient


class HVSMRTesthandler(object):

    pixel_dta_type = 'float32'
    pad_size = config_hvsmr.pad_size
    new_voxel_spacing = HVSMR2016DataSet.new_voxel_spacing

    def __init__(self, exper_config, search_mask=None, nclass=3,
                 fold_id=0, debug=False, use_cuda=True, batch_size=None):
        self.dataset = HVSMR2016DataSet(config_hvsmr, search_mask=config_hvsmr.dflt_image_name + ".nii",
                                        fold_id=fold_id, preprocess="rescale",
                                        debug=False, val_set_only=True, verbose=False)
        self.debug = debug
        # copied from TestHandler ACDC
        self.config = exper_config
        self.data_dir = os.path.join(self.config.root_dir, exper_config.data_dir)
        self.filtered_umaps_dir = os.path.join(self.config.root_dir, config_hvsmr.u_map_dir)
        self.num_of_classes = nclass
        self.fold_id = fold_id
        self.abs_path_fold = os.path.join(self.data_dir, "fold")
        self.images = self.dataset.val_images
        # contains patient_ids
        self.img_file_names = self.dataset.val_image_names
        self.labels = self.dataset.val_labels
        self.num_labels_per_class = []
        self.voxel_spacing = self.dataset.val_spacings
        self.b_pred_labels = None
        self.b_filtered_pred_labels = None
        self.b_pred_probs = None
        self.b_stddev_map = None
        # we'll loop through the test images and "current" state will be stored in these variables with prefix b_
        self.b_image = None
        self.b_image_id = None
        self.b_image_name = None  # store the filename from "above" used as identification during testing
        self.b_labels = None
        self.b_labels_multiclass = None
        self.b_labels_per_class = None
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
        # dictionary with key is patientID and value is imageID that we assign when loading the stuff
        self.trans_dict = OrderedDict()
        self.num_of_images = 0

        self.slice_counter = 0
        self.num_of_models = 1
        self.use_cuda = use_cuda

    def get_test_pair(self, patient_id):
        try:
            image_num = self.dataset.trans_dict[patient_id]
        except KeyError:
            print("ERROR - key {} does not exist in translation dict".format(patient_id))
            print("Following keys exist:")
            print(self.trans_dict.keys())
        return self.images[image_num], self.labels[image_num]

    def _split_class_labels(self, labels):
        """

        :param labels: numpy array of shape [width, height, #slices]. Values are multi-class labels {0...nclass}
                        we want to convert these to [nclass, width, height, #slices]
        :return: returns tensor [nclass, width, height, #slices] for reference labels (gt) and
        a "count" of the pixels/voxels that belong to that slice per class (simulating volume regression)
        """
        labels_per_class = np.zeros((self.num_of_classes, labels.shape[0], labels.shape[1],
                                     labels.shape[2]))
        b_num_labels_per_class = np.zeros((self.num_of_classes, labels.shape[2]))
        for cls_idx in np.arange(self.num_of_classes):
            labels_per_class[cls_idx, :, :, :] = (labels == cls_idx).astype('int16')
            if cls_idx != 0:
                b_num_labels_per_class[cls_idx] = \
                    np.count_nonzero(labels_per_class[cls_idx, :, :, :], axis=(0, 1))

        return labels_per_class, b_num_labels_per_class

    def batch_generator(self, image_num=0, use_labels=True):
        """
            Remember that self.dataset.image is a list, containing images with shape [2, height, width, depth]
            Object self.labels is also a list but with shape [8, height, width, depth]

        """
        self.b_image_id = image_num
        self.b_filtered_umap = False
        b_label = None
        self.slice_counter = -1
        self.b_image = self.images[image_num]
        num_of_slices = self.b_image.shape[2]
        # pad the original image to fit the dilated convolutions
        self.b_image = np.pad(self.b_image, ((HVSMRTesthandler.pad_size, HVSMRTesthandler.pad_size),
                                             (HVSMRTesthandler.pad_size, HVSMRTesthandler.pad_size), (0, 0)),
                                              'constant', constant_values=(0,)).astype(HVSMRTesthandler.pixel_dta_type)

        self.b_image_name = self.img_file_names[image_num]
        self.b_orig_spacing = self.voxel_spacing[image_num]
        self.b_acc_slices = np.zeros((self.num_of_classes, num_of_slices))
        self.b_hd_slices = np.zeros((self.num_of_classes, num_of_slices))
        if use_labels:
            self.b_labels_multiclass = self.labels[image_num]
            self.b_labels, self.b_num_labels_per_class = self._split_class_labels(self.b_labels_multiclass)
            self.b_pred_probs = np.zeros_like(self.b_labels)
            self.b_pred_labels = np.zeros_like(self.b_labels)

        self.b_stddev_map = np.zeros_like(self.b_labels)

        for slice_id in np.arange(num_of_slices):
            b_image_slice = torch.FloatTensor(torch.from_numpy(
                self.b_image[np.newaxis, np.newaxis, :, :, slice_id]).float())
            # we need an extra dimension as batch-dim, although we'll only process one-by-one

            if self.b_labels is not None:
                b_labels_per_class_slice = torch.FloatTensor(torch.from_numpy(
                    self.b_labels[:, :, :, slice_id]).float())
                labels_multiclass_slice = torch.FloatTensor(torch.from_numpy(self.b_labels_multiclass[:, :, slice_id]).float())
                b_labels_per_class_slice = b_labels_per_class_slice.unsqueeze(0)
                labels_multiclass_slice = labels_multiclass_slice.unsqueeze(0)
                # print(b_labels_per_class_slice.shape, b_label_multiclass_slice.shape)
            if self.use_cuda:
                b_image_slice = b_image_slice.cuda()
                if self.b_labels is not None:
                    b_labels_per_class_slice = b_labels_per_class_slice.cuda()
                    labels_multiclass_slice = labels_multiclass_slice.cuda()
            self.slice_counter += 1
            yield b_image_slice, b_labels_per_class_slice, labels_multiclass_slice

    def set_pred_labels(self, pred_probs, verbose=False, do_filter=False):
        """
        We determine the labels based on the softmax predictions and also store the softmax probs as a last
        step.

        :param pred_probs: [1, num_of_classes, width, height]

        Also note that object self.b_labels has dims [num_classes, width, height, slices]

        :param do_filter: filter predicted labels on connected components. This is an important post processing step
                          especially when doing this at the end for the 3D label object
        :param verbose:

        :return:
        """

        if not isinstance(pred_probs, np.ndarray):
            pred_probs = pred_probs.data.cpu().numpy()
        # remember dim0 of pred_probs is 1, so we squeeze it by taking index "0". Hence the argmax is over
        # axis 0, because we lost the original dim0
        pred_labels = np.argmax(pred_probs[0, :, :, :], axis=0)

        for cls in np.arange(self.num_of_classes):
            pred_labels_cls = (pred_labels == cls).astype(np.int)
            if do_filter:
                # use 6-connected components to filter out blobs that are not relevant for segmentation
                pred_labels_cls = filter_connected_components(pred_labels_cls, cls, verbose=verbose)

            self.b_pred_labels[cls, :, :, self.slice_counter] = pred_labels_cls

        self.b_pred_probs[:, :, :, self.slice_counter] = pred_probs

    def set_stddev_map(self, slice_std):
        """
            Important: we assume slice_std is a numpy array with shape [num_classes, width, height]

        """
        self.b_stddev_map[:, :, :, self.slice_counter] = slice_std

    def compute_slice_accuracy(self, slice_idx=None, compute_hd=False, store_results=True):
        """

            Compute dice coefficients for the complete 3D volume
            (1) self.b_label contains the ground truth: [num_of_classes, x, y, z]
            (2) self.b_pred_labels contains predicted softmax scores [num_of_classes, x, y, z]

            compute_hd: Boolean indicating whether or not to compute Hausdorff distance

            store_results:
        """
        if slice_idx is None:
            slice_idx = self.slice_counter
        # Remember self.num_of_classes is 3
        dices = np.zeros(self.num_of_classes)
        hausdff = np.zeros(self.num_of_classes)
        for cls in np.arange(self.num_of_classes):

            dices[cls] = dice_coefficient(self.b_labels[cls, :, :, slice_idx],
                                          self.b_pred_labels[cls, :, :, slice_idx])
            if compute_hd:
                # only compute distance if both contours are actually in images
                if 0 != np.count_nonzero(self.b_pred_labels[cls, :, :, slice_idx]) and \
                        0 != np.count_nonzero(self.b_labels[cls, :, :, slice_idx]):
                    hausdff[cls] = hd(self.b_pred_labels[cls, :, :, slice_idx], self.b_labels[cls, :, :, slice_idx],
                                      voxelspacing=HVSMRTesthandler.new_voxel_spacing, connectivity=1)
                else:
                    hausdff[cls] = 0

        if store_results:
            self.b_hd_slices[:, slice_idx] = hausdff
            self.b_acc_slices[:, slice_idx] = dices

        return dices, hausdff

    def get_accuracy(self, compute_hd=False, compute_seg_errors=False, do_filter=True,
                     compute_slice_metrics=False):
        """

            Compute dice coefficients for the complete 3D volume
            (1) self.b_label contains the ground truth: [num_of_classes, x, y, z]
            (2) self.b_pred_labels contains predicted softmax scores [x, y, z]

            compute_hd: Boolean indicating whether or not to compute Hausdorff distance
            compute_seg_errors: boolean indicating whether or not to compute segmentation errors

            :returns: always: 2 numpy arrays dice [num_of_classes], hd [num_of_classes]
                      if we compute seg-errors then in addition: [num_of_classes, #slices] -> errors per class per slice
        """
        dices = np.zeros(self.num_of_classes)
        hausdff = np.zeros(self.num_of_classes)
        num_of_slices = self.b_labels.shape[3]
        if compute_slice_metrics:
            dice_slices = np.zeros((self.num_of_classes, num_of_slices))
            hd_slices = np.zeros((self.num_of_classes, num_of_slices))
        if compute_seg_errors:
            # if we compute seg-errors then shape of results is [classes, #slices]
            seg_errors = np.zeros((self.num_of_classes, num_of_slices))
        for cls in np.arange(self.num_of_classes):
            if do_filter:
                # use 6-connected components to filter out blobs that are not relevant for segmentation
                self.b_pred_labels[cls] = filter_connected_components(self.b_pred_labels[cls], cls, verbose=False)

            dices[cls] = dice_coefficient(self.b_labels[cls, :, :, :], self.b_pred_labels[cls, :, :, :])

            if compute_slice_metrics:
                for s_idx in np.arange(num_of_slices):
                    dice_slices[cls, s_idx] = dice_coefficient(self.b_labels[cls, :, :, s_idx],
                                                               self.b_pred_labels[cls, :, :, s_idx])

            if compute_hd:
                # only compute distance if both contours are actually in images
                if 0 != np.count_nonzero(self.b_pred_labels[cls, :, :, :]) and \
                        0 != np.count_nonzero(self.b_labels[cls, :, :, :]):
                    hausdff[cls] = hd(self.b_pred_labels[cls, :, :, :], self.b_labels[cls, :, :, :],
                                      voxelspacing=self.b_new_spacing, connectivity=1)
                else:
                    hausdff[cls] = 0.

                if compute_slice_metrics:
                    for s_idx in np.arange(num_of_slices):
                        # only compute distance if both contours are actually in images
                        if 0 != np.count_nonzero(self.b_pred_labels[cls, :, :, s_idx]) and \
                                0 != np.count_nonzero(self.b_labels[cls, :, :, s_idx]):
                            hd_slices[cls, s_idx] = hd(self.b_pred_labels[cls, :, :, s_idx], self.b_labels[cls, :, :, s_idx],
                                                voxelspacing=HVSMRTesthandler.new_voxel_spacing, connectivity=1)
                        else:
                            hd_slices[cls] = 0

            if compute_seg_errors:
                true_labels_cls = self.b_labels[cls]
                pred_labels_cls = self.b_pred_labels[cls]
                errors = pred_labels_cls != true_labels_cls
                errors = np.count_nonzero(np.reshape(errors, (-1, errors.shape[2])), axis=0)
                seg_errors[cls] = errors

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

    def save_pred_labels(self, output_dir,  mc_dropout=False, used_entropy=False):
        """

        :param output_dir: is actually the exper_id e.g. 20180503_13_22_18_dcnn_mc_f2p005....
        :param u_threshold: indicating whether we store a filtered label mask...if <> 0

        :param mc_dropout: boolean indicating whether we obtained the predictions by means of mc-dropout
        :param used_entropy: did we use entropy maps to generate the predicted labels (filtering/referral)?
                if so, we need to distinguish the predictions from the "other" predictions
        saves the predicted label maps for an image. shape [8classes, width, height, #slices]
        """
        pred_lbl_output_dir = os.path.join(self.config.root_dir, os.path.join(output_dir, config_hvsmr.pred_lbl_dir))
        if not os.path.isdir(pred_lbl_output_dir):
            os.makedirs(pred_lbl_output_dir)

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

    def save_pred_probs(self, output_dir, mc_dropout=False):
        """
        Save a 3D volume of the mean softmax probabilities
        :return:
        """
        pred_prob_output_dir = os.path.join(self.config.root_dir, os.path.join(output_dir, config_hvsmr.pred_lbl_dir))
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

    @staticmethod
    def get_testset_instance(config_obj, fold_id, use_cuda=True):
        return HVSMRTesthandler(config_obj, fold_id=fold_id, use_cuda=use_cuda)