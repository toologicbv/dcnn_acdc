import torch
import numpy as np
import copy
from matplotlib import pyplot as plt
from matplotlib import cm
from plotting.color_maps import transparent_cmap
from common.detector.config import config_detector
from common.detector.box_utils import BoundingBox
from common.detector.helper import create_grid_heat_map
from common.hvsmr.helper import convert_to_multilabel


class BatchHandler(object):

    def __init__(self, data_set, is_train=True, cuda=False, verbose=False, keep_bounding_boxes=False,
                 backward_freq=1, num_of_max_pool_layers=None):
        """
            data_set  of object type SliceDetectorDataSet

        """
        self.cuda = cuda
        self.is_train = is_train
        self.number_of_patients = data_set.get_size(is_train=is_train)
        self.num_of_channels = data_set.num_of_channels
        self.data_set = data_set
        self.loss = torch.zeros(1)
        self.num_sub_batches = torch.zeros(1)
        self.backward_freq = backward_freq
        self.batch_dta_slice_ids = []
        self.sample_range = self.data_set.get_size(is_train=is_train)
        self.verbose = verbose
        self.batch_bounding_boxes = None
        self.keep_bounding_boxes = keep_bounding_boxes
        self.batch_label_slices = []
        self.batch_size = 0
        self.batch_images = None
        self.batch_labels_per_voxel = None
        self.batch_patient_ids = []
        # useful in order to find back the original patient+slice IDs /  only used for test-batch
        self.batch_patient_slice_id = []
        # stores indications for base/apex=1 or middle=0
        self.batch_extra_labels = None
        self.target_labels_per_roi = None
        self.target_labels_stats_per_roi = None
        # during testing, we skip slices that do not contain any automatic segmentations
        self.num_of_skipped_slices = 0
        # when calling with keep_batch=True, we also want to store the predicted probs during testing
        # only during eval time
        self.batch_pred_probs = None
        self.batch_pred_labels = None
        self.batch_gt_labels = None
        # dictionary with key=slice_id and value list of indices that refer to the numpy array indices of batch_pred_probs
        # and batch_pred_labels, batch_gt_labels. we need these to evaluate the performance PER SLICE (FROC curve)
        # e.g.
        self.batch_slice_pred_probs= {}
        self.batch_slice_gt_labels = {}
        # same as above but then separately for each patient:
        self.batch_patient_pred_probs = {}
        self.batch_patient_gt_labels = {}
        # in order to link slice_ids from the dataset used to generate batches, to the patient_id (for analysis only)
        self.trans_dict = {}
        # we store the last slice_id we returned for the test batch, so we can chunk the complete test set if
        # necessary. Remember this is an index referring to the slices in list dataset.test_images
        self.last_test_list_idx = None
        if num_of_max_pool_layers is None:
            self.num_of_max_pool_layers = 3
        else:
            self.num_of_max_pool_layers = num_of_max_pool_layers

        if self.cuda:
            self.loss = self.loss.cuda()
            self.num_sub_batches = self.num_sub_batches.cuda()

    def add_loss(self, loss):
        self.num_sub_batches += 1
        self.loss += loss

    def mean_loss(self):
        self.loss = 1./self.num_sub_batches * self.loss

    def reset(self):
        self.loss = torch.zeros(1)
        self.num_sub_batches = torch.zeros(1)
        if self.cuda:
            self.loss = self.loss.cuda()
            self.num_sub_batches = self.num_sub_batches.cuda()

    def add_probs(self, probs, slice_id=None):
        """

        :param probs: has shape [1, w, h] and contains only the probabilities for the positive(true) class
        :param slice_id:
        :return: None
        """
        self.batch_pred_probs.append(probs)
        if slice_id is not None:
            self.batch_slice_pred_probs[slice_id] = probs.flatten()

    def add_gt_labels_slice(self, gt_labels, slice_id):
        self.batch_slice_gt_labels[slice_id] = gt_labels

    def flatten_batch_probs(self):
        # Only used during testing
        # batch_pred_probs is a list of numpy arrays with shape [1, w_grid, h_grid]. For our analysis of the model
        # performance we flatten here all probs and concatenate them into ONE np array
        flattened_probs = np.empty(0)
        for probs in self.batch_pred_probs:
            flattened_probs = np.concatenate((flattened_probs, probs.flatten()))
        return flattened_probs

    def fill_trans_dict(self):
        self.trans_dict = {}
        # only fill the translation dictionary (slice_id -> patient_id) for the slices we are processing in this batch
        for slice_id in self.batch_dta_slice_ids:
            for patient_id, slice_info in self.data_set.trans_dict.iteritems():
                # slice_info is a 2-tuple with is_train indication and list of slice ids
                # we're only interested in train or test images. depends on the batch we're evaluating
                if slice_info[0] == self.is_train:
                    if slice_id in slice_info[1]:
                        self.trans_dict[slice_id] = patient_id
                        break

    def add_pred_labels(self, pred_labels):
        self.batch_pred_labels.extend(pred_labels)

    def add_gt_labels(self, gt_labels):
        self.batch_gt_labels.extend(gt_labels)

    def add_patient_slice_gt_labels(self, patient_id, slice_id, gt_labels):
        if patient_id not in self.batch_patient_gt_labels.keys():
            self.batch_patient_gt_labels[patient_id] = {slice_id: gt_labels.flatten()}
        else:
            # patient as key already exists
            self.batch_patient_gt_labels[patient_id][slice_id] = gt_labels.flatten()

    def add_patient_slice_pred_probs(self, patient_id, slice_id, pred_probs):
        if patient_id not in self.batch_patient_pred_probs.keys():
            self.batch_patient_pred_probs[patient_id] = {slice_id: pred_probs.flatten()}
        else:
            # patient as key already exists
            self.batch_patient_pred_probs[patient_id][slice_id] = pred_probs.flatten()

    @property
    def do_backward(self):
        if self.backward_freq is not None:
            return self.backward_freq == self.num_sub_batches
        else:
            return True

    def __call__(self, batch_size=None, do_balance=True, keep_batch=False, disrupt_chnnl=None):
        """
        Construct a batch of shape [batch_size, 3channels, w, h]
                                   3channels: (1) input image
                                              (2) generated (raw/unfiltered) u-map/entropy map
                                              (3) predicted segmentation mask

        :param batch_size:
        :param disrupt_chnnl:  integer {0, 1, 2}: see meaning of channels above

        :param do_balance: if True batch is balanced w.r.t. slices containing positive target areas and not (1:3)
        :param keep_batch: boolean, only used during TESTING. Indicates whether we keep the batch items in
                           self.batch_images, self.target_labels_per_roi (lists)
        :return: input batch and corresponding references
        """
        self.batch_bounding_boxes = np.zeros((batch_size, 4))
        self.batch_label_slices = []
        self.target_labels = None
        self.batch_size = batch_size
        if self.is_train:
            # (1) batch_images will have the shape [#batches, num_of_input_chnls, w, h]
            # (2) batch_labels_per_voxel: currently not in use
            # (3) target_labels_per_roi: is a dictionary (key batch-item-nr) of dictionaries (key grid spacing)
            self.batch_images, self.batch_labels_per_voxel, self.target_labels_per_roi = \
                self._generate_train_batch(batch_size, do_balance=do_balance)
            if self.cuda:
                self.batch_images = self.batch_images.cuda()
                for g_spacing, target_lbls in self.target_labels_per_roi.iteritems():
                    # we don't use grid-spacing "1" key (overall binary indication whether patch contains target
                    # voxels (pos/neg example)
                    if g_spacing != 1:
                        target_lbls = target_lbls.cuda()
                        self.target_labels_per_roi[g_spacing] = target_lbls
            return self.batch_images, self.target_labels_per_roi
        else:
            # during testing we process whole slices and loop over the patient_ids in the test set.
            return self._generate_test_batch(batch_size, keep_batch=keep_batch, disrupt_chnnl=disrupt_chnnl)

    def _generate_train_batch(self, batch_size, do_balance=True):
        """

        :param batch_size:
        :param do_balance: if set to True, batch is separated between negative and positive examples based on
                            fraction_negatives (currently 2/3)
        :return: (1) batch_imgs: torch.FloatTensors [batch_size, 3, patch_size, patch_size]
                 (2) np_batch_lbls: binary numpy array [batch_size, patch_size, patch_size]
                 (3) target_labels: dictionary (key batch-item-nr) of dictionary torch.LongTensors.
                                                                keys 1, 4, 8 (currently representing grid spacing)
        """
        num_of_negatives = int(batch_size * config_detector.fraction_negatives)
        num_of_positives = batch_size - num_of_negatives
        negative_idx = []
        positive_idx = []
        batch_imgs = torch.zeros((batch_size, self.num_of_channels, config_detector.patch_size[0],
                                  config_detector.patch_size[1]))
        # this dictionary only contains the binary indications for each batch item (does contain voxels or not)
        # contains currently three key values that represent the grid spacing:
        #                       1 = complete patch, we don't use this
        #                       4 = after maxPool 2, 4x4 patches hence, grid spacing 4
        #                       8 = after maxPool 3, 8x8 patches hence, grid spacing 8
        # Each dict entry contains torch.LongTensor array with binary indications
        target_labels = {1: np.zeros(batch_size)}
        # store the number of positive gt labels per grid. we use this to analyze the errors the model makes
        self.target_labels_stats_per_roi = {1: np.zeros(batch_size)}
        for i in np.arange(2, self.num_of_max_pool_layers + 1):
            grid_spacing = int(2 ** i)
            g = int(config_detector.patch_size[0] / grid_spacing)
            target_labels[grid_spacing] = torch.zeros((batch_size, g * g), dtype=torch.long)
            self.target_labels_stats_per_roi[grid_spacing] = np.zeros((batch_size, g * g))
        # this array holds the patch for each label_slice (binary). Is a numpy array, we only need this object
        # to determine the final target labels per grid/roi
        np_batch_lbls = np.zeros((batch_size, config_detector.patch_size[0], config_detector.patch_size[1]))
        max_search_iters = batch_size * 5
        cannot_balance = False
        do_continue = True
        i = 0
        # start with collecting image slice indices
        while do_continue:
            slice_num = np.random.randint(0, self.sample_range, size=1, dtype=np.int)[0]
            # train_lbl_rois has shape [N,4] and contains box_four notation for target areas
            num_of_target_rois = self.data_set.train_lbl_rois[slice_num].shape[0]
            pred_lbls_exist = (np.sum(self.data_set.train_pred_lbl_rois[slice_num]) != 0).astype(np.bool)
            i += 1
            # we enforce the ratio between positives-negatives up to a certain effort (max_search_iters)
            # and when we indicated that we want to balance anyway
            # Logic: (1) if we still need to collect positives but the slice does not contain target rois
            #            and we collected already enough negatives, then we skip this slice.
            #        (2) or: we already have enough positives and but enough negatives and this slice contains
            #            target rois, we will skip this slice.
            if do_balance and (num_of_positives != 0 and num_of_target_rois == 0 and \
                               num_of_negatives == 0 and i <= max_search_iters) or \
                    (num_of_positives == 0 and num_of_target_rois != 0 and \
                     num_of_negatives != 0 and i <= max_search_iters):
                # no positive target area in slice and we already have enough negatives. Hence, continue
                if self.verbose:
                    print("Continue - #negatives {} out of {}".format(num_of_negatives, len(negative_idx)))
                continue
            else:
                if i > max_search_iters and not cannot_balance:
                    cannot_balance = True
                    # print("WARNING - Reached max-iters {}".format(i))
            # We need to make sure that if we still need negatives, and we don't have target rois to detect
            # then we AT LEAST need predicted labels. It's one or the other. In case we didn't predict anything
            # AND we didn't make any errors, then skip this slice
            if num_of_target_rois == 0 and not pred_lbls_exist and i <= max_search_iters and num_of_negatives != 0:
                continue

            if num_of_target_rois == 0:
                num_of_negatives -= 1
                # make a tuple with (1) slice-number (2) slice-coordinates describing area automatic mask (4 numbers)
                negative_idx.append(tuple((slice_num, self.data_set.train_pred_lbl_rois[slice_num])))
            else:
                if self.verbose:
                    print("#Positives {}".format(num_of_target_rois))
                num_of_positives -= 1
                roi_idx = np.random.randint(0, num_of_target_rois, size=1, dtype=np.int)[0]
                # make a tuple with (1) slice-number (2) slice-coordinates describing roi area (4 numbers)
                positive_idx.append(tuple((slice_num, self.data_set.train_lbl_rois[slice_num][roi_idx])))

            if len(positive_idx) + len(negative_idx) == batch_size:
                do_continue = False
        if cannot_balance:
            print("WARNING - Negative/positive ratio {}/{}".format(len(negative_idx), len(positive_idx)))
        b = 0
        for slice_area_spec in positive_idx:
            # print("INFO - Positives slice num {}".format(slice_area_spec))
            batch_imgs[b], np_batch_lbls[b], overall_lbls = self._create_train_batch_item(slice_area_spec,
                                                                                             is_positive=True,
                                                                                             batch_item_nr=b)
            target_labels = self._generate_batch_labels(np_batch_lbls[b], target_labels, batch_item_nr=b,
                                                        is_positive=overall_lbls)
            target_labels[1][b] = overall_lbls
            b += 1
            self.batch_dta_slice_ids.append(slice_area_spec[0])
        for slice_area_spec in negative_idx:
            # print("INFO - Negatives slice num {}".format(slice_area_spec))
            # _create_train_batch_item returns: torch.FloatTensor [3, patch_size, patch_size],
            #                                   Numpy array [patch_size, patch_size]
            #                                   Binary scalar value indicating pos/neg slice
            batch_imgs[b], np_batch_lbls[b], overall_lbls = self._create_train_batch_item(slice_area_spec,
                                                                                             is_positive=False,
                                                                                             batch_item_nr=b)
            target_labels[1][b] = overall_lbls
            b += 1
            self.batch_dta_slice_ids.append(slice_area_spec[0])

        return batch_imgs, np_batch_lbls, target_labels

    def _create_train_batch_item(self, slice_area_spec, is_positive, batch_item_nr):
        """

        :param slice_area_spec: is a tuple (1) slice number (2) target area described by box-four specification
                                i.e. [x.start, y.start, x.stop, y.stop]
        :param is_positive: boolean, indicating a slice which contains at least one positive target voxel
        :param batch_item_nr: the current batch item number we're processing
        :return: (1) input_channels_patch, torch.FloatTensor of shape [3, patch_size, patch_size]
                 (2) lbl_slice: Numpy binary array containing the target pixels [patch_size, patch_size].
                                we only need this array to compute the target roi labels in _generate_train_labels
                 (3) target_label: Binary value indicating whether the slice is positive or negative. Not used
        """
        # first we sample a pixel from the area of interest
        slice_num = slice_area_spec[0]
        input_channels = self.data_set.train_images[slice_num]
        label = self.data_set.train_labels[slice_num]
        _, w, h = input_channels.shape
        do_continue = True
        area_spec = slice_area_spec[1]  # is 1x4 np.array
        max_iters = 0
        half_width = config_detector.patch_size[0] / 2
        half_height = config_detector.patch_size[1] / 2
        while do_continue:

            x = np.random.randint(area_spec[0], area_spec[2], size=1, dtype=np.int)[0]
            y = np.random.randint(area_spec[1], area_spec[3], size=1, dtype=np.int)[0]
            max_iters += 1
            if (x + half_width <= w and x - half_width >= 0) and \
               (y + half_height <= h and y - half_height >= 0):
                slice_x = slice(x - half_width, x + half_width, None)
                slice_y = slice(y - half_height, y + half_height, None)
                if is_positive and self.verbose:
                    print("Slice {} (w/h {}/{}), x,y ({}, {})".format(slice_num, w, h, x, y))
                    print("area_spec ", area_spec)
                    print("slice_x ", slice_x)
                    print("slice_y ", slice_y)
                # input_channels has size [num_channels, w, h]
                input_channels_patch = input_channels[:, slice_x, slice_y]
                lbl_slice = label[slice_x, slice_y]
                target_label = 1 if np.count_nonzero(lbl_slice) != 0 else 0
                if is_positive and target_label == 0 and self.verbose:
                    print("WARNING - No labels slice {}".format(slice_num))
                do_continue = False
            else:
                if max_iters > 50:
                    print("WARNING - Problem need to break out of loop in BatchHandler._create_train_batch_item")
                    half_width = w / 2
                    half_height = h / 2
                    slice_x = slice(half_width - config_detector.patch_size[0],
                                    half_width + config_detector.patch_size[1], None)
                    slice_y = slice(half_height - config_detector.patch_size[0],
                                    half_height + config_detector.patch_size[1], None)
                    input_channels_patch = input_channels[:, slice_x, slice_y]
                    target_label = 1 if np.count_nonzero(lbl_slice) != 0 else 0
                    do_continue = False
                    lbl_slice = label[slice_x, slice_y]
                else:
                    continue
            if self.keep_bounding_boxes:
                roi_box_of_four = BoundingBox.convert_slices_to_box_four(slice_x, slice_y)

        if self.keep_bounding_boxes:
            self.batch_bounding_boxes[batch_item_nr] = roi_box_of_four
            self.batch_label_slices.append(lbl_slice)

        return torch.FloatTensor(torch.from_numpy(input_channels_patch).float()), lbl_slice, target_label

    def _generate_batch_labels(self, lbl_slice, target_labels, batch_item_nr, is_positive,
                               target_labels_stats_per_roi=None):
        """
        we already calculated the label for the entire patch (assumed to be square).
        Here, we compute the labels for the grid that is produced after maxPool 2 and maxPool 3
        E.g.: patch_size 72x72, has a grids of 2x2 (after maxPool 1) and 4x4 after maxPool 2

        :param lbl_slice:
        :param target_labels: dictionary with keys grid spacing. Currently 1, 4 and 8
        :param batch_item_nr: sequence number for batch item (ID)
        :param is_positive:
        :param target_labels_stats_per_roi: dictionary with keys grid spacing. Values store num of positive voxels
                                            per grid-block (used for evaluation purposes)
        :return: target_labels:
                 A dictionary of torch.LongTensors. Dict keys are grid spacings after maxPooling operations.
                 i.e.: 4 = grid spacing after maxPool 2, 8 = after maxPool 3
                 Each key contains torch tensor of shape [batch_size, grid_size, grid_size]
                target_labels_stats_per_roi if not None
        """

        w, h = lbl_slice.shape
        if is_positive == 0:
            raise ValueError("ERROR - _generate_train_labels. is_positive must be equal to True/1, not {}"
                             "".format(is_positive))

        if is_positive:
            all_labels = {}
            # First split label slice vertically
            for i in np.arange(2, self.num_of_max_pool_layers + 1):
                grid_spacing = int(2**i)
                # omit the [0, ...] at the front of the array, we don't want to split there
                grid_spacings_w = np.arange(0, w, grid_spacing)[1:]
                v_blocks = np.vsplit(lbl_slice, grid_spacings_w)
                all_labels[grid_spacing] = []
                # Second split label slice horizontally
                for block in v_blocks:
                    grid_spacings_h = np.arange(0, h, grid_spacing)[1:]
                    h_blocks = np.hsplit(block, grid_spacings_h)
                    all_labels[grid_spacing].extend(h_blocks)

            for grid_spacing, label_patches in all_labels.iteritems():
                grid_target_labels = np.zeros(len(label_patches))
                # REMEMBER: label_patches is a list of e.g. 81 in case of maxPool 3 layer which has
                # a final feature map size of 9x9.
                for i, label_patch in enumerate(label_patches):
                    label_patch = np.array(label_patch)
                    num_of_positives = np.count_nonzero(label_patch)
                    if target_labels_stats_per_roi is not None:
                        target_labels_stats_per_roi[grid_spacing][batch_item_nr][i] = num_of_positives
                    grid_target_labels[i] = 1 if 0 != num_of_positives else 0

                target_labels[grid_spacing][batch_item_nr] = \
                    torch.LongTensor(torch.from_numpy(grid_target_labels).long())

        if target_labels_stats_per_roi is not None:
            # Should be only necessary during Testing
            return target_labels, target_labels_stats_per_roi
        else:
            return target_labels

    def _generate_test_batch(self, batch_size=8, keep_batch=False, disrupt_chnnl=None, location_only="ALL"):
        """

        :param batch_size:
        :param keep_batch:
        :param disrupt_chnnl:
        :param location_only: M=only middle slices; AB=only BASE/APEX slices, otherwise ALL slices
        :return:
        """
        if self.last_test_list_idx is None:
            self.last_test_list_idx = 0
        else:
            # we need to increase the index by one to start with the "next" slice from the test set
            self.last_test_list_idx += 1
        self.num_of_skipped_slices = 0
        self.batch_dta_slice_ids = []
        self.batch_pred_probs = []
        self.batch_pred_labels = []
        self.batch_gt_labels = []
        self.batch_patient_ids = []
        self.batch_slice_areas = []
        # stores 1=apex/base or 0=middle slice
        self.batch_extra_labels = []
        self.batch_images, self.target_labels_per_roi = [], []
        self.target_labels_stats_per_roi = {}
        self.batch_patient_pred_probs = {}
        self.batch_patient_gt_labels = {}
        for i in np.arange(2, self.num_of_max_pool_layers + 1):
            grid_spacing = int(2 ** i)
            self.target_labels_stats_per_roi[grid_spacing] = []
        if disrupt_chnnl is not None:
            print("WARNING - Disrupting input channel {}".format(disrupt_chnnl))
        # we loop over the slices in the test set, starting
        for list_idx in np.arange(self.last_test_list_idx, self.last_test_list_idx + batch_size):
            target_labels = {}
            target_labels_stats_per_roi = {}
            # get input images should have 3 channels [3, w, h] and corresponding binary image [w, h]
            if disrupt_chnnl is not None:
                input_channels = copy.deepcopy(self.data_set.test_images[list_idx])
                _, w, h = input_channels.shape
                input_channels[disrupt_chnnl] = np.random.randn(w, h)
            else:
                input_channels = self.data_set.test_images[list_idx]
            label = self.data_set.test_labels[list_idx]
            base_apex_slice = self.data_set.test_labels_extra[list_idx]
            roi_area_spec = self.data_set.test_pred_lbl_rois[list_idx]
            slice_x, slice_y = BoundingBox.convert_to_slices(roi_area_spec)

            if slice_x.start == 0 and slice_x.stop == 0:
                # IMPORTANT: WE SKIP SLICES THAT DO NOT CONTAIN ANY AUTOMATIC SEGMENTATIONS!
                # does not contain any automatic segmentation mask, continue
                self.num_of_skipped_slices += 1
                continue
            if location_only == "AB" and not base_apex_slice:
                # ONLY PROCESS BASE/APEX SLICES
                self.num_of_skipped_slices += 1
                continue
            elif location_only == "M" and base_apex_slice:
                # SKIP BASE or APEX slices
                self.num_of_skipped_slices += 1
                continue
            else:
                # we test ALL SLICES
                pass

            # test_patient_slice_id contains 3-tuple (patient_id, slice_id, ES=0/ED=1)
            patient_id, slice_id, _ = self.data_set.test_patient_slice_id[list_idx]
            self.batch_slice_areas.append(tuple((slice_x, slice_y)))
            self.batch_extra_labels.append(base_apex_slice)
            self.batch_patient_slice_id.append(self.data_set.test_patient_slice_id[list_idx])

            # now slice input and label according to roi specifications (automatic segmentation mask roi)
            input_channels_patch = input_channels[:, slice_x, slice_y]
            _, w, h = input_channels_patch.shape
            lbl_slice = label[slice_x, slice_y]
            # does the patch contain any target voxels?
            contains_pos_voxels = 1 if np.count_nonzero(lbl_slice) != 0 else 0
            # store the overall indication whether our slice contains any voxels to be inspected
            target_labels[1] = np.array([contains_pos_voxels])
            target_labels_stats_per_roi[1] = np.array([contains_pos_voxels])
            # construct PyTorch tensor and add a dummy batch dimension in front
            input_channels_patch = torch.FloatTensor(torch.from_numpy(input_channels_patch[np.newaxis]).float())
            self.batch_dta_slice_ids.append(list_idx)
            # initialize dictionary target_labels with (currently) three keys for grid-spacing 1 (overall), 4, 8
            for i in np.arange(2, self.num_of_max_pool_layers + 1):
                grid_spacing = int(2 ** i)
                g_w = int(w / grid_spacing)
                g_h = int(h / grid_spacing)
                # print("INFO - spacing {} - grid size {}x{}={}".format(grid_spacing, g_w, g_h, g_w * g_h))
                # looks kind of awkward, but we always use a batch size of 1
                target_labels[grid_spacing] = torch.zeros((1, g_w * g_h), dtype=torch.long)
                # we need the dummy batch dimension (which is always 1) for the _generate_batch_labels method
                target_labels_stats_per_roi[grid_spacing] = np.zeros((1, g_w * g_h))

            if contains_pos_voxels:
                target_labels, target_labels_stats_per_roi = self._generate_batch_labels(lbl_slice, target_labels,
                                                                                         batch_item_nr=0,
                                                                                         is_positive=contains_pos_voxels,
                                                                                         target_labels_stats_per_roi=
                                                                                         target_labels_stats_per_roi)
            else:
                # no target voxels to inspect in ROI
                pass

            # if on GPU
            if self.cuda:
                input_channels_patch = input_channels_patch.cuda()
                for g_spacing, target_lbls in target_labels.iteritems():
                    # we don't use grid-spacing key "1" (overall binary indication whether patch contains target
                    # voxels (pos/neg example)
                    if g_spacing != 1:
                        target_labels[g_spacing] = target_lbls.cuda()
            # we keep the batch details in lists. Actually only used during debugging to make sure the patches
            # are indeed what we expect them to be.
            if keep_batch:
                self.batch_images.append(input_channels_patch)
                self.target_labels_per_roi.append(target_labels)
                self.batch_label_slices.append(lbl_slice)
                for grid_sp in target_labels_stats_per_roi.keys():
                    if grid_sp != 1:
                        self.target_labels_stats_per_roi[grid_sp].extend(target_labels_stats_per_roi[grid_sp].flatten())
            yield input_channels_patch, target_labels
            self.last_test_list_idx = list_idx

    def visualize_batch(self, grid_spacing=8, index_range=None, base_apex_only=False, sr_threshold=0.5,
                        exper_handler=None):

        mycmap = transparent_cmap(plt.get_cmap('jet'), alpha=0.4)
        error_cmap = transparent_cmap(plt.get_cmap('jet'), alpha=0.6)
        if index_range is None:
            index_range = [0, self.batch_size]
        slice_idx_generator = np.arange(index_range[0], index_range[1])
        number_of_slices = slice_idx_generator.shape[0]

        width = 16
        height = number_of_slices * 8
        print("number_of_slices {} height, width {}, {}".format(number_of_slices, height, width))
        columns = 4
        rows = number_of_slices * 2  # 2 because we do double row plotting
        row = 0
        fig = plt.figure(figsize=(width, height))
        heat_map = None
        for idx in slice_idx_generator:
            slice_num = self.batch_dta_slice_ids[idx]
            base_apex_slice = self.batch_extra_labels[idx]
            if self.batch_label_slices is not None and len(self.batch_label_slices) != 0:
                target_slice = self.batch_label_slices[idx]
            else:
                print("WARNING - self.batch_label_slices is empty")
            if not self.is_train:
                # during testing batch_images is a list with numpy arrays of shape [1, 3, w, h]
                # during training it's soley a numpy array of shape [batch_size, 3, w, h]
                # Hence, during testing we need to get rid of the first dummy batch dimension
                # idx = list item, first 0 = dummy batch dimension always equal to 1, second 0 = image channel
                image_slice = self.batch_images[idx][0][0]
                uncertainty_slice = self.batch_images[idx][0][1]
                # first list key/index and then dict key 1 for overall indication
                target_lbl_binary = self.target_labels_per_roi[idx][1][0]
                target_lbl_binary_grid = self.target_labels_per_roi[idx][grid_spacing][0].data.cpu().numpy()
                if len(self.batch_pred_probs) != 0:
                    # the model softmax predictions are store in batch property (list) batch_pred_probs if we enabled
                    # keep_batch during testing. The shape of np array is [1, w/grid_spacing, h/grid_spacing]
                    # and we need to get rid of first batch dimension.
                    p = self.batch_pred_probs[idx]
                    pred_probs = np.squeeze(p)
                    w, h = image_slice.shape
                    heat_map, grid_map, target_lbl_grid = create_grid_heat_map(pred_probs, grid_spacing, w, h,
                                                                               target_lbl_binary_grid,
                                                                               prob_threshold=sr_threshold)
            else:
                # training
                image_slice = self.batch_images[idx][0]
                uncertainty_slice = self.batch_images[idx][1]
                target_lbl_binary = self.target_labels_per_roi[1][idx]
                target_lbl_binary_grid = self.target_labels_per_roi[grid_spacing][idx]
            w, h = image_slice.shape
            if base_apex_only and not base_apex_slice:
                continue
            ax1 = plt.subplot2grid((rows, columns), (row, 0), rowspan=2, colspan=2)
            ax1.set_title("Slice {} ({}) (base/apex={})".format(slice_num, idx + 1, base_apex_slice))
            ax1.imshow(image_slice, cmap=cm.gray)
            # ax1.imshow(target_slice, cmap=mycmap, vmin=0, vmax=1)
            ax1.set_xticks(np.arange(-.5, h, grid_spacing))
            ax1.set_yticks(np.arange(-.5, w, grid_spacing))
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            plt.grid(True)

            ax2 = plt.subplot2grid((rows, columns), (row, 2), rowspan=2, colspan=2)
            ax2.imshow(image_slice, cmap=cm.gray)
            if self.trans_dict is not None:
                # contains tuple (patient_id, slice_id (increased + 1), 0=ES/1=ED)
                pat_slice_id = self.batch_patient_slice_id[idx]
                seg_error_volume = exper_handler.get_pred_labels_errors(patient_id=pat_slice_id[0])
                seg_error_volume = convert_to_multilabel(seg_error_volume, bg_cls_idx=[0, 4])
                seg_error_slice = seg_error_volume[pat_slice_id[2], :, :, pat_slice_id[1] - 1]
                slice_x, slice_y = self.batch_slice_areas[idx]
                seg_error_slice = seg_error_slice[slice_x, slice_y]
            else:
                pat_slice_id = None
            ax2.set_title("Contains ta-voxels {} {}".format(int(target_lbl_binary), pat_slice_id))

            if heat_map is not None:
                fontsize = 10 if grid_spacing == 4 else 15
                ax1.imshow(uncertainty_slice, cmap=mycmap)
                ax2.imshow(target_slice, cmap=mycmap)
                ax2.imshow(seg_error_slice, cmap=error_cmap)

                # automatic seg-mask
                # ax2.imshow(self.batch_images[idx][0][2], cmap=mycmap)
                ax2.imshow(heat_map, cmap=mycmap, vmin=0, vmax=1)
                for i, map_index in enumerate(zip(grid_map[0], grid_map[1])):
                    z_i = target_lbl_binary_grid[i]
                    if z_i == 1:
                        # BECAUSE, we are using ax2 with imshow, we need to swap x, y coordinates of map_index
                        ax2.text(map_index[1], map_index[0], '{}'.format(z_i), ha='center', va='center', fontsize=25,
                                 color="y")
            else:
                ax2.imshow(uncertainty_slice, cmap=mycmap)
                _, grid_map, _ = create_grid_heat_map(None, grid_spacing, w, h, target_lbl_binary_grid,
                                                      prob_threshold=0.5)
                for i, map_index in enumerate(zip(grid_map[0], grid_map[1])):
                    z_i = target_lbl_binary_grid[i]
                    if z_i == 1:
                        # BECAUSE, we are using ax2 with imshow, we need to swap x, y coordinates of map_index
                        ax1.text(map_index[1], map_index[0], '{}'.format(z_i), ha='center', va='center',
                                 fontsize=fontsize, color="b")

            ax2.set_xticks(np.arange(-.5, h, grid_spacing))
            ax2.set_yticks(np.arange(-.5, w, grid_spacing))
            ax2.set_xticklabels([])
            ax2.set_yticklabels([])
            plt.grid(True)
            # plt.axis("off")
            row += 2
        plt.show()

