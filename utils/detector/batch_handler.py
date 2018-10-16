import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from plotting.color_maps import transparent_cmap
from common.detector.config import config_detector
from common.detector.box_utils import BoundingBox


class BatchHandler(object):

    fraction_negatives = 2./3

    def __init__(self, data_set, is_train=True, cuda=False, verbose=False, keep_bounding_boxes=False,
                 backward_freq=1):
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
        self.current_slice_ids = []
        self.sample_range = self.data_set.get_size(is_train=is_train)
        self.verbose = verbose
        self.batch_bounding_boxes = None
        self.keep_bounding_boxes = keep_bounding_boxes
        self.batch_label_slices = []
        self.batch_size = 0
        self.batch_images = None
        self.batch_labels_per_voxel = None
        self.target_labels_per_roi = None
        if config_detector.num_of_max_pool is None:
            self.num_of_max_pool_layers = 3
        else:
            self.num_of_max_pool_layers = config_detector.num_of_max_pool

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

    @property
    def do_backward(self):
        if self.backward_freq is not None:
            return self.backward_freq == self.num_sub_batches
        else:
            return True

    def __call__(self, batch_size=None, do_balance=True):
        """
        Construct a batch of shape [batch_size, 3channels, w, h]
                                   3channels: (1) input image
                                              (2) predicted segmentation mask
                                              (3) generated (raw/unfiltered) u-map/entropy map

        :param batch_size:
        :param patient_id: if not None than we use all slices
        :param do_balance: if True batch is balanced w.r.t. slices containing positive target areas and not (1:3)
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
        else:
            # during testing we process whole slices and loop over the patient_ids in the test set.
            self.batch_images, self.batch_labels_per_voxel, self.target_labels_per_roi = \
                self._generate_test_batch(batch_size)

        if self.cuda:
            self.batch_images = self.batch_images.cuda()
            for g_spacing, target_lbls in self.target_labels_per_roi.iteritems():
                # we don't use grid-spacing "1" key (overall binary indication whether patch contains target
                # voxels (pos/neg example)
                if g_spacing != 1:
                    target_lbls = target_lbls.cuda()
                    self.target_labels_per_roi[g_spacing] = target_lbls
        return self.batch_images, self.target_labels_per_roi

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
        num_of_negatives = int(batch_size * BatchHandler.fraction_negatives)
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
        for i in np.arange(2, self.num_of_max_pool_layers + 1):
            grid_spacing = int(2 ** i)
            g = int(config_detector.patch_size[0] / grid_spacing)
            target_labels[grid_spacing] = torch.zeros((batch_size, g * g), dtype=torch.long)
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
            target_labels = self._generate_train_labels(np_batch_lbls[b], target_labels, batch_item_nr=b,
                                                        is_positive=overall_lbls)
            target_labels[1][b] = overall_lbls
            b += 1
            self.current_slice_ids.append(slice_area_spec[0])
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
            self.current_slice_ids.append(slice_area_spec[0])

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

    def _generate_train_labels(self, lbl_slice, target_labels, batch_item_nr, is_positive):
        """
        we already calculated the label for the entire patch (assumed to be square).
        Here, we compute the labels for the grid that is produced after maxPool 2 and maxPool 3
        E.g.: patch_size 72x72, has a grids of 2x2 (after maxPool 1) and 4x4 after maxPool 2

        :param lbl_slice:
        :param target_labels: dictionary with keys grid spacing. Currently 4 and 8
        :param batch_item_nr: sequence number for batch item (ID)
        :param is_positive:
        :return: target_labels:
                 A dictionary of torch.LongTensors. Dict keys are grid spacings after maxPooling operations.
                 i.e.: 4 = grid spacing after maxPool 2, 8 = after maxPool 3
                 Each key contains torch tensor of shape [batch_size, grid_size, grid_size]
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
                grid_spacings = np.arange(0, w, grid_spacing)[1:]
                v_blocks = np.vsplit(lbl_slice, grid_spacings)
                all_labels[grid_spacing] = []
                # Second split label slice horizontally
                for block in v_blocks:
                    h_blocks = np.hsplit(block, grid_spacings)
                    all_labels[grid_spacing].extend(h_blocks)

            for grid_spacing, label_patches in all_labels.iteritems():
                grid_target_labels = np.zeros(len(label_patches))
                # REMEMBER: label_patches is a list of e.g. 81 in case of maxPool 3 layer which has
                # a final feature map size of 9x9.
                for i, label_patch in enumerate(label_patches):
                    label_patch = np.array(label_patch)
                    grid_target_labels[i] = 1 if 0 != np.count_nonzero(label_patch) else 0
                target_labels[grid_spacing][batch_item_nr] = \
                    torch.LongTensor(torch.from_numpy(grid_target_labels).long())

        return target_labels

    def visualize_batch(self, grid_spacing=8):
        mycmap = transparent_cmap(plt.get_cmap('jet'))

        width = 16
        height = self.batch_size * 8
        columns = 4
        rows = self.batch_size * 2  # 2 because we do double row plotting
        row = 0
        fig = plt.figure(figsize=(width, height))

        for idx in np.arange(self.batch_size):
            slice_num = self.current_slice_ids[idx]
            image_slice = self.batch_images[idx][0]
            w, h = image_slice.shape
            uncertainty_slice = self.batch_images[idx][1]
            # pred_labels_slice = self.batch_images[idx][2]
            target_lbl_binary = self.target_labels_per_roi[1][idx]
            target_slice = self.batch_label_slices[idx]
            ax1 = plt.subplot2grid((rows, columns), (row, 0), rowspan=2, colspan=2)
            ax1.set_title("Slice {} ({})".format(slice_num, idx + 1))
            ax1.imshow(image_slice, cmap=cm.gray)
            ax1.imshow(uncertainty_slice, cmap=mycmap)
            plt.axis("off")

            ax2 = plt.subplot2grid((rows, columns), (row, 2), rowspan=2, colspan=2)
            ax2.imshow(image_slice, cmap=cm.gray)
            ax2.set_title("Target label {}".format(int(target_lbl_binary)))
            ax2.imshow(target_slice, cmap=mycmap)

            ax2.set_xticks(np.arange(-.5, h, grid_spacing))
            ax2.set_yticks(np.arange(-.5, w, grid_spacing))
            # ax2.set_xticklabels(np.arange(1, h + 1, grid_spacing))
            ax2.set_xticklabels([])
            # ax2.set_yticklabels(np.arange(1, w + 1, grid_spacing))
            ax2.set_yticklabels([])
            plt.grid(True)
            # plt.axis("off")
            row += 2
        plt.show()

