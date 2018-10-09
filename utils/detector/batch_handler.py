import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from plotting.color_maps import transparent_cmap
from common.detector.config import config_detector
from common.detector.box_utils import BoundingBox


class BatchHandler(object):

    def __init__(self, data_set, is_train=True, cuda=False, verbose=False, keep_bounding_boxes=False):
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
        self.backward_freq = None
        self.current_slice_ids = []
        self.sample_range = self.data_set.get_size(is_train=is_train)
        self.verbose = verbose
        self.batch_bounding_boxes = None
        self.keep_bounding_boxes = keep_bounding_boxes
        self.batch_label_slices = []
        self.batch_size = 0
        self.batch_images = None
        self.batch_labels = None

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

    def __call__(self, batch_size=None, backward_freq=None, patient_id=None, do_balance=True,
                 keep_rois=False):
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
        self.backward_freq = backward_freq
        self.batch_bounding_boxes = None
        self.batch_label_slices = []
        self.batch_size = batch_size
        if self.is_train:
            # during training we sample slices to extract our patches of size config_detector.patch_size
            self.batch_images, self.batch_labels = self._generate_train_batch(batch_size, do_balance=do_balance)
        else:
            # during testing we process whole slices and loop over the patient_ids in the test set.
            raise NotImplementedError()
        return self.batch_images, self.batch_labels

    def _generate_train_batch(self, batch_size, do_balance=True):
        num_of_negatives = int(batch_size * 2. / 3)
        num_of_positives = batch_size - num_of_negatives
        negative_idx = []
        positive_idx = []
        np_batch_imgs = np.zeros((batch_size, self.num_of_channels, config_detector.patch_size[0],
                                  config_detector.patch_size[1]))
        np_batch_lbls = np.zeros(batch_size)
        max_search_iters = 125
        do_continue = True
        i = 0
        # start with collecting image slice indices
        while do_continue:
            slice_num = np.random.randint(0, self.sample_range, size=1, dtype=np.int)[0]
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
                if i > max_search_iters:
                    print("WARNING - Reached max-iters {}".format(i))
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
            self.current_slice_ids.append(slice_num)
            if len(positive_idx) + len(negative_idx) == batch_size:
                do_continue = False

        b = 0
        for slice_area_spec in positive_idx:
            # print("INFO - Positives slice num {}".format(slice_area_spec))
            np_batch_imgs[b], np_batch_lbls[b] = self._create_train_batch_item(slice_area_spec)
            b += 1
        for slice_area_spec in negative_idx:
            # print("INFO - Negatives slice num {}".format(slice_area_spec))
            np_batch_imgs[b], np_batch_lbls[b] = self._create_train_batch_item(slice_area_spec)
            b += 1

        return np_batch_imgs, np_batch_lbls

    def _create_train_batch_item(self, slice_area_spec):
        # slice_area_spec is a tuple (1) slice number (2) target area described by box-four specification
        # i.e. [x.start, y.start, x.stop, y.stop]
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
            if area_spec[2] <= area_spec[0]:
                if self.verbose:
                    print("Ho! {} <= {}".format(area_spec[2], area_spec[0]))
            x = np.random.randint(area_spec[0], area_spec[2], size=1, dtype=np.int)[0]
            y = np.random.randint(area_spec[1], area_spec[3], size=1, dtype=np.int)[0]
            max_iters += 1
            if (x + half_width <= w and x - half_width >= 0) and \
               (y + half_height <= h and y - half_height >= 0):
                slice_x = slice(x - half_width, x + half_width, None)
                slice_y = slice(y - half_height, y + half_height, None)
                # input_channels has size [num_channels, w, h]
                input_channels_patch = input_channels[:, slice_x, slice_y]
                lbl_slice = label[slice_x, slice_y]
                target_label = 1 if np.count_nonzero(lbl_slice) != 0 else 0
                do_continue = False
            else:
                if max_iters > 50:
                    print("WARNING - Problem need to break out of loop in "
                                  "BatchHandler._create_train_batch_item")
                    half_width = w / 2
                    half_height = h / 2
                    slice_x = slice(half_width - config_detector.patch_size[0],
                                    half_width + config_detector.patch_size[1], None)
                    slice_y = slice(half_height - config_detector.patch_size[0],
                                    half_height + config_detector.patch_size[1], None)
                    input_channels_patch = input_channels[:, slice_x, slice_y]
                    target_label = 1 if np.count_nonzero(lbl_slice) != 0 else 0
                    do_continue = False
                    lbl_slice = None
                else:
                    continue
            if self.keep_bounding_boxes:
                roi_box_of_four = BoundingBox.convert_slices_to_box_four(slice_x, slice_y)

        if self.keep_bounding_boxes:
            self.batch_bounding_boxes = roi_box_of_four
            self.batch_label_slices.append(lbl_slice)

        return input_channels_patch, target_label

    def visualize_batch(self):
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
            uncertainty_slice = self.batch_images[idx][1]
            # pred_labels_slice = self.batch_images[idx][2]
            target_lbl_binary = self.batch_labels[idx]
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
            plt.axis("off")
            row += 2
        plt.show()

