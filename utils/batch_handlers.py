import sys
if "/home/jogi/.local/lib/python2.7/site-packages" in sys.path:
    sys.path.remove("/home/jogi/.local/lib/python2.7/site-packages")

import matplotlib.pyplot as plt
from matplotlib import cm
import abc
import os
import numpy as np
import torch
from torch.autograd import Variable
from in_out.read_save_images import write_numpy_to_image
from config.config import config
# from in_out.load_data import ACDC2017DataSet


class BatchStatistics(object):

    # TO DO: should be traced in the Dataset object, so that we can use it here
    max_img_slices = 20

    def __init__(self, trans_imgid_name):
        """
        WE ARE CURRENTLY ONLY TRACKING THE USED IMAGE AND SLICE ID IN THE BATCHES.
        LATER WE WANT TO TRACK THE OFFSETs WHEN WE HOPEFULLY GUIDE TRAINING TO AREAS IN THE IMAGE
        WHERE THE UNCERTAINTY IS HIGH.
        :param trans_imgid_name: dictionary that translates patientID to imageIDs
        """
        self.num_of_images = len(trans_imgid_name)
        self.image_names = trans_imgid_name
        self.img_slice_stats = np.zeros((self.num_of_images, BatchStatistics.max_img_slices))
        # overall frequencies of sliceIDs during training (e.g. for guided training the first and last
        # slices should have a higher frequency
        self.slice_frequencies_train = None
        # actual slice frequencies in the dataset (e.g. slice 1 75 times...)
        self.slice_frequencies_dataset = None
        # distribution of number of slices per (train) image
        self.num_of_slice_freq = None
        # distribution of slices (how many times appears slice 1, remember for all images with num_of_slice >=1
        self.slice_freq = None
        # 1D numpy array of unique maximal sliceIDs, will be used to create a histgram of #slices frequencies
        self.unique_sliceIDs = None
        # the maximum sliceID for the complete dataset
        self.max_sliceid = 0

    def update_stats(self, batch_stats):
        """

        :param batch_stats: shape [batch_size, 2], where index of dim1: 0=imageID and 1=sliceID
        :return:

            NOTE: we're assuming that imageIDs and sliceIDs start from 0, so this is immediately
            compatible with the indices of the img_slice_stats matrix

        """
        for idx in np.arange(batch_stats.shape[0]):
            i = int(batch_stats[idx, 0])
            j = int(batch_stats[idx, 1])
            self.img_slice_stats[i, j] += 1

    def compute_slice_histogram(self):
        # REMEMBER: self.img_slice_stats is a matrix of shape [number of images (100), 20] <- we assumed that no
        # image has more than 20 slices!

        # determine the maximum number of images we trained on (should be 75)
        # although this returns 75, this is correct for slicing numpy arrays
        max_image_id = np.argmax(self.img_slice_stats[:, 0] == 0)
        img_slice_stats = self.img_slice_stats[:max_image_id, :]
        # determine for each image the maximum number of slices, note: the returned idx is the first which is equal
        # to zero, which give us the correct #slices, hence we can use "max_sliceid" for slicing
        max_sliceid_images = np.argmax(img_slice_stats == 0, axis=1)
        self.max_sliceid = np.max(max_sliceid_images)
        self.unique_sliceIDs = np.unique(max_sliceid_images)
        self.num_of_slice_freq, _ = np.histogram(max_sliceid_images, bins=self.unique_sliceIDs.shape[0])
        # finally we slice the columns, so we end up with a matrix [:max_image_id, :max_sliceid]
        img_slice_stats = img_slice_stats[:, :self.max_sliceid]
        # now sum over images/rows
        slice_freq = np.sum(img_slice_stats, axis=0)
        denominator = np.zeros(self.max_sliceid)
        for imgID in np.arange(max_image_id):
            col = max_sliceid_images[imgID]
            denominator[:col] += np.ones(col)
        # we need a counter that specifies for each slice how many times it occurs in the overall dataset (train).
        # "slice_frequencies_train". we'll use this later to scale the frequencies of the outliers e.g. when
        # we detected slice 17 as an outlier twice but it occurs only twice in the complete dataset, than this is
        # higher compared to twice finding slice 1, which occurs 75 times in the dataset
        counter = np.zeros(self.max_sliceid)
        for i, num_of_slices in enumerate(self.unique_sliceIDs):
            loc_counter = np.ones(num_of_slices) * self.num_of_slice_freq[i]
            counter[:num_of_slices] += loc_counter
        # USED IN plot_outlier_slice_hists
        self.slice_frequencies_dataset = 1./counter
        # the actual frequencies of each slice used during training
        self.slice_frequencies_train = slice_freq * (1. / denominator)

    def plot_histgrams(self, width=16, height=8, do_save=False, do_show=False, model_name=None):

        rows = 1
        columns = 2
        fig = plt.figure(figsize=(width, height))
        ax = fig.gca()
        fig.suptitle("SliceID histograms obtained during training", **config.title_font_medium)

        ax5 = plt.subplot2grid((rows, columns), (0, 0), rowspan=1, colspan=1)
        ax5.bar(self.unique_sliceIDs, self.num_of_slice_freq, label=r"Densities", color='g', alpha=0.2)
        ax5.legend(loc="best", prop={'size': 12})
        ax5.set_xlabel("# of slices per image", **config.axis_font)
        ax5.set_xticks(self.unique_sliceIDs)
        ax5.set_ylabel("Density")
        ax5.set_title("Distribution of #slices per image", **config.title_font_small)

        ax6 = plt.subplot2grid((rows, columns), (0, 1), rowspan=1, colspan=1)
        sliceIDs = np.arange(1, self.max_sliceid + 1)
        ax6.bar(sliceIDs, self.slice_frequencies_train, label=r"Densities", color='b', alpha=0.2)
        ax6.legend(loc="best", prop={'size': 12})
        ax6.set_xlabel("SliceID", **config.axis_font)
        ax6.set_xticks(sliceIDs)
        ax6.set_ylabel("Density")
        ax6.set_title("Distribution of sliceIDs", **config.title_font_small)

        if do_save:
            pass

        if do_show:
            plt.show()


class BatchHandler(object):
    __metaclass__ = abc.ABCMeta

    # static class variable to count the batches
    id = 0

    @abc.abstractmethod
    def cuda(self):
        pass


class TwoDimBatchHandler(BatchHandler):

    # we use a zero-padding of 65 on both dimensions, equals 130 positions
    patch_size_with_padding = 281
    patch_size = 150
    pixel_dta_type = "float32"

    def __init__(self, exper, test_run=False, batch_size=None, num_classes=8):
        if batch_size is None:
            self.batch_size = int(exper.run_args.batch_size)
        else:
            self.batch_size = int(batch_size)

        self.test_run = test_run
        # the number of classes to 8 but sometimes we'll need to work with half of the classes (4 for ES and ED)
        self.num_classes = num_classes
        self.ps_wp = TwoDimBatchHandler.patch_size_with_padding
        self.patch_size = TwoDimBatchHandler.patch_size
        self.is_cuda = exper.run_args.cuda
        self.offsets = []

        # batch image patch
        self.b_images = None
        self.b_labels = None
        # batch reference image for the different classes (so for each reference class 1 image)
        self.b_labels_per_class = None
        self.b_num_labels_per_class = None
        self.b_pred_labels_per_class = None
        # this objects holds for each image-slice the separate class labels, so one set for each class
        self.b_labels_per_class = None
        self.config = exper.config
        # we keep track of the images/slices we are using in each batch, 2nd dim for tuple (imageID, sliceID)
        self.batch_stats = np.zeros((self.batch_size, 2))

    def cuda(self):
        self.b_images = self.b_images.cuda()
        self.b_labels_per_class = self.b_labels_per_class.cuda()
        self.b_num_labels_per_class = self.b_num_labels_per_class.cuda()
        self.b_labels = self.b_labels.cuda()

    def set_pred_labels(self, pred_per_class):
        # in order to save memory we cast the object as NUMPY array
        self.b_pred_labels_per_class = pred_per_class.data.cpu().numpy()

    def get_images(self):
        return self.b_images

    def get_labels(self):
        return self.b_labels_per_class

    def get_labels_multiclass(self):
        return self.b_labels

    def get_num_labels_per_class(self):
        return self.b_num_labels_per_class

    def generate_batch_2d(self, images, labels, img_slice_ids=None, num_of_slices=None, save_batch=False, logger=None,
                          slice_range=None):
        """
        Variable images and labels are LISTS containing image slices, hence len(images) = number of total slices
        Each image slice (and label slice) has the following tensor dimensions:

         :param images:    [2 (ES and ED channels), x-axis, y-axis]
         :param labels:    [2 (ES and ED channels),  x-axis, y-axis]
         :param img_slice_ids: list that contains tuples of (imgID, sliceID) which we use later for statistics
         :param slice_range: we use this range of sliceIDs during validation in order to make sure that we always
         process the same sliceIDs during validation (why? because we don't validate always the complete set and
         we use a random selection of the slices (see below ind variable)

         IMPORTANT NOTE: first dim has size 2 and index "0" = ES image
                                                  index "1" = ED image

        This applies to image and label.

        :param num_of_slices: the length of our dataset in fact. If None, will be computed based on list images.
        Useful when we train with outlier slices and more efficient

         Note that labels contains segmentation class values between 0 and 3 for the four classes.
         We will convert these values into binary values below and hence add (next to the batch dimension dim-0)
         a second dimension with the number of classes (in our case 4 * 2 = 8), see b_labels_per_class object.
        """

        b_images = np.zeros((self.batch_size, 2, self.ps_wp, self.ps_wp))
        b_labels_per_class = np.zeros((self.batch_size, self.num_classes, self.patch_size + 1, self.patch_size + 1))
        b_labels_multiclass = np.zeros((self.batch_size, 2, self.patch_size + 1, self.patch_size + 1))
        b_num_labels_per_class = np.zeros((self.batch_size, self.num_classes))
        if num_of_slices is None:
            num_of_slices = len(images)

        checksum = 0
        for idx in range(self.batch_size):
            if slice_range is None:
                ind = np.random.randint(0, num_of_slices)
            else:
                ind = slice_range[idx]
            checksum += ind
            # print("Batch: ind={} and len={}".format(ind, num_of_slices))
            # img_nums.append(str(ind))
            img = images[ind]
            label = labels[ind]
            if img_slice_ids is not None:
                # img_slice_ids is a tuple()
                img_sliceID = np.array(img_slice_ids[ind])
            else:
                img_sliceID = np.zeros(2)
            # track imageID/sliceID
            self.batch_stats[idx, :] = img_sliceID
            # note img tensor has 3-dim and the first is equal to 2, because we concatenated ED and ES images
            offx = np.random.randint(0, img.shape[1] - self.ps_wp)
            offy = np.random.randint(0, img.shape[2] - self.ps_wp)
            self.offsets.append(tuple((offx, offy)))
            if logger is not None:
                logger.info("Random: {} {} {}".format(ind, offx, offy))
            img = img[:, offx:offx + self.ps_wp, offy:offy + self.ps_wp]

            b_images[idx, :, :, :] = img
            # next convert class labels to binary class labels
            label_ed = label[0, offx:offx + self.patch_size + 1, offy:offy + self.patch_size + 1]
            label_es = label[1, offx:offx + self.patch_size + 1, offy:offy + self.patch_size + 1]
            b_labels_multiclass[idx, 0] = label_ed.astype('int16')
            b_labels_multiclass[idx, 1] = label_es.astype('int16')
            half_classes = int(self.num_classes / 2)
            for cls_idx in np.arange(half_classes):
                # store ED class labels in first 4 positions of dim-1
                b_labels_per_class[idx, cls_idx, :, :] = (label_ed == cls_idx).astype('int16')
                if cls_idx != 0:
                    b_num_labels_per_class[idx, cls_idx] = np.count_nonzero(b_labels_per_class[idx, cls_idx, :, :])
                # sotre ES class labels in positions 4-7 of dim-1
                b_labels_per_class[idx, cls_idx+half_classes, :, :] = (label_es == cls_idx).astype('int16')
                if cls_idx != 0:
                    b_num_labels_per_class[idx, cls_idx+half_classes] = \
                        np.count_nonzero(b_labels_per_class[idx, cls_idx+half_classes, :, :])

        # print("Images used {}".format(",".join(img_nums)))
        self.b_images = Variable(torch.FloatTensor(torch.from_numpy(b_images).float()), volatile=self.test_run)
        self.b_labels_per_class = Variable(torch.FloatTensor(torch.from_numpy(b_labels_per_class).float()),
                                           volatile=self.test_run)
        self.b_num_labels_per_class = Variable(torch.FloatTensor(torch.from_numpy(b_num_labels_per_class).float()),
                                               volatile=self.test_run)

        self.b_labels = Variable(torch.LongTensor(torch.from_numpy(b_labels_multiclass).long()), volatile=self.test_run)
        if save_batch:
            self.save_batch_img_to_files()

        if self.is_cuda:
            self.cuda()

        del b_images
        del b_labels_per_class
        del b_num_labels_per_class
        del label_ed
        del label_es

    def save_batch_img_to_files(self, save_dir=None):
        num_of_classes = self.b_labels_per_class.size(1)
        print("Batch-size {} / classes {}".format(self.b_labels_per_class.size(0), num_of_classes))
        for i in np.arange(self.batch_size):
            # each input contains 2 images: 0=ES and 1=ED
            for phase in np.arange(2):
                filename_img = os.path.join(self.config.data_dir, "b" + str(i+1).zfill(2) + "_img_ph"
                                            + str(phase) + ".nii")
                offx = self.config.pad_size
                offy = self.config.pad_size
                img = self.b_images[i].data.cpu().numpy()[phase, offx:offx+self.patch_size + 1,
                      offy:offy+self.patch_size + 1]

                # we don't need to swap the axis because the image is 2D only
                write_numpy_to_image(img, filename=filename_img)
                cls_offset = phase * 4
                for cls in np.arange(num_of_classes / 2):
                    if cls != 0 and cls != 4:
                        cls_lbl = self.b_labels_per_class[i, cls_offset+cls].data.cpu().numpy()

                        filename_lbl = os.path.join(self.config.data_dir, "b" + str(i + 1).zfill(2) + "_lbl_ph"
                                                    + str(phase) + "_cls" + str(cls) + ".nii")
                        write_numpy_to_image(cls_lbl.astype("float32"), filename=filename_lbl)
                        # if object that store predictions is non None we save them as well for analysis
                        if self.b_pred_labels_per_class is not None:
                            pred_cls_lbl = self.b_pred_labels_per_class[i, cls_offset+cls]
                            filename_lbl = os.path.join(self.config.data_dir, "b_pred" + str(i + 1).zfill(2) + "_lbl_ph"
                                                        + str(phase) + "_cls" + str(cls) + ".nii")
                            write_numpy_to_image(pred_cls_lbl.astype("float32"), filename=filename_lbl)

    def visualize_batch(self, width=8, height=6, num_of_images=None):
        """

        Visualizing an image + label(s) from the batch in order to visually inspect the batch.
        Note, b_images is [batch_size, 2, x, y]
              b_labels_per_class is [batch_size, 8, x, y]
        """

        if num_of_images is None:
            num_of_images = self.batch_size

        fig = plt.figure(figsize=(width, height))
        counter = 1
        half_classes = int(self.num_classes / 2)
        columns = half_classes + 1
        if self.b_pred_labels_per_class is not None:
            rows = 4
            plot_preds = True
        else:
            rows = 2
            plot_preds = False
        num_of_subplots = rows * num_of_images * columns  # +1 because the original image is included
        print("Number of subplots {}".format(num_of_subplots))
        for idx in np.arange(num_of_images):
            img = self.b_images[idx].data.cpu().numpy()
            img_ed = img[0]  # INDEX 0 = end-diastole image
            img_es = img[1]  # INDEX 1 = end-systole image
            ax1 = plt.subplot(num_of_subplots, columns, counter)
            ax1.set_title("End-diastole image and reference")
            offx = self.config.pad_size
            offy = self.config.pad_size
            if offy < 0:
                offy = 0
            img_ed = img_ed[offx:offx+self.patch_size + 1, offy:offy+self.patch_size + 1]
            plt.imshow(img_ed, cmap=cm.gray)
            counter += 1
            labels = self.b_labels_per_class[idx].data.cpu().numpy()
            if plot_preds:
                pred_labels = self.b_pred_labels_per_class[idx]
            for cls1 in np.arange(half_classes):
                _ = plt.subplot(num_of_subplots, columns, counter)
                plt.imshow(labels[cls1], cmap=cm.gray)
                if plot_preds:
                    _ = plt.subplot(num_of_subplots, columns, counter + columns)
                    plt.imshow(pred_labels[cls1], cmap=cm.gray)
                counter += 1

            cls1 += 1
            counter += columns
            ax2 = plt.subplot(num_of_subplots, columns, counter)
            ax2.set_title("End-systole image and reference")
            img_es = img_es[offx:offx + self.patch_size + 1, offy:offy + self.patch_size + 1]
            plt.imshow(img_es, cmap=cm.gray)

            counter += 1
            for cls2 in np.arange(half_classes):
                _ = plt.subplot(num_of_subplots, columns, counter)
                plt.imshow(labels[cls1 + cls2], cmap=cm.gray)
                if plot_preds:
                    _ = plt.subplot(num_of_subplots, columns, counter + columns)
                    plt.imshow(pred_labels[cls1 + cls2], cmap=cm.gray)
                counter += 1

            counter += columns
        plt.show()
