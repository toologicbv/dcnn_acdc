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
        # batch reference image for the different classes (so for each reference class 1 image)
        self.b_labels_per_class = None
        self.b_pred_labels_per_class = None
        # this objects holds for each image-slice the separate class labels, so one set for each class
        self.b_labels_per_class = None
        self.config = exper.config
        # we keep track of the images/slices we are using in each batch, 2nd dim for tuple (imageID, sliceID)
        self.batch_stats = np.zeros((self.batch_size, 2))

    def cuda(self):
        self.b_images = self.b_images.cuda()
        self.b_labels_per_class = self.b_labels_per_class.cuda()

    def set_pred_labels(self, pred_per_class):
        # in order to save memory we cast the object as NUMPY array
        self.b_pred_labels_per_class = pred_per_class.data.cpu().numpy()

    def get_images(self):
        return self.b_images

    def get_labels(self):
        return self.b_labels_per_class

    def generate_batch_2d(self, images, labels, img_slice_ids=None, save_batch=False, logger=None):
        """
        Variable images and labels are LISTS containing image slices, hence len(images) = number of total slices
        Each image slice (and label slice) has the following tensor dimensions:

         images:    [2 (ES and ED channels), x-axis, y-axis]
         labels:    [2 (ES and ED channels),  x-axis, y-axis]

         IMPORTANT NOTE: first dim has size 2 and index "0" = ES image
                                                  index "1" = ED image

        This applies to image and label

         Note that labels contains segmentation class values between 0 and 3 for the four classes.
         We will convert these values into binary values below and hence add (next to the batch dimension dim-0)
         a second dimension with the number of classes (in our case 4 * 2 = 8), see b_labels_per_class object.
        """

        b_images = np.zeros((self.batch_size, 2, self.ps_wp, self.ps_wp))
        b_labels_per_class = np.zeros((self.batch_size, self.num_classes, self.patch_size + 1, self.patch_size + 1))

        num_images = len(images)
        # img_nums = []
        for idx in range(self.batch_size):
            ind = np.random.randint(0, num_images)
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
            half_classes = int(self.num_classes / 2)
            for cls_idx in np.arange(half_classes):
                # store ED class labels in first 4 positions of dim-1
                b_labels_per_class[idx, cls_idx, :, :] = (label_ed == cls_idx).astype('int16')
                # sotre ES class labels in positions 4-7 of dim-1
                b_labels_per_class[idx, cls_idx+half_classes, :, :] = (label_es == cls_idx).astype('int16')

        # print("Images used {}".format(",".join(img_nums)))
        self.b_images = Variable(torch.FloatTensor(torch.from_numpy(b_images).float()), volatile=self.test_run)
        self.b_labels_per_class = Variable(torch.FloatTensor(torch.from_numpy(b_labels_per_class).float()),
                                           volatile=self.test_run)

        if save_batch:
            self.save_batch_img_to_files()

        if self.is_cuda:
            self.cuda()

        del b_images
        del b_labels_per_class

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
