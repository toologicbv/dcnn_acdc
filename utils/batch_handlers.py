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
from in_out.load_data import write_numpy_to_image
# from in_out.load_data import ACDC2017DataSet


class BatchHandler(object):
    __metaclass__ = abc.ABCMeta

    # static class variable to count the batches
    id = 0

    @abc.abstractmethod
    def cuda(self):
        pass

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def backward(self, *args):
        pass


class TwoDimBatchHandler(BatchHandler):

    # we use a zero-padding of 65 on both dimensions, equals 130 positions
    patch_size_with_padding = 281
    patch_size = 150
    pixel_dta_type = "float32"

    def __init__(self, exper, is_train=True, batch_size=None, num_classes=4):
        if batch_size is None:
            self.batch_size = exper.run_args.batch_size
        else:
            self.batch_size = batch_size

        self.is_train = is_train
        self.num_classes = num_classes
        self.ps_wp = TwoDimBatchHandler.patch_size_with_padding
        self.patch_size = TwoDimBatchHandler.patch_size
        self.is_cuda = exper.run_args.cuda

        # batch image patch
        self.b_images = None
        # batch reference image for the different classes (so for each reference class 1 image)
        self.b_labels_per_class = None
        # this objects holds for each image-slice the separate class labels, so one set for each class
        self.b_labels_per_class = None
        self.config = exper.config

    def cuda(self):
        self.b_images = self.b_images.cuda()
        self.b_labels_per_class = self.b_labels_per_class.cuda()

    def __call__(self, exper, network):
        print("Batch size {}".format(exper.run_args.batch_size))

    def backward(self, *args):
        pass

    def generate_batch_2d(self, images, labels, save_batch=False):
        """
        Variable images and labels are LISTS containing image slices, hence len(images) = number of total slices
        Each image slice (and label slice) has the following tensor dimensions:

         images:    [2 (ES and ED channels), x-axis, y-axis]
         labels:    [2 (ES and ED channels),  x-axis, y-axis]

         Note that labels contains segmentation class values between 0 and 3 for the four classes.
         We will convert these values into binary values below and hence add (next to the batch dimension dim-0)
         a second dimension with the number of classes (in our case 4 * 2 = 8), see b_labels_per_class object.
        """

        b_images = np.zeros((self.batch_size, 2, self.ps_wp, self.ps_wp))
        b_labels_per_class = np.zeros((self.batch_size, self.num_classes * 2, self.patch_size + 1, self.patch_size + 1))

        num_images = len(images)
        # img_nums = []
        for idx in range(self.batch_size):
            ind = np.random.randint(0, num_images)
            # img_nums.append(str(ind))
            img = images[ind]
            label = labels[ind]
            # note img tensor has 3-dim and the first is equal to 2, because we concatenated ED and ES images
            offx = np.random.randint(0, img.shape[1] - self.ps_wp)
            offy = np.random.randint(0, img.shape[2] - self.ps_wp)

            img = img[:, offx:offx + self.ps_wp, offy:offy + self.ps_wp]

            b_images[idx, :, :, :] = img
            # next convert class labels to binary class labels
            label_es = label[0, offx:offx + self.patch_size + 1, offy:offy + self.patch_size + 1]
            label_ed = label[1, offx:offx + self.patch_size + 1, offy:offy + self.patch_size + 1]
            for cls_idx in np.arange(self.num_classes):
                # store ES class labels in first 4 positions of dim-1
                b_labels_per_class[idx, cls_idx, :, :] = (label_es == cls_idx).astype('int16')
                # sotre ED class labels in positions 4-7 of dim-1
                b_labels_per_class[idx, cls_idx+self.num_classes, :, :] = (label_ed == cls_idx).astype('int16')

        # print("Images used {}".format(",".join(img_nums)))
        self.b_images = Variable(torch.FloatTensor(torch.from_numpy(b_images).float()))
        self.b_labels_per_class = Variable(torch.FloatTensor(torch.from_numpy(b_labels_per_class).float()))

        if save_batch:
            self.save_batch_img_to_files()

        if self.is_cuda:
            self.cuda()

        del b_images
        del b_labels_per_class

    def save_batch_img_to_files(self):
        for i in np.arange(self.batch_size):
            filename_img = os.path.join(self.config.data_dir, "b_img" + str(i+1).zfill(2) + ".nii")
            write_numpy_to_image(self.b_images[i].data.cpu().numpy(), filename=filename_img)

            lbl = self.b_labels[i].data.cpu().numpy()
            print(np.unique(lbl))
            for l in np.unique(lbl):
                if l != 0:
                    cls_lbl = self.b_labels_per_class[i, l].data.cpu().numpy()
                    filename_lbl = os.path.join(self.config.data_dir, "b_lbl" + str(i + 1).zfill(2) + "_" + str(l) + ".nii")
                    lbl = np.pad(cls_lbl, 65, 'constant', constant_values=(0,)).astype("float32")
                    write_numpy_to_image(lbl, filename=filename_lbl)

    def visualize_batch(self, width=8, height=6, num_of_images=1):
        """

        Visualizing an image + label(s) from the batch in order to visually inspect the batch.
        Note, b_images is [batch_size, 2, x, y]
              b_labels_per_class is [batch_size, 8, x, y]
        """

        fig = plt.figure(figsize=(width, height))
        counter = 1
        columns = self.num_classes + 1
        num_of_subplots = 2 * num_of_images * self.num_classes

        for idx in np.arange(num_of_images):
            # start to inspect only the ES image (index 0)
            img = self.b_images[idx].data.cpu().numpy()

            ax1 = plt.subplot(num_of_subplots, columns, counter)
            plt.imshow(img[0], cmap=cm.gray)
            counter += 1
            labels = self.b_labels_per_class[idx].data.cpu().numpy()
            for cls1 in np.arange(self.num_classes):
                _ = plt.subplot(num_of_subplots, columns, counter, sharex=ax1, sharey=ax1)
                plt.imshow(labels[cls1], cmap=cm.gray)
                counter += 1

            cls1 += 1
            ax2 = plt.subplot(num_of_subplots, columns, counter)
            plt.imshow(img[1], cmap=cm.gray)
            counter += 1
            for cls2 in np.arange(self.num_classes):
                _ = plt.subplot(num_of_subplots, columns, counter, sharex=ax2, sharey=ax2)
                plt.imshow(labels[cls1 + cls2], cmap=cm.gray)
                counter += 1
        plt.show()
