import abc
import os
import time
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
        self.b_labels = None
        # this objects holds for each image-slice the separate class labels, so one set for each class
        self.b_labels_per_class = None
        self.config = exper.config

    def cuda(self):
        self.b_images = self.b_images.cuda()
        self.b_labels = self.b_labels.cuda()

    def __call__(self, exper, network):
        print("Batch size {}".format(exper.run_args.batch_size))

    def backward(self, *args):
        pass

    def generate_batch_2d(self, images, labels, save_batch=False):
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
            # I WAS HERE......<<<<<<<<<<<< TOMORROW
            label = label[offx:offx + self.patch_size + 1, offy:offy + self.patch_size + 1]

            b_labels[idx, 0, :, :] = label

            for cls in range(self.num_classes):
                b_labels_per_class[idx, cls, :, :] = (label == cls).astype('int16')
        # print("Images used {}".format(",".join(img_nums)))
        self.b_images = Variable(torch.FloatTensor(torch.from_numpy(b_images).float()))
        self.b_labels_per_class = Variable(torch.LongTensor(torch.from_numpy(b_labels_per_class.astype(int))))

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


