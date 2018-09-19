import os
import numpy as np
import torch

from in_out.hvsmr.load_data import write_numpy_to_image, HVSMR2016DataSet
from utils.batch_handlers import TwoDimBatchHandler


class HVSMRTwoDimBatchHandler(TwoDimBatchHandler):

    # we use a zero-padding of 65 on both dimensions, equals 130 positions
    patch_size_with_padding = 201
    patch_size = 70
    pixel_dta_type = "float32"

    def __init__(self, exper, batch_size=None, num_classes=3):
        super(HVSMRTwoDimBatchHandler, self).__init__(exper, batch_size=batch_size, num_classes=num_classes)
        # Important, because the super class set these to ACDC dataset specific values.
        self.ps_wp = HVSMRTwoDimBatchHandler.patch_size_with_padding
        self.patch_size = HVSMRTwoDimBatchHandler.patch_size

    def generate_batch_2d(self, images, labels, save_batch=False, num_of_slices=None, slice_range=None):
        b_images = np.zeros((self.batch_size, 1, self.ps_wp, self.ps_wp))
        b_labels_per_class = np.zeros((self.batch_size, self.num_classes, self.patch_size + 1, self.patch_size + 1))
        b_labels = np.zeros((self.batch_size, self.patch_size + 1, self.patch_size + 1))
        if num_of_slices is None:
            num_of_slices = len(images)
        # img_nums = []
        for idx in range(self.batch_size):
            if slice_range is None:
                ind = np.random.randint(0, num_of_slices)
            else:
                ind = slice_range[idx]

            # img_nums.append(str(ind))
            img = images[ind]
            label = labels[ind]

            offx = np.random.randint(0, img.shape[0] - self.ps_wp)
            offy = np.random.randint(0, img.shape[1] - self.ps_wp)

            img = img[offx:offx + self.ps_wp, offy:offy + self.ps_wp]

            b_images[idx, 0, :, :] = img
            label = label[offx:offx + self.patch_size + 1, offy:offy + self.patch_size + 1]

            b_labels[idx, :, :] = label

            for cls in range(self.num_classes):
                b_labels_per_class[idx, cls, :, :] = (label == cls).astype('int16')
        # print("Images used {}".format(",".join(img_nums)))
        self.b_images = torch.FloatTensor(torch.from_numpy(b_images).float())
        self.b_labels = torch.LongTensor(torch.from_numpy(b_labels.astype(np.int)).long())
        self.b_labels_per_class = torch.LongTensor(torch.from_numpy(b_labels_per_class.astype(int)))
        if save_batch:
            self.save_batch_img_to_files()

        if self.is_cuda:
            self.cuda()

        del b_images
        del b_labels
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
                    filename_lbl = os.path.join(self.config.data_dir, "b_lbl" + str(i + 1).zfill(2) + "_"
                                                + str(l) + ".nii")
                    lbl = np.pad(cls_lbl, HVSMR2016DataSet.pad_size, 'constant', constant_values=(0,)).astype("float32")
                    write_numpy_to_image(lbl, filename=filename_lbl)
