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
import torch
if "/home/jogi/.local/lib/python2.7/site-packages" in sys.path:
    sys.path.remove("/home/jogi/.local/lib/python2.7/site-packages")
import matplotlib.pyplot as plt
from matplotlib import cm


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
        self.labels = []
        self.spacings = []
        self.b_pred_labels = None
        self.b_pred_probs = None
        self.b_uncertainty_map = None
        self.b_image = None
        self.b_labels = None
        self.b_new_spacing = None
        self.b_orig_spacing = None
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
            print("{} - {}".format(idx, img_file))
            # first frame is always the end-systolic MRI scan, filename ends with "1"
            mri_scan_es, origin, spacing = self.load_func(img_file, data_type=ACDC2017TestHandler.pixel_dta_type,
                                                          swap_axis=True)
            self.spacings.append(spacing)
            mri_scan_es = self._preprocess(mri_scan_es, spacing, poly_order=3, do_pad=True)
            # print("INFO - Loading ES-file {}".format(img_file))
            reference_es, _, _ = self.load_func(ref_file, data_type=ACDC2017TestHandler.pixel_dta_type,
                                                swap_axis=True)
            reference_es = self._preprocess(reference_es, spacing, poly_order=0, do_pad=False)

            # do the same for the End-diastole pair of images
            img_file, ref_file = file_list[idx+1]
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
        self.b_orig_spacing = self.spacings[image_num]
        self.b_new_spacing = tuple((ACDC2017TestHandler.new_voxel_spacing, ACDC2017TestHandler.new_voxel_spacing,
                                    self.b_orig_spacing[2]))

        if use_labels:
            self.b_labels = self.labels[image_num]
            self.b_pred_probs = np.zeros_like(self.b_labels)
            self.b_pred_labels = np.zeros_like(self.b_labels)
        # TO DO: actually it could happen now that b_uncertainty has undefined shape (because b_labels is not used
        self.b_uncertainty_map = np.zeros_like(self.b_labels)
        batch_dim = self.b_image.shape[3]

        for slice in np.arange(batch_dim):
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

    def set_pred_labels(self, pred_probs, pred_stddev=None, threshold=0.2):
        if isinstance(pred_probs.data, torch.cuda.FloatTensor) or isinstance(pred_probs.data, torch.FloatTensor):
            pred_probs = pred_probs.data.cpu().numpy()
        pred_labels_es = np.argmax(pred_probs[:, 0:self.num_of_classes, :, :], axis=1)
        pred_labels_ed = np.argmax(pred_probs[:, self.num_of_classes:self.num_of_classes+self.num_of_classes,
                                   :, :], axis=1)
        if pred_stddev is not None:
            # ommit pixels with high uncertainty and "just" set them to the most common class = background
            pred_stddev_es = np.mean(pred_stddev[0:self.num_of_classes, :, :], axis=0, keepdims=True)
            true_lbl_es = np.argmax(pred_probs[:, 0:self.num_of_classes, :, :], axis=1)
            pred_labels_es[pred_stddev_es >= threshold] = true_lbl_es[pred_stddev_es >= threshold]
            pred_stddev_ed = np.mean(pred_stddev[self.num_of_classes:self.num_of_classes+self.num_of_classes,
                                     :, :], axis=0, keepdims=True)
            discarded_es = np.count_nonzero(pred_stddev_es >= threshold)
            discarded_ed = np.count_nonzero(pred_stddev_ed >= threshold)
            true_lbl_ed = np.argmax(pred_probs[:, self.num_of_classes:self.num_of_classes+self.num_of_classes,
                                    :, :], axis=1)
            # print("Sum labels before {}".format(np.sum(pred_labels_ed[pred_stddev_ed >= threshold])))
            pred_labels_ed[pred_stddev_ed >= threshold] = true_lbl_ed[pred_stddev_ed >= threshold]
            # print("Sum labels after {}".format(np.sum(pred_labels_ed[pred_stddev_ed >= threshold])))
            print("Discarded ES/ED {} / {}".format(discarded_es, discarded_ed))

        for cls in np.arange(self.num_of_classes):
            self.b_pred_labels[cls, :, :, self.slice_counter] = pred_labels_es == cls
            self.b_pred_labels[cls + self.num_of_classes, :, :, self.slice_counter] = pred_labels_ed == cls

        self.b_pred_probs[:, :, :, self.slice_counter] = pred_probs

    def set_uncertainty_map(self, slice_std):
        """
            Important: we assume slice_std is a numpy array with shape [num_classes, width, height]

        """
        # print(self.b_uncertainty_map.shape)
        self.b_uncertainty_map[:, :, :, self.slice_counter] = slice_std

    def get_accuracy(self, compute_hd=False):
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
            dices[cls] = dice_coefficient(self.b_labels[cls, :, :, :], self.b_pred_labels[cls, :, :, :])
            dices[cls + self.num_of_classes] = dice_coefficient(self.b_labels[cls + self.num_of_classes, :, :, :],
                                                                self.b_pred_labels[cls + self.num_of_classes, :, :, :])
            if compute_hd:
                # only compute distance if both contours are actually in images
                if 0 != np.count_nonzero(self.b_pred_labels[cls, :, :, :]) and \
                        0 != np.count_nonzero(self.b_labels[cls, :, :, :]):
                    hausdff[cls] = hd(self.b_pred_labels[cls, :, :, :], self.b_labels[cls, :, :, :],
                                      voxelspacing=self.b_new_spacing, connectivity=1)
                if 0 != np.count_nonzero(self.b_pred_labels[cls + self.num_of_classes, :, :, :]) and \
                        0 != np.count_nonzero(self.b_labels[cls + self.num_of_classes, :, :, :]):
                    hausdff[cls + self.num_of_classes] = \
                        hd(self.b_pred_labels[cls + self.num_of_classes, :, :, :],
                           self.b_labels[cls + self.num_of_classes, :, :, :],
                           voxelspacing=self.b_new_spacing, connectivity=1)

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
            uncertainty = self.b_uncertainty_map[:, :, :, idx]
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
