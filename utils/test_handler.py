import sys
import numpy as np
from config.config import config
import os
import glob
from tqdm import tqdm
from in_out.read_save_images import load_mhd_to_numpy, write_numpy_to_image
from utils.img_sampling import resample_image_scipy
from torch.autograd import Variable
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
        self.b_image = None
        self.b_labels = None
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

    def generate_batch(self, image_num=0, use_labels=False, use_volatile=False, save_batch=False):
        """
            Remember that self.image is a list, containing images with shape [2, height, width, depth]
            Object self.labels is also a list but with shape [8, height, width, depth]

        """
        self.b_labels = None
        batch_dim = self.images[image_num].shape[3]
        img = self.images[image_num]
        self.b_image = np.reshape(img, (batch_dim, 2, img.shape[1], img.shape[2]))
        if use_labels:
            labels = self.labels[image_num]
            self.b_labels = np.reshape(labels, (labels.shape[3], labels.shape[0], labels.shape[1], labels.shape[2]))
            self.b_pred_labels = np.zeros_like(self.b_labels)

        if save_batch:
            self.save_batch_img_to_files()

    def batch_generator(self, image_num=0, use_labels=True, use_volatile=True):
        """
            Remember that self.image is a list, containing images with shape [2, height, width, depth]
            Object self.labels is also a list but with shape [8, height, width, depth]

        """
        b_label = None
        self.slice_counter = -1
        self.b_image = self.images[image_num]

        if use_labels:
            self.b_labels = self.labels[image_num]
            self.b_pred_labels = np.zeros_like(self.b_labels)
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

    def set_pred_labels(self, pred_labels):

        if isinstance(pred_labels.data, torch.cuda.FloatTensor) or isinstance(pred_labels.data, torch.FloatTensor):
            pred_labels = pred_labels.data.cpu().numpy()

        self.b_pred_labels[:, :, :, self.slice_counter] = pred_labels

    def visualize_test_slices(self, width=8, height=6, num_of_images=None):

        fig = plt.figure(figsize=(width, height))
        counter = 1
        columns = self.num_of_classes + 1
        if self.b_pred_labels is not None:
            rows = 4
            plot_preds = True


    def save_batch_img_to_files(self, num_of_images=None, save_dir=None):

        print("Batch-size {} ".format(len(self.images)))
        if num_of_images is None:
            num_of_images = len(self.images)
        if save_dir is None:
            save_dir = self.config.data_dir

        for i in np.arange(num_of_images):
            # each input contains 2 images: 0=ED and 1=ES
            for phase in np.arange(2):
                if phase == 0:
                    suffix = "ed"
                else:
                    suffix = "es"
                filename_img = os.path.join(save_dir, str(i+1).zfill(2) + "_img_ph_"
                                            + suffix + ".nii")
                img = self.images[i][phase, :, :, :]
                orig_spacing = self.spacings[i]
                new_spacing = tuple((ACDC2017TestHandler.new_voxel_spacing, ACDC2017TestHandler.new_voxel_spacing,
                                     orig_spacing[2]))
                # we need to swap the axis because the image is 3D only
                write_numpy_to_image(img, filename=filename_img, swap_axis=True, spacing=new_spacing)
                cls_offset = phase * 4
                for cls in np.arange(self.num_of_classes):
                    if cls != 0 and cls != 4:
                        cls_lbl = self.labels[i][cls_offset + cls]
                        cls_lbl = np.pad(cls_lbl, ((ACDC2017TestHandler.pad_size, ACDC2017TestHandler.pad_size - 1),
                                               (ACDC2017TestHandler.pad_size, ACDC2017TestHandler.pad_size - 1),
                                               (0, 0)),
                                       'constant', constant_values=(0,)).astype(ACDC2017TestHandler.pixel_dta_type)

                        filename_lbl = os.path.join(save_dir, str(i + 1).zfill(2) + "_lbl_ph"
                                                    + suffix + "_cls" + str(cls) + ".nii")
                        write_numpy_to_image(cls_lbl.astype("float32"), filename=filename_lbl, swap_axis=True,
                                             spacing=new_spacing)
                        # if object that store predictions is non None we save them as well for analysis
                        if self.b_pred_labels is not None:
                            pred_cls_lbl = self.b_pred_labels[i][cls_offset+cls]
                            filename_lbl = os.path.join(self.config.data_dir, str(i + 1).zfill(2) + "_pred_lbl_ph"
                                                        + suffix + "_cls" + str(cls) + ".nii")
                            write_numpy_to_image(pred_cls_lbl.astype("float32"), filename=filename_lbl, swap_axis=True,
                                                 spacing=new_spacing)


# print("Root path {}".format(os.environ.get("REPO_PATH")))
# dataset = ACDC2017TestHandler(exper_config=config, search_mask=config.dflt_image_name + ".mhd", fold_ids=[0],
#                              debug=False, batch_size=4)

# del dataset
