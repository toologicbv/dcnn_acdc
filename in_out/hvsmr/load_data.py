import sys
from collections import OrderedDict
import numpy as np
import os
import glob
import SimpleITK as sitk

from tqdm import tqdm

from torch.utils.data import Dataset
from common.hvsmr.config import config_hvsmr

from in_out.load_data import BaseImageDataSet
from in_out.read_save_images import load_mhd_to_numpy


def write_numpy_to_image(np_array, filename, swap_axis=False, spacing=None):

    if swap_axis:
        np_array = np.swapaxes(np_array, 0, 2)
    img = sitk.GetImageFromArray(np_array)
    if spacing is not None:
        img.SetSpacing(spacing)
    sitk.WriteImage(img, filename)
    print("Successfully saved image to {}".format(filename))


def crawl_dir(in_dir, load_func="load_itk", pattern="*.mhd", logger=None):
    """
    Searches for files that match the pattern-parameter and assumes that there also
    exist a <filename>.raw for this mhd file
    :param in_dir:
    :param load_func:
    :param pattern:
    :param logger:
    :return: python list with
    """
    im_arrays = []
    gt_im_arrays = []

    pattern = os.path.join(in_dir, pattern)
    for fname in glob.glob(pattern):
        mri_scan, origin, spacing = load_func(fname)
        logger.info("Loading {}".format(fname, ))
        im_arrays.append((mri_scan, origin, spacing))

    return im_arrays, gt_im_arrays


def rescale_image(img, perc_low=5, perc_high=95, axis=None):
    # flatten 3D image to 1D and determine percentiles for recaling
    lower, upper = np.percentile(img, [perc_low, perc_high], axis=axis)
    # set new normalized image
    img = (img - lower) * 1. / (upper - lower)
    return img


def normalize_image(img, axis=None):
    img = (img - np.mean(img, axis=axis)) / np.std(img, axis=axis)
    return img


class HVSMR2016DataSet(BaseImageDataSet):

    train_path = "train"
    val_path = "validate"
    image_path = "images"
    label_path = "reference"

    pixel_dta_type = 'float32'
    pad_size = config_hvsmr.pad_size
    new_voxel_spacing = 0.65
    label_background = config_hvsmr.class_lbl_background
    label_myocardium = config_hvsmr.class_lbl_myocardium
    label_bloodpool = config_hvsmr.class_lbl_bloodpool

    def __init__(self, exper_config, search_mask=None, nclass=3, load_func=load_mhd_to_numpy,
                 fold_id=0, preprocess="normalize", debug=False, do_augment=True,
                 load_type="nifty", logger=None):
        """
        The images are already resampled to an isotropic 3D size of 0.65mm x 0.65 x 0.65


        :param data_dir: root directory
        :param search_mask:
        :param nclass:
        :param fold_id:
        :param exper_config:
        :param load_func: currently only load_mhd_to_numpy is supported
        :param preprocess: takes arguments "normalize" or "rescale"
        :param load_type: takes "nifty" or "numpy"
        """
        super(HVSMR2016DataSet, self).__init__()
        self.name = "HVSMR"
        self.logger = logger
        self.data_dir = os.path.join(exper_config.root_dir, exper_config.data_dir)
        self.fold_id = fold_id
        self.abs_path_fold = os.path.join(self.data_dir, "fold")
        self.num_of_classes = nclass
        self.class_count = np.zeros((2, self.num_of_classes))  # 0=train-set 1=validation set
        self.norm_scale = preprocess
        self.search_mask = search_mask
        if self.search_mask is None:
            self.search_mask = exper_config.dflt_image_name + ".nii"
        self.load_func = load_func
        self.load_type = load_type
        self.do_augment = do_augment
        self.num_of_augmentations = 3  # four rotations 90, 180, 270 degrees of rotations
        self.image_names = []
        # Note, this list will contain len() image slices...2D!
        # IMPORTANT: this is NOT the total number of slices in train_images or val_images, but in fact the number
        # of patients or files loaded from disk (where ES/ED files (2) count as 1)!
        self.num_of_images = 0
        self.train_images = []
        # the actual number of slices in train_images
        self.train_num_slices = 0
        self.train_labels = []
        # dictionary with key is patientID and value is imageID that we assign when loading the stuff
        self.trans_dict = OrderedDict()
        # we use the img-slice id, belonging to a training image-patch to track the statistics (how much to we train
        # on certain image/slices? We'll store tuples
        self.train_img_slice_ids = []
        # mean width, height, #slices per image
        self.img_stats = np.zeros(3)
        self.img_slice_stats = {}
        self.train_spacings = []
        self.val_images = []
        self.val_labels = []
        self.val_img_slice_ids = []
        # patient_ids in the validation/test set
        self.val_image_names = []
        # the actual number of slices in val_images
        self.val_num_slices = 0
        self.val_spacings = []
        self.debug = debug
        self.num_images_train = 0
        self.num_images_val = 0
        self.voxelspacing = tuple((HVSMR2016DataSet.new_voxel_spacing, HVSMR2016DataSet.new_voxel_spacing))
        self._set_pathes()

        # print("Validation indices {}".format(self.val_image_indices))
        if self.load_type == "nifty":
            # Important detail: we need to swap axis 0 and 2 of the HVSMR2016 files
            self.load_images_from_dir(swap_axis=True)
        elif self.load_type == "numpy":
            self.load_numpy_arr_from_dir()
            if len(self.images) == 0:
                self.info("Info - cannot find any numpy npz files. Looking for raw files...")
                self.load_images_from_dir(swap_axis=True)
        else:
            raise ValueError("Load mode {} is not supported".format(self.load_type))

    def _set_pathes(self):

        if self.load_type == "numpy":
            self.rel_image_path = HVSMR2016DataSet.image_path + "_np"
            self.rel_label_path = HVSMR2016DataSet.label_path + "_np"
        else:
            self.rel_image_path = HVSMR2016DataSet.image_path
            self.rel_label_path = HVSMR2016DataSet.label_path

        if self.debug:
            # load images from a directory that only contains a couple of images
            self.rel_image_path = HVSMR2016DataSet.image_path + "_quick"
            self.rel_label_path = HVSMR2016DataSet.label_path + "_quick"

    def load_images_from_dir(self, swap_axis=True):
        files_loaded = 0
        self.info("INFO - Using folds {} - busy loading images/references...this may take a while!".format(self.fold_id))
        train_file_list, val_file_list = self._get_file_lists()
        # load training set
        files_loaded += self._load_file_list(train_file_list, is_train=True, swap_axis=swap_axis)
        self.train_num_slices = len(self.train_images)
        # load validation/test set
        files_loaded += self._load_file_list(val_file_list, is_train=False, swap_axis=swap_axis)
        self.info("INFO - Using fold {} - loaded {} files: {} studies in train set, {} in validation set".format(
            self.fold_id, files_loaded, self.num_images_train, self.num_images_val))
        self.train_num_slices = len(self.train_images)
        self.val_num_slices = len(self.val_images)

    def _get_file_lists(self):

        train_file_list = []
        val_file_list = []
        self.train_path = os.path.join(self.abs_path_fold + str(self.fold_id),
                                       os.path.join(HVSMR2016DataSet.train_path, self.rel_image_path))
        self.val_path = os.path.join(self.abs_path_fold + str(self.fold_id),
                                     os.path.join(HVSMR2016DataSet.val_path, self.rel_image_path))

        search_mask_img = os.path.join(self.train_path, self.search_mask)
        self.info("INFO - Creating file list. Search for {} ".format(search_mask_img))
        for train_file in glob.glob(search_mask_img):
            ref_file = train_file.replace(self.rel_image_path, self.rel_label_path)
            ref_file = ref_file.replace("_img", "_lbl")
            train_file_list.append(tuple((train_file, ref_file)))

        # get validation images and labels
        search_mask_img = os.path.join(self.val_path, self.search_mask)
        for val_file in glob.glob(search_mask_img):
            ref_file = val_file.replace(self.rel_image_path, self.rel_label_path)
            ref_file = ref_file.replace("_img", "_lbl")
            val_file_list.append(tuple((val_file, ref_file)))

        return train_file_list, val_file_list

    def _load_file_list(self, file_list, is_train=True, swap_axis=True, verbose=False):
        files_loaded = 0
        file_list.sort()

        # Note: file_list contains 200 entries if we load 100 images.
        if self.norm_scale == "normalize":
            self.info("INFO - Normalizing images intensity values (is_train={})".format(is_train))
        elif self.norm_scale == "rescale":
            self.info("INFO - Rescaling images intensity values (is_train={})".format(is_train))
        else:
            self.info("INFO - Images are NOT normalized/rescales!")

        for idx in tqdm(np.arange(0, len(file_list))):
            img_file, ref_file = file_list[idx]
            if verbose:
                self.info("INFO - Loading image {}".format(img_file))
            mri_scan, origin, spacing = self.load_func(img_file, data_type=HVSMR2016DataSet.pixel_dta_type,
                                                       swap_axis=swap_axis)
            if self.norm_scale == "normalize":
                mri_scan = normalize_image(mri_scan, axis=None)
            elif self.norm_scale == "rescale":
                mri_scan = rescale_image(mri_scan, axis=None)

            # get rid off _iso_img and take only the patient name
            patient_id = img_file[:img_file.find("_")]
            if is_train:
                self.image_names.append(patient_id)
                self.num_images_train += 1
                count_idx = 0
            else:
                self.val_image_names.append(patient_id)
                self.num_images_val += 1
                count_idx = 1
            self.trans_dict[patient_id] = self.num_of_images
            if verbose:
                self.info("INFO - Loading reference {}".format(ref_file))
            reference, origin, spacing = self.load_func(ref_file, data_type=HVSMR2016DataSet.pixel_dta_type,
                                                        swap_axis=swap_axis)

            self._augment_data(mri_scan, reference, pad_size=HVSMR2016DataSet.pad_size, is_train=is_train)

            for class_label in range(self.num_of_classes):
                self.class_count[count_idx, class_label] += np.sum(reference == class_label)

            files_loaded += 1

        return files_loaded

    def _augment_data(self, image, label, pad_size=0, is_train=True):
        """
        Adds all original and rotated image slices to self.images and self.labels objects
        :param image:
        :param label:
        :param pad_size:
        :return:
        """

        def rotate_slice(img_slice, lbl_slice, is_train=True):
            # PAD IMAGE
            for rots in range(4):
                # no padding here but when we extract patches during BatchGeneration
                section = np.pad(img_slice, pad_size, 'constant', constant_values=(0,)).astype(
                    HVSMR2016DataSet.pixel_dta_type)
                if is_train:
                    self.train_images.append(section)
                    self.train_labels.append(lbl_slice)
                else:
                    # store the complete image for testing as well
                    self.val_images.append(section)
                    self.val_labels.append(lbl_slice)

                # rotate for next iteration
                img_slice = np.rot90(img_slice)
                lbl_slice = np.rot90(lbl_slice)

        # for each image-slice rotate the img four times. We're doing that for all three orientations
        for z in range(image.shape[2]):
            label_slice = label[:, :, z]
            image_slice = image[:, :, z]
            rotate_slice(image_slice, label_slice, is_train=is_train)

        for y in range(image.shape[1]):
            label_slice = np.squeeze(label[:, y, :])
            image_slice = np.squeeze(image[:, y, :])
            rotate_slice(image_slice, label_slice, is_train=is_train)

        for x in range(image.shape[0]):
            label_slice = np.squeeze(label[x, :, :])
            image_slice = np.squeeze(image[x, :, :])
            rotate_slice(image_slice, label_slice, is_train=is_train)

    def load_numpy_arr_from_dir(self, file_prefix=None, abs_path=None):
        if file_prefix is None:
            file_prefix = config_hvsmr.numpy_save_filename

        if abs_path is None:
            abs_path = self.data_dir

        if self.mode == "train":
            out_dir = os.path.join(abs_path, "train")
        else:
            out_dir = os.path.join(abs_path, "test")

        search_mask = os.path.join(out_dir, file_prefix + "*.npz")
        print(">>>>>>>>>>>> Info - Looking for files with search_mask {}".format(search_mask))
        for fname in glob.glob(search_mask):
            print(">>>>>>>>>>>>>>> Info - Loading numpy objects from {}".format(fname))
            numpy_ar = np.load(fname)
            self.images.extend(list(numpy_ar["images"]))
            self.labels.extend(list(numpy_ar["labels"]))
            if self.class_count is None:
                self.class_count = numpy_ar["class_count"]

    def save_to_numpy(self, file_prefix=None, abs_path=None):

        if file_prefix is None:
            file_prefix = config_hvsmr.numpy_save_filename

        if abs_path is None:
            abs_path = self.data_dir

        if self.mode == "train":
            out_filename = os.path.join(abs_path, "train")
        else:
            out_filename = os.path.join(abs_path, "test")

        try:
            chunksize = 1000
            start = 0
            end = chunksize
            for c, chunk in enumerate(np.arange(chunksize)):
                filename = os.path.join(out_filename, file_prefix + str(chunk) + ".npz")
                print("> > > Info - Save (chunked) data to directory {}".format(filename))
                np.savez(filename, images=self.images[start:end],
                         labels=self.labels[start:end],
                         class_count=self.class_count)

                start += chunksize
                end += chunksize
                if c == 3:
                    break
        except IOError:
            raise IOError("Can't save {}".format(filename))

    def create_test_slices(self):
        test_img_slices, test_lbl_slices = [], []

        for i in np.arange(len(self.test_images)):
            image = np.pad(self.test_images[i], ((0, 0),
                                                 (HVSMR2016DataSet.pad_size, HVSMR2016DataSet.pad_size),
                                                 (HVSMR2016DataSet.pad_size, HVSMR2016DataSet.pad_size)),
                           'constant', constant_values=(0,)).astype(HVSMR2016DataSet.pixel_dta_type)
            for x in np.arange(self.test_images[i].shape[0]):
                slice_padded = image[x, :, :]
                lbl_slice = self.test_labels[i][x, :, :]
                test_img_slices.append(slice_padded)
                test_lbl_slices.append(lbl_slice)

        return test_img_slices, test_lbl_slices

    def __len__(self, is_train=True):
        if is_train:
            return self.num_images_train
        else:
            return self.num_images_val

    def images(self, train=True):
        if train:
            return self.train_images
        else:
            return self.val_images

    def get_num_of_slices(self, train=True):
        if train:
            return len(self.train_images)
        else:
            return len(self.val_images)

    def labels(self, train=True):
        if train:
            return self.train_labels
        else:
            return self.val_labels

    @staticmethod
    def remove_padding(image):
        return image[HVSMR2016DataSet.pad_size:-HVSMR2016DataSet.pad_size,
                     HVSMR2016DataSet.pad_size:-HVSMR2016DataSet.pad_size]


if __name__ == '__main__':
    dataset = HVSMR2016DataSet(config_hvsmr, search_mask=config_hvsmr.dflt_image_name + ".nii",
                               fold_id=0, preprocess="rescale",
                               debug=True)

    del dataset

