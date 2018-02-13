import sys
import SimpleITK as sitk
import numpy as np
import os
import glob
if "/home/jogi/.local/lib/python2.7/site-packages" in sys.path:
    sys.path.remove("/home/jogi/.local/lib/python2.7/site-packages")

from sklearn.model_selection import KFold

from torch.autograd import Variable
import torch
from torch.utils.data import Dataset
from config.config import config
from utils.img_sampling import resample_image_scipy
from in_out.read_save_images import save_img_as_mhg
from in_out.read_save_images import load_mhd_to_numpy
# from losses.dice_metric import dice_coeff


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


class BaseImageDataSet(Dataset):
    def __init__(self, data_dir, conf_obj=None):
        if data_dir is None:
            if not conf_obj is None:
                data_dir = config.data_dir
            else:
                raise ValueError("parameter {} is None, cannot load images".format(data_dir))
        assert os.path.exists(data_dir), "{} does not exist".format(data_dir)
        self.data_dir = data_dir
        self.search_mask = None
        self.images = None
        self.images_raw = None
        self.labels = None
        self.origins = None
        self.spacings = None

    def __getitem__(self, index):
        assert index <= self.__len__()
        return tuple((self.images[index], self.labels[index]))

    def __len__(self):
        return len(self.images)

    def crawl_directory(self):
        pass

    def save_to_numpy(self, file_prefix=None, abs_path=None):
        if file_prefix is None:
            file_prefix = "img_np"
        if abs_path is None:
            abs_path = config.data_dir
        if not os.path.exists(abs_path):
            raise IOError("{} does not exist".format(abs_path))

        for idx in np.arange(self.__len__()):
            filename = os.path.join(abs_path, file_prefix + str(idx+1) + ".npz")
            image = self.images[idx]
            label = self.labels[idx]
            try:
                np.savez(filename, image=image, label=label)
            except IOError:
                raise IOError("Can't save {}".format(filename))


class ACDC2017DataSet(BaseImageDataSet):

    train_path = "train"
    val_path = "validate"
    image_path = "images_iso"
    label_path = "reference_iso"

    pixel_dta_type = 'float32'
    pad_size = config.pad_size
    new_voxel_spacing = 1.4

    def __init__(self, config, search_mask=None, nclass=4, load_func=load_mhd_to_numpy,
                 fold_ids=[0], preprocess=False, debug=False):

        super(BaseImageDataSet, self).__init__()
        self.data_dir = os.path.join(config.root_dir, config.data_dir)
        self.search_mask = search_mask
        self.num_of_classes = nclass
        self.fold_ids = fold_ids
        self.load_func = load_func
        self.abs_path_fold = os.path.join(self.data_dir, "fold")

        self.train_images = []
        self.train_labels = []
        self.train_spacings = []
        self.val_images = []
        self.val_labels = []
        self.val_spacings = []
        self.preprocess = preprocess
        self.debug = debug
        self._set_pathes()
        # The images from Jelmer are already normalized (per image) but they are not isotropic
        # Hence with the "pre-process" option the "Jelmer" images are loaded and resampled to an in-plane
        # 1.4 mm^2 spacing
        if preprocess:
            self.pre_process()
        else:
            self.load_files()

    def _set_pathes(self):
        if self.preprocess:
            ACDC2017DataSet.image_path = ACDC2017DataSet.image_path.replace("_iso", "")
            ACDC2017DataSet.label_path = ACDC2017DataSet.label_path.replace("_iso", "")

        if self.debug:
            # load images from a directory that only contains a couple of images
            ACDC2017DataSet.image_path = ACDC2017DataSet.image_path + "_test"
            ACDC2017DataSet.label_path = ACDC2017DataSet.label_path + "_test"

    def _get_file_lists(self):

        train_file_list = []
        val_file_list = []
        for fold_id in self.fold_ids:
            self.train_path = os.path.join(self.abs_path_fold + str(fold_id),
                                           os.path.join(ACDC2017DataSet.train_path, ACDC2017DataSet.image_path))
            self.val_path = os.path.join(self.abs_path_fold + str(fold_id),
                                         os.path.join(ACDC2017DataSet.val_path, ACDC2017DataSet.image_path))
            # get training images and labels
            search_mask_img = os.path.join(self.train_path, self.search_mask)
            # print("INFO - >>> Search with dir+pattern {} <<<".format(search_mask_img))
            for train_file in glob.glob(search_mask_img):
                ref_file = train_file.replace(ACDC2017DataSet.image_path, ACDC2017DataSet.label_path)
                train_file_list.append(tuple((train_file, ref_file)))

            # get validation images and labels
            search_mask_img = os.path.join(self.val_path, self.search_mask)
            for val_file in glob.glob(search_mask_img):
                ref_file = val_file.replace(ACDC2017DataSet.image_path, ACDC2017DataSet.label_path)
                val_file_list.append(tuple((val_file, ref_file)))

        return train_file_list, val_file_list

    def _load_file_list(self, file_list, is_train=True):
        files_loaded = 0
        file_list.sort()

        for idx in np.arange(0, len(file_list), 2):
            # tuple contains [0]=train file name and [1] reference file name
            img_file, ref_file = file_list[idx]
            # first frame is always the end-systolic MRI scan, filename ends with "1"
            mri_scan_es, origin, spacing = self.load_func(img_file, data_type=ACDC2017DataSet.pixel_dta_type,
                                                          swap_axis=True)
            # print("INFO - Loading ES-file {}".format(img_file))
            reference_es, origin, spacing = self.load_func(ref_file, data_type=ACDC2017DataSet.pixel_dta_type,
                                                           swap_axis=True)
            # do the same for the End-Systolic pair of images
            img_file, ref_file = file_list[idx+1]
            mri_scan_ed, origin, spacing = self.load_func(img_file, data_type=ACDC2017DataSet.pixel_dta_type,
                                                          swap_axis=True)
            # print("INFO - Loading ED_file {}".format(img_file))
            reference_ed, origin, spacing = self.load_func(ref_file, data_type=ACDC2017DataSet.pixel_dta_type,
                                                           swap_axis=True)
            # AUGMENT data and add to train, validation or test if applicable
            self._augment_data(mri_scan_ed, reference_ed, mri_scan_es, reference_es,
                               is_train=is_train)

            files_loaded += 2
        return files_loaded

    def load_files(self):
        files_loaded = 0

        train_file_list, val_file_list = self._get_file_lists()
        files_loaded += self._load_file_list(train_file_list, is_train=True)
        files_loaded += self._load_file_list(val_file_list, is_train=False)
        print("INFO - Using folds {} - loaded {} files: {} slices in train set, {} slices in validation set".format(
            self.fold_ids, files_loaded, len(self.train_images), len(self.val_images)))

    def _augment_data(self, image_ed, label_ed, image_es, label_es, is_train=False):
        """
        Augments image slices by rotating z-axis slices for 90, 180 and 270 degrees

        image_ed, label_ed, image_es and label_es are 3 dimensional tensors [x,y,z]
        label tensors contain the class labels for the segmentation from 0-3 for the 4 segmentation classes
        0 = background, 1 = left ventricle, 2 = right ventricle, 3 = myocardium

        """

        def rotate_slice(img_ed_slice, lbl_ed_slice, img_es_slice, lbl_es_slice,
                         is_train=False):

            for rots in range(4):
                pad_img_ed_slice = np.pad(img_ed_slice, ACDC2017DataSet.pad_size, 'constant',
                                          constant_values=(0,)).astype(ACDC2017DataSet.pixel_dta_type)
                pad_img_es_slice = np.pad(img_es_slice, ACDC2017DataSet.pad_size, 'constant',
                                          constant_values=(0,)).astype(ACDC2017DataSet.pixel_dta_type)
                # we make a 3dim tensor (first dim has one-size) and concatenate ED and ES image
                pad_img_slice = np.concatenate((np.expand_dims(pad_img_ed_slice, axis=0),
                                                np.expand_dims(pad_img_es_slice, axis=0)))
                # same concatenation for the label files of ED and ES
                label_slice = np.concatenate((np.expand_dims(lbl_ed_slice, axis=0),
                                                np.expand_dims(lbl_es_slice, axis=0)))
                if is_train:
                    self.train_images.append(pad_img_slice)
                    self.train_labels.append(label_slice)
                else:
                    self.val_images.append(pad_img_slice)
                    self.val_labels.append(label_slice)
                # rotate for next iteration
                img_ed_slice = np.rot90(img_ed_slice)
                lbl_ed_slice = np.rot90(lbl_ed_slice)
                img_es_slice = np.rot90(img_es_slice)
                lbl_es_slice = np.rot90(lbl_es_slice)

        # for each image-slice rotate the img four times. We're doing that for all three orientations
        for z in range(image_ed.shape[2]):
            image_ed_slice = image_ed[:, :, z]
            label_ed_slice = label_ed[:, :, z]
            image_es_slice = image_es[:, :, z]
            label_es_slice = label_es[:, :, z]

            rotate_slice(image_ed_slice, label_ed_slice, image_es_slice, label_es_slice, is_train)

    def _resample_images(self, file_list):
        files_loaded = 0
        file_list.sort()
        for i, file_tuple in enumerate(file_list):
            # tuple contains [0]=train file name and [1] reference file name
            print("INFO - Re-sampling file {}".format(file_tuple[0]))
            mri_scan, origin, spacing = self.load_func(file_tuple[0], data_type=ACDC2017DataSet.pixel_dta_type,
                                                       swap_axis=True)
            zoom_factors = tuple((spacing[0] / ACDC2017DataSet.new_voxel_spacing,
                                 spacing[1] / ACDC2017DataSet.new_voxel_spacing, 1))
            new_spacing = tuple((ACDC2017DataSet.new_voxel_spacing, ACDC2017DataSet.new_voxel_spacing,
                                 spacing[2]))
            out_filename = file_tuple[0]
            out_filename = out_filename.replace("images", "images_iso")
            mri_scan = resample_image_scipy(mri_scan, new_spacing=zoom_factors, order=3)
            save_img_as_mhg(mri_scan, new_spacing, origin, out_filename, swap_axis=True)
            # print("INFO - Loading file {}".format(file_tuple[1]))
            reference, origin, spacing = self.load_func(file_tuple[1], data_type=ACDC2017DataSet.pixel_dta_type,
                                                        swap_axis=True)
            out_filename = file_tuple[1]
            out_filename = out_filename.replace("reference", "reference_iso")
            reference = resample_image_scipy(reference, new_spacing=zoom_factors, order=0)
            save_img_as_mhg(reference, new_spacing, origin, out_filename, swap_axis=True)
            files_loaded += 1
        return files_loaded

    def pre_process(self):
        print("INFO - Resampling images")
        train_file_list, val_file_list = self._get_file_lists()
        files_loaded = self._resample_images(train_file_list)
        files_loaded += self._resample_images(val_file_list)
        print("INFO - Using folds {} - loaded {} files".format(self.fold_ids, files_loaded))

    def __len__(self):
        return len(self.train_images), len(self.val_images)

    def images(self, train=True):
        if train:
            return self.train_images
        else:
            return self.val_images

    def labels(self, train=True):
        if train:
            return self.train_labels
        else:
            return self.val_labels


# dataset = ACDC2017DataSet(config=config, search_mask=config.dflt_image_name + ".mhd", fold_id=0,
#                           preprocess=False)

# del dataset

