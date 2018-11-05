import sys
from collections import OrderedDict
import numpy as np
import os
import glob
if "/home/jogi/.local/lib/python2.7/site-packages" in sys.path:
    sys.path.remove("/home/jogi/.local/lib/python2.7/site-packages")

from tqdm import tqdm

from torch.utils.data import Dataset
from config.config import config
from utils.img_sampling import resample_image_scipy
from in_out.read_save_images import save_img_as_mhg
from in_out.read_save_images import load_mhd_to_numpy


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
    # flatten 3D image to 1D and determine percentiles for rescaling
    lower, upper = np.percentile(img, [perc_low, perc_high], axis=axis)
    # set new normalized image
    img = (img - lower) * 1. / (upper - lower)
    return img


def normalize_image(img, axis=None):
    img = (img - np.mean(img, axis=axis)) / np.std(img, axis=axis)
    return img


def detect_incomplete_slices(labels_es, labels_ed):
    es_rv_count, es_myo_count, es_lv_count = np.count_nonzero(labels_es == config.class_lbl_RV), \
                                             np.count_nonzero(labels_es == config.class_lbl_myo), \
                                             np.count_nonzero(labels_es == config.class_lbl_LV)
    ed_rv_count, ed_myo_count, ed_lv_count = np.count_nonzero(labels_ed == config.class_lbl_RV), \
                                             np.count_nonzero(labels_ed == config.class_lbl_myo), \
                                             np.count_nonzero(labels_ed == config.class_lbl_LV)
    high_incompleteness_es = False
    incomplete_es = False
    high_incompleteness_ed = False
    incomplete_ed = False
    if es_rv_count == 0 or es_myo_count == 0 or es_lv_count == 0:
        incomplete_es = True
        if (es_rv_count == 0 and es_myo_count == 0) or (es_rv_count == 0 and es_lv_count == 0):
            high_incompleteness_es = True

    if ed_rv_count == 0 or ed_myo_count == 0 or ed_lv_count == 0:
        incomplete_ed = True
        if (ed_rv_count == 0 and ed_myo_count == 0) or (ed_rv_count == 0 and ed_lv_count == 0):
            high_incompleteness_ed = True

    return incomplete_es, high_incompleteness_es, incomplete_ed, high_incompleteness_ed


def compute_missing_stats(labels_es, labels_ed, image_stats, add_c=1):
    es_rv_count, es_myo_count, es_lv_count = np.count_nonzero(labels_es == config.class_lbl_RV), \
                                             np.count_nonzero(labels_es == config.class_lbl_myo), \
                                             np.count_nonzero(labels_es == config.class_lbl_LV)
    ed_rv_count, ed_myo_count, ed_lv_count = np.count_nonzero(labels_ed == config.class_lbl_RV), \
                                             np.count_nonzero(labels_ed == config.class_lbl_myo), \
                                             np.count_nonzero(labels_ed == config.class_lbl_LV)
    if es_rv_count == 0:
        image_stats["es_wo_rv"] += add_c
    if es_myo_count == 0:
        image_stats["es_wo_myo"] += add_c
    if es_lv_count == 0:
        image_stats["es_wo_lv"] += add_c
    if es_rv_count != 0 and es_myo_count != 0 and es_lv_count != 0:
        image_stats["es_all"] += add_c
    if es_rv_count == 0 and es_myo_count == 0 and es_lv_count == 0:
        image_stats["es_wo_all"] += add_c
    if ed_rv_count == 0:
        image_stats["ed_wo_rv"] += add_c
    if ed_myo_count == 0:
        image_stats["ed_wo_myo"] += add_c
    if ed_lv_count == 0:
        image_stats["ed_wo_lv"] += add_c
    if ed_rv_count != 0 and ed_myo_count != 0 and ed_lv_count != 0:
        image_stats["ed_all"] += add_c
    if ed_rv_count == 0 and ed_myo_count == 0 and ed_lv_count == 0:
        image_stats["ed_wo_all"] += add_c


class BaseImageDataSet(Dataset):

    def __init__(self):

        self.search_mask = None
        self.origins = None
        self.spacings = None
        self.logger = None

    def __getitem__(self, index):
        assert index <= self.__len__()
        return tuple((self.images[index], self.labels[index]))

    def __len__(self):
        return len(self.images)

    def crawl_directory(self):
        pass

    def info(self, message):
        if self.logger is None:
            print(message)
        else:
            self.logger.info(message)

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

    """
        IMPORTANT: For each patient we load four files (1) ED-image (2) ED-reference (3) ES-image (4) ES-reference
                   !!! ED image ALWAYS ends with 01 index
                   !!! ES image has higher index
                   
                   HENCE ES IMAGE IS THE SECOND WE'RE LOADING in the _load_file_list method
    """

    def __init__(self, exper_config, search_mask=None, nclass=4, load_func=load_mhd_to_numpy,
                 fold_ids=[0], preprocess=False, debug=False, do_augment=True, incomplete_only=False,
                 do_flip=False):
        super(BaseImageDataSet, self).__init__()
        self.name = "ACDC"
        self.data_dir = os.path.join(exper_config.root_dir, exper_config.data_dir)
        self.rel_image_path = None
        self.rel_label_path = None
        self.incomplete_only = incomplete_only
        self.search_mask = search_mask
        self.num_of_classes = nclass
        self.fold_ids = fold_ids
        self.load_func = load_func
        self.abs_path_fold = os.path.join(self.data_dir, "fold")
        # extend data set with augmentations. For test images we don't want this to be performed
        self.do_augment = do_augment
        self.do_flip = do_flip
        self.num_of_augmentations = 3  # four rotations 90, 180, 270 degrees of rotations
        self.image_names = []
        self.incomplete_stats = {"es_wo_rv": 0, "es_wo_myo": 0, "es_wo_lv": 0, "es_wo_all": 0, "es_all": 0,
                                 "ed_wo_rv": 0, "ed_wo_myo": 0, "ed_wo_lv": 0, "ed_wo_all": 0, "ed_all": 0}
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
            self.rel_image_path = ACDC2017DataSet.image_path.replace("_iso", "")
            self.rel_label_path = ACDC2017DataSet.label_path.replace("_iso", "")
        else:
            self.rel_image_path = ACDC2017DataSet.image_path
            self.rel_label_path = ACDC2017DataSet.label_path

        if self.debug:
            # load images from a directory that only contains a couple of images
            self.rel_image_path = ACDC2017DataSet.image_path + "_test"
            self.rel_label_path = ACDC2017DataSet.label_path + "_test"

    def _get_file_lists(self):

        train_file_list = []
        val_file_list = []
        for fold_id in self.fold_ids:
            self.train_path = os.path.join(self.abs_path_fold + str(fold_id),
                                           os.path.join(ACDC2017DataSet.train_path, self.rel_image_path))
            self.val_path = os.path.join(self.abs_path_fold + str(fold_id),
                                         os.path.join(ACDC2017DataSet.val_path, self.rel_image_path))
            if self.preprocess:
                iso_img_path = self.train_path.replace("images", "images_iso")
                if not os.path.isdir(iso_img_path):
                    os.mkdir(iso_img_path)
                    iso_lbl_path = iso_img_path.replace("images_iso", "reference_iso")
                    os.mkdir(iso_lbl_path)
                iso_img_path = self.val_path.replace("images", "images_iso")
                if not os.path.isdir(iso_img_path):
                    os.mkdir(iso_img_path)
                    iso_lbl_path = iso_img_path.replace("images_iso", "reference_iso")
                    os.mkdir(iso_lbl_path)
            # get training images and labels
            search_mask_img = os.path.join(self.train_path, self.search_mask)
            print("INFO - >>> Search for {} <<<".format(search_mask_img))
            for train_file in glob.glob(search_mask_img):
                ref_file = train_file.replace(self.rel_image_path, self.rel_label_path)
                train_file_list.append(tuple((train_file, ref_file)))

            # get validation images and labels
            search_mask_img = os.path.join(self.val_path, self.search_mask)
            for val_file in glob.glob(search_mask_img):
                ref_file = val_file.replace(self.rel_image_path, self.rel_label_path)
                val_file_list.append(tuple((val_file, ref_file)))

        return train_file_list, val_file_list

    def _load_file_list(self, file_list, is_train=True):
        files_loaded = 0
        file_list.sort()
        # we use image_num to track the imageID (batch statistics). can't use idx because ES/ED count for
        # one image (we concatenate) and file_list contains 2 x images because of separate ES/ED files on disk
        image_num = 0
        # Note: file_list contains 200 entries if we load 100 images.
        for idx in tqdm(np.arange(0, len(file_list), 2)):
            # tuple contains [0]=train file name and [1] reference file name
            img_file, ref_file = file_list[idx]
            # IMPORTANT: first frame is always the End-Diastole MRI scan, filename ends with "1"

            mri_scan_ed, origin, spacing = self.load_func(img_file, data_type=ACDC2017DataSet.pixel_dta_type,
                                                          swap_axis=True)
            self.img_stats += mri_scan_ed.shape
            num_of_slices = mri_scan_ed.shape[-1]
            if num_of_slices in self.img_slice_stats.keys():
                self.img_slice_stats[num_of_slices] += 1
            else:
                self.img_slice_stats[num_of_slices] = 1
            # ed_abs_file_name = os.path.dirname(img_file)  # without actual filename, just directory
            ed_file_name = os.path.splitext(os.path.basename(img_file))[0]
            # print("ED: Image file-name {}".format(es_file_name))
            # get rid off _frameXX and take only the patient name
            patientID = ed_file_name[:ed_file_name.find("_")]
            self.image_names.append(patientID)
            if not is_train:
                self.val_image_names.append(patientID)
            self.trans_dict[patientID] = self.num_of_images
            # print("INFO - Loading ES-file {}".format(img_file))
            reference_ed, origin, spacing = self.load_func(ref_file, data_type=ACDC2017DataSet.pixel_dta_type,
                                                           swap_axis=True)
            # IMPORTANT: do the same for the End-Systole pair of images
            img_file, ref_file = file_list[idx+1]
            # ed_file_name = os.path.splitext(os.path.basename(img_file))[0]
            # print("ES: Image file-name {}".format(ed_file_name))
            mri_scan_es, origin, spacing = self.load_func(img_file, data_type=ACDC2017DataSet.pixel_dta_type,
                                                          swap_axis=True)
            # print("INFO - Loading ED_file {}".format(img_file))
            reference_es, origin, spacing = self.load_func(ref_file, data_type=ACDC2017DataSet.pixel_dta_type,
                                                           swap_axis=True)
            # AUGMENT data and add to train, validation or test if applicable
            if self.do_augment:
                self._augment_data(mri_scan_ed, reference_ed, mri_scan_es, reference_es,
                                   is_train=is_train, img_id=self.num_of_images)
            else:
                # add "raw" images
                self._add_raw_image(mri_scan_ed, reference_ed, mri_scan_es, reference_es,
                                    is_train=is_train, img_id=self.num_of_images)
            self.num_of_images += 1
            files_loaded += 2

        return files_loaded

    def load_files(self):
        files_loaded = 0
        print("INFO - Using folds {} - busy loading images/references...this may take a while!".format(self.fold_ids))
        train_file_list, val_file_list = self._get_file_lists()
        if len(train_file_list) == 0:
            print("INFO - No iso-files found therefore generating them from fold directorie(s)")
            # iso images do not already exist...presumably
            self.preprocess = True
            self._set_pathes()
            self.pre_process()
            train_file_list, val_file_list = self._get_file_lists()
        # load training set
        files_loaded += self._load_file_list(train_file_list, is_train=True)
        self.train_num_slices = len(self.train_images)
        # load validation/test set
        files_loaded += self._load_file_list(val_file_list, is_train=False)
        self.val_num_slices = len(self.val_images)
        self.img_stats *= 1./self.num_of_images
        self.img_stats = self.img_stats.astype(np.int16)
        if self.incomplete_only:
            print("WARNING - Loading only images WITH INCOMPLETE slices (missing RV/MYO or LV)")
        print("INFO - Using folds {} - loaded {} files: {} studies in train set, {} studies in validation set"
              " (augment={}/flip={})".format(self.fold_ids, files_loaded, self.train_num_slices, self.val_num_slices,
                                             self.do_augment, self.do_flip))

        print("INFO - Mean width/height/#slices per image {}/{}/{}".format(self.img_stats[0], self.img_stats[1],
                                                                           self.img_stats[2]))
        self.show_image_stats()

    def _augment_data(self, image_ed, label_ed, image_es, label_es, is_train=False, img_id=None):
        """
        Augments image slices by rotating z-axis slices for 90, 180 and 270 degrees

        image_ed, label_ed, image_es and label_es are 3 dimensional tensors [x,y,z]
        label tensors contain the class labels for the segmentation from 0-3 for the 4 segmentation classes
        0 = background, 1 = left ventricle, 2 = right ventricle, 3 = myocardium

        """

        def rotate_slice(img_ed_slice, lbl_ed_slice, img_es_slice, lbl_es_slice,
                         is_train=False, img_slice_id=None):

            for rots in range(4):
                pad_img_ed_slice = np.pad(img_ed_slice, ACDC2017DataSet.pad_size, 'constant',
                                          constant_values=(0,)).astype(ACDC2017DataSet.pixel_dta_type)
                pad_img_es_slice = np.pad(img_es_slice, ACDC2017DataSet.pad_size, 'constant',
                                          constant_values=(0,)).astype(ACDC2017DataSet.pixel_dta_type)
                # we make a 3dim tensor (first dim has one-size) and concatenate ES and ED image
                pad_img_slice = np.concatenate((np.expand_dims(pad_img_es_slice, axis=0),
                                                np.expand_dims(pad_img_ed_slice, axis=0)))
                # same concatenation for the label files of ED and ES
                label_slice = np.concatenate((np.expand_dims(lbl_es_slice, axis=0),
                                                np.expand_dims(lbl_ed_slice, axis=0)))
                # NOTE: Also creating a 180 degree flipped image.
                # we need to ndarray.copy() the flipped tensor, otherwise we run into annoying errors
                # reported here: https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663/4
                if self.do_flip:
                    flipped_pad_img_slice = np.flip(pad_img_slice, axis=2).copy()
                    # remember label has shape [#classes, width, height, #slices]
                    flipped_label_slice = np.flip(label_slice, axis=2).copy()
                if is_train:
                    self.train_images.append(pad_img_slice)
                    self.train_labels.append(label_slice)
                    # img_slice_id is a combination of imageid and sliceID e.g. (10, 1)
                    self.train_img_slice_ids.append(img_slice_id)
                    if self.do_flip:
                        # add flipped image as well
                        self.train_images.append(flipped_pad_img_slice)
                        self.train_labels.append(flipped_label_slice)
                        # img_slice_id is a combination of imageid and sliceID e.g. (10, 1)
                        self.train_img_slice_ids.append(img_slice_id)
                else:
                    self.val_images.append(pad_img_slice)
                    self.val_labels.append(label_slice)
                    self.val_img_slice_ids.append(img_slice_id)
                    if self.do_flip:
                        # add flipped image as well
                        self.val_images.append(flipped_pad_img_slice)
                        self.val_labels.append(flipped_label_slice)
                        # img_slice_id is a combination of imageid and sliceID e.g. (10, 1)
                        self.val_img_slice_ids.append(img_slice_id)
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

            # Note: z is the sliceID. PLEASE also note that we're constructing a tuple of imgID and sliceID
            # that we'll store for each image patch
            if self.incomplete_only:
                incomplete_es, high_incomplete_es, incomplete_ed, high_incomplete_ed = \
                    detect_incomplete_slices(label_es_slice, label_ed_slice)
                if incomplete_es or high_incomplete_es or incomplete_ed or high_incomplete_ed:
                    if high_incomplete_es:
                        num_augs = 8
                    elif incomplete_es:
                        num_augs = 4
                    elif high_incomplete_ed:
                        num_augs = 3
                    else:
                        num_augs = 1
                    for _ in range(num_augs):
                        rotate_slice(image_ed_slice, label_ed_slice, image_es_slice, label_es_slice, is_train,
                                     img_slice_id=tuple((img_id, z)))
                    compute_missing_stats(label_es_slice, label_ed_slice, self.incomplete_stats, add_c=num_augs)

            else:
                compute_missing_stats(label_es_slice, label_ed_slice, self.incomplete_stats)
                rotate_slice(image_ed_slice, label_ed_slice, image_es_slice, label_es_slice, is_train,
                             img_slice_id=tuple((img_id, z)))

    def _add_raw_image(self, image_ed, label_ed, image_es, label_es, is_train=False, img_id=None):

        img = np.concatenate((np.expand_dims(image_es, axis=0),
                              np.expand_dims(image_ed, axis=0)))
        # same concatenation for the label files of ED and ES
        label = np.concatenate((np.expand_dims(label_es, axis=0),
                                np.expand_dims(label_ed, axis=0)))
        if is_train:
            self.train_images.append(img)
            self.train_labels.append(label)
            # img_slice_id is a combination of imageid and sliceID e.g. (10, 1)
        else:
            self.val_images.append(img)
            self.val_labels.append(label)

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
            # using scipy.interpolation.zoom
            mri_scan = resample_image_scipy(mri_scan, new_spacing=zoom_factors, order=3)
            # using scipy.misc.imresize
            # mri_scan = resample_image_scipy(mri_scan, new_spacing=zoom_factors, order=None, use_func="imresize",
            #                                interp="lanczos")
            save_img_as_mhg(mri_scan, new_spacing, origin, out_filename, swap_axis=True)
            # print("INFO - Loading file {}".format(file_tuple[1]))
            reference, origin, spacing = self.load_func(file_tuple[1], data_type=ACDC2017DataSet.pixel_dta_type,
                                                        swap_axis=True)
            out_filename = file_tuple[1]
            out_filename = out_filename.replace("reference", "reference_iso")
            # using scipy.interpolation.zoom
            reference = resample_image_scipy(reference, new_spacing=zoom_factors, order=0)
            # using scipy.misc.imresize
            # reference = resample_image_scipy(reference, new_spacing=zoom_factors, order=None, use_func="imresize",
            #                                 interp="nearest")
            save_img_as_mhg(reference, new_spacing, origin, out_filename, swap_axis=True)
            files_loaded += 1
        return files_loaded

    def pre_process(self):
        print("INFO - Resampling images to isometric size")
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

    def show_image_stats(self):
        total_num_of_slices = (self.get_num_of_slices(train=True) + self.get_num_of_slices(train=False)) / 4.
        print("---------------------- Image incompleteness statistics -------------------------")
        print("ES: slices missing RV/MYO/LV {}/{}/{} ({:.2f}/{:.2f}/{:.2f}) "
              "/ complete {} wo-all {}".format(self.incomplete_stats["es_wo_rv"], self.incomplete_stats["es_wo_myo"],
                                               self.incomplete_stats["es_wo_lv"],
                                               self.incomplete_stats["es_wo_rv"]/total_num_of_slices,
                                               self.incomplete_stats["es_wo_myo"]/total_num_of_slices,
                                               self.incomplete_stats["es_wo_lv"]/total_num_of_slices,
                                               self.incomplete_stats["es_all"],
                                               self.incomplete_stats["es_wo_all"]))
        print("ED: slices missing RV/MYO/LV {}/{}/{} ({:.2f}/{:.2f}/{:.2f}) "
              "/ complete {} wo-all {}".format(self.incomplete_stats["ed_wo_rv"], self.incomplete_stats["ed_wo_myo"],
                                               self.incomplete_stats["ed_wo_lv"],
                                               self.incomplete_stats["ed_wo_rv"] / total_num_of_slices,
                                               self.incomplete_stats["ed_wo_myo"] / total_num_of_slices,
                                               self.incomplete_stats["ed_wo_lv"] / total_num_of_slices,
                                               self.incomplete_stats["ed_all"],
                                               self.incomplete_stats["ed_wo_all"]))

    @staticmethod
    def remove_padding(image):
        if image.ndim == 2:
            return image[ACDC2017DataSet.pad_size:-ACDC2017DataSet.pad_size,
                         ACDC2017DataSet.pad_size:-ACDC2017DataSet.pad_size]
        elif image.ndim == 4:
            return image[:, ACDC2017DataSet.pad_size:-ACDC2017DataSet.pad_size,
                         ACDC2017DataSet.pad_size:-ACDC2017DataSet.pad_size, :]

# dataset = ACDC2017DataSet(exper_config=config, search_mask=config.dflt_image_name + ".mhd", fold_ids=[0],
#                          preprocess=True, debug=False)

# del dataset

