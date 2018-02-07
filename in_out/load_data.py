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

    def __init__(self, config, search_mask=None, nclass=3, load_func=load_mhd_to_numpy,
                 fold_id=1, preprocess=False):

        super(BaseImageDataSet, self).__init__()
        self.data_dir = os.path.join(config.root_dir, config.data_dir)
        self.search_mask = search_mask
        self.num_of_classes = nclass
        self.current_fold = fold_id
        self.load_func = load_func
        self.fold_id = fold_id
        self.abs_path_fold = os.path.join(self.data_dir, "fold" + str(fold_id))

        self.train_images = []
        self.train_labels = []
        self.train_spacings = []
        self.val_images = []
        self.val_labels = []
        self.val_spacings = []
        self.preprocess = preprocess
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

        self.train_path = os.path.join(self.abs_path_fold,
                                       os.path.join(ACDC2017DataSet.train_path, ACDC2017DataSet.image_path))
        self.val_path = os.path.join(self.abs_path_fold,
                                     os.path.join(ACDC2017DataSet.val_path, ACDC2017DataSet.image_path))

    def _get_file_lists(self):

        train_file_list = []
        val_file_list = []
        # get training images and labels
        search_mask_img = os.path.join(self.train_path, self.search_mask)
        print("INFO - >>> Search with dir+pattern {} <<<".format(search_mask_img))
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
        print(len(file_list))
        for idx in np.arange(0, len(file_list), 2):
            # tuple contains [0]=train file name and [1] reference file name
            img_file, ref_file = file_list[idx]
            # first frame is always the end-systolic MRI scan, filename ends with "1"
            mri_scan_es, origin, spacing = self.load_func(img_file, data_type=ACDC2017DataSet.pixel_dta_type,
                                                          swap_axis=True)
            print("INFO - Loading ES-file {}".format(img_file))
            reference_es, origin, spacing = self.load_func(ref_file, data_type=ACDC2017DataSet.pixel_dta_type,
                                                           swap_axis=True)
            # do the same for the End-Systolic pair of images
            img_file, ref_file = file_list[idx+1]
            mri_scan_ed, origin, spacing = self.load_func(img_file, data_type=ACDC2017DataSet.pixel_dta_type,
                                                          swap_axis=True)
            print("INFO - Loading ED_file {}".format(img_file))
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
        print("INFO - Using fold{} - loaded {} files: {} slices in train set, {} slices in validation set".format(
            self.fold_id, files_loaded, len(self.train_images), len(self.val_images)))

    def _augment_data(self, image_ed, label_ed, image_es, label_es, is_train=False):
        """
        Augments image slices by rotating z-axis slices for 90, 180 and 270 degrees

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
        print("INFO - Using fold{} - loaded {} files".format(self.fold_id, files_loaded))


class HVSMR2016CardiacMRI(BaseImageDataSet):

    pixel_dta_type = 'float32'
    pad_size = config.pad_size
    label_background = config.class_lbl_background
    label_myocardium = config.class_lbl_myocardium
    label_bloodpool = config.class_lbl_bloodpool

    def __init__(self, data_dir, search_mask=None, nclass=3, transform=False, conf_obj=None,
                 load_func=load_mhd_to_numpy, norm_scale="normalize", mode="train", load_type="raw",
                 kfold=5, val_fold=1):
        """
        The images are already resampled to an isotropic 3D size of 0.65mm x 0.65 x 0.65


        :param data_dir: root directory
        :param search_mask:
        :param nclass:
        :param transform:
        :param conf_obj:
        :param load_func: currently only load_mhd_to_numpy is supported
        :param norm_scale: takes arguments "normalize" or "rescale"
        :param mode: takes "train", "test", "valid"
        """
        super(HVSMR2016CardiacMRI, self).__init__(data_dir, conf_obj)
        self.kfold = kfold
        self.val_fold = val_fold
        self.transform = transform
        self.norm_scale = norm_scale
        self.search_mask = search_mask
        self.load_func = load_func
        self.load_type = load_type
        self.mode = mode
        # Note, this list will contain len() image slices...2D!
        self.images = []
        self.labels = []
        self.val_images = []
        self.val_labels = []
        self.test_images = []
        self.test_labels = []
        self.origins = []
        self.spacings = []
        self.test_origins = []
        self.test_spacings = []
        self.no_class = nclass
        self.class_count = np.zeros(self.no_class)
        self.val_image_indices =  self._prepare_cross_validation()
        # print("Validation indices {}".format(self.val_image_indices))
        if self.load_type == "raw":
            # Important detail: we need to swap axis 0 and 2 of the HVSMR2016 files
            self.load_images_from_dir(swap_axis=True)
        elif self.load_type == "numpy":
            self.load_numpy_arr_from_dir()
            if len(self.images) == 0:
                print("Info - cannot find any numpy npz files. Looking for raw files...")
                self.load_images_from_dir(swap_axis=True)
        else:
            raise ValueError("Load mode {} is not supported".format(self.load_type))

    def __getitem__(self, index):
        assert index <= self.__len__()
        return tuple((self.images[index], self.labels[index]))

    def _prepare_cross_validation(self):
        folds = KFold(n_splits=self.kfold)
        fold_arrays = [val_set for _, val_set in folds.split(np.arange(10))]
        return fold_arrays[self.val_fold]

    def _get_file_lists(self):

        img_file_list = []
        label_file_list = []

        if self.mode == "train":
            input_dir = os.path.join(self.data_dir, "train")
        elif self.mode == "test":
            input_dir = os.path.join(self.data_dir, "test")
        else:
            raise ValueError("Loading mode {} is currently not supported (train/test)".format(self.mode))

        search_mask = os.path.join(input_dir, self.search_mask)

        for fname in glob.glob(search_mask):
            img_file_list.append(fname)
            label_file_list.append(fname.replace("image", "label"))

        return img_file_list, label_file_list

    def load_images_from_dir(self, swap_axis=False):
        """
            Searches for files that match the search_mask-parameter and assumes that there are also
            reference aka label images accompanied with each image

        """
        files_loaded = 0

        img_file_list, label_file_list = self._get_file_lists()
        for i, file_name in enumerate(img_file_list):
            print("> > > Loading image+label from {}".format(file_name))
            mri_scan, origin, spacing = self.load_func(file_name, data_type=HVSMR2016CardiacMRI.pixel_dta_type)
            # the Nifty files from the challenge that Jelmer provided have (z,y,x) and hence z,x must be swapped
            if swap_axis:
                mri_scan = np.swapaxes(mri_scan, 0, 2)

            if self.norm_scale == "normalize":
                # print("> > > Info - Normalizing images intensity values")
                mri_scan = normalize_image(mri_scan, axis=None)

            elif self.norm_scale == "rescale":
                mri_scan = rescale_image(mri_scan, axis=None)
                # print("> > > Info - Rescaling images intensity values")
            else:
                # no rescaling or normalization
                print("> > > Info - No rescaling or normalization applied to image!")
            # add a front axis to the numpy array, will use that to concatenate the image slices
            self.origins.append(origin)
            self.spacings.append(spacing)
            print("{} / {}".format(file_name,  label_file_list[i]))
            label, _, _ = self.load_func(label_file_list[i])
            if swap_axis:
                label = np.swapaxes(label, 0, 2)
            for class_label in range(self.no_class):
                self.class_count[class_label] += np.sum(label == class_label)
            # augment the image with additional rotated slices
            if i in self.val_image_indices:
                val_set = True
            else:
                val_set = False
            # AUGMENT data and add to train, validation or test if applicable
            self._augment_data(mri_scan, label, pad_size=HVSMR2016CardiacMRI.pad_size, isval=val_set)

            files_loaded += 1

        if len(self.val_images) == 0:
            samples = int(0.1 * len(self.images))
            self.val_images = self.images[0:samples]
            self.val_labels = self.labels[0:samples]

    def _augment_data(self, image, label, pad_size=0, isval=False):
        """
        Adds all original and rotated image slices to self.images and self.labels objects
        :param image:
        :param label:
        :param pad_size:
        :return:
        """

        def rotate_slice(img_slice, lbl_slice, isval=False):
            # PAD IMAGE
            for rots in range(4):
                # no padding here but when we extract patches during BatchGeneration
                section = np.pad(img_slice, pad_size, 'constant', constant_values=(0,)).astype(
                    HVSMR2016CardiacMRI.pixel_dta_type)
                if not isval:
                    self.images.append(section)
                    self.labels.append(lbl_slice)
                else:
                    self.val_images.append(section)
                    self.val_labels.append((lbl_slice))

                # rotate for next iteration
                img_slice = np.rot90(img_slice)
                lbl_slice = np.rot90(lbl_slice)

        if isval:
            # store the complete image for testing as well
            self.test_images.append(image)
            self.test_labels.append(label)

        # for each image-slice rotate the img four times. We're doing that for all three orientations
        for z in range(image.shape[2]):
            label_slice = label[:, :, z]
            image_slice = image[:, :, z]
            rotate_slice(image_slice, label_slice, isval)

        for y in range(image.shape[1]):
            label_slice = np.squeeze(label[:, y, :])
            image_slice = np.squeeze(image[:, y, :])
            rotate_slice(image_slice, label_slice, isval)

        for x in range(image.shape[0]):
            label_slice = np.squeeze(label[x, :, :])
            image_slice = np.squeeze(image[x, :, :])
            rotate_slice(image_slice, label_slice, isval)

    def load_numpy_arr_from_dir(self, file_prefix=None, abs_path=None):
        if file_prefix is None:
            file_prefix = config.numpy_save_filename

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
            file_prefix = config.numpy_save_filename

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

    def get_test_image(self, swap_axis=True):
        save_mode = self.mode
        self.mode = "test"
        img_files, label_files = self._get_file_lists()
        for i, file_name in enumerate(img_files):
            print("> > > Loading image+label from {}".format(file_name))
            mri_scan, origin, spacing = self.load_func(file_name, data_type=HVSMR2016CardiacMRI.pixel_dta_type)
            # the Nifty files from the challenge that Jelmer provided have (z,y,x) and hence z,x must be swapped
            if swap_axis:
                mri_scan = np.swapaxes(mri_scan, 0, 2)

            if self.norm_scale == "normalize":
                # print("> > > Info - Normalizing images intensity values")
                mri_scan = normalize_image(mri_scan, axis=None)

            elif self.norm_scale == "rescale":
                mri_scan = rescale_image(mri_scan, axis=None)
                # print("> > > Info - Rescaling images intensity values")
            else:
                # no rescaling or normalization
                print("> > > Info - No rescaling or normalization applied to image!")
            # add a front axis to the numpy array, will use that to concatenate the image slices
            self.test_origins.append(origin)
            self.test_spacings.append(spacing)
            self.test_images.append(mri_scan)
            label, _, _ = self.load_func(label_files[i])
            if swap_axis:
                label = np.swapaxes(label, 0, 2)
            self.test_labels.append(label)

        self.mode = save_mode

    def create_test_slices(self):
        test_img_slices, test_lbl_slices = [], []

        for i in np.arange(len(self.test_images)):
            image = np.pad(self.test_images[i], ((0, 0),
                                                 (HVSMR2016CardiacMRI.pad_size, HVSMR2016CardiacMRI.pad_size),
                                                 (HVSMR2016CardiacMRI.pad_size, HVSMR2016CardiacMRI.pad_size)),
                           'constant', constant_values=(0,)).astype(HVSMR2016CardiacMRI.pixel_dta_type)
            for x in np.arange(self.test_images[i].shape[0]):
                slice_padded = image[x, :, :]
                lbl_slice = self.test_labels[i][x, :, :]
                # slice_padded = np.pad(img_slice, HVSMR2016CardiacMRI.pad_size, 'constant', constant_values=(0,)).astype(
                #    HVSMR2016CardiacMRI.pixel_dta_type)
                test_img_slices.append(slice_padded)
                test_lbl_slices.append(lbl_slice)

        return test_img_slices, test_lbl_slices

    @staticmethod
    def get_pred_class_labels(predictions, classes=None, axis=1):
        """
            predictions, autograd.Variable or numpy array with dim:
             [batch_size, num_of_classes, width, height]

            The parameter "axis" specifies the axis over which we take the maximum in order to determine
            the segmentation class. The method only supports axis-values 0 and 1.

            Important: the return object "overlays" has dimension [num_of_classes, batch_size, width, height]

        :param predictions:
        :param classes:
        :return:
        """
        # if PyTorch Variable, convert to numpy array
        if isinstance(predictions, Variable):
            predictions = predictions.data.cpu().squeeze().numpy()

        if classes is None:
            classes = [HVSMR2016CardiacMRI.label_myocardium, HVSMR2016CardiacMRI.label_bloodpool]

        pred_idx = np.argmax(predictions, axis=axis)
        print(np.unique(pred_idx))
        print("In method get_pred_class_labels, shape of input ", predictions.shape)
        if axis == 1:
            overlays = np.zeros((len(classes) + 1, predictions.shape[0], predictions.shape[2],
                                 predictions.shape[3]))
        elif axis == 0:
            overlays = np.zeros((len(classes) + 1, predictions.shape[1], predictions.shape[2],
                                 predictions.shape[3]))
        else:
            raise ValueError("axis value {} is not supported".format(axis))

        for cls in classes:
            pred_cls_labels = pred_idx == cls
            overlays[cls, :, :, :] = pred_cls_labels

        return overlays


dataset = ACDC2017DataSet(config=config, search_mask=config.dflt_image_name + ".mhd", fold_id=0,
                          preprocess=False)

# del dataset

