import sys
import numpy as np
from config.config import config
import os
import glob
from tqdm import tqdm

if "/home/jogi/.local/lib/python2.7/site-packages" in sys.path:
    sys.path.remove("/home/jogi/.local/lib/python2.7/site-packages")

from in_out.read_save_images import load_mhd_to_numpy
from utils.img_sampling import resample_image_scipy


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

        self.data_dir = os.path.join(exper_config.root_dir, exper_config.data_dir)
        self.search_mask = search_mask
        self.num_of_classes = nclass
        self.fold_ids = fold_ids
        self.load_func = load_func
        self.abs_path_fold = os.path.join(self.data_dir, "fold")
        self.images = []
        self.labels = []
        self.spacings = []
        self.debug = debug
        self._set_pathes()

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
            for train_file in tqdm(glob.glob(search_mask_img)):
                ref_file = train_file.replace(ACDC2017TestHandler.image_path, ACDC2017TestHandler.label_path)
                file_list.append(tuple((train_file, ref_file)))

            # get validation images and labels
            search_mask_img = os.path.join(self.val_path, self.search_mask)
            for val_file in tqdm(glob.glob(search_mask_img)):
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
            mri_scan_es = self._resample_images(mri_scan_es, spacing, poly_order=3, do_pad=True)
            # print("INFO - Loading ES-file {}".format(img_file))
            reference_es, _, _ = self.load_func(ref_file, data_type=ACDC2017TestHandler.pixel_dta_type,
                                                swap_axis=True)
            reference_es = self._resample_images(reference_es, spacing, poly_order=0, do_pad=False)

            # do the same for the End-diastole pair of images
            img_file, ref_file = file_list[idx+1]
            print("{} - {}".format(idx+1, img_file))
            mri_scan_ed, _, _ = self.load_func(img_file, data_type=ACDC2017TestHandler.pixel_dta_type,
                                               swap_axis=True)
            mri_scan_ed = self._resample_images(mri_scan_ed, spacing, poly_order=3, do_pad=True)

            reference_ed, _, _ = self.load_func(ref_file, data_type=ACDC2017TestHandler.pixel_dta_type,
                                                swap_axis=True)
            reference_ed = self._resample_images(reference_ed, spacing, poly_order=0, do_pad=False)
            # concatenate both images for further processing
            images = np.concatenate((np.expand_dims(mri_scan_ed, axis=0),
                                     np.expand_dims(mri_scan_es, axis=0)))
            # same concatenation for the label files of ED and ES
            labels = np.concatenate((np.expand_dims(reference_ed, axis=0),
                                     np.expand_dims(reference_es, axis=0)))
            self.images.append(images)
            self.labels.append(labels)

        print("INFO - Successfully loaded {} ED/ES patient pairs".format(len(self.images)))

    def _resample_images(self, image, spacing, poly_order=3, do_pad=True):

        # print("Spacings ", spacing)
        zoom_factors = tuple((spacing[0] / ACDC2017TestHandler.new_voxel_spacing,
                             spacing[1] / ACDC2017TestHandler.new_voxel_spacing, 1))

        image = resample_image_scipy(image, new_spacing=zoom_factors, order=poly_order)
        # we only pad images not the references aka labels...hopefully
        if do_pad:
            image = np.pad(image, ((ACDC2017TestHandler.pad_size, ACDC2017TestHandler.pad_size-1),
                                   (ACDC2017TestHandler.pad_size, ACDC2017TestHandler.pad_size - 1), (0, 0)),
                           'constant', constant_values=(0,)).astype(ACDC2017TestHandler.pixel_dta_type)

        return image


dataset = ACDC2017TestHandler(exper_config=config, search_mask=config.dflt_image_name + ".mhd", fold_ids=[0],
                              debug=False, batch_size=4)

del dataset
