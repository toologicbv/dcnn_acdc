from in_out.read_save_images import save_img_as_mhg
from utils.img_sampling import resample_image_scipy
from in_out.load_data import load_mhd_to_numpy

import SimpleITK as sitk
import scipy.ndimage.interpolation as interpol

in_filename = "/home/jorg/repository/dcnn_acdc/data/Folds/fold1/train/images/patient001_frame01.mhd"
# in_filename = "/home/jorg/repository/dcnn_mri_seg/data/HVSMR2016/train/0_image.nii"
mri_scan, origin, spacing = load_mhd_to_numpy(in_filename, data_type="float32", swap_axis=True)
new_voxel_spacing = 1.4

zoom_factors = tuple((spacing[0] / new_voxel_spacing,
                                  spacing[1] / new_voxel_spacing, 1))
print("Before ", mri_scan.shape)

image = interpol.zoom(mri_scan, zoom_factors, order=3)
save_spacing = tuple((new_voxel_spacing, new_voxel_spacing, spacing[2]))
print("After ", image.shape)

save_img_as_mhg(image, spacing=save_spacing, origin=origin,
                abs_out_filename="/home/jorg/repository/dcnn_acdc/data/Folds/fold1/train/images_iso/image1.mhd",
                swap_axis=True)


in_filename = "/home/jorg/repository/dcnn_acdc/data/Folds/fold1/train/reference/patient001_frame01.mhd"
mri_scan, origin, spacing = load_mhd_to_numpy(in_filename, data_type="float32", swap_axis=True)

print("Before ", mri_scan.shape)
reference = interpol.zoom(mri_scan, zoom_factors, order=0)
print("After ", reference.shape)

save_img_as_mhg(reference, spacing=save_spacing, origin=origin,
                abs_out_filename="/home/jorg/repository/dcnn_acdc/data/Folds/fold1/train/images_iso/reference1.mhd",
                swap_axis=True)
