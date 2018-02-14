import SimpleITK as sitk
import numpy as np


def load_mhd_to_numpy(filename, data_type="float32", swap_axis=False):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    mri_scan = sitk.GetArrayFromImage(itkimage).astype(data_type)
    if swap_axis:
        mri_scan = np.swapaxes(mri_scan, 0, 2)

    # Read the origin of the mri_scan
    origin = np.array(list(itkimage.GetOrigin()))

    # Read the spacing along each dimension
    spacing = np.array(list(itkimage.GetSpacing()))

    return mri_scan, origin, spacing


def write_numpy_to_image(np_array, filename, swap_axis=False, spacing=None):

    if swap_axis:
        np_array = np.swapaxes(np_array, 0, 2)
    img = sitk.GetImageFromArray(np_array)
    if spacing is not None:
        img.SetSpacing(spacing)
    sitk.WriteImage(img, filename)
    print("Successfully saved image to {}".format(filename))


def save_mhd(image, filename, verbose=True):
    try:
        sitk.WriteImage(image, filename, True)
        if verbose:
            print("Successfully saved image to {}".format(filename))
    except:
        raise IOError("Can't save image to {}.".format(filename))


def save_img_as_mhg(image, spacing, origin, abs_out_filename=None, swap_axis=False):

    if isinstance(image, np.ndarray):
        if swap_axis:
            image = np.swapaxes(image, 0, 2)
        image = sitk.GetImageFromArray(image)

    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    save_mhd(image, abs_out_filename)