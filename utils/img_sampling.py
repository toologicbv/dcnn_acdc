import scipy.ndimage.interpolation as interpol
from scipy.misc import imresize
import numpy as np
import SimpleITK as sitk
from in_out.read_save_images import save_img_as_mhg


def resample_image_scipy(image, new_spacing, order=3, use_func="zoom", interp=None):
    """
    Assuming new_spacing is a tuple of 3 elements for x,y,z axis

    """
    if use_func == "scipy":
        image = interpol.zoom(image, (new_spacing[0], new_spacing[1], new_spacing[2]), order=order)
    else:
        if interp is None:
            raise ValueError("ERROR - function imresize requires parameter <interp> to be specified ")
        image = imresize(image, (new_spacing[0], new_spacing[1], new_spacing[2]), interp=interp)

    return image


def resample(image, spacingx, spacingy, spacingz, interpolator = sitk.sitkLinear):
    """

    Procedure copied from Jelmer's ConvNet Utilities

    """
    targetspacing = np.array([spacingx, spacingy, spacingz])
    image = sitk.Cast(image, sitk.sitkFloat32)
    sourcespacing = np.array([image.GetSpacing()])
    sourcesize = np.array([image.GetSize()])
    scale = sourcespacing / targetspacing
    image.SetOrigin((0.0, 0.0, 0.0))
    targetsize = sourcesize * scale
    targetsize = targetsize.astype(int)
    transform = sitk.Transform(3, sitk.sitkScale)
    transform.SetParameters((1.0, 1.0, 1.0))
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(targetspacing)
    resampler.SetDefaultPixelValue(0)
    resampler.SetSize(tuple(targetsize[0]))
    resampler.SetTransform(transform)
    resampler.SetInterpolator(interpolator)
    out = resampler.Execute(image)
    out = sitk.Cast(out, sitk.sitkInt16)
    return out
