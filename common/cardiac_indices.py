
import numpy as np


def compute_ejection_fraction(target_vol_es, target_vol_ed, spacings):
    """

    :param target_vol_es: binary numpy array indicating target voxels of tissue structure at ES [w, h, #slices]
    :param target_vol_ed: binary numpy array indicating target voxels of tissue structure at ED [w, h, #slices]
    :param spacings: numpy array [x-spacing, y-spacing, z-spacing]
    :return: volume at ES, volume at ED, Ejection fraction = (1 - (ESV/EDV) ) * 100

    """

    num_of_voxels_es = np.count_nonzero(target_vol_es)
    num_of_voxels_ed = np.count_nonzero(target_vol_ed)
    esv = np.prod(spacings) * num_of_voxels_es
    edv = np.prod(spacings) * num_of_voxels_ed
    ef = (1. - esv/float(edv)) * 100
    return esv, edv, ef
