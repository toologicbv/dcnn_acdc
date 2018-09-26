import numpy as np
import os
from common.detector.config import config_detector
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, \
    generate_binary_structure
import math


def compute_slice_distance_transform_for_structure(reference, voxelspacing=None, connectivity=1):
    """
    We compute the
    :param reference: we assume shape [width, height] and binary encoding (1=voxels of target structure)
    :param voxelspacing:
    :param connectivity:
    :return:
    """
    reference = np.atleast_1d(reference.astype(np.bool))
    # binary structure
    footprint = generate_binary_structure(reference.ndim, connectivity)
    if 0 == np.count_nonzero(reference):
        raise RuntimeError('The reference array does not contain any binary object.')

    reference_border = np.logical_xor(reference, binary_erosion(reference, structure=footprint, iterations=1))
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)

    return dt


def generate_dt_maps(exper_handler, patient_id=None, voxelspacing=1.4, bg_classes=[0, 4]):

    if exper_handler.test_set is None:
        exper_handler.get_test_set()
    if patient_id is None:
        p_range = exper_handler.test_set.trans_dict.keys()
    else:
        p_range = [patient_id]
    dt_output_dir = os.path.join(exper_handler.exper.config.root_dir,
                                 os.path.join(exper_handler.exper.output_dir, config_detector.dt_map_dir))
    if not os.path.isdir(dt_output_dir):
        os.mkdir(dt_output_dir)

    for p_id in p_range:
        _, labels = exper_handler.test_set.get_test_pair(p_id)
        dt_slices = np.zeros_like(labels)
        num_of_classes, w, h, num_of_slices = labels.shape
        for slice_id in np.arange(num_of_slices):
            # determine apex-base slice. we currently use a simple heuristic. first and last slice are base-apex
            if slice_id == 0 or slice_id == num_of_slices - 1:
                non_base_apex_slice = 0
            else:
                non_base_apex_slice = 1
            for cls_idx in np.arange(1, num_of_classes):
                # IMPORTANT: skip the background classes!
                if cls_idx not in bg_classes:
                    label_slice = labels[cls_idx, :, :, slice_id]
                    if 0 != np.count_nonzero(label_slice):
                        dt_slices[cls_idx, :, :, slice_id] = \
                            compute_slice_distance_transform_for_structure(label_slice, voxelspacing=voxelspacing)
                        # we use the "non_base_apex_slice" index to fetch the inter-observer margin from the config
                        # object. index=0: the margin we use for the apex/base slices, it's greater than for non-A/B
                        #         index=1: the margin we use for the non-apex/base slices, it's smaller than the first
                        inter_observ_margin = config_detector.acdc_inter_observ_var[cls_idx][non_base_apex_slice]
                        # dt_slices[cls_idx, :, :, slice_id] -= inter_observ_margin
                    else:
                        penalty = math.sqrt((h**2 + w**2)) * voxelspacing
                        print("Penalty {:.2f}".format(penalty))
                        dt_slices[cls_idx, :, :, slice_id] = penalty

        file_name = os.path.join(dt_output_dir, p_id + "_dt_map.npz")
        try:
            np.savez(file_name, dt_map=dt_slices)
        except IOError:
            print("ERROR - cannot save hd map to {}".format(file_name))
            raise

    if patient_id is not None:
        return dt_slices
