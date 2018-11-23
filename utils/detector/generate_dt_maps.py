import numpy as np
from common.detector.config import config_detector
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, \
    generate_binary_structure
import math


def compute_slice_distance_transform_for_structure(reference, voxelspacing=None, connectivity=1, ):
    """
    We compute the
    :param reference: we assume shape [width, height] and binary encoding (1=voxels of target structure)
    :param voxelspacing:
    :param connectivity:
    :return:
    """
    reference = np.atleast_1d(reference.astype(np.bool))
    inside_obj_mask = np.zeros_like(reference).astype(np.bool)
    # binary structure
    footprint = generate_binary_structure(reference.ndim, connectivity)
    if 0 == np.count_nonzero(reference):
        raise RuntimeError('The reference array does not contain any binary object.')

    inside_voxels_indices = binary_erosion(reference, structure=footprint, iterations=1)
    inside_obj_mask[inside_voxels_indices] = 1
    reference_border = np.logical_xor(reference, inside_voxels_indices)
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)

    return dt, inside_obj_mask


def generate_dt_maps(labels, voxelspacing=1.4, bg_classes=[0, 4], slice_label_areas=None):
    """

    :param labels: shape [8, w, h, #slices]
    :param voxelspacing:
    :param bg_classes:
    :param slice_label_areas: [2, #slices]: contains tissue roi area for each slice in volume for ES and ED
    :return: distance transform maps for a particular tissue class per slice
    """

    dt_slices = np.zeros_like(labels)
    num_of_classes, w, h, num_of_slices = labels.shape
    for slice_id in np.arange(num_of_slices):
        # determine apex-base slice. we currently use a simple heuristic. first and last slice are base-apex
        if slice_id == 0 or slice_id == num_of_slices - 1:
            non_base_apex_slice = 0
        else:
            non_base_apex_slice = 1
        if slice_label_areas is not None:
            # print(slice_label_areas[:, slice_id])
            slice_label_area_es, slice_label_area_ed = slice_label_areas[:, slice_id]
            scale_bin_es = np.digitize(slice_label_area_es, config_detector.label_area_bins)
            scale_bin_ed = np.digitize(slice_label_area_ed, config_detector.label_area_bins)
        else:
            slice_label_area_es, slice_label_area_ed = -1, -1
            scale_bin_es, scale_bin_ed = config_detector.label_area_num_bins, config_detector.label_area_num_bins

        for cls_idx in np.arange(1, num_of_classes):
            # IMPORTANT: skip the background classes!
            if cls_idx not in bg_classes:
                if cls_idx < num_of_classes / 2:
                    scale_bin = scale_bin_es
                    s_area = slice_label_area_es
                else:
                    scale_bin = scale_bin_ed
                    s_area = slice_label_area_ed
                label_slice = labels[cls_idx, :, :, slice_id]
                if 0 != np.count_nonzero(label_slice):
                    dt_slices[cls_idx, :, :, slice_id], inside_obj_mask = \
                        compute_slice_distance_transform_for_structure(label_slice, voxelspacing=voxelspacing)
                    dt = dt_slices[cls_idx, :, :, slice_id]
                    # we use the "non_base_apex_slice" index to fetch the inter-observer margin from the config
                    # object. index=0: the margin we use for the apex/base slices, it's greater than for non-A/B
                    #         index=1: the margin we use for the non-apex/base slices, it's smaller than the first
                    # we substract the margin from the distance-transform for all voxels.
                    inter_observ_margin = config_detector.acdc_inter_observ_var[cls_idx][non_base_apex_slice]
                    if slice_label_areas is not None:
                        margin_bins = np.linspace(voxelspacing, inter_observ_margin, config_detector.label_area_num_bins)
                        error_margin = margin_bins[scale_bin - 1]
                    else:
                        error_margin = inter_observ_margin
                    # print("INFO area {:.2f}, scale_bin {}, error_margin {:.2f}, "
                    #      "inter_observer_m {:.2f}".format(s_area, scale_bin, error_margin, inter_observ_margin))
                    # config_detector.label_area_bins
                    if error_margin > 5.:
                        dt[~inside_obj_mask] -= error_margin
                        dt[inside_obj_mask] -= 5  # we only substract
                    else:
                        dt -= error_margin
                    # dt -= 5.  # 5 mm is what we accept
                    dt[dt < 0] = 0
                    dt_slices[cls_idx, :, :, slice_id] = dt
                else:
                    penalty = math.sqrt((h**2 + w**2)) * voxelspacing
                    # print("Penalty {:.2f}".format(penalty))
                    dt_slices[cls_idx, :, :, slice_id] = penalty
    return dt_slices


def determine_target_voxels(auto_seg, reference, dt_map, cls_indices=None, bg_classes=[0, 4]):
    """
        Parameters:
        auto_seg [num_of_classes, w, h, #slices]
        reference [num_of_classes, w, h, #slices]
        dt_map (distance transfer) [num_of_classes, w, h, #slices]
        cls_indices (tissue class indices that we process) e.g. for ACDC [1, 2, 3, 5, 6, 7]
        In general we assume that 0=background class
    """
    num_of_classes, w, h, num_of_slices = auto_seg.shape
    target_areas = np.zeros_like(auto_seg)
    if cls_indices is None:
        cls_indices = list(np.arange(1, num_of_classes))
        # remove bg-class for ES
        for rm_idx in bg_classes:
            if rm_idx in cls_indices:
                cls_indices.remove(rm_idx)
    else:
        if not isinstance(cls_indices, list):
            cls_indices = [cls_indices]
    for cls_idx in cls_indices:
        for slice_id in np.arange(num_of_slices):
            auto_seg_slice = auto_seg[cls_idx, :, :, slice_id]
            reference_slice = reference[cls_idx, :, :, slice_id]
            dt_map_slice = (dt_map[cls_idx, :, :, slice_id] > 0).astype(np.bool)
            # for the proposal roi's we use the uncertainty info and the automatic segmentation mask
            # for the target roi's we use the reference & automatic labels (for training only)
            seg_errors = auto_seg_slice != reference_slice
            roi_target_map = np.logical_and(dt_map_slice, seg_errors)
            target_areas[cls_idx, :, :, slice_id] = roi_target_map

    return target_areas
