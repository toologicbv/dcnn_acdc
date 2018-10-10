from scipy import ndimage
from scipy.ndimage.measurements import label
import numpy as np
import matplotlib.patches as patches
from common.detector.config import config_detector


# def check_box_overlap(box1, box2):
#     left = max(x1, x2)
#     right = min(x1 + w1, x2 + w2)
#     bottom = max(y1, y2)
#     top = min(y1 + h1, y2 + h2)
#     height = top - bottom
#     width = right - left
#
#     if height < 0 or width < 0:
#         return True
#     else:
#         return False


def find_multiple_connected_rois(label_slice, padding=1, min_size=config_detector.min_roi_area):
    """

    :param label_slice: has shape [w, h] and label values are binary (i.e. no distinction between tissue classes)

    :param padding: in fact extending the roi area that we detected. Because the target structure is of connectivity
                    six (3x3), we at least extend by 1 pixel to both sides
    :param min_size: the minimum area size of the 2D 4-connected component
    :return:
    """
    structure = [[0, 1, 0],
                     [1, 1, 1],
                     [0, 1, 0]]
    cc_labels, n_comps = label(label_slice, structure=structure)
    roi_boxes = np.empty((0, 4))
    roi_box_areas = []
    roi_binary_mask = np.zeros_like(label_slice)
    for i_comp in np.arange(1, n_comps + 1):
        comp_mask = cc_labels == i_comp
        roi_slice_x, roi_slice_y = ndimage.find_objects(comp_mask)[0]
        comp_mask_size = np.count_nonzero(comp_mask)
        # roi_slice_x contains [x_low, x_high], roi_slice_y contains [y_low, y_high]
        # roi boxes [Nx4] with x_low, y_low, x_high, y_high
        roi_box = BoundingBox(roi_slice_x, roi_slice_y, padding=padding)
        if comp_mask_size >= min_size:
            roi_box_areas.append(roi_box.area)
            roi_boxes = np.concatenate((roi_boxes, roi_box.box_four[np.newaxis])) if roi_boxes.size else \
                roi_box.box_four[np.newaxis]
            if comp_mask_size <= config_detector.max_roi_area_for_fill:
                roi_binary_mask[roi_box.slice_x, roi_box.slice_y] = 1
            else:
                roi_binary_mask[cc_labels == i_comp] = 1
    del cc_labels
    del comp_mask

    return roi_boxes, roi_binary_mask, roi_box_areas


def find_bbox_object(multi_label_slice, threshold_pixel_value=0, padding=0):
    # multi_label_slice slice [w, h]. all pixels != 0 belong to the automatic segmentation mask
    # threshold_pixel_value (float): we're trying these bboxes also for the uncertainty maps.
    # but basically all pixels have values above 0. Experimenting with this.
    binary_mask_slice = (multi_label_slice > threshold_pixel_value).astype(np.bool)
    if 0 != np.count_nonzero(binary_mask_slice):
        roi_slice_x, roi_slice_y = ndimage.find_objects(binary_mask_slice == 1)[0]
    else:
        roi_slice_x, roi_slice_y = slice(0, 0, None), slice(0, 0, None)
        padding = 0

    roi_box = BoundingBox(roi_slice_x, roi_slice_y, padding=padding)

    return roi_box


class BoundingBox(object):

    def __init__(self, slice_x, slice_y, padding=0):
        # roi_slice_x contains [x_low, x_high], roi_slice_y contains [y_low, y_high]
        # roi box_four [Nx4] with x_low, y_low, x_high, y_high
        self.empty = False
        slice_x = slice(slice_x.start - padding, slice_x.stop + padding, None)
        slice_y = slice(slice_y.start - padding, slice_y.stop + padding, None)
        self.slice_x = slice_x
        self.slice_y = slice_y
        if slice_x.stop - slice_x.start == 0:
            self.empty = True
        self.padding = padding
        self.xy_left = tuple((slice_y.start, slice_x.start))
        # actually we switched height and width because we're
        self.height = slice_x.stop - slice_x.start
        self.width = slice_y.stop - slice_y.start
        self.area = self.height * self.width
        self.box_four = np.array([slice_x.start, slice_y.start,
                                  slice_x.stop, slice_y.stop])
        # create the default rectangular that we can use for plotting (red edges, linewidth=1)
        self.rectangular_patch = self.get_matplotlib_patch()

    def get_matplotlib_patch(self, color='r', linewidth=1):
        rect = patches.Rectangle(self.xy_left, self.width, self.height, linewidth=linewidth, edgecolor=color,
                                 facecolor='none')
        return rect

    @staticmethod
    def create(box_four, padding=0):
        """

        :param box_four: np array of shape [4] with x_low, y_low, x_high, y_high
        :param padding:
        :return: BoundingBox object
        """
        slice_x, slice_y = BoundingBox.convert_to_slices(box_four)
        slice_y = slice(box_four[1], box_four[3], None)
        return BoundingBox(slice_x, slice_y, padding=padding)

    @staticmethod
    def convert_to_slices(box_four):
        slice_x = slice(box_four[0], box_four[2], None)
        slice_y = slice(box_four[1], box_four[3], None)
        return slice_x, slice_y

    @staticmethod
    def convert_slices_to_box_four(slice_x, slice_y):
        return np.array([slice_x.start, slice_y.start, slice_x.stop, slice_y.stop])


def find_box_four_rois(label_slice):
    """

    :param label_slice: has shape [w, h] and label values are binary (i.e. no distinction between tissue classes)


    :return:
    """
    structure = [[0, 1, 0],
                     [1, 1, 1],
                     [0, 1, 0]]
    cc_labels, n_comps = label(label_slice, structure=structure)
    roi_boxes = np.empty((0, 4))

    for i_comp in np.arange(1, n_comps + 1):
        comp_mask = cc_labels == i_comp
        roi_slice_x, roi_slice_y = ndimage.find_objects(comp_mask)[0]
        roi_box = BoundingBox(roi_slice_x, roi_slice_y, padding=0)
        roi_boxes = np.concatenate((roi_boxes, roi_box.box_four[np.newaxis])) if roi_boxes.size else \
            roi_box.box_four[np.newaxis]

    return roi_boxes
