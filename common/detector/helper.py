from scipy import ndimage
import numpy as np
import matplotlib.patches as patches


def find_bbox_object(multi_label_slice, threshold_pixel_value=0):
    # multi_label_slice slice [w, h]. all pixels != 0 belong to the automatic segmentation mask
    # threshold_pixel_value (float): we're trying these bboxes also for the uncertainty maps.
    # but basically all pixels have values above 0. Experimenting with this.
    binary_mask_slice = (multi_label_slice > threshold_pixel_value).astype(np.bool)

    roi_slice_x, roi_slice_y = ndimage.find_objects(binary_mask_slice == 1)[0]

    roi_box = BoundingBox(roi_slice_x, roi_slice_y)

    return roi_box


class BoundingBox(object):

    padding = 5

    def __init__(self, slice_x, slice_y):
        # roi_slice_x contains [x_low, x_high], roi_slice_y contains [y_low, y_high]
        # roi box_four [Nx4] with x_low, y_low, x_high, y_high
        self.xy_left = tuple((slice_y.start - BoundingBox.padding, slice_x.start - BoundingBox.padding))
        self.height = slice_x.stop - slice_x.start + 2 * BoundingBox.padding
        self.width = slice_y.stop - slice_y.start + 2 * BoundingBox.padding
        self.area = self.height * self.width
        self.box_four = np.array([slice_x.start - BoundingBox.padding, slice_y.start - BoundingBox.padding,
                                  slice_x.stop, slice_y.stop])
        # create the default rectangular that we can use for plotting (red edges, linewidth=1)
        self.rectangular_patch = self.get_matplotlib_patch()

    def get_matplotlib_patch(self, color='r', linewidth=1):
        rect = patches.Rectangle(self.xy_left, self.width, self.height, linewidth=linewidth, edgecolor=color,
                                 facecolor='none')
        return rect

