import numpy as np


def create_grid_heat_map(pred_probs, grid_spacing, w, h, target_lbl_binary_grid, prob_threshold=0.):
    """

    :param pred_probs:
    :param grid_spacing:
    :param w:
    :param h:
    :param target_lbl_binary_grid: same shape as pred_probs [w / grid-spacing, h/grid-spacing]
                                    contains gt labels for each grid block. Used here to reformat so that it fits
                                    the heat_map coordinates. We use this during plotting
    :param prob_threshold:
    :return: heat-map of original patch or image size. we fill the grid-blocks with the values from the predicted
                        probablities.
    """
    heat_map = np.zeros((w, h))
    target_lbl_grid = np.zeros(target_lbl_binary_grid.shape)
    grid_map_x, grid_map_y = [], []
    grid_spacings_w = np.arange(0, w + grid_spacing, grid_spacing)[1:]
    grid_spacings_h = np.arange(0, h + grid_spacing, grid_spacing)[1:]
    # Second split label slice horizontally
    start_w = 0
    pred_idx_w = 0
    for w_offset in grid_spacings_w:
        pred_idx_h = 0
        start_h = 0
        for h_offset in grid_spacings_h:
            if pred_probs is not None:
                block_prob = pred_probs[pred_idx_w, pred_idx_h]
            else:
                block_prob = 0
            if block_prob >= prob_threshold:
                heat_map[start_w:w_offset, start_h:h_offset] = block_prob
            grid_map_x.append((start_w + w_offset) / 2.)
            grid_map_y.append((start_h + h_offset) / 2.)
            start_h = h_offset
            pred_idx_h += 1
        start_w = w_offset
        pred_idx_w += 1
    # print("grid_map_x {} grid_map_y {}".format(len(grid_map_x), len(grid_map_y)))
    # print("pred_idx_w {} pred_idx_h {}".format(pred_idx_w, pred_idx_h))
    return heat_map, [np.array(grid_map_x), np.array(grid_map_y)], None
