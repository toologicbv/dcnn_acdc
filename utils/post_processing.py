import scipy.ndimage as scnd
import numpy as np


def filter_connected_components(pred_labels, cls=None, verbose=False, threshold=0., rank=None):
    """

    :param pred_labels:
    :param cls: currently not in use, only for debug purposes
    :param verbose:
    :return:
    """
    if rank is None:
        rank = pred_labels.ndim

    if rank == 2:
        structure = [[0, 1, 0],
                     [1, 1, 1],
                     [0, 1, 0]]
    elif rank == 3:
        structure = [[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                     [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                     [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]
    else:
        raise ValueError("Input has more than 3 dimensions which is not supported by this function")

    if threshold != 0:
        # used for filtering 3D class uncertainty maps, masking voxels that are below uncertainty (tolerated)
        # threshold
        mask = pred_labels >= threshold
    else:
        mask = pred_labels == 1

    cc_labels, n_comps = scnd.measurements.label(mask, structure=structure)
    if verbose:
        print('INFO - Class {}: FCC {} components'.format(cls, n_comps))
    if n_comps > 1:
        sel_comp = 0
        in_comp = 0
        # Find the largest connected component
        for i_comp in range(1, n_comps + 1):
            if np.sum(cc_labels == i_comp) > in_comp:
                sel_comp = i_comp
                in_comp = np.sum(cc_labels == i_comp)
        if verbose:
            print('INFO - Class {}: Select component {}'.format(cls, sel_comp))
        # set all pixels/voxels to 0 (background) if they don't belong to the LARGEST connected-component
        for i_comp in range(1, n_comps + 1):
            if i_comp != sel_comp:
                pred_labels[cc_labels == i_comp] = 0

    return pred_labels
