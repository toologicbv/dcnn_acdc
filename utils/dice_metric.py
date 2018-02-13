import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable

from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.metrics.classification import UndefinedMetricWarning


# https://github.com/EKami/carvana-challenge/blob/master/src/nn/losses.py
def soft_dice_score(prob_c, true_label_c):
    """

    Computing the soft-dice-loss for a SPECIFIC class according to:

    DICE_c = \frac{\sum_{i=1}^{N} (R_c(i) * A_c(i) ) }{ \sum_{i=1}^{N} (R_c(i) +   \sum_{i=1}^{N} A_c(i)  }

    Input: (1) probs: 4-dim tensor [batch_size, num_of_classes, width, height]
               contains the probabilities for each class
           (2) true_binary_labels, 4-dim tensor with the same dimensionalities as probs, but contains binary
           labels for a specific class

           Remember that classes 0-3 belongs to ES images (phase) and 4-7 to ED images

    """
    eps = 1.0e-6

    nominator = torch.sum(true_label_c * prob_c)
    denominator = torch.sum(true_label_c) + torch.sum(prob_c) + eps
    return nominator/denominator


def dice_coefficient(pred_labels, true_labels, cls=0):
    """
    Compute the Dice aka F1-score

    We assume predicted and true labels are PyTorch Variables for ONE specific class.

    Both


    """

    if not (isinstance(pred_labels, Variable) or isinstance(true_labels, torch.FloatTensor)):
        raise TypeError('expected torch.autograd.Variable or torch.FloatTensor, but got: {}'
                        .format(torch.typename(true_labels)))
    else:
        np_true_labels = true_labels.data.cpu().squeeze().numpy()
        np_pred_labels = pred_labels.data.cpu().squeeze().numpy()

    intersection = np.sum((np_pred_labels == 1) * (np_true_labels == 1))
    denominator = np.sum(np_pred_labels == 1) + np.sum(np_true_labels == 1)
    try:
        dice = 2 * float(intersection) / float(denominator)
    except ZeroDivisionError:
        print("WARNING - Division by zero in procedure dice_coefficient!")
        dice = 0.
    return dice
