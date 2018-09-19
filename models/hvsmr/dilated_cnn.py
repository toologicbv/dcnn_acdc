import numpy as np
import torch

from models.dilated_cnn import BaseDilated2DCNN
from utils.dice_metric import soft_dice_score, dice_coefficient
from common.losses import compute_brier_score
from common.hvsmr.config import config_hvsmr
from utils.medpy_metrics import hd


class HVSMRDilated2DCNN(BaseDilated2DCNN):

    def __init__(self, architecture, optimizer=torch.optim.Adam, lr=1e-4, weight_decay=0.,
                 use_cuda=False, verbose=True, cycle_length=0, loss_function="soft-dice",
                 use_reg_loss=False, use_dual_head=True):
        super(HVSMRDilated2DCNN, self).__init__(architecture=architecture, optimizer=optimizer, lr=lr,
                                                weight_decay=weight_decay, use_cuda=use_cuda,
                                                verbose=verbose, cycle_length=cycle_length,
                                                loss_function=loss_function, use_reg_loss=use_reg_loss,
                                                use_dual_head=use_dual_head)
        self.np_dice_coeffs = np.zeros(architecture["channels"][-1])

    def get_loss_cross_entropy(self, log_softmax_predictions, labels_multiclass):
        # print("DEBUG - get_loss_cross_entropy ", log_softmax_predictions.shape, labels_multiclass.shape)
        return self.lossfunc(log_softmax_predictions, labels_multiclass)

    def do_train(self, batch):
        self.zero_grad()

        b_predictions, b_log_soft_preds = self(batch.get_images())
        if self.loss_function == "cross-entropy":
            b_loss = self.get_loss_cross_entropy(b_log_soft_preds, batch.get_labels_multiclass())
            _ = self.get_loss(b_predictions, batch.get_labels(), zooms=config_hvsmr.voxelspacing)
        else:
            b_loss = self.get_loss(b_predictions, batch.get_labels(), zooms=config_hvsmr.voxelspacing)

        # compute gradients w.r.t. model parameters
        b_loss.backward(retain_graph=False)
        if self.use_scheduler:
            self.lr_scheduler.step()

        self.optimizer.step()
        return b_loss

    def get_accuracy(self):
        # returns numpy array of shape [2], we skip the background class
        return self.np_dice_coeffs[1:]

    def get_dice_losses(self, average=False):
        # this is the same as get_accuracy, but we need this for compatibility during validation
        return self.get_accuracy()

    def get_loss(self, predictions, labels, zooms=None, compute_hd=False):
        """
            the input tensor is in our case [batch_size, num_of_classes, height, width]
            the labels are                  [batch_size, num_of_classes, height, width]
            :param predictions
            :param labels
            :param zooms
            :param compute_hd

        """
        num_of_classes = predictions.size(1)
        batch_size = predictions.size(0)
        dices = torch.FloatTensor(num_of_classes)
        losses = torch.FloatTensor(num_of_classes)
        class_counts = np.zeros(num_of_classes)
        if self.use_cuda:
            losses = losses.cuda()
        # compute sum of regression predictions per image-slice and class
        self.hausdorff_list = []

        # determine the predicted labels.
        _, pred_labels = torch.max(predictions, dim=1)

        for cls in np.arange(num_of_classes):
            if self.loss_function == "soft-dice":
                losses[cls] = soft_dice_score(predictions[:, cls, :, :], labels[:, cls, :, :])
            elif self.loss_function == "brier":
                losses[cls] = compute_brier_score(predictions[:, cls, :, :], labels[:, cls, :, :])
            elif self.loss_function == "cross-entropy":
                # we compute the cross-entropy in a different method
                pass
            else:
                raise ValueError("ERROR - {} as loss functional is not supported!".format(self.loss_function))
            # brier_score[cls] = compute_brier_score(predictions[:, cls, :, :], labels[:, cls, :, :])
            # for the dice coefficient we need to determine the class labels of the predictions
            # remember that the object labels contains binary labels for each class (hence dim=1 has size 3)
            # here we determine the max index for each prediction over the 3 classes, and then set all labels
            # to zero that are not relevant for this dice coeff.
            pred_labels_cls = pred_labels == cls
            class_counts[cls] = torch.nonzero(pred_labels_cls).size(0)
            dices[cls] = dice_coefficient(pred_labels_cls, labels[:, cls, :, :])

            # IMPORTANT: we DON't compute the HD for the background class
            if cls != config_hvsmr.class_lbl_background:
                if compute_hd:
                    # need this counter to calculate the mean afterwards, some cls contours can be empty
                    batch_hd_cls = []
                    # hausdorff procedure only takes numpy arrays as input
                    # the hausdorff method must be computed per image, hence we loop through batch (dim0)
                    for i_idx in np.arange(batch_size):
                        np_gt_labels = labels[i_idx, cls, :, :].data.cpu().numpy()
                        np_pred_labels = pred_labels_cls[i_idx].data.cpu().numpy()
                        # only compute distance if both contours are actually in images
                        if 0 != np.count_nonzero(np_gt_labels) and 0 != np.count_nonzero(np_pred_labels):
                            batch_hd_cls.append(hd(np_pred_labels, np_gt_labels, voxelspacing=zooms, connectivity=1))

                    # compute mean for this class over batches
                    self.hausdorff_list.append(batch_hd_cls)

        self.np_dice_losses = losses.data.cpu().numpy()
        self.np_dice_coeffs = dices.data.cpu().numpy()
        # print("INFO - CLASS COUNTS {} {} {}".format(class_counts[0], class_counts[1], class_counts[2]))
        # NOTE: we want to minimize the loss, but the soft-dice-loss is actually increasing when our predictions
        # get better. HENCE, we need to multiply by minus one here.
        if self.loss_function == "soft-dice":
            return (-1.) * torch.sum(losses)
        elif self.loss_function == "cross-entropy":
            # as we mentioned before, we compute CE in different method, but we needed to compute dice and hd
            return torch.zeros(1)
        else:
            # summing the mean loss for RV & LV as if we would have the cavity volumes. See whether this "helps"
            losses = torch.mean(losses)
            return losses
