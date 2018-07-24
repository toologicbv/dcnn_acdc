
import torch
import torch.nn as nn
from config.config import config
import numpy as np
from config.config import OPTIMIZER_DICT
from building_blocks import Basic2DCNNBlock
from models.building_blocks import ConcatenateCNNBlock, ConcatenateCNNBlockWithRegression
from utils.dice_metric import soft_dice_score, dice_coefficient
from models.lr_schedulers import CycleLR
from utils.medpy_metrics import hd
from common.losses import compute_brier_score


class BaseDilated2DCNN(nn.Module):

    def __init__(self, architecture, optimizer=torch.optim.Adam, lr=1e-4, weight_decay=0.,
                 use_cuda=False, verbose=True, cycle_length=0, loss_function="soft-dice",
                 use_reg_loss=False):
        super(BaseDilated2DCNN, self).__init__()
        self.architecture = architecture
        self.use_cuda = use_cuda
        self.use_regression_loss = use_reg_loss
        self.num_conv_layers = self.architecture['num_of_layers']
        self.verbose = verbose
        self.model = self._build_dcnn()
        self.lr_scheduler = None
        self.optimizer = OPTIMIZER_DICT[optimizer](
            self.parameters(), lr=lr, weight_decay=weight_decay)
        if cycle_length != 0:
            self.lr_scheduler = CycleLR(self.optimizer, alpha_zero=lr, cycle_length=cycle_length)
            self.use_scheduler = True
        else:
            self.use_scheduler = False

        self.np_dice_losses = None
        self.np_dice_coeffs = None
        self.np_reg_loss = None
        self.hausdorff_list = None
        self.loss_function = loss_function
        self.lossfunc = nn.NLLLoss()

    def cuda(self):
        super(BaseDilated2DCNN, self).cuda()

    def _build_dcnn(self):

        layer_list = []
        num_conv_layers = self.architecture['num_of_layers']
        for l_id in np.arange(num_conv_layers):
            if l_id == 0:
                in_channels = self.architecture['input_channels']
            else:
                # get previous output channel size

                in_channels = self.architecture['channels'][l_id - 1]
                # if self.architecture['non_linearity'][l_id - 1].__name__ == "CReLU":
                #    print("INFO - double input channels")
                #    in_channels = int(in_channels * 2)
            if self.verbose:
                print("Constructing layer {}".format(l_id+1))
            if l_id < num_conv_layers - 1:
                layer_list.append(Basic2DCNNBlock(in_channels, self.architecture['channels'][l_id],
                                                  self.architecture['kernels'][l_id],
                                                  stride=self.architecture['stride'][l_id],
                                                  dilation=self.architecture['dilation'][l_id],
                                                  apply_batch_norm=self.architecture['batch_norm'][l_id],
                                                  apply_non_linearity=self.architecture['non_linearity'][l_id],
                                                  prob_dropout=self.architecture['dropout'][l_id],
                                                  verbose=self.verbose))
                if self.use_cuda:
                    layer_list[-1].cuda()
            else:
                if not self.use_regression_loss:
                    # for ACDC data the last layer is a concatenation of two 2D-CNN layers
                    layer_list.append(ConcatenateCNNBlock(in_channels, self.architecture['channels'][l_id],
                                                          self.architecture['kernels'][l_id],
                                                          stride=self.architecture['stride'][l_id],
                                                          dilation=self.architecture['dilation'][l_id],
                                                          apply_batch_norm=self.architecture['batch_norm'][l_id],
                                                          apply_non_linearity=self.architecture['non_linearity'][l_id],
                                                          axis=1, verbose=self.verbose))
                else:
                    layer_list.append(ConcatenateCNNBlockWithRegression(
                        in_channels,
                        self.architecture['channels'][l_id],
                        self.architecture['kernels'][l_id],
                        stride=self.architecture['stride'][l_id],
                        dilation=self.architecture['dilation'][l_id],
                        apply_batch_norm=self.architecture['batch_norm'][l_id],
                        apply_non_linearity=self.architecture['non_linearity'][l_id],
                        axis=1, verbose=self.verbose))

                if self.use_cuda:
                    layer_list[-1].cuda()

        return nn.Sequential(*layer_list)

    def forward(self, input):
        """

        :param input:
        :return: (1) the raw output in order to compute loss with PyTorch cross-entropy (see comment below)
                 (2) the softmax output
        """

        out = self.model(input)
        # our last layer ConcatenateCNNBlock already contains the two Softmax layers

        return out

    def get_loss_cross_entropy(self, log_softmax_predictions, labels_multiclass):
        losses = torch.FloatTensor(2)
        if self.use_cuda:
            losses = losses.cuda()
        b_pred_es = log_softmax_predictions[:, :4, :, :]
        b_pred_ed = log_softmax_predictions[:, 4:, :, :]
        b_gt_labels_es = labels_multiclass[:, 0, :, :]
        b_gt_labels_ed = labels_multiclass[:, 1, :, :]
        losses[0] = self.lossfunc(b_pred_es, b_gt_labels_es)
        losses[1] = self.lossfunc(b_pred_ed, b_gt_labels_ed)

        return torch.sum(losses)

    def get_loss(self, predictions, labels, zooms=None, compute_hd=False, regression_maps=None,
                 num_of_labels_per_class=None):
        """
            IMPORTANT: predictions and labels have 4 dimensions (see below).
                       The second dim contains the split between ES and ED phase
                            ==> index 0-3 = ES
                            ==> index 4-7 = ED

            we need to reshape the tensors because CrossEntropy expects 2D tensor (N, C) where C is num of classes
            the input tensor is in our case [batch_size, 2 * num_of_classes, height, width]
            the labels are                  [batch_size, 2 * num_of_classes, height, width]
            :param predictions
            :param labels
            :param zooms
            :param compute_hd
            When computing a kind of regression loss for the pixels we use objects:
            num_of_labels_per_class: [batch_size, 8classes]
            object contains the number of pixels for the specific slice per class
            :param regression_maps: has shape [batch_size, 8, width, height]
            :param num_of_labels_per_class
            loss_func is: (1) softdice=soft_dice_score or (2) brier=compute_brier_score
        """
        batch_size = labels.size(0)
        num_of_classes = labels.size(1)
        half_classes = int(num_of_classes / 2)
        losses = torch.FloatTensor(num_of_classes)
        dices = torch.FloatTensor(num_of_classes)
        pixel_reg_loss = torch.FloatTensor(batch_size, half_classes)
        # compute sum of regression predictions per image-slice and class
        reg_loss_weight = 0.05
        adjusted_class_idx = [-1, 0, -1, 1, -1, 2, -1, 3]
        if self.use_regression_loss:
            regression_maps = torch.sum(regression_maps.view(batch_size, half_classes, -1), dim=2)

        self.hausdorff_list = []
        if self.use_cuda:
            losses = losses.cuda()
            pixel_reg_loss = pixel_reg_loss.cuda()

        # determine the predicted labels. IMPORTANT do this separately for each ES and ED which means
        # we have to split the network output on dim1 in to parts of size 4
        _, pred_labels_es = torch.max(predictions[:, 0:num_of_classes / 2, :, :], dim=1)
        _, pred_labels_ed = torch.max(predictions[:, num_of_classes / 2:num_of_classes, :, :], dim=1)

        for cls in np.arange(labels.size(1)):
            if self.loss_function == "soft-dice":
                losses[cls] = soft_dice_score(predictions[:, cls, :, :], labels[:, cls, :, :])
            elif self.loss_function == "brier":
                losses[cls] = compute_brier_score(predictions[:, cls, :, :], labels[:, cls, :, :])
            elif self.loss_function == "cross-entropy":
                pass
            else:
                raise ValueError("ERROR - {} as loss functional is not supported!".format(self.loss_function))
            # brier_score[cls] = compute_brier_score(predictions[:, cls, :, :], labels[:, cls, :, :])
            # for the dice coefficient we need to determine the class labels of the predictions
            # remember that the object labels contains binary labels for each class (hence dim=1 has size 8)
            # here we determine the max index for each prediction over the 8 classes, and then set all labels
            # to zero that are not relevant for this dice coeff.
            if cls < num_of_classes / 2:
                pred_labels = pred_labels_es == cls
            else:
                # it looks kind of awkward but, cls stays in the range [0-7] but the two tensors pred_labels_es
                # and pred_labels_ed contain class values each from [0-3] and not from [0-7]. Hence for the ED
                # labels (which are situated in the second part of tensor "label" which actually has size 8 in dim1)
                # we make sure the comparison accounts for this (cls - num_of_classes / 2)
                pred_labels = pred_labels_ed == (cls - half_classes)
            dices[cls] = dice_coefficient(pred_labels, labels[:, cls, :, :])
            # Compute an adhoc regression loss for the num of pixels belonging to a certain seg-class
            if self.use_regression_loss:
                adj_cls_idx = adjusted_class_idx[cls]
                if adj_cls_idx != -1:
                    # computing the number of predicted pixels per class. pred_labels should have shape
                    # [batch_size, width, height] with True values containing. we first need to flatten the
                    # tensor to [batch_size, rest] and then sum over the columns (so we get #nonzeros per row/batch)
                    # pred_num_of_labels[:, cls] = torch.sum(regression_maps[:, cls].view(batch_size, -1), dim=1)
                    # we compute the L1 norm
                    pixel_reg_loss[:, adj_cls_idx] = torch.abs(regression_maps[:, adj_cls_idx] -
                                                               num_of_labels_per_class[:, cls])
                else:
                    pixel_reg_loss[:, adj_cls_idx] = 0

            # IMPORTANT: we DON't compute the HD for the background class
            if cls != config.class_lbl_background and cls != half_classes:
                if compute_hd:
                    # need this counter to calculate the mean afterwards, some cls contours can be empty
                    batch_hd_cls = []
                    # hausdorff procedure only takes numpy arrays as input
                    # the hausdorff method must be computed per image, hence we loop through batch (dim0)
                    for i_idx in np.arange(labels.size(0)):
                        np_gt_labels = labels[i_idx, cls, :, :].data.cpu().numpy()
                        np_pred_labels = pred_labels[i_idx].data.cpu().numpy()
                        # only compute distance if both contours are actually in images
                        if 0 != np.count_nonzero(np_gt_labels) and 0 != np.count_nonzero(np_pred_labels):
                            batch_hd_cls.append(hd(np_pred_labels, np_gt_labels, voxelspacing=zooms, connectivity=1))

                    # compute mean for this class over batches
                    self.hausdorff_list.append(batch_hd_cls)

        self.np_dice_losses = losses.data.cpu().numpy()
        self.np_dice_coeffs = dices.data.cpu().numpy()

        # NOTE: we want to minimize the loss, but the soft-dice-loss is actually increasing when our predictions
        # get better. HENCE, we need to multiply by minus one here.
        if self.loss_function == "soft-dice":
            return (-1.) * torch.sum(losses)
        elif self.loss_function == "cross-entropy":
            pass
        else:
            # summing the mean loss for RV & LV as if we would have the cavity volumes. See whether this "helps"
            losses = torch.mean(losses[0:half_classes]) + torch.mean(losses[half_classes:])
            if self.use_regression_loss:
                reg_loss = reg_loss_weight * torch.sum(torch.mean(pixel_reg_loss, dim=1))
                self.np_reg_loss = reg_loss.data.cpu().numpy()
                losses += reg_loss
            return losses

    def do_train(self, batch):
        self.zero_grad()
        if not self.use_regression_loss:
            b_predictions, b_log_soft_preds = self(batch.get_images())
            if self.loss_function == "cross-entropy":
                b_loss = self.get_loss_cross_entropy(b_log_soft_preds, batch.get_labels_multiclass())
                _ = self.get_loss(b_predictions, batch.get_labels())
                # acc = self.get_accuracy()
                # print("Current dice accuracies ES {:.3f}/{:.3f}/{:.3f} \t"
                #             "ED {:.3f}/{:.3f}/{:.3f} ".format(acc[0], acc[1], acc[2], acc[3], acc[4], acc[5]))
            else:
                b_loss = self.get_loss(b_predictions, batch.get_labels())
                # ent_loss = self.get_loss_cross_entropy(b_log_soft_preds, batch.get_labels_multiclass())
                # print("INFO - CrossEntropyLoss {:.3f}".format(ent_loss.data.cpu().numpy()[0]))
        else:
            b_predictions, b_out_regression = self(batch.get_images())
            b_loss = self.get_loss(b_predictions, batch.get_labels(), regression_maps=b_out_regression,
                                   num_of_labels_per_class=batch.get_num_labels_per_class())
        # compute gradients w.r.t. model parameters
        b_loss.backward(retain_graph=False)
        if self.use_scheduler:
            self.lr_scheduler.step()

        self.optimizer.step()
        return b_loss

    def do_test(self, images, labels, voxel_spacing=None, compute_hd=False, num_of_labels_per_class=None,
                mc_dropout=False, multi_labels=None):
        """
        voxel_spacing: we assume the image slices are 2D and have isotropic spacing. Hence voxel_spacing is a scaler
                       and for AC-DC equal to 1.4

        :returns validation loss (torch.Tensor) and the label predictions [batch_size, classes, width, height]
                 also as torch.Tensor
        """
        self.eval(mc_dropout=mc_dropout)
        if not self.use_regression_loss:
            b_predictions, b_log_soft_preds = self(images)
            test_loss = self.get_loss(b_predictions, labels, zooms=voxel_spacing, compute_hd=compute_hd)
            if self.loss_function == "cross-entropy" and multi_labels is not None:
                test_loss = self.get_loss_cross_entropy(b_log_soft_preds, multi_labels)
        else:
            b_predictions, b_reg_maps = self(images)
            test_loss = self.get_loss(b_predictions, labels, zooms=voxel_spacing, compute_hd=compute_hd,
                                      regression_maps=b_reg_maps, num_of_labels_per_class=num_of_labels_per_class)
        self.train()
        return test_loss, b_predictions

    def get_dice_losses(self, average=False):

        if not average:
            return self.np_dice_losses
        else:
            # REMEMBER: first 3 values are ES results, and last 3 values ED results
            # NOTE: returns two scalar values, one for ES one for ED
            split = int(self.np_dice_losses.shape[0] * 0.5)
            return np.mean(self.np_dice_losses[0:split]), np.mean(self.np_dice_losses[split:])

    def get_reg_loss(self):
        return self.np_reg_loss

    def get_accuracy(self):
        # returns numpy array of shape [6], 3 for ES and 3 for ED
        return np.concatenate((self.np_dice_coeffs[1:4], self.np_dice_coeffs[5:8]))

    def get_brier_score(self):
        return self.brier_score

    def get_hausdorff(self, compute_statistics=True):
        num_of_classes = len(self.hausdorff_list)
        # store mean and stdev
        hd_stats = np.zeros(num_of_classes)
        if compute_statistics:
            for cls in np.arange(num_of_classes):
                # can happen, with small batch-sizes that e.g. for RV class there were no contours, check here
                if len(self.hausdorff_list[cls]) != 0:
                    hd = np.array(self.hausdorff_list[cls])
                    hd_stats[cls] = np.mean(hd)
                else:
                    hd_stats[cls] = 0.

        return hd_stats, self.hausdorff_list

    def cuda(self):
        super(BaseDilated2DCNN, self).cuda()

    def zero_grad(self):
        super(BaseDilated2DCNN, self).zero_grad()

    def save_params(self, absolute_path):
        torch.save(self.state_dict(), absolute_path)

    def get_lr(self):
        if self.lr_scheduler is not None:
            lr = self.lr_scheduler.lr
        else:
            lr = self.optimizer.defaults["lr"]
        return lr

    def sum_grads(self, verbose=False):
        sum_grads = 0.
        for name, param in self.named_parameters():
            if param.grad is not None:
                sum_grads += torch.sum(torch.abs(param.grad.data))
            else:
                if verbose:
                    print("WARNING - No gradients for parameter >>> {} <<<".format(name))

        return sum_grads

    def train(self, mode=True, mc_dropout=False):
        """Sets the module in training mode.

        This has any effect only on modules such as Dropout or BatchNorm.

        Returns:
            Module: self
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        # we only execute this if we're in dropout mode (sampling) and we're NOT in training mode
        if mc_dropout and not mode:
            for module_name, module in self.named_modules():
                if "layer_drop" in module_name:
                    module.training = True
            # print("WARNING MC-DROPOUT - dropout layers are still in training mode")

        return self

    def eval(self, mc_dropout=False):
        """Sets the module in evaluation mode.

        This has any effect only on modules such as Dropout or BatchNorm.
        """
        return self.train(False, mc_dropout=mc_dropout)


class DilatedCNN(nn.Module):

    def __init__(self, use_cuda=False):
        super(DilatedCNN, self).__init__()
        self.use_cuda = use_cuda

        self.layer1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, dilation=1)
        self.elu1 = nn.ELU(inplace=True)
        self.layer2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, dilation=1)
        self.elu2 = nn.ELU(inplace=True)
        self.layer3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, dilation=2)
        self.elu3 = nn.ELU(inplace=True)
        self.layer4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, dilation=4)
        self.elu4 = nn.ELU(inplace=True)
        self.layer5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, dilation=8)
        self.elu5 = nn.ELU(inplace=True)
        self.layer6 = nn.Conv2d(32, 32, kernel_size=3, stride=1, dilation=16)
        self.elu6 = nn.ELU(inplace=True)
        self.layer7 = nn.Conv2d(32, 32, kernel_size=3, stride=1, dilation=32)
        self.elu7 = nn.ELU(inplace=True)
        self.layer8 = nn.Conv2d(32, 32, kernel_size=3, stride=1, dilation=1)
        self.elu8 = nn.ELU(inplace=True)
        self.bn8 = nn.BatchNorm2d(32)
        self.layer_drop8 = nn.Dropout2d(p=0.5)
        self.layer9 = nn.Conv2d(32, 192, kernel_size=1, stride=1, dilation=1)
        self.elu9 = nn.ELU(inplace=True)
        self.bn9 = nn.BatchNorm2d(192)
        self.layer_drop9 = nn.Dropout2d(p=0.5)
        self.layer10 = nn.Conv2d(192, 3, kernel_size=1, stride=1, dilation=1)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.loss_function = nn.NLLLoss2d()
        if self.use_cuda:
            self.cuda()
        print(">>> Everything is set-up!")

    def forward(self, input):
        out = self.layer1(input)
        out = self.elu1(out)
        out = self.layer2(out)
        out = self.elu2(out)
        out = self.layer3(out)
        out = self.elu3(out)
        out = self.layer4(out)
        out = self.elu4(out)
        out = self.layer5(out)
        out = self.elu5(out)
        out = self.layer6(out)
        out = self.elu6(out)
        out = self.layer7(out)
        out = self.elu7(out)
        out = self.layer8(out)
        out = self.elu8(out)
        out = self.bn8(out)
        out = self.layer_drop8(out)
        out = self.layer9(out)
        out = self.elu9(out)
        out = self.bn9(out)
        out = self.layer_drop9(out)
        out = self.layer10(out)
        out = self.log_softmax(out)
        return out

    def get_loss(self, predictions, labels):
        # we need to reshape the tensors because CrossEntropy expects 2D tensor (N, C) where C is num of classes
        # the input tensor is in our case [batch_size, num_of_classes, height, width]
        # the labels are                  [batch_size, 1, height, width]
        labels = labels.view(labels.size(0), labels.size(2), labels.size(3))
        # print("Loss sizes ", predictions.size(), labels.size())
        return self.loss_function(predictions, labels)

    def cuda(self):
        super(DilatedCNN, self).cuda()

    def zero_grad(self):
        super(DilatedCNN, self).zero_grad()

    def sum_grads(self, verbose=False):
        sum_grads = 0
        for name, param in self.named_parameters():
            if param.grad is not None:
                sum_grads += torch.sum(torch.abs(param.grad.data))
            else:
                if verbose:
                    print("WARNING - No gradients for parameter >>> {} <<<".format(name))

        return sum_grads

# dcnn_model = BaseDilated2DCNN(use_cuda=True)
# dcnn_model = DilatedCNN(use_cuda=True)