
import torch
import torch.nn as nn
from torch.autograd import Variable
from config.config import DEFAULT_DCNN_2D, config
import numpy as np
from config.config import OPTIMIZER_DICT
from building_blocks import Basic2DCNNBlock
from models.building_blocks import ConcatenateCNNBlock
from utils.dice_metric import soft_dice_score, dice_coefficient
from models.lr_schedulers import CycleLR


class BaseDilated2DCNN(nn.Module):

    def __init__(self, architecture=DEFAULT_DCNN_2D, optimizer=torch.optim.Adam, lr=1e-4, weight_decay=0.,
                 use_cuda=False, verbose=True, cycle_length=0):
        super(BaseDilated2DCNN, self).__init__()
        self.architecture = architecture
        self.use_cuda = use_cuda
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
        if self.use_cuda:
            self.cuda()

    def _build_dcnn(self):

        layer_list = []
        num_conv_layers = self.architecture['num_of_layers']
        for l_id in np.arange(num_conv_layers):
            if l_id == 0:
                in_channels = self.architecture['input_channels']
            else:
                # get previous output channel size
                in_channels = self.architecture['channels'][l_id - 1]
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
            else:
                # for ACDC data the last layer is a concatenation of two 2D-CNN layers
                layer_list.append(ConcatenateCNNBlock(in_channels, self.architecture['channels'][l_id],
                                                      self.architecture['kernels'][l_id],
                                                      stride=self.architecture['stride'][l_id],
                                                      dilation=self.architecture['dilation'][l_id],
                                                      apply_batch_norm=self.architecture['batch_norm'][l_id],
                                                      apply_non_linearity=self.architecture['non_linearity'][l_id],
                                                      axis=1, verbose=self.verbose))

        return nn.Sequential(*layer_list)

    def forward(self, input):
        """

        :param input:
        :return: (1) the raw output in order to compute loss with PyTorch cross-entropy (see comment below)
                 (2) the softmax output
        """
        if not isinstance(input, torch.autograd.variable.Variable):
            raise ValueError("input is not of type torch.autograd.variable.Variable")

        out = self.model(input)
        # our last layer ConcatenateCNNBlock already contains the two Softmax layers

        return out

    def get_loss(self, predictions, labels, is_train=True):
        """
            we need to reshape the tensors because CrossEntropy expects 2D tensor (N, C) where C is num of classes
            the input tensor is in our case [batch_size, 2 * num_of_classes, height, width]
            the labels are                  [batch_size, 2 * num_of_classes, height, width]
        """
        num_of_classes = labels.size(1)
        losses = Variable(torch.FloatTensor(num_of_classes))
        dices = Variable(torch.FloatTensor(num_of_classes))
        if self.use_cuda:
            losses = losses.cuda()
        # determine the predicted labels. IMPORTANT do this separately for each ES and ED which means
        # we have to split the network output on dim1 in to parts of size 4
        _, pred_labels_es = torch.max(predictions[:, 0:num_of_classes / 2, :, :], dim=1)
        _, pred_labels_ed = torch.max(predictions[:, num_of_classes / 2:num_of_classes, :, :], dim=1)
        for cls in np.arange(labels.size(1)):
            losses[cls] = soft_dice_score(predictions[:, cls, :, :], labels[:, cls, :, :])
            # for the dice coefficient we need to determine the class labels of the predictions
            # remember that the object labels contains binary labels for each class (hence dim=1 has size 8)
            # here we determine the max index for each prediction over the 8 classes, and then set all labels
            # to zero that are not relevant for this dice coeff.
            # IMPORTANT: we DON't compute the DICE for the background class
            if cls != config.class_lbl_background:
                if cls < num_of_classes / 2:
                    pred_labels = pred_labels_es == cls
                else:
                    # it looks kind of awkward but, cls stays in the range [0-7] but the two tensors pred_labels_es
                    # and pred_labels_ed contain class values each from [0-3] and not from [0-7]. Hence for the ED
                    # labels (which are situated in the second part of tensor "label" which acually has size 8 in dim1)
                    # we make sure the comparison accounts for this (cls - num_of_classes / 2)
                    pred_labels = pred_labels_ed == (cls - num_of_classes / 2)
                dices[cls] = dice_coefficient(pred_labels, labels[:, cls, :, :])

        self.np_dice_losses = losses.data.cpu().numpy()
        self.np_dice_coeffs = dices.data.cpu().numpy()

        # NOTE: we want to minimize the loss, but the soft-dice-loss is actually increasing when our predictions
        # get better. HENCE, we need to multiply by minus one here.
        return (-1.) * torch.sum(losses)

    def do_train(self, batch):
        self.zero_grad()
        b_out = self(batch.get_images())
        b_loss = self.get_loss(b_out, batch.get_labels())
        # compute gradients w.r.t. model parameters
        b_loss.backward(retain_graph=False)
        if self.use_scheduler:
            self.lr_scheduler.step()

        self.optimizer.step()
        return b_loss

    def do_validate(self, batch):
        self.eval()
        b_predictions = self(batch.get_images())
        val_loss = self.get_loss(b_predictions, batch.get_labels())
        self.train()
        return val_loss

    def get_dice_losses(self, average=False):
        if not average:
            return self.np_dice_losses
        else:
            split = int(self.np_dice_losses.shape[0] * 0.5)
            return np.mean(self.np_dice_losses[0:split]), np.mean(self.np_dice_losses[split:])

    def get_accuracy(self):
        return np.concatenate((self.np_dice_coeffs[1:4], self.np_dice_coeffs[5:8]))

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
        sum_grads = 0
        for name, param in self.named_parameters():
            if param.grad is not None:
                sum_grads += torch.sum(torch.abs(param.grad.data))
            else:
                if verbose:
                    print("WARNING - No gradients for parameter >>> {} <<<".format(name))

        return sum_grads


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