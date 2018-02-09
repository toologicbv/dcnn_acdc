
import torch
import torch.nn as nn
from torch.autograd import Variable
from config.config import DEFAULT_DCNN_2D
import numpy as np
from building_blocks import Basic2DCNNBlock
from models.building_blocks import ConcatenateCNNBlock
from utils.dice_metric import soft_dice_score


class BaseDilated2DCNN(nn.Module):

    def __init__(self, architecture=DEFAULT_DCNN_2D, use_cuda=False, verbose=True):
        super(BaseDilated2DCNN, self).__init__()
        self.architecture = architecture
        self.use_cuda = use_cuda
        self.num_conv_layers = self.architecture['num_of_layers']
        self.verbose = verbose
        self.model = self._build_dcnn()

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

    def get_loss(self, predictions, labels):
        """
            we need to reshape the tensors because CrossEntropy expects 2D tensor (N, C) where C is num of classes
            the input tensor is in our case [batch_size, 2 * num_of_classes, height, width]
            the labels are                  [batch_size, 2 * num_of_classes, height, width]
        """
        num_of_classes = labels.size(1)
        losses = Variable(torch.FloatTensor(num_of_classes))
        if self.use_cuda:
            losses = losses.cuda()
        for cls in np.arange(labels.size(1)):
            losses[cls] = soft_dice_score(predictions, labels, cls=cls)

        # regloss = don't forget
        # loss = -1.0 * loss_ES - 1.0 * loss_ED + 0.0001 * regloss
        return torch.sum(losses)

    def cuda(self):
        super(BaseDilated2DCNN, self).cuda()

    def zero_grad(self):
        super(BaseDilated2DCNN, self).zero_grad()

    def save_params(self, absolute_path):
        torch.save(self.state_dict(), absolute_path)

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