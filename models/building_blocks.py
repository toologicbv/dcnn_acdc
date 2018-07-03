import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch


class CReLU(nn.Module):
    def __init__(self):
        super(CReLU, self).__init__()

    def forward(self, x):
        return torch.cat((F.relu(x), F.relu(-x)), x.dim()-1)


class ConcatenateCNNBlock(nn.Module):
    """
        Special layer/block designed for concatenation of two Conv2d layers that are combined with a
        non-linearity (assuming LogSoftmax or Softmax for classification purposes.
        We used this e.g. for the ACDC challenge data to predict four segmentation class labels
        for two-channels input images (end-systole and end-diastole images)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=(1, 1),
                 prob_dropout=0., apply_batch_norm=False, apply_non_linearity=None, bias=True, axis=1, verbose=False):
        super(ConcatenateCNNBlock, self).__init__()

        self.apply_batch_norm = apply_batch_norm
        self.conv_layer1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                     dilation=dilation, bias=bias)
        self.conv_layer2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                     dilation=dilation, bias=bias)
        self.axis = axis
        self.verbose = verbose
        self.apply_non_linearity = apply_non_linearity
        if prob_dropout > 0.:
            self.apply_dropout = True
            self.layer_drop = nn.Dropout2d(p=prob_dropout)
        else:
            self.apply_dropout = False
            self.layer_drop = None

        if self.verbose:
            print("INFO - Adding {} layer".format(self.__class__.__name__))

        if self.apply_non_linearity is not None:
            # assuming that the only non-linearity we're encountering here is Softmax or LogSoftmax
            # if not (isinstance(apply_non_linearity, nn.LogSoftmax) or
            #        isinstance(apply_non_linearity, nn.Softmax)):
            #    raise ValueError("{} is not supported as non-linearity".format(apply_non_linearity.__class__.__name__))
            if self.verbose:
                print("INFO - apply {}".format(apply_non_linearity.__name__))
            self.non_linearity = self.apply_non_linearity(dim=1)
            self.log_softmax = nn.LogSoftmax(dim=1)

        if self.apply_batch_norm:
            if self.verbose:
                print("INFO - apply batch-normalization")
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

    def cuda(self):
        super(ConcatenateCNNBlock, self).cuda()

    def reset_weights(self):
        init.xavier_normal(self.conv_layer1.weight.data)
        init.xavier_normal(self.conv_layer2.weight.data)
        if self.conv_layer1.bias is not None:
            self.conv_layer1.bias.data.fill_(0)
        if self.conv_layer2.bias is not None:
            self.conv_layer2.bias.data.fill_(0)

    def forward(self, tensor_in):
        if self.apply_dropout:
            tensor_in = self.layer_drop(tensor_in)

        out1 = self.conv_layer1(tensor_in)
        out2 = self.conv_layer2(tensor_in)
        if self.apply_batch_norm:
            out1 = self.bn1(out1)
            out2 = self.bn2(out2)

        if self.apply_non_linearity is not None:
            # The model uses the so called soft-Dice loss function. Calculation of the loss requires
            # that we calculate probabilities for each pixel, and hence we use the Softmax for this, which should
            # be specified as the non-linearity in the dictionary of the config object
            out1_log_softmax = self.log_softmax(out1)
            out2_log_softmax = self.log_softmax(out2)
            out_log_softmax = torch.cat((out1_log_softmax, out2_log_softmax), dim=self.axis)
            out1 = self.non_linearity(out1)
            out2 = self.non_linearity(out2)
        else:
            out_log_softmax = None

        return torch.cat((out1, out2), dim=self.axis), out_log_softmax


class ConcatenateCNNBlockWithRegression(ConcatenateCNNBlock):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=(1, 1),
                 prob_dropout=0., apply_batch_norm=False, apply_non_linearity=None, bias=True, axis=1, verbose=False):
        super(ConcatenateCNNBlockWithRegression, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                                                padding=padding, dilation=dilation,
                                                                prob_dropout=prob_dropout,
                                                                apply_batch_norm=apply_batch_norm,
                                                                apply_non_linearity=apply_non_linearity,
                                                                bias=bias, axis=axis, verbose=verbose)
        # the previous layer provides us with 128 channels, which we will concatenate with 4class channels from
        # this layer (ConcatenateCNNBlock) = 132, which we want to reduce to 1! channel for the regression of
        # pixels
        reg_channels = 128
        prob_dropout = 0.1
        self.layer3a = nn.Sequential(nn.Conv2d(in_channels + 4, reg_channels, kernel_size, stride=stride, padding=0,
                                     dilation=dilation, bias=bias), nn.BatchNorm2d(reg_channels), nn.ELU())
        self.layer3b = nn.Sequential(nn.Conv2d(in_channels + 4, reg_channels, kernel_size, stride=stride, padding=0,
                                               dilation=dilation, bias=bias), nn.BatchNorm2d(reg_channels), nn.ELU())
        self.layer4a = nn.Conv2d(reg_channels, 2, kernel_size, stride=stride, padding=0, dilation=dilation, bias=bias)
        self.layer4b = nn.Conv2d(reg_channels, 2, kernel_size, stride=stride, padding=0, dilation=dilation, bias=bias)

    def forward(self, tensor_in):
        """

        :param tensor_in: [batch_size, 128, width, height]
        :return:
        """
        # returns a concatenated tensor with shape [batch_size, 8classes, width, height]
        if self.apply_dropout:
            tensor_in = self.layer_drop(tensor_in)

        out1 = self.conv_layer1(tensor_in)
        out2 = self.conv_layer2(tensor_in)
        if self.apply_batch_norm:
            out1 = self.bn1(out1)
            out2 = self.bn2(out2)

        if self.apply_non_linearity is not None:
            # The model uses the so called soft-Dice loss function. Calculation of the loss requires
            # that we calculate probabilities for each pixel, and hence we use the Softmax for this, which should
            # be specified as the non-linearity in the dictionary of the config object
            out1 = self.non_linearity(out1)
            out2 = self.non_linearity(out2)
        # concatenate the features maps of tensor_in with the softmax-probability heat maps we generated in this
        # forward step: the "fused" tensors have shape [batch_size, 132, width, height]
        # out1_fused = torch.cat((tensor_in, out1[:, 1:2, :, :], out1[:, 3:4, :, :]), dim=self.axis)
        out1_fused = torch.cat((tensor_in, out1), dim=self.axis)
        # out2_fused = torch.cat((tensor_in, out2[:, 1:2, :, :], out2[:, 3:4, :, :]), dim=self.axis)
        out2_fused = torch.cat((tensor_in, out2), dim=self.axis)
        out1_fused = self.layer3a(out1_fused)
        out2_fused = self.layer3b(out2_fused)
        # in the final step the maps are compressed to [batch_size, 4, width, height]
        out1_fused = self.layer4a(out1_fused)
        out2_fused = self.layer4b(out2_fused)
        # 1. tensor: [batch_size, 8, width, height]     2. tensor: [batch_size, 4, width, height] only RV+LV for ES/ED
        return torch.cat((out1, out2), dim=self.axis), torch.cat((out1_fused, out2_fused), dim=self.axis)


class Basic2DCNNBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=(1, 1), apply_batch_norm=False,
                 prob_dropout=0., apply_non_linearity=None, verbose=False):
        super(Basic2DCNNBlock, self).__init__()
        if apply_non_linearity.__name__ == "CReLU":
            conv_out_channels = int(out_channels / 2)
        else:
            conv_out_channels = out_channels

        self.verbose = verbose
        self.conv_layer = nn.Conv2d(in_channels, conv_out_channels, kernel_size, stride=stride, padding=padding,
                                    dilation=dilation, bias=True)
        self.apply_non_linearity = apply_non_linearity
        self.apply_batch_norm = apply_batch_norm
        if prob_dropout > 0.:
            self.apply_dropout = True
        else:
            self.apply_dropout = False

        if self.apply_non_linearity is not None:
            if self.verbose:
                print(">>> apply non linearity <<<")
            self.non_linearity = apply_non_linearity()
        if self.apply_batch_norm:
            if self.verbose:
                print(">>> apply batch-normalization <<<")
            self.bn = nn.BatchNorm2d(out_channels)
        if self.apply_dropout:
            if self.verbose:
                print(">>> apply dropout <<<")
            self.layer_drop = nn.Dropout2d(p=prob_dropout)

    def cuda(self):
        super(Basic2DCNNBlock, self).cuda()

    def reset_weights(self):
        init.xavier_normal(self.conv_layer.weight.data)
        if self.conv_layer.bias is not None:
            self.conv_layer.bias.data.fill_(0)

    def forward(self, x):

        out = self.conv_layer(x)
        if self.apply_non_linearity is not None:
            out = self.non_linearity(out)
        if self.apply_batch_norm:
            out = self.bn(out)
        if self.apply_dropout:
            out = self.layer_drop(out)

        return out
