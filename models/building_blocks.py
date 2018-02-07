import torch.nn as nn
from torch.nn import init


class Basic2DCNNBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=(1, 1), apply_batch_norm=False,
                 prob_dropout=0., apply_non_linearity=False, verbose=False):
        super(Basic2DCNNBlock, self).__init__()
        self.verbose = verbose
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                    dilation=dilation, bias=True)
        self.apply_non_linearity = apply_non_linearity
        self.apply_batch_norm = apply_batch_norm
        if prob_dropout > 0.:
            self.apply_dropout = True
        else:
            self.apply_dropout = False

        if self.apply_non_linearity:
            if self.verbose:
                print(">>> apply non linearity <<<")
            self.non_linearity = nn.ELU()
        if self.apply_batch_norm:
            if self.verbose:
                print(">>> apply batch-normalization <<<")
            self.bn = nn.BatchNorm2d(out_channels)
        if self.apply_dropout:
            if self.verbose:
                print(">>> apply dropout <<<")
            self.layer_drop = nn.Dropout2d(p=prob_dropout)

        # self.reset_weights()

    def reset_weights(self):
        init.xavier_normal(self.conv_layer.weight.data)
        if self.conv_layer.bias is not None:
            self.conv_layer.bias.data.fill_(0)

    def forward(self, x):
        out = self.conv_layer(x)
        if self.apply_non_linearity:
            out = self.non_linearity(out)
        if self.apply_batch_norm:
            out = self.bn(out)
        if self.apply_dropout:
            out = self.layer_drop(out)

        return out
