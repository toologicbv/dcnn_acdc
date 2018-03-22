import torch.nn.functional as F
import torch.nn as nn
import torch


class CReLU(nn.Module):
    def __init__(self):
        super(CReLU, self).__init__()

    def forward(self, x):
        return torch.cat((F.relu(x), F.relu(-x)), dim=1)

    @property
    def __name__(self):
        return "CReLU"
