import math
import torch
import torch.nn as nn
import torch.functional as F
import torchvision


class DegenerateSliceDetector(nn.Module):

    def __init__(self, architecture, init_weights=True):
        super(DegenerateSliceDetector, self).__init__()
        self.num_of_spp_levels = architecture["num_of_spp_levels"]
        self.num_of_classes = architecture["num_of_classes"]
        model_name = getattr(torchvision.models, architecture["base_model"])
        self.base_model = model_name(pretrained=False)
        self.classifier = nn.Sequential(
            SpatialPyramidPoolLayer(num_of_levels=self.num_of_spp_levels, pool_type="max_pool"),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, self.num_of_classes),
        )
        self.modelName = 'Degenerate Slice Detector'
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x_in):
        print("INFO - DegenerateSliceDetector: x_in ", x_in.size())
        features = self.base_model(x_in)
        y = self.classifier(features)
        return y


class SpatialPyramidPoolLayer(nn.Module):

    def __init__(self, num_of_levels=3, pool_type='max_pool'):
        """
        num_of_levels: number of pooling levels
        pool_type: max- or average pooling is supported

        returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
        """
        super(SpatialPyramidPoolLayer, self).__init__()
        self.num_of_levels = num_of_levels
        self.pool_type = pool_type

    def forward(self, x_in):
        bs, c, h, w = x_in.size()
        pooling_layers = []
        for i in range(self.num_levels):
            level = 2 ** i
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.floor(h / level), math.floor(w / level))
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x_in, kernel_size=kernel_size,
                                      stride=stride).view(bs, -1)
            else:
                tensor = F.avg_pool2d(x_in, kernel_size=kernel_size,
                                      stride=stride).view(bs, -1)
            print("SpatialPyramidPoolLayer - forward - tensor.size.out ", tensor.size())
            pooling_layers.append(tensor)
        x = torch.cat(pooling_layers, dim=-1)
        return x
