import torch.nn as nn
import math
import torch
from models.detector.vgg_style_model import make_layers
from common.detector.config import config_detector
from common.detector.config import OPTIMIZER_DICT


class RegionDetector(nn.Module):

    def __init__(self, model_cfg, lr=0.001, init_weights=True, device=None):
        super(RegionDetector, self).__init__()
        # get model architecture parameters
        self.sgd_optimizer = model_cfg['optimizer']
        self.lr = None
        self.nclasses = model_cfg['num_of_classes']
        self.num_input_channels = model_cfg['num_of_input_channels']
        self.use_batch_norm = model_cfg['use_batch_norm']
        self.weight_decay = model_cfg['weight_decay']

        # The base VGG net. 3 convolutional layers (F=3x3; p=1) with BatchNorm + ReLU + MaxPooling (2x2).
        # So we have an output_stride (compression) factor of 8. e.g. 80x80 patch size results in 10x10 activation map
        self.base_model = make_layers(num_of_input_channels=self.num_input_channels, cfg=model_cfg['base'],
                                      batch_norm=self.use_batch_norm)
        # get the last sequential module, -1=MaxPooling, -2=[Conv2d, Batchnorm, ReLU]
        # print(self.base_model)
        if self.use_batch_norm:
            last_base_layer = self.base_model[-3]
        else:
            last_base_layer = self.base_model[-2]
        # param.shape in comprehension should have shape (#output_chnls, #inputchnls, kernel_size, kernel_size)
        # hence, we take index 0 (second [0] below)
        num_of_channels_last_layer = [param.shape for param in last_base_layer.parameters()][0][0]

        # flexible feature map size, we're using fully convolutional layers
        self.classifier = nn.Sequential(
            nn.Conv2d(num_of_channels_last_layer, 128, kernel_size=1, padding=0),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Conv2d(128, self.nclasses, kernel_size=1, padding=0),
        )
        if init_weights:
            self._initialize_weights()

        # finally prepare optimizer
        self.set_learning_rate(lr=lr)

    def forward(self, x):
        x = self.base_model(x)
        x = self.classifier(x)
        return x

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

    def set_learning_rate(self, lr):
        if self.sgd_optimizer == "sparse_adam":
            self.sgd_optimizer = OPTIMIZER_DICT[self.sgd_optimizer](
                self.parameters(), lr=lr)
        else:
            self.sgd_optimizer = OPTIMIZER_DICT[self.sgd_optimizer](
                self.parameters(), lr=lr, weight_decay=self.weight_decay, betas=(0.9, 0.999))
        self.lr = lr


if __name__ == '__main__':
    detector_model = RegionDetector(config_detector.detector_cfg, init_weights=True)
    device = torch.device("cuda")
    detector_model = detector_model.to(device)
    dummy_x = (torch.randn(1, 2, 80, 80)).cuda()
    out = detector_model(dummy_x)
    print("Output has shape ", out.shape)
