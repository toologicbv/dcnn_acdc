import torch.nn as nn
import math


__all__ = [
    'SimpleCNN', 'SimpleCNNV1',
]


class SimpleCNN(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(SimpleCNN, self).__init__()
        self.features = features
        # TODO: we do not yet know how dense we compress the input tiles (2D). This is taken from VGG net and assume
        # 7x7 matrix size.
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
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


def make_layers(cfg, batch_norm=False):
    """

    :param cfg: list of parameters:
        Indices:
        0 = number of input channels

    :param batch_norm:
    :return:
    """
    layers = []
    in_channels = cfg[0]
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [3, 16, 'M', 32, 'M', 64, 64, 'M'],
    'B': [3, 64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [3, 64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [3, 64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def SimpleCNNV1(batch_norm=False, pretrained=False, **kwargs):
    """SimpleCNNV1 model (configuration "A")

    Args:
        :param pretrained: If True, returns a model pre-trained on ImageNet
        :param batch_norm: (boolean) whether or not to use batch normalization

    """
    if pretrained:
        kwargs['init_weights'] = False
    model = SimpleCNN(make_layers(cfg['A'], batch_norm=batch_norm), **kwargs)
    if pretrained:
        raise NotImplementedError("ERROR - pretrained argument not yet implemented. Set to FALSE!")
        # model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model
