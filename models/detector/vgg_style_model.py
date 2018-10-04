import torch.nn as nn


def make_layers(num_of_input_channels, cfg, batch_norm=False):
    """

    :param cfg: list of parameters:
        Indices:
        0 = number of input channels

    :param batch_norm:
    :return:
    """
    layers = []
    in_channels = num_of_input_channels
    # 1st position specifies #input channels (to start with) hence we omit that index in the loop
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


