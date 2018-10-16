import torch.nn as nn
import math
import torch
from models.detector.vgg_style_model import make_layers
from common.detector.config import config_detector
from common.detector.config import OPTIMIZER_DICT


class RegionDetector(nn.Module):

    def __init__(self, model_cfg, lr=0.001, init_weights=True):
        super(RegionDetector, self).__init__()
        # get model architecture parameters
        self.sgd_optimizer_name = model_cfg['optimizer']
        self.optimizer = None
        self.lr = None
        self.nclasses = model_cfg['num_of_classes']
        self.num_input_channels = model_cfg['num_of_input_channels']
        self.use_batch_norm = model_cfg['use_batch_norm']
        self.weight_decay = model_cfg['weight_decay']
        self.backward_freq = model_cfg["backward_freq"]
        if "drop_prob" in model_cfg.keys():
            self.drop_prob = model_cfg["drop_prob"]
        else:
            self.drop_prob = 0.5
        self.lr = lr
        # The base VGG net. 3 convolutional layers (F=3x3; p=1) with BatchNorm + ReLU + MaxPooling (2x2).
        # So we have an output_stride (compression) factor of 8. e.g. 80x80 patch size results in 10x10 activation map
        self.base_model = make_layers(num_of_input_channels=self.num_input_channels, cfg=model_cfg['base'],
                                      batch_norm=self.use_batch_norm)
        self.classifier = None
        self.classifier_extra = None
        self._make_classifiers(model_cfg)
        if init_weights:
            self._initialize_weights()

        # finally prepare optimizer
        self.set_learning_rate(lr=lr)
        self.softmax_layer = nn.Softmax(dim=1)
        self.log_softmax_layer = nn.LogSoftmax(dim=1)
        self.loss_function = model_cfg["classification_loss"]
        self.use_extra_classifier = model_cfg["use_extra_classifier"]
        self._compute_num_trainable_params()
        print("INFO - RegionDetector - debug - total #parameters {}".format(self.model_total_params))

    def _make_classifiers(self, model_cfg):
        # get the last sequential module, -1=MaxPooling, -2=[Conv2d, Batchnorm, ReLU]
        # print(self.base_model)
        if self.use_batch_norm:
            last_base_layer = self.base_model[-3]
        else:
            last_base_layer = self.base_model[-2]
        # param.shape in comprehension should have shape (#output_chnls, #inputchnls, kernel_size, kernel_size)
        # hence, we take index 0 (second [0] below)
        num_of_channels_last_layer = [param.shape for param in last_base_layer.parameters()][0][0]
        print("INFO - RegionDetector - debug - num_of_channels_last_layer {}".format(num_of_channels_last_layer))
        if model_cfg["model_id"] == "rd1":
            self.classifier = nn.Sequential(
                nn.Conv2d(num_of_channels_last_layer, 128, kernel_size=1, padding=0),
                nn.ReLU(True),
                nn.Dropout(p=self.drop_prob),
                nn.Conv2d(128, self.nclasses, kernel_size=1, padding=0),
            )

        elif model_cfg["model_id"] == "rd2":
            self.classifier_extra = nn.Sequential(
                nn.Conv2d(num_of_channels_last_layer, 32, kernel_size=1, padding=0),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Dropout(p=self.drop_prob),
                nn.Conv2d(128, self.nclasses, kernel_size=1, padding=0),
            )
            # flexible feature map size, we're using fully convolutional layers
            self.classifier = nn.Sequential(
                nn.Conv2d(num_of_channels_last_layer, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                # This is actually fully convolutional part instead of fc
                nn.Conv2d(64, 128, kernel_size=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Dropout(p=self.drop_prob),
                nn.Conv2d(128, self.nclasses, kernel_size=1, padding=0),
            )

    def forward(self, x):
        x = self.base_model(x)
        if self.use_extra_classifier:
            x_extra = self.classifier_extra(x)
            out_extra = {"softmax": self.softmax_layer(x_extra),
                         "log_softmax": self.log_softmax_layer(x_extra)}
        else:
            out_extra = None
        x = self.classifier(x)
        # we return softmax probs and log-softmax. The last for computation of NLLLoss
        out = {"softmax": self.softmax_layer(x),
               "log_softmax": self.log_softmax_layer(x)}
        return out, out_extra

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

        if self.sgd_optimizer_name == "sparse_adam":
            self.optimizer = OPTIMIZER_DICT[self.sgd_optimizer_name](
                self.parameters(), lr=lr)
        else:
            self.optimizer = OPTIMIZER_DICT[self.sgd_optimizer_name](
                self.parameters(), lr=lr, weight_decay=self.weight_decay, betas=(0.9, 0.999))
        self.lr = lr

    def get_loss(self, pred_lbls, lbls, average=False):
        # The input given through a forward call is expected to contain log-probabilities of each class
        b_loss = self.loss_function(pred_lbls, lbls)
        if average:
            # assuming first dim contains batch dimension
            b_loss = torch.mean(b_loss, dim=0)
        return b_loss

    def do_forward_pass(self, x_input, y_labels, y_labels_extra=None):
        out, out_extra = self(x_input)
        batch_size, channels, _, _ = out["log_softmax"].size()
        loss = self.get_loss(out["log_softmax"].view(batch_size, channels, -1), y_labels, average=False)
        if out_extra is not None and y_labels_extra is not None:
            loss_extra = self.get_loss(out_extra["log_softmax"].view(batch_size, channels, -1), y_labels_extra,
                                       average=False)
            loss += loss_extra

        return loss, out["softmax"]

    def _compute_num_trainable_params(self):
        self.model_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def sum_grads(self, verbose=False):
        sum_grads = 0.
        for name, param in self.named_parameters():
            if param.grad is not None:
                sum_grads += torch.sum(torch.abs(param.grad.data))
            else:
                if verbose:
                    print("WARNING - No gradients for parameter >>> {} <<<".format(name))

        return sum_grads


if __name__ == '__main__':
    detector_model = RegionDetector(config_detector.detector_cfg, init_weights=True)
    device = torch.device("cuda")
    detector_model = detector_model.to(device)
    dummy_x = (torch.randn(1, 2, 72, 72)).cuda()
    out = detector_model(dummy_x)
    print("Output has shape ", out.shape)
