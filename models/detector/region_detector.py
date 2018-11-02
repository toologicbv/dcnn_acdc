import torch.nn as nn
import math
import copy
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
        self.use_fn_loss = model_cfg["use_fn_loss"]
        if self.use_fn_loss:
            self.fn_penalty_weight = model_cfg["fn_penalty_weight"]
            self.fp_penalty_weight = model_cfg["fp_penalty_weight"]
        else:
            self.fn_penalty_weight = 0
            self.fp_penalty_weight = 0
        self._compute_num_trainable_params()

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
                nn.Conv2d(128, 128, kernel_size=1, padding=0),
                nn.ReLU(True),
                nn.Dropout(p=self.drop_prob),
                nn.Conv2d(128, self.nclasses, kernel_size=1, padding=0),
            )

        elif model_cfg["model_id"] == "rd2":
            self.classifier_extra = nn.Sequential(
                nn.Conv2d(num_of_channels_last_layer, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                # This is actually fully convolutional part instead of fc
                nn.Conv2d(64, 128, kernel_size=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Dropout(p=self.drop_prob),
                nn.Conv2d(128, 128, kernel_size=1, padding=0),
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

        elif model_cfg["model_id"] == "rd3":
            self.classifier = nn.Sequential(
                nn.Conv2d(num_of_channels_last_layer, 128, kernel_size=1, padding=0),
                nn.ReLU(True),
                nn.Dropout(p=self.drop_prob),
                nn.Conv2d(128, 128, kernel_size=1, padding=0),
                nn.ReLU(True),
                nn.Dropout(p=self.drop_prob),
                nn.Conv2d(128, self.nclasses, kernel_size=1, padding=0),
                # LARGE MODEL
                # nn.Conv2d(num_of_channels_last_layer, 512, kernel_size=1, padding=0),
                # nn.ReLU(True),
                # nn.Dropout(p=self.drop_prob),
                # nn.Conv2d(512, 512, kernel_size=1, padding=0),
                # nn.ReLU(True),
                # nn.Dropout(p=self.drop_prob),
                # nn.Conv2d(512, self.nclasses, kernel_size=1, padding=0),
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

    def get_loss(self, log_pred_probs, lbls, average=False, pred_probs=None):
        """

        :param log_pred_probs: LOG predicted probabilities [batch_size, 2, w * h]
        :param lbls: ground truth labels [batch_size, w * h]
        :param average:
        :param pred_probs: [batch_size, 2, w * h]
        :return: torch scalar
        """
        # print("INFO - get_loss - log_pred_probs.shape, lbls.shape ", log_pred_probs.shape, lbls.shape)
        # NOTE: this was a tryout (not working) for hard negative mining
        # batch_loss_indices = RegionDetector.hard_negative_mining(pred_probs, lbls)
        # b_loss_idx_preds = batch_loss_indices.unsqueeze(1).expand_as(log_pred_probs)
        # The input given through a forward call is expected to contain log-probabilities of each class
        b_loss = self.loss_function(log_pred_probs, lbls)

        if self.use_fn_loss:
            # pred_probs last 2 dimensions need to be merged because lbls has shape [batch_size, w, h ]
            pred_probs = pred_probs.view(pred_probs.size(0), 2, -1)
            fn_soft = pred_probs[:, 0] * lbls.float()
            fn_nonzero = torch.nonzero(fn_soft.data).size(0)
            if fn_nonzero != 0:
                t = torch.sum(fn_soft)
                fn_soft = torch.sum(fn_soft) * 1 / float(fn_nonzero)
                # print("EXTRA INFO b_loss {:.3f} + loss={:.3f} - "
                #      "fn_soft {:.3f} fn_nonzero {}".format(b_loss.item(), fn_soft.item(), t.item(), fn_nonzero))
            else:
                fn_soft = torch.mean(fn_soft)
            # same for false positive
            ones = torch.ones(lbls.size()).cuda()
            fp_soft = (ones - lbls.float()) * pred_probs[:, 1]
            fp_nonzero = torch.nonzero(fp_soft).size(0)
            if fp_nonzero != 0:
                fp_soft = torch.sum(fp_soft) * 1 / float(fp_nonzero)
            else:
                fp_soft = torch.mean(fp_soft)
            b_loss = b_loss + self.fn_penalty_weight * fn_soft + self.fp_penalty_weight * fp_soft
        if average:
            # assuming first dim contains batch dimension
            b_loss = torch.mean(b_loss, dim=0)
        return b_loss

    def do_forward_pass(self, x_input, y_labels, y_labels_extra=None):
        out, out_extra = self(x_input)
        batch_size, channels, _, _ = out["log_softmax"].size()
        loss = self.get_loss(out["log_softmax"].view(batch_size, channels, -1), y_labels, average=False,
                             pred_probs=out["softmax"])
        if out_extra is not None and y_labels_extra is not None:
            loss_extra = self.get_loss(out_extra["log_softmax"].view(batch_size, channels, -1), y_labels_extra,
                                       average=False, pred_probs=out_extra["softmax"])
            loss += loss_extra

        return loss, [out["softmax"], None if out_extra is None else out_extra["softmax"]]

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

    @staticmethod
    def hard_negative_mining(pred_probs, gt_targets):
        """Return negative indices that is 3x the number as positive indices.
        Args:
          pred_probs: (tensor) predicted softmax probabilities [batch_size, 2, w * h].
          gt_targets: (tensor) ground truth labels [batch_size, w * h]
        Return:
          (tensor) indices of false negatives with high softmax (high confidence)
        """
        # we only use the softmax responses indicating this is a positive grid.

        soft_resp = (pred_probs[:, 1]).clone()
        soft_resp = soft_resp.view(soft_resp.size(0), -1)
        positve_idx = gt_targets != 0
        batch_size, num_of_grids = positve_idx.size()

        # set pos grids = 0, the rest are negative examples
        soft_resp[positve_idx] = 0
        soft_resp = soft_resp.view(batch_size, -1)  # [batch_size, g * h]

        _, idx = soft_resp.sort(1, descending=True)  # sort by neg softmax-response
        _, rank = idx.sort(1)  # [batch_size, g * h]

        num_pos = positve_idx.long().sum(1)  # [batch_size, 1]
        num_pos = num_pos.unsqueeze(1)
        num_neg = torch.clamp(3 * num_pos, max=num_of_grids - 1)  # [batch_size, 1]
        negative_idx = rank < num_neg  # [batch_size, g * h]
        return negative_idx


if __name__ == '__main__':
    detector_model = RegionDetector(config_detector.detector_cfg, init_weights=True)
    device = torch.device("cuda")
    detector_model = detector_model.to(device)
    dummy_x = (torch.randn(1, 2, 72, 72)).cuda()
    out = detector_model(dummy_x)
    print("Output has shape ", out.shape)
