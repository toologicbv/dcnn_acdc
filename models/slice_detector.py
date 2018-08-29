import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from config.config import OPTIMIZER_DICT


class DegenerateSliceDetector(nn.Module):

    def __init__(self, architecture, lr=0.02, init_weights=True, verbose=False, num_of_input_chnls=3):
        super(DegenerateSliceDetector, self).__init__()
        self.verbose = verbose
        self.num_of_input_chnls = num_of_input_chnls
        self.spp_pyramid = architecture["spp_pyramid"]
        self.num_of_classes = architecture["num_of_classes"]
        self.drop_perc = architecture["drop_percentage"]
        self.weight_decay = architecture["weight_decay"]
        self.sgd_optimizer = architecture["optimizer"]
        self.fp_penalty_weight = architecture["fp_penalty_weight"]
        self.backward_freq = architecture["backward_freq"]
        self.lr = lr
        # We assume that the last conv-block has this number of channels (could be more dynamic, yes)
        self.channels_last_layer = 512
        # compute the fixed length of the vector representation after the SPP layer
        # total num of parameters = "num of channels last conv-layer" * num_of_spatial_bins
        # how to calculate number of spatial bins
        # spp_pyramid is assumed to be an array e.g. [4, 2, 1] which results in a pyramid of
        # {4x4, 2x2, 1x1} which is equal to 16+4+1 = 21 bins
        # see SPP paper for details: https://arxiv.org/abs/1406.4729
        fc_no_params = self.channels_last_layer * np.sum(np.array(self.spp_pyramid)**2)
        model_name = getattr(torchvision.models, architecture["base_model"])
        self.base_model = model_name(pretrained=False)
        if self.num_of_input_chnls != 3:
            # Begin change number of input channels in first conv layer, because we'll use 2 instead of 3
            features = list(self.base_model.features)
            del features[0]
            # add the input conv layer to the front of the list
            features = [nn.Conv2d(self.num_of_input_chnls, 64, kernel_size=3, stride=1, padding=1)] + features
            self.base_model.features = nn.Sequential(*features)
            # End change number of input channels
        # Begin exchange last MaxPool layer
        # print("Last feature layer: ", self.base_model.features[-1].__class__.__name__)
        # need to convert nn.Sequential back into list object, in order to exchange module
        features = list(self.base_model.features)
        del features[-1]
        features += [SpatialPyramidPoolLayer(spp_pyramid=self.spp_pyramid, pool_type="max_pool",
                                             verbose=self.verbose)]
        self.base_model.features = nn.Sequential(*features)
        del features
        # End exchange last MaxPool2d layer
        # Create classifier sequential module
        self.base_model.classifier = nn.Sequential(
            nn.Linear(fc_no_params, 4096),
            nn.ReLU(True),
            nn.Dropout(p=self.drop_perc),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=self.drop_perc),
            nn.Linear(4096, self.num_of_classes),
        )
        if self.verbose:
            print(self.base_model)

        self.softmax_layer = nn.Softmax(dim=1)
        self.log_softmax_layer = nn.LogSoftmax(dim=1)
        self.loss_function = nn.NLLLoss()
        self.optimizer = None
        self.set_learning_rate(lr=lr)
        self.modelName = 'Degenerate Slice Detector'
        self.features = None
        if init_weights:
            self._initialize_weights()

    def set_learning_rate(self, lr):
        if self.sgd_optimizer == "sparse_adam":
            self.optimizer = OPTIMIZER_DICT[self.sgd_optimizer](
                self.parameters(), lr=lr)
        else:
            self.optimizer = OPTIMIZER_DICT[self.sgd_optimizer](
                self.parameters(), lr=lr, weight_decay=self.weight_decay, betas=(0.9, 0.999))
        self.lr = lr

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

    def forward(self, x_in, keep_features=False):

        features = self.base_model.features(x_in)
        if keep_features:
            self.features = features
        y = self.base_model.classifier(features)
        # we return softmax probs and log-softmax. The last for computation of NLLLoss
        out = {"softmax": self.softmax_layer(y),
               "log_softmax": self.log_softmax_layer(y)}

        return out

    def get_loss(self, output, lbls, average=False):
        # The input given through a forward call is expected to contain log-probabilities of each class
        b_loss = self.loss_function(output["log_softmax"], lbls)
        # add SOFT FP and FN to the loss
        if self.fp_penalty_weight is not None:
            ones = torch.ones(lbls.size(0))
            ones = ones.cuda()
            fp_soft = (ones - lbls.float()) * output["softmax"][:, 1]
            fp_nonzero = np.count_nonzero(fp_soft.data.cpu().numpy())
            if fp_nonzero != 0:
                fp_soft = torch.sum(fp_soft) * 1/float(fp_nonzero)
            else:
                fp_soft = torch.mean(fp_soft)
            fn_soft = output["softmax"][:, 0] * lbls.float()
            fn_nonzero = np.count_nonzero(fn_soft.data.cpu().numpy())
            if fn_nonzero != 0:
                fn_soft = torch.sum(fn_soft) * 1/float(fn_nonzero)
            else:
                fn_soft = torch.mean(fn_soft)
            # print("fp_soft, fn_soft ", fp_soft.item(), fn_soft.item())
            b_loss = b_loss + fn_soft + self.fp_penalty_weight * fp_soft
        if average:
            # assuming first dim contains batch dimension
            b_loss = torch.mean(b_loss, dim=0)
        return b_loss

    def do_forward_pass(self, x_input, y_labels):
        out = self(x_input)
        loss = self.get_loss(out, y_labels, average=False)

        return loss, out["softmax"]

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
    def print_architecture_params(architecture, logger=None):
        for param_name, param_value in architecture.iteritems():
            msg_str = "INFO - model-param {}: {}".format(param_name, param_value)
            if logger is None:
                print(msg_str)
            else:
                logger.info(msg_str)


class SpatialPyramidPoolLayer(nn.Module):

    def __init__(self, spp_pyramid=[4, 2, 1], pool_type='max_pool', verbose=False):
        """
        num_of_levels: number of pooling levels
        pool_type: max- or average pooling is supported

        returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
        """
        super(SpatialPyramidPoolLayer, self).__init__()
        self.spp_pyramid = spp_pyramid
        self.pool_type = pool_type
        self.verbose = verbose

    def forward(self, x_in):

        bs, c, h, w = x_in.size()

        pooling_layers = []
        for i in range(len(self.spp_pyramid)):
            level = self.spp_pyramid[i]
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            if i == 3:
                print("KERNEL size (level={}): ({}/{})".format(level, kernel_size[0], kernel_size[1]))
            stride = (math.floor(h / level), math.floor(w / level))
            padding = (math.floor((int(kernel_size[0]) * level - h + 1) / 2),
                       math.floor((int(kernel_size[1]) * level - w + 1) / 2))
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x_in, kernel_size=kernel_size,
                                      stride=stride, padding=padding).view(bs, -1)
            else:
                tensor = F.avg_pool2d(x_in, kernel_size=kernel_size,
                                      stride=stride, padding=padding).view(bs, -1)
            pooling_layers.append(tensor)
            if self.verbose:
                print("Input [c, h, w] [{}, {}, {}] tensor.out [{}]".format(c, h, w, tensor.size(1)))

        x = torch.cat(pooling_layers, dim=-1)
        if self.verbose:
            print("SpatialPyramidPoolLayer - forward - x.size.out ", x.size())
        return x

    def extra_repr(self):
        r"""Set the extra representation of the module

        To print customized extra information, you should reimplement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        pyramid_str = "[" + ",".join([str(i) for i in self.spp_pyramid]) + "]"
        return "pool-type=" + self.pool_type + ", num-of-levels=" + pyramid_str
