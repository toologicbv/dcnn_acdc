import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from config.config import OPTIMIZER_DICT


class DegenerateSliceDetector(nn.Module):

    def __init__(self, architecture, lr=0.02, init_weights=True):
        super(DegenerateSliceDetector, self).__init__()
        self.num_of_spp_levels = architecture["num_of_spp_levels"]
        self.num_of_classes = architecture["num_of_classes"]
        self.drop_perc = architecture["drop_percentage"]
        self.weight_decay = architecture["weight_decay"]
        self.sgd_optimizer = architecture["optimizer"]
        model_name = getattr(torchvision.models, architecture["base_model"])
        self.base_model = model_name(pretrained=False)
        self.base_model.classifier = nn.Sequential(
            SpatialPyramidPoolLayer(num_of_levels=self.num_of_spp_levels, pool_type="max_pool"),
            nn.Linear(24064, 4096),
            nn.ReLU(True),
            nn.Dropout(p=self.drop_perc),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=self.drop_perc),
            nn.Linear(4096, self.num_of_classes),
        )
        self.softmax_layer = nn.Softmax(dim=1)
        self.log_softmax_layer = nn.LogSoftmax(dim=1)
        self.loss_function = nn.NLLLoss()
        if self.sgd_optimizer == "sparse_adam":
            self.optimizer = OPTIMIZER_DICT[self.sgd_optimizer](
                self.parameters(), lr=lr)
        else:
            self.optimizer = OPTIMIZER_DICT[self.sgd_optimizer](
                self.parameters(), lr=lr, weight_decay=self.weight_decay, betas=(0.9, 0.999))
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

        features = self.base_model.features(x_in)
        y = self.base_model.classifier(features)
        # we return softmax probs and log-softmax. The last for computation of NLLLoss
        out = {"softmax": self.softmax_layer(y),
               "log_softmax": self.log_softmax_layer(y)}

        return out

    def get_loss(self, pred, lbls, average=False):
        # The input given through a forward call is expected to contain log-probabilities of each class

        b_loss = self.loss_function(pred, lbls)
        if average:
            # assuming first dim contains batch dimension
            b_loss = torch.mean(b_loss, dim=0)
        return b_loss

    def do_train(self, x_in, y_lbl, batch, verbose=False):
        out = self(x_in)
        loss = self.get_loss(out["log_softmax"], y_lbl, average=False)
        batch.add_loss(loss)
        if batch.do_backward:
            self.zero_grad()
            batch.mean_loss()
            # print("do_train - loss {:.3f}".format(batch.loss.item()))
            batch.loss.backward(retain_graph=False)
            self.optimizer.step()
            if verbose:
                print("do_train - sum-grads {:.3f}".format(self.sum_grads()))
            batch.reset()
        return loss

    def sum_grads(self, verbose=False):
        sum_grads = 0.
        for name, param in self.named_parameters():
            if param.grad is not None:
                sum_grads += torch.sum(torch.abs(param.grad.data))
            else:
                if verbose:
                    print("WARNING - No gradients for parameter >>> {} <<<".format(name))

        return sum_grads


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

    def forward(self, x_in, verbose=False):

        bs, c, h, w = x_in.size()
        pooling_layers = []
        for i in range(self.num_of_levels):
            level = 2 ** i
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.floor(h / level), math.floor(w / level))
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x_in, kernel_size=kernel_size,
                                      stride=stride).view(bs, -1)
            else:
                tensor = F.avg_pool2d(x_in, kernel_size=kernel_size,
                                      stride=stride).view(bs, -1)
            if verbose:
                print("SpatialPyramidPoolLayer - forward - tensor.size.out ", tensor.size())
            pooling_layers.append(tensor)
        x = torch.cat(pooling_layers, dim=-1)
        return x
