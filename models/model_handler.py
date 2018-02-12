from models.dilated_cnn import BaseDilated2DCNN
import torch
import shutil
import os
import torch.nn as nn
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal(m.weight)
        m.bias.data.zero_()


def save_checkpoint(exper_hdl, state, is_best, prefix=None, filename='checkpoint{}.pth.tar'):
    filename = filename.format(str(state["epoch"]).zfill(5))
    if prefix is not None:
        file_name = os.path.join(exper_hdl.exper.chkpnt_dir, prefix + filename)
    else:
        file_name = os.path.join(exper_hdl.exper.chkpnt_dir, filename)

    exper_hdl.logger.info("INFO - Saving model at epoch {} to {}".format(state["epoch"], file_name))
    torch.save(state, file_name)
    if is_best:
        shutil.copyfile(file_name, file_name + '_model_best.pth.tar')


def load_model(exper_hdl):

    if exper_hdl.exper.run_args.model == 'dcnn':
        exper_hdl.logger.info("Creating new model BaseDilated2DCNN: {}".format(exper_hdl.exper.run_args.model))
        model = BaseDilated2DCNN(optimizer=exper_hdl.exper.config.optimizer, lr=exper_hdl.exper.run_args.lr,
                                 weight_decay=exper_hdl.exper.run_args.weight_decay,
                                 use_cuda=exper_hdl.exper.run_args.cuda,
                                 cycle_length=exper_hdl.exper.run_args.cycle_length)

        model.apply(weights_init)
    else:
        raise ValueError("{} name is unknown and hence cannot be created".format(exper_hdl.exper.run_args.model))

    return model