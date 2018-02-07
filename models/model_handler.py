from models.dilated_cnn import BaseDilated2DCNN, DilatedCNN
import torch
import shutil
import os
import torch.nn as nn


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


def load_model(exper_hdl, simple=False):

    if exper_hdl.exper.run_args.model == 'dcnn':

        if simple:
            exper_hdl.logger.info("Creating new model DilatedCNN: {}".format(exper_hdl.exper.run_args.model))
            model = DilatedCNN(use_cuda=exper_hdl.exper.run_args.cuda)
        else:
            exper_hdl.logger.info("Creating new model BaseDilated2DCNN: {}".format(exper_hdl.exper.run_args.model))
            model = BaseDilated2DCNN(use_cuda=exper_hdl.exper.run_args.cuda)

        model.apply(weights_init)
    else:
        raise ValueError("{} name is unknown and hence cannot be created".format(exper_hdl.exper.run_args.model))

    return model