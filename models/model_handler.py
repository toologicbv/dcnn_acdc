from models.dilated_cnn import BaseDilated2DCNN
from models.slice_detector import DegenerateSliceDetector
import torch
import shutil
import os
import torch.nn as nn



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight)
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


def load_model(exper_hdl, verbose=False):

    if exper_hdl.logger is None:
        use_logger = False
    else:
        use_logger = True

    model_architecture = exper_hdl.exper.config.get_architecture(model=exper_hdl.exper.run_args.model,
                                                                 drop_prob=exper_hdl.exper.run_args.drop_prob)
    if exper_hdl.exper.run_args.model[:4] == 'dcnn':
        message = "Creating new model BaseDilated2DCNN: {} " \
                  "with architecture {}".format(exper_hdl.exper.run_args.model,
                   model_architecture["description"])
        if use_logger:
            exper_hdl.logger.info(message)
        else:
            print(message)
        model = BaseDilated2DCNN(architecture=model_architecture,
                                 optimizer=exper_hdl.exper.config.optimizer,
                                 lr=exper_hdl.exper.run_args.lr,
                                 weight_decay=exper_hdl.exper.run_args.weight_decay,
                                 use_cuda=exper_hdl.exper.run_args.cuda,
                                 cycle_length=exper_hdl.exper.run_args.cycle_length,
                                 loss_function=exper_hdl.exper.run_args.loss_function,
                                 verbose=verbose,
                                 use_reg_loss=exper_hdl.exper.run_args.use_reg_loss)

        model.apply(weights_init)
    else:
        raise ValueError("{} name is unknown and hence cannot be created".format(exper_hdl.exper.run_args.model))

    return model


def load_slice_detector_model(exper_hdl, verbose=False):

    if exper_hdl.logger is None:
        use_logger = False
    else:
        use_logger = True

    if exper_hdl.exper.run_args.model[:5] == 'sdvgg':
        exper_hdl.exper.config.get_architecture(base_model=exper_hdl.exper.run_args.model)
        message = "Creating new model DegenerateSliceDetector"
        if use_logger:
            exper_hdl.logger.info(message)
        else:
            print(message)
        model = DegenerateSliceDetector(exper_hdl.exper.config.architecture, lr=exper_hdl.exper.run_args.lr,
                                        num_of_input_chnls=exper_hdl.exper.run_args.num_input_chnls)

    else:
        raise ValueError("{} name is unknown and hence cannot be created".format(exper_hdl.exper.run_args.model))

    exper_hdl.device = torch.device("cuda" if exper_hdl.exper.run_args.cuda else "cpu")
    # assign model to CPU or GPU if available
    model = model.to(exper_hdl.device)
    return model
