import models.dilated_cnn
import models.hvsmr.dilated_cnn
from models.slice_detector import DegenerateSliceDetector
from models.detector.region_detector import RegionDetector
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

        if "hvsmr" in exper_hdl.exper.run_args.model:
            str_classname = "HVSMRDilated2DCNN"
            model_class = getattr(models.hvsmr.dilated_cnn, str_classname)
            # for ACDC dataset we use 2 times 4 classes, because we feed the network with 2 input channels
            # for HVSMR dataset we only use 3 classes and 1 input channel, hence use_dua_head is off
            use_dual_head = False
        else:
            str_classname = "BaseDilated2DCNN"
            model_class = getattr(models.dilated_cnn, str_classname)
            use_dual_head = True
        model = model_class(architecture=model_architecture,
                            optimizer=exper_hdl.exper.config.optimizer,
                            lr=exper_hdl.exper.run_args.lr,
                            weight_decay=exper_hdl.exper.run_args.weight_decay,
                            use_cuda=exper_hdl.exper.run_args.cuda,
                            cycle_length=exper_hdl.exper.run_args.cycle_length,
                            loss_function=exper_hdl.exper.run_args.loss_function,
                            verbose=verbose,
                            use_reg_loss=exper_hdl.exper.run_args.use_reg_loss,
                            use_dual_head=use_dual_head,
                            use_loss_attenuation=exper_hdl.exper.run_args.use_loss_attenuation)

        model.apply(weights_init)
        message = "INFO - MODEL - Creating new model {}: {} " \
                  "with architecture {}".format(str_classname, exper_hdl.exper.run_args.model,
                                                model_architecture["description"])
        if use_logger:
            exper_hdl.logger.info(message)
        else:
            print(message)
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
        model.print_architecture_params(exper_hdl.exper.config.architecture,
                                        logger=exper_hdl.logger)

    else:
        raise ValueError("{} name is unknown and hence cannot be created".format(exper_hdl.exper.run_args.model))

    exper_hdl.device = torch.device("cuda" if exper_hdl.exper.run_args.cuda else "cpu")
    # assign model to CPU or GPU if available
    model = model.to(exper_hdl.device)
    return model


def load_region_detector_model(exper_hdl, verbose=False):

    if exper_hdl.logger is None:
        use_logger = False
    else:
        use_logger = True

    if exper_hdl.exper.run_args.model[:2] == 'rd':
        exper_hdl.exper.config.get_architecture(model_name=exper_hdl.exper.run_args.model)
        message = "Creating new model RegionDetector: {}".format(exper_hdl.exper.config.architecture["description"])
        if use_logger:
            exper_hdl.logger.info(message)
            exper_hdl.logger.info("Network configuration details")
            for c_key, c_value in exper_hdl.exper.config.architecture.iteritems():
                c_msg = "{} = {}".format(c_key, str(c_value))
                exper_hdl.logger.info(c_msg)
            config_det_msg = "#max-pool = {}".format(exper_hdl.exper.config.num_of_max_pool)
            exper_hdl.logger.info(config_det_msg)
            config_det_msg = "fraction_negatives = {}".format(exper_hdl.exper.config.fraction_negatives)
            exper_hdl.logger.info(config_det_msg)
            config_det_msg = "max_grid_spacing = {}".format(exper_hdl.exper.config.max_grid_spacing)
            exper_hdl.logger.info(config_det_msg)

        else:
            print(message)
        model = RegionDetector(exper_hdl.exper.config.architecture, lr=exper_hdl.exper.run_args.lr, init_weights=True)
        # call static method when you want parameters to be printed
        if verbose:
            DegenerateSliceDetector.print_architecture_params(exper_hdl.exper.config.architecture,
                                                              logger=exper_hdl.logger)

    else:
        raise ValueError("{} is an unknown model-name and hence cannot be created".format(exper_hdl.exper.run_args.model))

    exper_hdl.device = torch.device("cuda" if exper_hdl.exper.run_args.cuda else "cpu")
    # assign model to CPU or GPU if available
    model = model.to(exper_hdl.device)
    return model