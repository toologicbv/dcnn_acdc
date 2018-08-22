import argparse
import torch
from common.dslices.config import config


def do_parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch VGG slice detection model')

    parser.add_argument('--model', default="sdvgg11_bn", choices=['sdvgg11', 'sdvgg11_bn', 'sdvgg16', 'sdvgg16_bn'])
    parser.add_argument('--version', type=str, default='v1')
    # basics
    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='use GPU')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: [0])')
    parser.add_argument('--root_dir', default=config.root_dir)
    parser.add_argument('--log_dir', default=None)

    # exper_dict: the 4 experiments we use to load u-maps/e-maps and predicted seg-masks
    parser.add_argument('--exper_dict', choices=['exper_brier', 'exper_softdice', 'exper_centropy'],
                        default="exper_brier",
                        help="Experiments that are used as input for image, predicted seg-mask and uncertainty-map")
    parser.add_argument('--fold_id', type=int, default=None, help="Fold ID of experiment")
    parser.add_argument('--type_of_map', choices=['u_map', 'e_map'], default="u_map",
                        help="Type of uncertainty map to use as input: u_map versus e_map")

    # in case we retrain a previous model/checkpoint this parameter specifies the experiment directory
    # relative path (w.r.t. logs/ directory e.g. "20180330_09_56_01_dcnn_mcv1_150000E_lr2e02"
    parser.add_argument('--retrain_exper', type=str, default=None)
    # if --retrain_exper is specified but without checkpoint, we'll try to load the last model in the checkpoints-dir
    parser.add_argument('--retrain_chkpnt', type=int, default=None)

    # print training measures and validation frequency
    parser.add_argument('--print_freq', type=int, default=10, metavar='N',
                        help='Frequency of printing training performance (expressed in epochs) (default: 10)')
    parser.add_argument('--val_freq', type=int, default=10, metavar='N',
                        help='Frequency of validation (expressed in epochs) (default: 10)')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    # Checkpoints and storing models
    parser.add_argument('--chkpnt', action='store_true')
    parser.add_argument('--quick_run', action='store_true')
    parser.add_argument('--chkpnt_freq', type=int, default=1000, metavar='N',
                        help='Checkpoint frequency (saving model state) (default: 1000)')

    args = parser.parse_args()
    args.cuda = args.use_cuda and torch.cuda.is_available()

    if args.model[:5] == "sdvgg":
        pass
    else:
        raise ValueError("Parameter value of model {} is not supported".format(args.model))

    assert args.root_dir is not None
    # set the dictionary that contains the experimental IDs for the four experiments serving as input to network
    if args.exper_dict == "exper_brier":
        args.exper_dict = config.exper_dict_brier
    elif args.exper_dict == "exper_softdice":
        args.exper_dict = config.exper_dict_softdice
    else:
        args.exper_dict = config.exper_dict_centropy

    return args
