import argparse
import torch
from common.dslices.config import config


def do_parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch VGG slice detection model')

    parser.add_argument('--model', default="sdvgg11", choices=['sdvgg11', 'sdvgg11_bn'])
    parser.add_argument('--version', type=str, default='v1')
    # in case we retrain a previous model/checkpoint this parameter specifies the experiment directory
    # relative path (w.r.t. logs/ directory e.g. "20180330_09_56_01_dcnn_mcv1_150000E_lr2e02"
    parser.add_argument('--retrain_exper', type=str, default=None)
    # if --retrain_exper is specified but without checkpoint, we'll try to load the last model in the checkpoints-dir
    parser.add_argument('--retrain_chkpnt', type=int, default=None)
    parser.add_argument('--root_dir', default=config.root_dir)
    parser.add_argument('--log_dir', default=None)

    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='use GPU')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: [0])')
    parser.add_argument('--fold_id', type=int, default=None, help="Fold ID of experiment")
    parser.add_argument('--print_freq', type=int, default=10, metavar='N',
                        help='Frequency of printing training performance (expressed in epochs) (default: 10)')
    parser.add_argument('--val_freq', type=int, default=10, metavar='N',
                        help='Frequency of validation (expressed in epochs) (default: 10)')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.0001)')

    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--drop_prob', '--dp', default=0.5, type=float,
                        metavar='x.xx', help='dropout probability (default: 0.5)')

    parser.add_argument('--chkpnt', action='store_true')
    parser.add_argument('--quick_run', action='store_true')
    parser.add_argument('--chkpnt_freq', type=int, default=100, metavar='N',
                        help='Checkpoint frequency (saving model state) (default: 100)')

    args = parser.parse_args()
    args.cuda = args.use_cuda and torch.cuda.is_available()

    if args.model[:5] == "sdvgg":
        pass
    else:
        raise ValueError("Parameter value of model {} is not supported".format(args.model))

    assert args.root_dir is not None
    return args
