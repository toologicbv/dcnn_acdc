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
    parser.add_argument('--type_of_map', choices=['u_map', 'e_map', "None"], default="u_map",
                        help="Type of uncertainty map to use as input: u_map versus e_map")
    parser.add_argument('--use_no_map', action='store_true', help="whether or not to use the map for prediction")
    parser.add_argument('--use_random_map', action='store_true', help="whether or not to use a random map")

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
    parser.add_argument('--chkpnt_freq', type=int, default=3000, metavar='N',
                        help='Checkpoint frequency (saving model state) (default: 3000)')

    args = parser.parse_args()
    args.cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_no_map:
        args.num_input_chnls = 2
    else:
        args.num_input_chnls = 3

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


run_dict = {'type_of_map': 'e_map',
            'model': "sdvgg11_bn",
            'exper_dict': config.exper_dict_brier,
            'version': "v1",
            'root_dir': config.root_dir,
            'fold_id': 0,
            'log_dir': None,
            'use_cuda': False,
            'cuda': False,
            'epochs': 10,
            'print_freq': 25,
            'batch_size': 8,
            'lr': 1e-4,
            'quick_run': False,
            'val_freq': 100,
            'retrain_chkpnt': False,
            'retrain_exper': None,
            'chkpnt_freq': 10,
            'chkpnt': False,
            'use_no_map': False,
            'use_random_map': False,
            'num_input_chnls': 3
}


def create_def_argparser(**kwargs):

    args = argparse.Namespace()
    args.type_of_map = kwargs['type_of_map']
    args.model = kwargs['model']
    args.fold_id = kwargs['fold_id']
    args.exper_dict = kwargs['exper_dict']
    args.version = kwargs['version']
    args.root_dir = kwargs['root_dir']
    args.log_dir = kwargs['log_dir']
    args.use_cuda = kwargs['use_cuda']
    args.cuda = kwargs['cuda']
    args.epochs = kwargs['epochs']
    args.print_freq = kwargs['print_freq']
    args.batch_size = kwargs['batch_size']
    args.lr = kwargs['lr']
    args.quick_run = kwargs['quick_run']
    args.val_freq = kwargs['val_freq']
    args.retrain_chkpnt = kwargs['retrain_chkpnt']
    args.retrain_exper = kwargs['retrain_exper']
    args.chkpnt_freq = kwargs['chkpnt_freq']
    args.chkpnt = kwargs['chkpnt']
    args.use_no_map = kwargs['use_no_map']
    args.use_random_map = kwargs['use_random_map']
    args.num_input_chnls = kwargs['num_input_chnls']

    return args

