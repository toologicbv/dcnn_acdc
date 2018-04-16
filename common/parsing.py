import argparse
import torch

from config.config import config
import os


run_dict = {'cmd': 'train',
            'model': "dcnn",
            'architecture': config.get_architecture(model="dcnn"),
            'version': "v1",
            'data_dir': config.data_dir,
            'use_cuda': True,
            'epochs': 10,
            'batch_size': 5,
            'lr': 1e-3,
            'retrain': False,
            'log_dir': None,
            'chkpnt': False,
            'val_fold_id': 1,
            'val_freq': 10,
            'chkpnt_freq': 10,
            'weight_decay': 0.,
            'cycle_length': 100,
            'drop_prob': 0.5,
            'guided_train': False
}


def create_def_argparser(**kwargs):

    args = argparse.Namespace()
    args.cmd = kwargs['cmd']
    args.model = kwargs['model']
    args.architecture = kwargs['architecture']
    args.version = kwargs['version']
    args.data_dir = kwargs['data_dir']
    args.use_cuda = kwargs['use_cuda']
    args.epochs = kwargs['epochs']
    args.batch_size = kwargs['batch_size']
    args.lr = kwargs['lr']
    args.retrain = kwargs['retrain']
    args.log_dir = kwargs['log_dir']
    args.val_fold_id = kwargs['val_fold_id']
    args.print_freq = kwargs['print_freq']
    args.val_freq = kwargs['val_freq']
    args.chkpnt_freq = kwargs['chkpnt_freq']
    args.weight_decay = kwargs['weight_decay']
    args.cycle_length = kwargs['cycle_length']
    args.quick_run = kwargs['quick_run']
    args.drop_prob = kwargs['drop_prob']
    args.loss_function = kwargs['loss_function']
    args.guided_train = kwargs['guided_train']
    args.cuda = args.use_cuda and torch.cuda.is_available()
    args.chkpnt = os.path.join(config.checkpoint_path, "default.tar")
    return args


def do_parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Dilated CNN')

    parser.add_argument('--model', default="dcnn", choices=['dcnn', 'dcnn_mc', 'dcnn_mcm', 'dcnn_mc_crelu'])
    parser.add_argument('--version', type=str, default='v1')
    # in case we retrain a previous model/checkpoint this parameter specifies the experiment directory
    # relative path (w.r.t. logs/ directory e.g. "20180330_09_56_01_dcnn_mcv1_150000E_lr2e02"
    parser.add_argument('--retrain_exper', type=str, default=None)
    parser.add_argument('--retrain_chkpnt', type=int, default=None)
    parser.add_argument('--root_dir', default=config.root_dir)
    parser.add_argument('--log_dir', default=None)
    parser.add_argument('--loss_function', type=str, choices=['softdice', 'brier'], default='softdice',
                        help='Loss function for training the model (default: softdice)')
    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='use GPU')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: [0])')
    parser.add_argument('--fold_ids', type=list, default="0", metavar='N',
                        help='which fold to use for validation ([1...5]) (default: 1)')
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

    parser.add_argument('--cycle_length', type=int, default=0, metavar='N',
                        help='Cycle length for update of leardcnn_mcning rate (and snapshot ensemble) (default: 0)')

    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--chkpnt', action='store_true')
    parser.add_argument('--guided_train', action="store_true", help="train extra on slice outliers.")
    parser.add_argument('--quick_run', action='store_true')
    parser.add_argument('--chkpnt_freq', type=int, default=100, metavar='N',
                        help='Checkpoint frequency (saving model state) (default: 100)')

    args = parser.parse_args()
    args.cuda = args.use_cuda and torch.cuda.is_available()

    if args.model[:4] == "dcnn":
        pass
    else:
        raise ValueError("Parameter value of model {} is not supported".format(args.model))

    assert args.root_dir is not None
    return args

