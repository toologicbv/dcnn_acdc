import argparse
import torch

from config.config import config
import os


run_dict = {'cmd': 'train',
            'model': "dcnn",
            'version': "v1",
            'data_dir': config.data_dir,
            'use_cuda': True,
            'epochs': 1,
            'batch_size': 2,
            'lr': 1e-4,
            'retrain': False,
            'log_dir': None,
            'chkpnt': False,
            'val_fold_id': 1,
            'val_freq': 10,
            'chkpnt_freq': 10,
            'weight_decay': 0.0001
}


def create_def_argparser(**kwargs):

    args = argparse.Namespace()
    args.cmd = kwargs['cmd']
    args.model = kwargs['model']
    args.version = kwargs['version']
    args.data_dir = kwargs['data_dir']
    args.use_cuda = kwargs['use_cuda']
    args.epochs = kwargs['epochs']
    args.batch_size = kwargs['batch_size']
    args.lr = kwargs['lr']
    args.retrain = kwargs['retrain']
    args.log_dir = kwargs['log_dir']
    args.val_fold_id = kwargs['val_fold_id']
    args.val_freq = kwargs['val_freq']
    args.val_freq = kwargs['chkpnt_freq']
    args.weight_decay = kwargs['weight_decay']

    args.cuda = args.use_cuda and torch.cuda.is_available()
    args.chkpnt = os.path.join(config.checkpoint_path, "default.tar")
    return args


def do_parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Dilated CNN')
    parser.add_argument('--cmd', choices=['train', 'trainv2', 'test'], default="train")
    parser.add_argument('--model', default="dcnn")
    parser.add_argument('--version', type=str, default='v1')
    parser.add_argument('--root_dir', default=config.root_dir)
    parser.add_argument('--log_dir', default=None)
    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='use GPU')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--val_fold_id', type=int, default=1, metavar='N',
                        help='which fold to use for validation ([1...5]) (default: 1)')
    parser.add_argument('--val_freq', type=int, default=10, metavar='N',
                        help='Frequency of validation (expressed in epochs) (default: 10)')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--lr_mode', type=str, default='step')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')

    parser.add_argument('--bn_sync', action='store_true')
    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--chkpnt', action='store_true')
    parser.add_argument('--chkpnt_freq', type=int, default=100, metavar='N',
                        help='Checkpoint frequency (saving model state) (default: 100)')


    args = parser.parse_args()
    args.cuda = args.use_cuda and torch.cuda.is_available()

    assert args.root_dir is not None
    return args

