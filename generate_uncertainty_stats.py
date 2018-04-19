import os
import argparse
import torch
import numpy as np
from utils.experiment import ExperimentHandler
from utils.generate_uncertainty_maps import UncertaintyMapsGenerator
from in_out.load_data import ACDC2017DataSet
from config.config import config


ROOT_DIR = os.getenv("REPO_PATH", "/home/jorg/repo/dcnn_acdc/")
LOG_DIR = os.path.join(ROOT_DIR, "logs")


def do_parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Generate Uncertainty Maps')

    parser.add_argument('--exper_id', default=None)
    parser.add_argument('--model_name', default="MC dropout p={}")
    parser.add_argument('--mc_samples', type=int, default=5, help="# of MC samples")
    parser.add_argument('--checkpoint', type=int, default=150000, help="Saved checkpoint")
    parser.add_argument('--cuda', action='store_true', default=False, help='use GPU')
    parser.add_argument('--reuse_maps', action='store_true', default=False, help='use existing U-maps')
    parser.add_argument('--u_threshold', type=float, default=0.1, help="Threshold to filter initial u-values.")
    parser.add_argument('--verbose', action='store_true', default=False, help='show debug messages')
    args = parser.parse_args()

    return args


def _print_flags(args, logger=None):
    """
    Prints all entries in argument parser.
    """
    for key, value in vars(args).items():
        if logger is not None:
            logger.info(key + ' : ' + str(value))
        else:
            print(key + ' : ' + str(value))

    if args.cuda:
        if logger is not None:
            logger.info(" *** RUNNING ON GPU *** ")
        else:
            print(" *** RUNNING ON GPU *** ")


def main():
    args = do_parse_args()
    SEED = 4325
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    if args.cuda:
        torch.backends.cudnn.enabled = True

    np.random.seed(SEED)
    exp_model_path = os.path.join(LOG_DIR, args.exper_id)
    exper = ExperimentHandler.load_experiment(exp_model_path)
    exper_handler = ExperimentHandler(exper, use_logfile=False)
    exper_handler.set_root_dir(ROOT_DIR)
    exper_handler.set_model_name(args.model_name.format(exper.run_args.drop_prob))
    _print_flags(args)

    # create dataset
    dataset = ACDC2017DataSet(exper_handler.exper.config, search_mask=config.dflt_image_name + ".mhd",
                              fold_ids=exper_handler.exper.run_args.fold_ids, preprocess=False,
                              debug=exper_handler.exper.run_args.quick_run)

    # IMPORTANT: current settings=we're loading VALIDATION set for outlier detection: use_train_set=False !!!
    _ = exper_handler.create_outlier_dataset(dataset, model=None, test_set=None,
                                             checkpoint=args.checkpoint, mc_samples=args.mc_samples,
                                             u_threshold=args.u_threshold, use_train_set=False,
                                             do_save_u_stats=True, use_high_threshold=True,
                                             do_save_outlier_stats=True, use_existing_umaps=args.reuse_maps)


if __name__ == '__main__':
    main()
