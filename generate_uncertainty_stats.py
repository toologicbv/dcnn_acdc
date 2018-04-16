import os
import argparse
import torch

from utils.experiment import ExperimentHandler
from utils.generate_uncertainty_maps import UncertaintyMapsGenerator

ROOT_DIR = os.getenv("REPO_PATH", "/home/jorg/repo/dcnn_acdc/")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
EXPERS = {"MC005_F2": "20180328_10_54_36_dcnn_mcv1_150000E_lr2e02",
          "MC005_F0": "20180330_09_56_01_dcnn_mcv1_150000E_lr2e02"}


def do_parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Generate Uncertainty Maps')

    parser.add_argument('--exper_id', default="MC005_F2")
    parser.add_argument('--model_name', default="MC dropout p={}")
    parser.add_argument('--mc_samples', type=int, default=10, help="# of MC samples")
    parser.add_argument('--cuda', action='store_true', default=False, help='use GPU')
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
    exp_model_path = os.path.join(ROOT_DIR, EXPERS[args.exper_id])
    exper = ExperimentHandler.load_experiment(exp_model_path)
    exper_handler = ExperimentHandler(exper, use_logfile=False)
    exper_handler.set_root_dir(ROOT_DIR)
    exper_handler.set_model_name(args.model_name.format(exper.run_args.drop_prob))
    _print_flags(args)
    maps_generator = UncertaintyMapsGenerator(exper_handler, verbose=args.verbose, mc_samples=args.mc_samples)
    maps_generator()


if __name__ == '__main__':
    main()
