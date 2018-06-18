import os
import argparse
import torch
import numpy as np
from utils.experiment import ExperimentHandler
from utils.test_handler import ACDC2017TestHandler
from utils.referral_handler import ReferralHandler
from in_out.load_data import ACDC2017DataSet
from config.config import config


ROOT_DIR = os.getenv("REPO_PATH", "/home/jorg/repo/dcnn_acdc/")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
exp_mc01_brier = {3: "20180426_14_14_57_dcnn_mc_f3p01_brier_150KE_lr2e02",
                  2: "20180426_14_14_39_dcnn_mc_f2p01_brier_150KE_lr2e02",
                  1: "20180426_14_13_46_dcnn_mc_f1p01_brier_150KE_lr2e02",
                  0: "20180418_15_02_05_dcnn_mcv1_150000E_lr2e02"}


def do_parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Generate Uncertainty Maps')

    parser.add_argument('--exper_id', default=None)
    parser.add_argument('--run_mode', choices=['outliers', 'u_maps_and_preds', 'figures_only', 'test_referrals'],
                        default="u_maps_only")
    parser.add_argument('--aggregate_func', choices=['max', 'mean'], default="max")
    parser.add_argument('--mc_samples', type=int, default=10, help="# of MC samples")
    parser.add_argument('--checkpoints', nargs='+', default=150000, help="Saved checkpoints")
    parser.add_argument('--cuda', action='store_true', default=False, help='use GPU')
    parser.add_argument('--reuse_maps', action='store_true', default=False, help='use existing U-maps')
    parser.add_argument('--save_actual_maps', action='store_true', default=False, help='save detailed u-maps')
    parser.add_argument('--do_filter_slices', action='store_true', default=False, help='filter slices for referral')
    parser.add_argument('--referral_thresholds', nargs='+', default=0.1, help="Referral thresholds used for figures.")
    parser.add_argument('--verbose', action='store_true', default=False, help='show debug messages')
    parser.add_argument('--generate_plots', action='store_true', default=False, help='generate plots for analysis')

    args = parser.parse_args()
    # convert string checkpoints to int checkpoints
    if isinstance(args.checkpoints, list):
        args.checkpoints = [int(c) for c in args.checkpoints]
    else:
        args.checkpoints = [int(args.checkpoints)]

    if isinstance(args.referral_thresholds, list):
        args.referral_thresholds = [float(c) for c in args.referral_thresholds]
    else:
        args.referral_thresholds = [int(args.referral_thresholds)]
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
    _print_flags(args)

    exper_args = exper.run_args
    info_str = "{} p={:.2f} fold={} loss={}".format(exper_args.model, exper_args.drop_prob, exper_args.fold_ids,
                                                    exper_args.loss_function)
    print("INFO - Experimental details extracted:: " + info_str)

    if args.run_mode == "outliers":
        # create dataset
        dataset = ACDC2017DataSet(exper_handler.exper.config, search_mask=config.dflt_image_name + ".mhd",
                                  fold_ids=exper_handler.exper.run_args.fold_ids, preprocess=False,
                                  debug=exper_handler.exper.run_args.quick_run)

        # IMPORTANT: current settings=we're loading VALIDATION set for outlier detection: use_train_set=False !!!
        _ = exper_handler.create_outlier_dataset(dataset, model=None, test_set=None,
                                                 checkpoint=args.checkpoints[0], mc_samples=args.mc_samples,
                                                 u_threshold=0., use_train_set=False,
                                                 do_save_u_stats=True, use_high_threshold=True,
                                                 do_save_outlier_stats=True, use_existing_umaps=args.reuse_maps,
                                                 do_analyze_slices=args.generate_plots)
    elif args.run_mode == "u_maps_and_preds":
        exper_handler.create_u_maps(model=None, checkpoints=args.checkpoints, mc_samples=args.mc_samples,
                                    u_threshold=0., referral_thresholds=args.referral_thresholds,
                                    do_save_u_stats=True, verbose=args.verbose,
                                    save_actual_maps=True, test_set=None, generate_figures=args.generate_plots,
                                    aggregate_func=args.aggregate_func,
                                    store_test_results=True)
    elif args.run_mode == "figures_only":
        args.referral_thresholds.sort()
        if args.referral_thresholds[0] <= 0.:
            raise ValueError("ERROR - argument referral_threshold needs to be greater than 0.")
        test_set = ACDC2017TestHandler.get_testset_instance(exper_handler.exper.config,
                                                            exper_handler.exper.run_args.fold_ids,
                                                            load_train=False, load_val=True,
                                                            batch_size=None, use_cuda=True)
        # in this case image_range REALLY must be a list e.g. [1, 3, 10] in order to select specific images
        exper_handler.generate_figures(test_set, image_range=None,
                                       referral_thresholds=args.referral_thresholds,
                                       patients=None)  # ["patient005", "patient022"])

    elif args.run_mode == "test_referrals":

        ref_test_set = ACDC2017TestHandler(exper_config=exper_handler.exper.config,
                                           search_mask=config.dflt_image_name + ".mhd", fold_ids=exper_args.fold_ids,
                                           debug=False, batch_size=25, use_cuda=True, load_train=False,
                                           load_val=True,
                                           use_iso_path=True)
        # refer only positive, uncertain pixels
        # ref_handler = ReferralHandler(exper_handler, test_set=ref_test_set,
        #                               referral_thresholds=args.referral_thresholds,
        #                               aggregate_func=args.aggregate_func,
        #                               verbose=False, do_save=True, num_of_images=None, pos_only=True,
        #                               patients=["patient005", "patient022"])
        # ref_handler.test(without_referral=True, verbose=True)
        # Refer all uncertain pixels
        ref_handler = ReferralHandler(exper_handler, test_set=ref_test_set,
                                      referral_thresholds=args.referral_thresholds,
                                      aggregate_func=args.aggregate_func, do_filter_slices=args.do_filter_slices,
                                      verbose=True, do_save=True, num_of_images=None, pos_only=False,
                                      patients=None)  # ["patient005", "patient062", "patient045"])
        ref_handler.test(without_referral=True, verbose=False)


if __name__ == '__main__':
    main()


"""

"""
#
# nohup python generate_uncertainty_stats.py --cuda --exper_id=20180418_15_02_05_dcnn_mcv1_150000E_lr2e02
# --checkpoints 150000 --mc_samples=10 --save_actual_maps --run_mode="u_maps_only"
# --aggregate_func=max > /home/jorg/tmp/20180418_15_02_05_dcnn_mcv1_150000E_lr2e02.log 2>&1&

"""
python generate_uncertainty_stats.py --cuda --exper_id=20180426_14_14_57_dcnn_mc_f3p01_brier_150KE_lr2e02 
 --run_mode="figures_only" --referral_thresholds 0.2 0.18
"""

"""
python generate_uncertainty_stats.py --cuda --exper_id=20180418_15_02_05_dcnn_mcv1_150000E_lr2e02 
--run_mode="u_maps_and_preds" --aggregate_func="max" --checkpoints 100000 110000 120000 130000 140000 150000

python generate_uncertainty_stats.py --cuda --exper_id=20180418_15_02_05_dcnn_mcv1_150000E_lr2e02 
--run_mode="test_referrals" --referral_thresholds 0.10 0.12 0.14 0.16 0.18 0.2 0.22 0.24 --aggregate_func="max" 

"""
