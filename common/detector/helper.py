import os
from utils.detector.exper_handler import ExperimentHandler
from common.detector.config import config_detector


def create_experiment(exper_id, verbose=False):

    log_dir = os.path.join(config_detector.root_dir, "logs")
    exp_model_path = os.path.join(log_dir, exper_id)
    exper_handler = ExperimentHandler()
    exper_handler.load_experiment(exp_model_path, use_logfile=False)
    exper_handler.set_root_dir(config_detector.root_dir)
    exper_args = exper_handler.exper.run_args
    if verbose:
        info_str = "{} fold={} loss={}".format(exper_args.model, exper_args.fold_ids,
                                               exper_args.loss_function)
        print("INFO - Experimental details extracted:: " + info_str)
    return exper_handler


