import os
from utils.experiment import ExperimentHandler
from common.dslices.config import config


def create_experiment(exper_id):

    log_dir = os.path.join(config.root_dir, "logs")
    exp_model_path = os.path.join(log_dir, exper_id)
    exper_handler = ExperimentHandler()
    exper_handler.load_experiment(exp_model_path, use_logfile=False)
    exper_handler.set_root_dir(config.root_dir)
    exper_args = exper_handler.exper.run_args
    info_str = "{} p={:.2f} fold={} loss={}".format(exper_args.model, exper_args.drop_prob, exper_args.fold_ids,
                                                    exper_args.loss_function)
    print("INFO - Experimental details extracted:: " + info_str)
    return exper_handler



