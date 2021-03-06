
from common.hvsmr.helper import create_experiment


class ExperHandlerEnsemble(object):
    """
        Object that holds the ExperimentHandlers from the previous Segmentation task,
        we need those handlers (for u-maps, e-maps, results)
    """

    def __init__(self, exper_dict):
        """

        :param exper_dict: dictionary with exper ids that point to log directories
        """
        self.seg_exper_handlers = {}
        self.exper_dict = exper_dict
        self.patient_fold = {}
        # we will load the corresponding object from the ReferralResults because they also contain the
        # non-referral dice scores per slice per patient. We use a "dummy" referral_threshold of 0.001
        # but any other available will also do the job. Object is assigned in load_dice_without_referral method
        # below.
        self.dice_score_slices = None
        for exper_id in exper_dict.values():
            if exper_id is not None and exper_id != "":
                exp_handler = create_experiment(exper_id)
                exp_handler.get_testset_ids()
                fold_id = int(exp_handler.exper.run_args.fold_ids[0])
                self.seg_exper_handlers[fold_id] = exp_handler
                for patient_id in exp_handler.test_set_ids.keys():
                    self.patient_fold[patient_id] = fold_id

    def get_patient_fold_id(self, patient_id):
        return self.patient_fold[patient_id]

    def __call__(self):
        for fold_id, exper_hdl in self.seg_exper_handlers.iteritems():
            yield fold_id, exper_hdl


def get_experiment_handlers():
    """
    Get slice detection handlers (experiments) for evaluation purposes

    :param root_dir:
    :param load_fold_id:
    :param seg_exper_ensemble
    :return:
    """

    exp_mc_brier = {3: "",
                      2: "",
                      1: "",
                      0: "20180919_23_03_53_dcnn_hvsmr_mc_f0p01_brier_10KE_lr1e03"}

    exp_mc_entropy = {3: "",
                      2: "",
                      1: "",
                      0: "20180920_17_04_37_dcnn_hvsmr_mc_f0p01_entrpy_10KE_lr5e04"}

    exp_mc_softdice = {3: "",
                        2: "",
                        1: "",
                        0: "20180921_08_52_02_dcnn_hvsmr_mc_f0p01_sdice_8KE_r_lr5e04"}

    exper_ensemble_brier = ExperHandlerEnsemble(exp_mc_brier)
    exper_ensemble_entropy = ExperHandlerEnsemble(exp_mc_entropy)
    exper_ensemble_softdice = ExperHandlerEnsemble(exp_mc_softdice)

    return exper_ensemble_brier, exper_ensemble_entropy, exper_ensemble_softdice