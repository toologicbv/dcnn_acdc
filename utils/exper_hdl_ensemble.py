from common.dslices.helper import create_experiment
from utils.referral_results import ReferralResults


class ExperHandlerEnsemble(object):
    """
        Object that holds the ExperimentHandlers from the previous Segmentation task, NOT the slice detection
        but we need those handlers (for u-maps, e-maps, results)
    """

    def __init__(self, exper_dict):
        self.seg_exper_handlers = {}
        self.exper_dict = exper_dict
        self.patient_fold = {}
        # we will load the corresponding object from the ReferralResults because they also contain the
        # non-referral dice scores per slice per patient. We use a "dummy" referral_threshold of 0.001
        # but any other available will also do the job. Object is assigned in load_dice_without_referral method
        # below.
        self.dice_score_slices = None
        for exper_id in exper_dict.values():
            exp_handler = create_experiment(exper_id)
            exp_handler.get_testset_ids()
            fold_id = int(exp_handler.exper.run_args.fold_ids[0])
            self.seg_exper_handlers[fold_id] = exp_handler
            for patient_id in exp_handler.test_set_ids.keys():
                self.patient_fold[patient_id] = fold_id

    def get_patient_fold_id(self, patient_id):
        return self.patient_fold[patient_id]

    def prepare_handlers(self, fold_id=None, patient_ids=None, type_of_map="e_map", force_reload=False,
                         for_detector_dtaset=False):
        """

        :param fold_id: we use the fold_id to determine for which seg-handler we're loading all the necessary
                        objects we need.

                IMPORTANT: If fold_id is None, patient_ids is None, then we're loading All ACDC patients!!!!

        :param patient_ids: list of patient_ids which we want to use for our experiment. Can contain train and test
                            patient_ids from different fold ids. We use this only, when we want to limit our
                            experiment to a small subset (for proof of concept).
        :param type_of_map: e_map or u_map
        :param force_reload: if TRUE, we force a reload from disk although the objects has already been loaded
                             by the handlers.
        :param for_detector_dtaset: boolean, limit the load process to uncertainty maps and predicted labels!
        :param quick_run: boolean, reduce number of patients for debug purposes
        :return:
        """

        if fold_id is not None:
            patient_ids = []
            for p_id, f_id in self.patient_fold.iteritems():
                if f_id == fold_id:
                    patient_ids.append(p_id)
        elif patient_ids is None:
            # ALL patients from ALL FOLDS!!!
            # we DON't filter, just using ALL patient IDs
            patient_ids = self.patient_fold.keys()
        else:
            if not isinstance(patient_ids, list):
                patient_ids = [patient_ids]

        patient_ids.sort()
        for p_id in patient_ids:
            fold_id = self.patient_fold[p_id]
            exper_hdl = self.seg_exper_handlers[fold_id]
            if type_of_map == "e_map":
                mc_dropout = False
                exper_hdl.get_entropy_maps(patient_id=p_id, force_reload=force_reload)
            elif type_of_map == "u_map":
                mc_dropout = True
                exper_hdl.get_referral_maps(0.001, per_class=False, aggregate_func="max", use_raw_maps=True,
                                            patient_id=p_id, load_ref_map_blobs=False)

            else:
                raise ValueError("ERROR - type of map {} is not supported".format(type_of_map))
            # For the RegionDetector dataset, we only need the uncertainty maps and predicted labels
            if not for_detector_dtaset:
                exper_hdl.get_dt_maps(patient_id=p_id, force_reload=force_reload)
                exper_hdl.get_target_roi_maps(patient_id=p_id, force_reload=force_reload, mc_dropout=mc_dropout)
                _ = exper_hdl.get_pred_prob_maps(patient_id=p_id, mc_dropout=mc_dropout)
            exper_hdl.get_pred_labels(patient_id=p_id, mc_dropout=mc_dropout, force_reload=force_reload)

    def load_dice_without_referral(self, type_of_map="u_map", referral_threshold=0.001):
        """
        We used this method for the SLICE detector experiments.
        Not in use for anything else at the moment!

        :param type_of_map:
        :param referral_threshold:
        :return:
        """
        if type_of_map == "e_map":
            use_entropy_maps = True
        else:
            use_entropy_maps = False

        ref_result_obj = ReferralResults(self.exper_dict, [referral_threshold], print_results=False,
                                         fold=None, slice_filter_type=None, use_entropy_maps=use_entropy_maps)
        self.dice_score_slices = ref_result_obj.org_dice_slices[referral_threshold]
