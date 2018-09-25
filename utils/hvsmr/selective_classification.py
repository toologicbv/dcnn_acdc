import dill
import os
import numpy as np
from scipy.stats import binned_statistic
from common.hvsmr.helper import detect_seg_errors
from common.hvsmr.config import config_hvsmr
import copy
from collections import OrderedDict


class SelectiveClassification(object):

    def __init__(self, exper_hdl_ensemble, num_of_thresholds=20, verbose=False, do_save=False,
                 num_of_images=None, aggregate_func="mean", patients=None, type_of_map=None,
                 force_reload=False):

        self.exper_hdl_ensemble = exper_hdl_ensemble
        self.type_of_map = type_of_map
        self.aggregate_func = aggregate_func
        # Important: how many measurements are we performing between rejection 0 voxels and all voxels
        self.num_of_thresholds = num_of_thresholds
        self.x_coverages = np.linspace(0, 1., self.num_of_thresholds)
        self.mean_risks = np.empty((0, self.num_of_thresholds))
        self.verbose = verbose
        self.force_reload = force_reload
        self.cov_risk_curv = {}
        self.patients = patients
        self.do_save = do_save
        self.loss_function = None

        # for each patient data, store the optimal C-R curve that could have been achieved
        # based on Geifman paper "Boosting uncertainty estimation..."
        self.optimal_curve = []
        self.save_output_dir = config_hvsmr.data_dir
        for fold_id, exper_hdl in self.exper_hdl_ensemble():
            self._prepare_handler(exper_hdl)

    def _get_map(self, exper_hdl, patient_id):
        if self.type_of_map == "emap":
            return exper_hdl.entropy_maps[patient_id]
        elif self.type_of_map == "umap":
            return exper_hdl.agg_umaps[patient_id]

    def _prepare_handler(self, exper_hdl):
        if exper_hdl.test_set is None or self.force_reload:
            exper_hdl.get_test_set()
        if exper_hdl.pred_labels is None or len(exper_hdl.pred_labels) == 0 or self.force_reload:
            exper_hdl.get_pred_labels()
        if exper_hdl.pred_prob_maps is None or len(exper_hdl.pred_prob_maps) or self.force_reload:
            # depending on the kind of map, we get the normal or MC-dropout predictions
            exper_hdl.get_pred_prob_maps(mc_dropout=True if self.type_of_map == "umap" else False)
        if self.type_of_map == "emap":
            exper_hdl.get_entropy_maps()
        elif self.type_of_map == "umap":
            # This doesn't work for ACDC dataset because there we generated different u-maps per threshold, stupid
            # Need to change this, if we apply the same for that dataset.
            exper_hdl.get_bayes_umaps(aggregate_func=self.aggregate_func)

        if self.loss_function is None:
            self.loss_function = exper_hdl.exper.run_args.loss_function

    def __call__(self):
        for fold_id, exper_hdl in self.exper_hdl_ensemble():
            for patient_id in exper_hdl.test_set_ids.keys():
                if self.patients is not None:
                    if patient_id not in self.patients.keys():
                        continue
                patient_coverages = []
                patient_risks = []
                cov_risk_curv = OrderedDict()
                print("Patient {}".format(patient_id))
                if patient_id not in exper_hdl.pred_labels.keys():
                    print("WARNING - No predicted labels found for this patient. Skipping")
                    continue
                _, labels = exper_hdl.test_set.get_test_pair(patient_id)
                pred_labels = exper_hdl.pred_labels[patient_id]
                map = self._get_map(exper_hdl, patient_id)
                map_max = np.max(map)
                # pred_probs has shape [num_of_classes, w, h], but we need the maximum prob per class to compute
                # the coverage
                pred_probs = np.max(exper_hdl.pred_prob_maps[patient_id], axis=0)
                w, h, num_of_slices = labels.shape
                errors_slices = np.zeros(num_of_slices)
                sel_errors_slices = np.zeros(num_of_slices)
                cov_slices = np.zeros(num_of_slices)
                optimal_error = np.zeros(num_of_slices)
                start = True
                for threshold in np.linspace(0, map_max, self.num_of_thresholds):
                    # print("INFO - Applying threshold {:.4f}".format(threshold))
                    for slice_id in np.arange(num_of_slices):
                        pred_probs_slice = pred_probs[:, :, slice_id]
                        pred_labels_slice = copy.deepcopy(pred_labels[:, :, :, slice_id])
                        gt_labels_slice = labels[:, :, slice_id]
                        map_slice = map[:, :, slice_id]
                        if start:
                            # we do this only once for each slice (i.e. for the first threshold).
                            seg_errors_slice = detect_seg_errors(gt_labels_slice, pred_labels_slice,
                                                                 is_multi_class=True)
                            num_of_seg_errors = np.count_nonzero(seg_errors_slice)
                            errors_slices[slice_id] = num_of_seg_errors
                            optimal_error[slice_id] = num_of_seg_errors * 1./(w * h)
                        # set voxels equal/above threshold to gt label
                        uncertain_voxels_idx = map_slice >= threshold
                        # Sum non-rejected probability mass
                        # cov_slices[slice_id] = np.sum(pred_probs_slice[~uncertain_voxels_idx])
                        # According to Greifman the empirical coverage is just the number of voxels not-rejected.
                        cov_slices[slice_id] = np.sum(~uncertain_voxels_idx)
                        pred_labels_slice = SelectiveClassification._set_selective_voxels(pred_labels_slice,
                                                                                          gt_labels_slice,
                                                                                          uncertain_voxels_idx)
                        # returns slice in which the incorrect labeled voxels are indicated by the gt class, using
                        # multi-class indices {1...nclass}
                        seg_errors_slice = detect_seg_errors(gt_labels_slice, pred_labels_slice,
                                                             is_multi_class=True)
                        sel_errors_slices[slice_id] = np.count_nonzero(seg_errors_slice)
                        if self.verbose:
                            print("INFO - Processing slice {}: {} : {}".format(slice_id + 1, errors_slices[slice_id],
                                                                               sel_errors_slices[slice_id]))
                    start = False
                    risk_perc = np.sum(sel_errors_slices) / float(np.sum(errors_slices))
                    coverage = np.mean(cov_slices * 1./(w * h))
                    cov_risk_curv[threshold] = np.array([coverage, risk_perc])
                    patient_coverages.append(coverage)
                    patient_risks.append(risk_perc)
                    if self.verbose:
                        print("INFO - patient {}: Risk% {:.3f} Coverage% {:.3f}".format(patient_id, risk_perc,
                                                                                        coverage))
                    self.cov_risk_curv[patient_id] = cov_risk_curv
                # End: for a patient applying all thresholds
                self.optimal_curve.append(np.mean(optimal_error))
                # prepare for interplolation, reverse ordering
                patient_coverages = np.array(patient_coverages)
                patient_risks = np.array(patient_risks)
                patient_risks = np.interp(self.x_coverages, patient_coverages, patient_risks)
                self.mean_risks = np.vstack((self.mean_risks, patient_risks)) if self.mean_risks.size else patient_risks
        self.optimal_curve = np.mean(self.optimal_curve)
        del patient_coverages
        del patient_risks
        if self.do_save:
            self.save(exper_hdl)

    def save(self):
        outfile = SelectiveClassification._create_out_filename(type_of_map=self.type_of_map,
                                                               aggregate_func=self.aggregate_func,
                                                               loss_function=self.loss_function)
        temp_ensemble = self.exper_hdl_ensemble
        self.exper_hdl_ensemble = None
        try:
            with open(outfile, 'wb') as f:
                dill.dump(self, f)
            print("INFO - Saved results to {}".format(outfile))
        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
            print("ERROR - can't save results to {}".format(outfile))

        self.exper_hdl_ensemble = temp_ensemble

    @staticmethod
    def _create_out_filename(type_of_map, aggregate_func, loss_function, env_config=None):
        if env_config is None:
            env_config = config_hvsmr
        loss_function = loss_function.replace("-", "")
        if type_of_map == "umap":
            map = type_of_map + "_" + aggregate_func
        else:
            map = type_of_map
        file_name = "sel_pred_" + map + "_" + loss_function
        outfile = os.path.join(env_config.data_dir, file_name + ".dll")
        return outfile

    @staticmethod
    def load(exper_hdl_ensemble, type_of_map, aggregate_func=None):

        # get an exper_handler that points to a directory (assuming we have a exper_dict that contains also empty
        # fold ids (in the beginning we didn't run the experiments for al the folds
        exper_hdl = next(iter(exper_hdl_ensemble.seg_exper_handlers.values()))
        loss_function = exper_hdl.exper.run_args.loss_function
        env_config = exper_hdl.exper.config
        file_to_load = SelectiveClassification._create_out_filename(type_of_map, aggregate_func, loss_function, env_config)
        try:
            with open(file_to_load, 'rb') as f:
                obj = dill.load(f)
        except IOError as (errno, strerror):
            print("ERROR - unable to load SelectiveClassification object from file {}".format(file_to_load))
            print "I/O error({0}): {1}".format(errno, strerror)
            raise
        return obj

    @staticmethod
    def _set_selective_voxels(pred_labels, gt_labels, high_uvalue_indices):
        """
        Correct all voxels indicated as highly uncertain with the ground truth label.

        :param pred_labels: [num_of_classes, w, h]
        :param gt_labels: [w, h] multiclass labels
        :param high_uvalue_indices: [w, h]
        :return: correct pred_labels w.r.t. high uncertain voxels
        """
        num_classes = pred_labels.shape[0]
        for cls in np.arange(num_classes):
            pred_labels[cls, high_uvalue_indices] = gt_labels[high_uvalue_indices] == cls

        return pred_labels
