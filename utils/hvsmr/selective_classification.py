import numpy as np
from common.hvsmr.helper import detect_seg_errors
import copy
from collections import OrderedDict


class SelectiveClassification(object):

    def __init__(self, exper_hdl_ensemble, referral_thresholds=None, verbose=False, do_save=False,
                 num_of_images=None, aggregate_func="max", patients=None, type_of_map=None):

        self.exper_hdl_ensemble = exper_hdl_ensemble
        self.type_of_map = type_of_map
        self.aggregate_func = aggregate_func
        self.referral_thresholds = referral_thresholds
        self.verbose = verbose
        self.cov_risk_curv = {}
        for fold_id, exper_hdl in self.exper_hdl_ensemble():
            self._prepare_handler(exper_hdl)

    def _get_map(self, exper_hdl, patient_id):
        if self.type_of_map == "emap":
            return exper_hdl.entropy_maps[patient_id]
        elif self.type_of_map == "umap":
            return exper_hdl.agg_umaps[patient_id]

    def _prepare_handler(self, exper_hdl):
        if exper_hdl.test_set is None:
            exper_hdl.get_test_set()
        if exper_hdl.pred_labels is None or len(exper_hdl.pred_labels) == 0:
            exper_hdl.get_pred_labels()
        if exper_hdl.pred_prob_maps is None or len(exper_hdl.pred_prob_maps):
            # depending on the kind of map, we get the normal or MC-dropout predictions
            exper_hdl.get_pred_prob_maps(mc_dropout=True if self.type_of_map == "umap" else False)
        if self.type_of_map == "emap":
            exper_hdl.get_entropy_maps()
        elif self.type_of_map == "umap":
            exper_hdl.get_bayes_umaps(aggregate_func=self.aggregate_func)

    def __call__(self):
        for fold_id, exper_hdl in self.exper_hdl_ensemble():
            for patient_id in exper_hdl.test_set_ids.keys():
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
                start = True
                for threshold in np.linspace(0, map_max, 40):
                    print("INFO - Applying threshold {:.4f}".format(threshold))
                    for slice_id in np.arange(num_of_slices):
                        pred_probs_slice = pred_probs[:, :, slice_id]
                        pred_labels_slice = copy.deepcopy(pred_labels[:, :, :, slice_id])
                        gt_labels_slice = labels[:, :, slice_id]
                        map_slice = map[:, :, slice_id]
                        if start:
                            seg_errors_slice = detect_seg_errors(gt_labels_slice, pred_labels_slice,
                                                                 is_multi_class=True)
                            errors_slices[slice_id] = np.count_nonzero(seg_errors_slice)
                        # set voxels equal/above threshold to gt label
                        uncertain_voxels_idx = map_slice >= threshold
                        cov_slices[slice_id] = np.sum(pred_probs_slice[~uncertain_voxels_idx])
                        pred_labels_slice = SelectiveClassification._set_selective_voxels(pred_labels_slice,
                                                                                          gt_labels_slice,
                                                                                          uncertain_voxels_idx)
                        seg_errors_slice = detect_seg_errors(gt_labels_slice, pred_labels_slice,
                                                             is_multi_class=True)
                        sel_errors_slices[slice_id] = np.count_nonzero(seg_errors_slice)
                        if self.verbose:
                            print("INFO - Processing slice {}: {} : {}".format(slice_id + 1, errors_slices[slice_id],
                                                                               sel_errors_slices[slice_id]))
                    start = False
                    risk_perc = np.sum(sel_errors_slices) / float(np.sum(errors_slices))
                    coverage = np.mean(cov_slices * 1./(w * h))
                    cov_risk_curv[threshold] = tuple((coverage, risk_perc))
                    if self.verbose:
                        print("INFO - patient {}: Risk% {:.3f} Coverage% {:.3f}".format(patient_id, risk_perc,
                                                                                        coverage))
                    self.cov_risk_curv[patient_id] = cov_risk_curv

    @staticmethod
    def _set_selective_voxels(pred_labels, gt_labels, high_uvalue_indices):
        """

        :param pred_labels: [num_of_classes, w, h]
        :param gt_labels: [w, h] multiclass labels
        :param high_uvalue_indices: [w, h]
        :return: correct pred_labels w.r.t. high uncertain voxels
        """
        num_classes = pred_labels.shape[0]
        for cls in np.arange(num_classes):
            pred_labels[cls, high_uvalue_indices] = gt_labels[high_uvalue_indices] == cls

        return pred_labels
