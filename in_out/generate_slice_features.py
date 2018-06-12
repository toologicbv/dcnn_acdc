import os
import numpy as np
from scipy import stats
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import generate_binary_structure
from collections import OrderedDict

from config.config import config
from utils.experiment import ExperimentHandler


class SliceFeatureGenerator(object):

    def __init__(self, exper_dict, root_dir=None, verbose=False, referral_threshold=None):

        if root_dir is None:
            self.root_dir = config.root_dir
        else:
            self.root_dir = root_dir
        self.exper_dict = exper_dict
        self.verbose = verbose
        self.num_of_bins = None
        self.num_of_patients = None
        self.total_num_of_slices = 0
        self.max_u_es = 0
        self.max_u_ed = 0
        # number of features in addition to histogram bins. we'll update this instance variable later when
        # generating the histogram stats (because we pass num_of_bins then to the method)
        # +1) entropy of hist; +2) number of uncertain voxels; +3) slice ID; +4) patient id +5) total sum uncertainties
        # +6) num of binary structures (morphology)
        self.num_of_features = 6
        # note: config.data_dir is "data/Fold/
        self.data_output_dir = os.path.join(config.root_dir, config.data_dir)
        self.u_map_hists = {}
        self.raw_u_maps = {}
        # dict of dict, key1=fold_id, key2=patient_id
        self.fold_test_patient_ids = {}
        self.patient_features = OrderedDict()
        self.referral_threshold = referral_threshold
        self._load_u_maps()

    def _load_u_maps(self):
        """
        we're loading the original unfiltered maps (per class a separate map) or the filtered maps we use
        for referral. In the latter case the referral_threshold is not None.
        In the former case we preprocess the raw maps in the "generate_features" method.

        :return:
        """
        max_u_es = 0
        max_u_ed = 0
        for fold_id, exper_output_dir in self.exper_dict.iteritems():
            exp_path = os.path.join(self.root_dir, os.path.join(config.log_root_path, exper_output_dir))
            exper = ExperimentHandler.load_experiment(exp_path)
            exper_handler = ExperimentHandler(exper, use_logfile=False)
            exper_handler.set_root_dir(self.root_dir)
            # fill dict exper_handler.test_set_ids with patient IDs of validation/test set
            exper_handler.get_testset_ids()
            self.fold_test_patient_ids[fold_id] = exper_handler.test_set_ids
            if self.referral_threshold is None:
                # get_u_maps loads raw u-maps of shape [2, 4classes, width, height, #slices]
                exper_handler.get_u_maps()
                u_maps = exper_handler.u_maps
            else:
                exper_handler.get_referral_maps(self.referral_threshold, per_class=False, )
                u_maps = exper_handler.referral_umaps
            for patient_id, u_map in u_maps.iteritems():
                self.raw_u_maps[patient_id] = u_map
                t_max_es = np.max(u_map[0].flatten())
                t_max_ed = np.max(u_map[1].flatten())
                if t_max_es > max_u_es:
                    max_u_es = t_max_es
                if t_max_ed > max_u_ed:
                    max_u_ed = t_max_ed
                self.total_num_of_slices += u_map.shape[-1]
        # order dict on patient_ids
        self.raw_u_maps = OrderedDict(sorted(self.raw_u_maps.items()))
        self.num_of_patients = len(self.raw_u_maps)
        self.max_u_es = max_u_es
        self.max_u_ed = max_u_ed
        if self.verbose:
            if self.referral_threshold is None:
                msg = "INFO - Successfully loaded {} raw u-maps (#slice={})"
            else:
                msg = "INFO - Successfully loaded {} filtered/referral u-maps (#slices={})"
            print(msg.format(self.num_of_patients, self.total_num_of_slices))
        del exper_handler
        del u_maps

    def generate_features(self, threshold=0., num_of_bins=15, normalize=True):
        self.num_of_bins = num_of_bins
        self.num_of_features += num_of_bins
        # temporary objects that store the frequency bin values for es/ed
        features_es = np.zeros((self.total_num_of_slices, self.num_of_features))
        features_ed = np.zeros((self.total_num_of_slices, self.num_of_features))

        def compute_entropy(hist_values):
            probs = hist_values * 1./np.sum(hist_values)
            return stats.entropy(probs)

        def pre_process(raw_u_map, u_threshold):
            # raw_u_map has shape [2, 4, width, height, #sllices]
            # we take the max over the four classes for ES and ED
            filtered_u_map = np.zeros((2, raw_u_map.shape[2], raw_u_map.shape[3], raw_u_map.shape[4]))
            filtered_u_map[0] = np.max(raw_u_map[0], axis=0)  # ES
            filtered_u_map[1] = np.max(raw_u_map[1], axis=0)  # ED
            filtered_u_map[filtered_u_map <= u_threshold] = 0.

            return filtered_u_map
        # we need to add 1 to num_of_bins because these will be the bin-edges
        if self.referral_threshold is not None:
            threshold = self.referral_threshold
        bin_edges_es = np.linspace(0 + threshold, self.max_u_es, num_of_bins + 1)
        bin_edges_ed = np.linspace(0 + threshold, self.max_u_ed, num_of_bins + 1)
        slice_counter = 0

        for patient_id, u_map in self.raw_u_maps.iteritems():
            # only filter and determine the maximum uncertainty per voxel if we haven't done so already, which is the
            # case if we load the referral u-maps aka filtered u-maps
            if self.referral_threshold is None:
                self.raw_u_maps[patient_id] = pre_process(u_map, u_threshold=threshold)
            num_of_slices = u_map.shape[-1]
            for slice_id in np.arange(num_of_slices):
                u_hist_es, _ = np.histogram(self.raw_u_maps[patient_id][0, :, :, slice_id].flatten(),
                                                       bins=bin_edges_es)
                binary_map_es = np.zeros(self.raw_u_maps[patient_id][0, :, :, slice_id].shape).astype(np.bool)
                mask_es = self.raw_u_maps[patient_id][0, :, :, slice_id] > threshold
                binary_map_es[mask_es] = True
                binary_structure_es = generate_binary_structure(binary_map_es.ndim, connectivity=2)
                bin_labels_es, num_of_objects_es = label(binary_map_es, binary_structure_es)
                features_es[slice_counter, :num_of_bins] = u_hist_es
                u_hist_ed, _ = np.histogram(self.raw_u_maps[patient_id][1, :, :, slice_id].flatten(),
                                                       bins=bin_edges_ed)
                binary_map_ed = np.zeros(self.raw_u_maps[patient_id][1, :, :, slice_id].shape).astype(np.bool)
                mask_ed = self.raw_u_maps[patient_id][1, :, :, slice_id] > threshold
                binary_map_ed[mask_ed] = True
                binary_structure_ed = generate_binary_structure(binary_map_ed.ndim, connectivity=2)
                bin_labels_ed, num_of_objects_ed = label(binary_map_ed, binary_structure_ed)
                features_ed[slice_counter, :num_of_bins] = u_hist_ed
                # number of uncertain pixels
                num_unc_voxels_es = np.sum(u_hist_es)
                num_unc_voxels_ed = np.sum(u_hist_ed)
                # total uncertainties
                total_unc_es = np.sum(self.raw_u_maps[patient_id][0, :, :, slice_id])
                total_unc_ed = np.sum(self.raw_u_maps[patient_id][1, :, :, slice_id])
                if num_unc_voxels_es != 0:
                    features_es[slice_counter, num_of_bins] = compute_entropy(u_hist_es)
                if num_unc_voxels_ed != 0:
                    features_ed[slice_counter, num_of_bins] = compute_entropy(u_hist_ed)
                features_es[slice_counter, num_of_bins + 1] = num_unc_voxels_es
                features_ed[slice_counter, num_of_bins + 1] = num_unc_voxels_ed
                features_es[slice_counter, num_of_bins + 2] = total_unc_es
                features_ed[slice_counter, num_of_bins + 2] = total_unc_ed
                features_es[slice_counter, num_of_bins + 3] = num_of_objects_es
                features_ed[slice_counter, num_of_bins + 3] = num_of_objects_ed
                if patient_id == "patient005":
                    print("Slice {} {} {}".format(slice_id+1, num_of_objects_es, num_of_objects_ed))
                    patch_size = []
                    for i in np.arange(1, num_of_objects_es + 1):
                        patch_size.append(np.count_nonzero(bin_labels_es == i))
                    patch_size = np.array(patch_size, dtype=np.float)
                    patch_size = patch_size * 1./num_unc_voxels_es
                    patch_size = sorted(patch_size, reverse=True)
                    print(patch_size)
                    patch_size = []
                    for i in np.arange(1, num_of_objects_ed + 1):
                        patch_size.append(np.count_nonzero(bin_labels_ed == i))
                    patch_size = np.array(patch_size, dtype=np.float)
                    patch_size = patch_size * 1. / num_unc_voxels_ed
                    patch_size = sorted(patch_size, reverse=True)
                    print(patch_size)

                features_es[slice_counter, num_of_bins + 4] = slice_id + 1
                features_ed[slice_counter, num_of_bins + 4] = slice_id + 1
                features_es[slice_counter, num_of_bins + 5] = int(patient_id.strip("patient"))
                features_ed[slice_counter, num_of_bins + 5] = int(patient_id.strip("patient"))
                slice_counter += 1

        # normalize each feature
        if normalize:
            features_es[:, :-1] = (features_es[:, :-1] - np.mean(features_es[:, :-1], axis=0)) / \
                                  np.std(features_es[:, :-1], axis=0)
            features_ed[:, :-1] = (features_es[:, :-1] - np.mean(features_ed[:, :-1], axis=0)) / \
                                  np.std(features_ed[:, :-1], axis=0)
        slice_counter = 0
        for patient_id, u_map in self.raw_u_maps.iteritems():
            num_of_slices = u_map.shape[-1]
            end_idx = slice_counter + num_of_slices
            # create numpy array of shape [2, #slices, #features]
            es_arr = np.expand_dims(features_es[slice_counter:end_idx], axis=0)
            ed_arr = np.expand_dims(features_ed[slice_counter:end_idx], axis=0)
            self.patient_features[patient_id] = np.concatenate((es_arr, ed_arr))
            if self.verbose:
                if patient_id in ["patient016", "patient100"]:
                    print(patient_id)
                    print(self.patient_features[patient_id].shape)
                    print(self.patient_features[patient_id][0, :, -2])
            slice_counter = end_idx
        print("Final value slice_counter {}".format(slice_counter))
        del features_es
        del features_ed
