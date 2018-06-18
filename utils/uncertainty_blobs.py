import os
import dill
import numpy as np
from tqdm import tqdm
from config.config import config
from utils.experiment import ExperimentHandler


class UncertaintyBlobStats(object):

    """
        Usage when CREATING a new object:
        referral_thresholds = [0.1, 0.12, 0.14, 0.16]
        ublob = UncertaintyBlobStats(exper_dict, referral_thresholds)
    """

    epsilon_blob_area = 10
    save_file_name = "UncertaintyBlobStats.dll"

    def __init__(self, exper_dict, referral_thresholds, filter_type="M", root_dir=None):
        self.exper_dict = exper_dict
        self.referral_thresholds = referral_thresholds
        if root_dir is None:
            self.root_dir = config.root_dir
        else:
            self.root_dir = root_dir
        # valid values for filter type are 1) M=mean 2) MD = median 3) MS = mean+std (default)
        self.filter_type = filter_type
        self.data_dir = None
        self.exper_handlers = {}
        self.median = {}
        self.mean = {}
        self.std = {}
        self.statistics_per_group = {'NOR': [], 'DCM': [], 'MINF': [], 'ARV': [], 'HCM': []}
        self.min_area_size = {}
        # first load the experiment handlers
        self._load_exper_handlers()
        # then for each referral threshold load for all patients (hence we need all exper_handlers) the
        # filtered u-maps and compute statistics for that threshold (mean, median, std)
        self._load_u_maps()

    def set_min_area_size(self, filter_type=None):

        if filter_type is not None:
            # overwrite current setting
            self.filter_type = filter_type

        for referral_threshold in self.referral_thresholds:
            if self.filter_type == "MS":
                es_filter_value = self.mean[referral_threshold][0] + self.std[referral_threshold][0]
                ed_filter_value = self.mean[referral_threshold][1] + self.std[referral_threshold][1]
            elif self.filter_type == "M":
                # mean
                es_filter_value = self.mean[referral_threshold][0]
                ed_filter_value = self.mean[referral_threshold][1]
            elif self.filter_type == "MD":
                # median
                es_filter_value = self.median[referral_threshold][0]
                ed_filter_value = self.median[referral_threshold][1]
            self.min_area_size[referral_threshold] = [es_filter_value, ed_filter_value]

    def _load_exper_handlers(self):
        for fold_id, exper_output_dir in self.exper_dict.iteritems():
            exp_path = os.path.join(self.root_dir, os.path.join(config.log_root_path, exper_output_dir))
            exper = ExperimentHandler.load_experiment(exp_path)
            exper_handler = ExperimentHandler(exper, use_logfile=False)
            exper_handler.set_root_dir(self.root_dir)
            if self.data_dir is None:
                self.data_dir = exper_handler.exper.config.data_dir
            self.exper_handlers[fold_id] = exper_handler

    def _load_u_maps(self):
        """
        we're loading the filtered maps in order to calcuate statistics for the uncertainty blob areas
        detected in the u-map slices. We use these statistics later to filter slices for referral

        :return:
        """
        patients = None
        for referral_threshold in tqdm(self.referral_thresholds):
            blob_values_es = []
            blob_values_ed = []
            statistics_per_group = {'NOR': [], 'DCM': [], 'MINF': [], 'ARV': [], 'HCM': []}
            blob_values_per_group_es = {'NOR': [], 'DCM': [], 'MINF': [], 'ARV': [], 'HCM': []}
            blob_values_per_group_ed = {'NOR': [], 'DCM': [], 'MINF': [], 'ARV': [], 'HCM': []}
            total_blobs_es = 0
            total_blobs_ed = 0
            for exper_handler in self.exper_handlers.values():
                exper_handler.get_referral_maps(referral_threshold, per_class=False, )
                if patients is None:
                    exper_handler.get_patients()
                    patients = exper_handler.patients
                filtered_u_maps = exper_handler.referral_umaps
                for patient_id, u_map in filtered_u_maps.iteritems():
                    patient_gp = patients[patient_id]
                    u_blobs = exper_handler.ref_map_blobs[patient_id]
                    slice_blobs_es = u_blobs[0]
                    slice_blobs_ed = u_blobs[1]
                    total_blobs_es += np.count_nonzero(slice_blobs_es)
                    total_blobs_ed += np.count_nonzero(slice_blobs_ed)
                    blob_values_es.append((slice_blobs_es * (slice_blobs_es > UncertaintyBlobStats.epsilon_blob_area)).sum(axis=1))
                    blob_values_ed.append((slice_blobs_ed * (slice_blobs_ed > UncertaintyBlobStats.epsilon_blob_area)).sum(axis=1))
                    blob_values_per_group_es[patient_gp].extend(blob_values_es[-1])
                    blob_values_per_group_ed[patient_gp].extend(blob_values_ed[-1])

            blob_values_es = np.concatenate(blob_values_es, axis=0)
            blob_values_ed = np.concatenate(blob_values_ed, axis=0)
            self.median[referral_threshold] = [np.median(np.array(blob_values_es)), np.median(np.array(blob_values_ed))]
            self.mean[referral_threshold] = [np.mean(np.array(blob_values_es)), np.mean(np.array(blob_values_ed))]
            self.std[referral_threshold] = [np.std(np.array(blob_values_es)), np.std(np.array(blob_values_ed))]
            # compute statistics per patient group
            for pgroup, blobs_es in blob_values_per_group_es.iteritems():
                blobs_ed = blob_values_per_group_ed[pgroup]
                blobs_es = np.array(blobs_es)
                blobs_ed = np.array(blobs_ed)
                mean_es, median_es, std_es = np.mean(blobs_es), np.median(blobs_es), np.std(blobs_es)
                mean_ed, median_ed, std_ed = np.mean(blobs_ed), np.median(blobs_ed), np.std(blobs_ed)
                statistics_per_group[pgroup] = np.array([[mean_es, median_es, std_es],
                                                              [mean_ed, median_ed, std_ed]])
            # dictionary (referral_threshold) of dictionaries (patient groups)
            self.statistics_per_group[referral_threshold] = statistics_per_group
        self.set_min_area_size()
        del blob_values_es
        del total_blobs_ed
        del patients
        del blob_values_per_group_ed
        del blob_values_per_group_es
        del statistics_per_group
        self.exper_handlers = []
        self.save()

    def save(self):
        outfile = os.path.join(self.data_dir, UncertaintyBlobStats.save_file_name)
        try:
            with open(outfile, 'wb') as f:
                dill.dump(self, f)
            print("INFO - Saved results to {}".format(outfile))
        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
            print("ERROR - can't save results to {}".format(outfile))

    @staticmethod
    def load(path_to_fold_root_dir):
        file_name = os.path.join(path_to_fold_root_dir, UncertaintyBlobStats.save_file_name)
        try:
            with open(file_name, 'rb') as f:
                ublob = dill.load(f)
        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
            print("ERROR - Can't open file {}".format(file_name))
            raise IOError

        return ublob

# blob_thresholds_median = {0.1: [500, 335], 0.12: [358, 224], 0.14: [242, 132], 0.16: [139,69]}
# blob_thresholds_mean = {0.1: [681, 549], 0.12: [537, 422], 0.14: [413, 312], 0.16: [306,229]}
# blob_thresholds_std = {0.1: [582, 639], 0.12: [530, 575], 0.14: [477, 513], 0.16: [426,453]}
