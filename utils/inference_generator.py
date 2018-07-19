import os
import time
import glob
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
from utils.test_handler import ACDC2017TestHandler
from common.common import create_logger
from config.config import config
from utils.test_results import TestResults


class InferenceGenerator(object):

    file_suffix = "_raw_umaps.npz"

    def __init__(self, exper_handler, test_set=None, mc_samples=10, checkpoints=None, verbose=False,
                 use_logger=False, u_threshold=0., store_test_results=False, aggregate_func="max"):

        self.store_test_results = store_test_results
        self.verbose = verbose
        self.u_threshold = u_threshold
        self.mc_samples = mc_samples
        self.checkpoints = checkpoints
        if self.checkpoints is None:
            self.checkpoints = [100000, 110000, 120000, 130000, 140000, 150000]
        self.exper_handler = exper_handler
        self.generate_figures = False
        self.aggregate_func = aggregate_func
        if use_logger and self.exper_handler.logger is None:
            self.exper_handler.logger = create_logger(self.exper_handler.exper, file_handler=True)
        # looks kind of awkward, but originally wanted to use an array of folds, but finally we only process 1
        # hence the index [0]
        self.fold_id = int(self.exper_handler.exper.run_args.fold_ids[0])
        if test_set is None:
            self.info("Loading validation set of fold {} as test "
                      "set.".format(self.fold_id))
            self.test_set = ACDC2017TestHandler.get_testset_instance(self.exper_handler.exper.config,
                                                                     self.fold_id,
                                                                     load_train=False, load_val=True,
                                                                     batch_size=None, use_cuda=True)
        else:
            self.test_set = test_set

        self.num_of_images = len(self.test_set.images)
        # set path in order to save results and figures
        self.umap_output_dir = os.path.join(self.exper_handler.exper.config.root_dir,
                                            os.path.join(self.exper_handler.exper.output_dir, config.u_map_dir))
        self.pred_lbl_output_dir = os.path.join(self.exper_handler.exper.config.root_dir,
                                                os.path.join(self.exper_handler.exper.output_dir,
                                                             config.pred_lbl_dir))
        if not os.path.isdir(self.umap_output_dir):
            os.makedirs(self.umap_output_dir)
        if self.store_test_results:
            self.test_results = TestResults(exper_handler.exper, use_dropout=True, mc_samples=mc_samples)
        else:
            self.test_results = None

    def __call__(self, clean_up=False, save_actual_maps=False, generate_figures=False, patient_ids=None):

        start_time = time.time()
        message = "INFO - Starting to generate uncertainty maps (agg-func={}) " \
                  "for {} images using {} samples and u-threshold {:.2f}".format(self.aggregate_func,
                                                                                 self.num_of_images, self.mc_samples,
                                                                                 self.u_threshold)
        saved = 0
        self.generate_figures = generate_figures
        self.info(message)
        # first delete all old files in the u_map directory
        if clean_up:
            self.info("NOTE: first cleaning up previous generated uncertainty maps!")
            self.clean_up_files()
        if self.mc_samples == 1:
            sample_weights = False
        else:
            sample_weights = True

        self.exper_handler.test_results = TestResults(self.exper_handler.exper, use_dropout=sample_weights,
                                                      mc_samples=self.mc_samples)
        # make a "pointer" to the test_results object of the exper_handler because we are using both
        self.test_results = self.exper_handler.test_results
        # debugging
        if patient_ids is not None:
            image_range = []
            for p_id in patient_ids:
                image_range.append(self.test_set.trans_dict[p_id])
            print("INFO - Using patient ids {}".format(", ".join(patient_ids)))
        else:
            image_range = np.arange(len(self.test_set.img_file_names))
        # image_range = np.arange(2, len(self.test_set.img_file_names))
        for image_num in tqdm(image_range):
            patient_id = self.test_set.img_file_names[image_num]
            print("Predictions for {} with #samples={} without referral (use-mc={})".format(patient_id,
                                                                                            self.mc_samples,
                                                                                            sample_weights))
            self._test(image_num, mc_samples=self.mc_samples, sample_weights=sample_weights,
                       store_test_results=True, save_pred_labels=True,
                       store_details=False)
            if save_actual_maps:
                saved += 1
                self.save()
        if self.store_test_results:
            self.exper_handler.test_results.compute_mean_stats()
            self.exper_handler.test_results.show_results()
        duration = time.time() - start_time
        if save_actual_maps:
            self.info("INFO - Successfully saved {} maps to {}".format(saved, self.umap_output_dir))
        if self.store_test_results:
            self.exper_handler.test_results.save_results(fold_ids=self.exper_handler.exper.run_args.fold_ids,
                                                         epoch_id=self.exper_handler.exper.epoch_id)

        self.info("INFO - Total duration of generation process {:.2f} secs".format(duration))

    def save(self):
        # NOTE: we're currently saving the uncertainty values per pixels and the stats
        # b_stddev_map: [8 classes, width, height, #slices]. we insert a new axis and split dim1 into 2 x 4
        # new shape [2, 4 classes, width, height, #slices]
        stddev_map = np.concatenate((self.test_set.b_stddev_map[np.newaxis, :4, :, :, :],
                                         self.test_set.b_stddev_map[np.newaxis, 4:, :, :, :]))
        # b_bald_map shape: [2, width, height, #slices]
        try:
            # the b_image_name contains the patientxxx ID! otherwise we have on identification
            filename = self.test_set.b_image_name + InferenceGenerator.file_suffix
            filename = os.path.join(self.umap_output_dir, filename)

            np.savez(filename, stddev_slice=self.test_set.b_uncertainty_stats["stddev"],
                     bald_slice=self.test_set.b_uncertainty_stats["bald"],
                     u_threshold=np.array(self.test_set.b_uncertainty_stats["u_threshold"]),
                     u_map=stddev_map)

        except IOError:
            print("Unable to save uncertainty maps to {}".format(filename))

    def clean_up_files(self):
        file_list = glob.glob(os.path.join(self.umap_output_dir, "*" + InferenceGenerator.file_suffix))
        for f in file_list:
            os.remove(f)

    def _show_results(self, image_num, test_accuracy):
        print("Image {} - "
              " dice(RV/Myo/LV):\tES {:.2f}/{:.2f}/{:.2f}\t"
              "ED {:.2f}/{:.2f}/{:.2f}".format(str(image_num + 1) + "-" + self.test_set.b_image_name,
                                               test_accuracy[1], test_accuracy[2],
                                               test_accuracy[3], test_accuracy[5],
                                               test_accuracy[6], test_accuracy[7]))

    @staticmethod
    def compute_mean_std_per_class(image_stats, u_type="stddev_slice", measure_idx=0):
        """
        Compute mean/std/min/max for stddev_slices over all images per class (except BG)
        for which we generated the uncertainty stats.
        This is useful when we rescale the absolute uncertainty values per slice (just the sum) and scale them
        to an interval between [0,1] in order to put uncertainties for the complete batch into perspective.
        :param image_stats: dictionary (key imgID)
        :param u_type: key of dictionary
        :param measure_idx: index specifying the measure we're using for the calculation (must be in range 0-3)
        :return: tensor [2 (ES/ED), 3 (classes), 4 (mean/std/min/max)]
        """
        u_es_per_class = {0: [], 1: [], 2: [], 3: []}
        u_ed_per_class = {0: [], 1: [], 2: [], 3: []}
        num_of_classes = 4
        if isinstance(image_stats, dict):
            image_range = image_stats.keys()
        else:
            # otherwise it must be a list. All this is necessary because we use this method for a result object
            # that can be a dictionary or a list.
            image_range = np.arange(len(image_stats))

        for img_key in image_range:
            # img_key is patient iD
            u_stats = image_stats[img_key][u_type]
            # 1st measure is total uncertainty for all slices!
            for cls in np.arange(0, num_of_classes):
                # first index ES/ED, second index class, third index uncertainty measure 0-idx = total uncertainty
                es_u_stats_cls = u_stats[0][cls][measure_idx]
                ed_u_stats_cls = u_stats[1][cls][measure_idx]
                u_es_per_class[cls].extend(es_u_stats_cls)
                u_ed_per_class[cls].extend(ed_u_stats_cls)

        # shape [2 phases (es/ed), 3 classes, 4 stat-values]
        cls_stats = np.zeros((2, num_of_classes, 4))
        # stats per ES/ED phase [2, 4 (stat-measures)]
        a_stats = np.zeros((2, 4))
        es_all = []
        ed_all = []
        for cls in np.arange(num_of_classes):
            es_all.extend(u_es_per_class[cls])
            u_es_per_class[cls] = np.array(u_es_per_class[cls])
            # overall stats for ES classes
            cls_stats[0, cls] = np.array(
                [np.mean(u_es_per_class[cls]), np.std(u_es_per_class[cls]), np.min(u_es_per_class[cls]), \
                 np.max(u_es_per_class[cls])])
            # overall stats for ED classes
            ed_all.extend(u_ed_per_class[cls])
            u_ed_per_class[cls] = np.array(u_ed_per_class[cls])
            cls_stats[1, cls] = np.array(
                [np.mean(u_ed_per_class[cls]), np.std(u_ed_per_class[cls]), np.min(u_ed_per_class[cls]), \
                 np.max(u_ed_per_class[cls])])
        es_all = np.array(es_all)
        ed_all = np.array(ed_all)
        a_stats[0, :] = [np.mean(es_all), np.std(es_all), np.min(es_all), np.max(es_all)]
        a_stats[1, :] = [np.mean(ed_all), np.std(ed_all), np.min(ed_all), np.max(ed_all)]
        return a_stats, cls_stats

    def info(self, message):
        if self.exper_handler.logger is None:
            print(message)
        else:
            self.exper_handler.logger.info(message)

    def _test(self, image_num, mc_samples=1, sample_weights=False,
              store_test_results=False, save_pred_labels=False,
              store_details=False):

        self.exper_handler.test(self.checkpoints, self.test_set, image_num=image_num,
                                sample_weights=sample_weights,
                                mc_samples=mc_samples, compute_hd=True,
                                u_threshold=self.u_threshold,
                                verbose=self.verbose,
                                store_details=store_details,
                                use_seed=False, do_filter=True,
                                save_pred_labels=save_pred_labels,
                                store_test_results=store_test_results)



