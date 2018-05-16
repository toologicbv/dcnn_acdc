import os
import time
import glob
from collections import OrderedDict
from tqdm import tqdm
import dill
import copy
import numpy as np
from utils.test_handler import ACDC2017TestHandler
from common.common import create_logger, setSeed
from config.config import config
from utils.test_results import TestResults
from plotting.main_seg_results import plot_seg_erros_uncertainties


class ImageOutliers(object):

    def __init__(self, patient_id, outliers_per_img_class_es, outliers_per_img_class_ed):
        self.patient_id = patient_id
        # outliers_per_img_class_es, outliers_per_img_class_ed are dicts with key (patient_id, slice_id, phase_id)
        # and values: list of classes 1=RV, 2=MYO, 3=LV

        # find outliers for this patient img-slices. key[1]=slice_id, values=list of classes
        self.out_img_cls_es = {key[1]: value for key, value in outliers_per_img_class_es.items() if patient_id in key}
        self.out_img_cls_ed = {key[1]: value for key, value in outliers_per_img_class_ed.items() if patient_id in key}
        if not self.out_img_cls_es:
            self.out_img_cls_es = None
        if not self.out_img_cls_ed:
            self.out_img_cls_ed = None
        if self.out_img_cls_es or self.out_img_cls_ed:
            self.has_outliers = True
        else:
            self.has_outliers = False

        del outliers_per_img_class_es
        del outliers_per_img_class_ed

    def get_slice_outliers(self, slice_id, phase):
        """
        returns a list of classes that are considered outliers. 1=RV, 2=MYO, 3=LV
        :param slice_id:
        :param phase:
        :return:
        """
        if phase == 0:
            if self.out_img_cls_es is not None and slice_id in self.out_img_cls_es:
                return self.out_img_cls_es[slice_id]
            else:
                return None
        else:
            if self.out_img_cls_ed is not None and slice_id in self.out_img_cls_ed:
                return self.out_img_cls_ed[slice_id]
            else:
                return None


class OutOfDistributionSlices(object):

    def __init__(self, outliers, outliers_per_img_es, outliers_per_img_ed, outliers_per_img_class_es,
                 outliers_per_img_class_ed):
        """

        :param outliers: is a dictionary with key=patiendID/sliceID. key values are tensors with shape
        [2 (ES/ED), 4 classes] containing the normalized uncertainty values (so per phase/class)

        :param outliers_per_img_es: dictionary (key patientxxx) with list of ES outlier slices
        :param outliers_per_img_ed: dictionary (key patientxxx) with list of ED outlier slices
        """
        self.outlier_slices = outliers
        self.outliers_per_img_es = outliers_per_img_es
        self.outliers_per_img_ed = outliers_per_img_ed
        self.outliers_per_img_class_es = outliers_per_img_class_es
        self.outliers_per_img_class_ed = outliers_per_img_class_ed
        # will store the image slices (augmented versions as well) that we extract from original dataset
        # hence in order to create the OutlierSlices we need the original dataset (see method create_dataset)
        self.images = []
        self.labels = []
        self.spacings = []
        # tuple (imgID, sliceID) that we will be using in the BatchHandler class to trace the imgID/sliceIDs that
        # we processed in each batch.
        self.img_slice_ids = []
        #
        self.trans_dict = {}
        # dictionary with key "patientID" that holds a sorted list with the outlier slice IDs.
        self.outliers_per_img = OrderedDict([])
        for img_slice_id, ph_cls_ustat in outliers.iteritems():
            # ph_cls_ustat has shape [2, 4 classes]
            # this is the union of ES and ED slices per image
            self.outliers_per_img.setdefault(img_slice_id[0], []).append(img_slice_id[1])
            self.outliers_per_img[img_slice_id[0]].sort()

        self.num_of_images = len(self.outliers_per_img)
        self.num_of_slices = 0

    def __len__(self):
        return self.num_of_slices

    def save(self, outfilename=None):
        """
        Save the self.outlier_slices dictionary which contains as keys (patientxxx, sliceID), and values
        consist of tuples (phase, class, u-value).
        We also save the self.outliers_per_img which contains the same information but the dict has onlye the
        "patientID" as key and stores a sorted list of sliceIDs, more convenient when we only want to know the
        outlier slices per image.
        :param outfilename: absolute filepath. Should be responsibility of exper_handler to provide this
        :return: n.a.
        """
        try:
            with open(outfilename, 'wb') as f:
                dill.dump([self.outliers_per_img, self.outlier_slices, self.outliers_per_img_es,
                           self.outliers_per_img_ed, self.outliers_per_img_class_es,
                           self.outliers_per_img_class_ed], f)
                print("INFO - Successfully saved OutOfDistributionSlices to {}".format(outfilename))
        except IOError:
            print("ERROR - unable to save object to {}".format(outfilename))

    @staticmethod
    def load(load_filename):
        """
        We're loading the statistics w.r.t. the outliers we detected during training.
        The following objects are returned:
            outliers_per_img: a dictionary (key patientxxx) which contains a list with the sliceIDs detected
                              Note, no separation of ES and ED in this lists.

            outlier_slices: a dictionary (key patientxxx, sliceID) with numpy arrays of shape [2 (es/ed), 4 classes]
                            the array contains the normalized uncertainty values per phase and class.
                            Not that although the array has shape 4 classes, the first index (background) will
                            always be zero. We currently load these stats in the experiment handler class
        :param load_filename:
        :return: I admit, no good programming here. We don't return an object but just the separate objects, in this
        case dictionaries that we need for the analysis. Please see __init__ method for description of these
        dictionaries we are returning.
        """
        try:
            with open(load_filename, 'rb') as f:
                # see method "save" for details about the two saved dictionaries
                saved_data = dill.load(f)
                outliers_per_img = saved_data[0]
                outlier_slices = saved_data[1]
                if len(saved_data) > 2:
                    outliers_per_img_es = saved_data[2]
                    outliers_per_img_ed = saved_data[3]
                    outliers_per_img_class_es = saved_data[4]
                    outliers_per_img_class_ed = saved_data[5]
                else:
                    outliers_per_img_es = None
                    outliers_per_img_ed = None
                    outliers_per_img_class_es = None
                    outliers_per_img_class_ed = None
        except IOError as err:
            print("ERROR - unable to load object from {}".format(load_filename))
        else:
            return outliers_per_img, outlier_slices, outliers_per_img_es, outliers_per_img_ed, \
                   outliers_per_img_class_es, outliers_per_img_class_ed

    def create_dataset(self, dataset, train=True, verbose=False):
        if verbose:
            print("INFO - ")
        for img_slice_id in self.outlier_slices.keys():
            # img_slice_id is a tuple (patientID, sliceID)
            patientID = img_slice_id[0]
            sliceID = img_slice_id[1]
            imgID = dataset.trans_dict[patientID]
            search_img_slice_id = tuple((imgID, sliceID))
            # important: we assume that we're creating this for the training set
            if train:
                first_idx = dataset.train_img_slice_ids.index(search_img_slice_id)
                images = dataset.train_images
                labels = dataset.train_labels
            else:
                first_idx = dataset.val_img_slice_ids.index(search_img_slice_id)
                images = dataset.val_images
                labels = dataset.val_labels
            # note: the index method only returns the first index of the occurrence of tuple in our list of
            # images/slices. But remember that the train/val set contains 4 version of a particular img/slice
            # because we augment the dataset with rotations (0, 90...270). So we get first_idx and the next three
            for idx in np.arange(first_idx, first_idx + dataset.num_of_augmentations + 1):
                self.images.append(images[idx])
                self.labels.append(labels[idx])
                self.img_slice_ids.append(search_img_slice_id)
                # print("Search {} - corresponding match in dataset {}".format(search_img_slice_id,
                #                                                             org_img_slice_ids[idx]))
        # total number of image slices in dataset
        self.num_of_slices = len(self.images)


class ImageUncertainties(object):

    def __init__(self, uncertainty_stats):
        # dictionary of dictionaries (see load method UncertaintyMapsGenerator). key1=imageID/patientID
        # key2 = "stddev_slice"/"bald_slice"/"u_threshold".
        # we only use "stddev_slice" here and calculate normalized statistics for one measure=total slice uncertainty
        # raw_uncertainties shape [2, 4 classes, 4 measures, #slices]
        # IMPORTANT: actually we're only interested in the total uncertainty measure, dim2 index 0.
        self.raw_uncertainties = uncertainty_stats
        # stats_per_phase: tensor shape [2, 4 measures]
        self.stats_per_phase = None
        # norm_stats_per_phase: tensor shape [2, 4 measures]
        self.norm_stats_per_phase = None
        # stats_per_cls: tensor shape [2, 4 classes, 4 measures]
        self.stats_per_cls = None
        # norm_uncertainty_per_phase: dict with imgID/patienID keys and values = tensor shape [2, #slices]
        self.norm_uncertainty_per_phase = None
        # norm_uncertainty_per_cls: dictionary, keys=imageID/patientID,
        # each entry contains tensor [2 (ES / ED), 4 classes, #slices]. IMPORTANT, when we compute the normalized
        # values, we skip all the other 3 measures, and only use index 0 = uncertainty
        self.norm_uncertainty_per_cls = None
        # dictionary, keys=imageID/patientID,
        # each entry contains tensor [2 (ES / ED), 4 classes, 2 stat-measures(mean / stddev)]
        self.norm_stats_per_cls = None
        # get the image IDs/patientIDs. keys of our first dictionary
        if isinstance(uncertainty_stats, dict):
            # patient IDs
            self.image_range = uncertainty_stats.keys()
            self.u_type = "stddev_slice"
        else:
            # otherwise it must be a list. All this is necessary because we use this method for a result object
            # that can be a dictionary or a list.
            # imageID just numbers 0...N
            self.image_range = np.arange(len(uncertainty_stats))
            self.u_type = "stddev"
        # dictionary (key patientxxx, sliceID) with numpy arr [2, 4 classes] uncertainty values
        self.outliers = None
        # dictionary (key patientxxx) with list of sliceIDs per phase (two different objects)
        self.outliers_per_img_es = None
        self.outliers_per_img_ed = None
        # dictionary (key patientxxx, sliceid, phase). values is list of classes
        self.outliers_per_img_class_es = None
        self.outliers_per_img_class_ed = None

    def set_stats(self, stats_per_phase, stats_per_cls):
        """
        These are two tensors that summarize uncertainty statistics for all images (normally test batch)

        :param stats_per_phase: Computed over all images for ES and ED phase: [2, 4 measures] (mean/std/min/max)
        :param stats_per_cls: Computed over all images for ES/ED per class: [2, 4 classes, 4 measures]
        :return: N.A.
        """
        self.stats_per_phase = stats_per_phase
        self.stats_per_cls = stats_per_cls

    @staticmethod
    def create_from_testresult(test_result, u_type="stddev"):
        """
        In order to create an ImageUncertainties object we need the original uncertainty stats object
        of the TestResult object. We also need the patientxxx IDs.
        We will convert the uncertainty_stats object (list of numpy arrays) into a dictionary with keys
        (patientxxx, sliceID): numpy array [2, 4 classes, #slices] for u_type stddev

        :param test_result: TestResults object. Actually the "get_uncertainty_stats" method of this object
        returns for u_tupe=stddev a list of numpy
        :param u_type: stddev or bald. we have only implemented stddev so far
        :return:
        """
        u_stats_dict = test_result.get_uncertainty_stats()
        image_uncertainties = ImageUncertainties(u_stats_dict)
        # Important we set u_type = "stddev" because that's the key test_result object uses
        image_uncertainties.u_type = u_type
        stats_per_phase, stats_per_cls = UncertaintyMapsGenerator. \
            compute_mean_std_per_class(u_stats_dict, u_type=image_uncertainties.u_type, measure_idx=0)
        image_uncertainties.set_stats(stats_per_phase, stats_per_cls)
        image_uncertainties.gen_normalized_stats()
        return image_uncertainties

    def gen_normalized_stats(self, measure_idx=0):

        def normalize_value(arr, mina, maxa):
            return (arr - mina) / (maxa - mina)

        if self.stats_per_cls is None:
            raise ValueError("Object self.stats_per_cls is None, can't compute normalized stats")

        # First compute stats per phase (already computed over all images), we just normalize what we have
        self.norm_stats_per_phase = np.zeros_like(self.stats_per_phase)
        es_min, es_max = self.stats_per_phase[0, 2], self.stats_per_phase[0, 3]
        ed_min, ed_max = self.stats_per_phase[1, 2], self.stats_per_phase[1, 3]
        # normalize es-mean/std, note that we append 0 and 1 because these are min/max after normalization
        self.norm_stats_per_phase[0] = np.array([normalize_value(self.stats_per_phase[0, 0], es_min, es_max),
                                                 normalize_value(self.stats_per_phase[0, 1], es_min, es_max), 0, 1])
        self.norm_stats_per_phase[1] = np.array([normalize_value(self.stats_per_phase[1, 0], ed_min, ed_max),
                                                 normalize_value(self.stats_per_phase[1, 1], ed_min, ed_max), 0, 1])
        # Second compute normalized values for all images-slices
        # u_norm_stats_cls has shape [2, 4 classes, 2 measures (mean/std)] and only needs to be computed once
        # for each class. Nevertheless we place the calculation in the loop of image_range, hence we've to prevent
        # the loop from computing these statistics over and over again
        u_norm_stats_cls = np.zeros((2, 4, 2))
        norm_uncertainty_cls = {}
        self.norm_uncertainty_per_phase = {}
        first = True

        for img_idx in self.image_range:
            # uncertainty_cls [2, 4 classes, 4 measures, #slices]
            uncertainty_cls = self.raw_uncertainties[img_idx][self.u_type]
            uncertainty_phase = np.mean(uncertainty_cls, axis=1)  # average over classes
            # some slight confusion, we're only interested in the total uncertainty value which is index 0 in
            # dimension 1 [2, 4 measures, #slices]
            uncertainty_phase = uncertainty_phase[:, measure_idx, :]
            # now normalize both phases for all slices of the image
            self.norm_uncertainty_per_phase[img_idx] = np.array([normalize_value(uncertainty_phase[0], es_min, es_max),
                                                                 normalize_value(uncertainty_phase[1], ed_min, ed_max)])
            num_of_classes = uncertainty_cls.shape[1]
            # shape u_norm_uncty_cls: [2 (ES/ED), 4classes, #slices], note we had 4 measures, but here we're only
            # using the total uncertainty, index 0
            u_norm_uncty_cls = np.zeros((uncertainty_cls.shape[0], uncertainty_cls.shape[1], uncertainty_cls.shape[3]))

            for cls in np.arange(num_of_classes):
                es_u_stats_cls = uncertainty_cls[0][cls][measure_idx]  # one value for each img-slice ES and below ED
                ed_u_stats_cls = uncertainty_cls[1][cls][measure_idx]
                # 3rd index of stats_per_cls specifies stat-measure: 0=mean, 1=std, 2=min, 3=max
                es_u_min_cls, es_u_max_cls = self.stats_per_cls[0, cls, 2], self.stats_per_cls[0, cls, 3]
                ed_u_min_cls, ed_u_max_cls = self.stats_per_cls[1, cls, 2], self.stats_per_cls[1, cls, 3]
                es_u_mean, es_u_stddev = self.stats_per_cls[0, cls, 0], self.stats_per_cls[0, cls, 1]
                ed_u_mean, ed_u_stddev = self.stats_per_cls[1, cls, 0], self.stats_per_cls[1, cls, 1]
                # RESCALE TOTAL UNCERTAINTY VALUE TO INTERVAL [0,1]
                es_u_mean = (es_u_mean - es_u_min_cls) / (es_u_max_cls - es_u_min_cls)
                ed_u_mean = (ed_u_mean - ed_u_min_cls) / (ed_u_max_cls - ed_u_min_cls)
                es_u_stddev = (es_u_stddev - es_u_min_cls) / (es_u_max_cls - es_u_min_cls)
                ed_u_stddev = (ed_u_stddev - ed_u_min_cls) / (ed_u_max_cls - ed_u_min_cls)
                # we only need to store these normalized mean/std for a class once. So only for first image
                if first:
                    u_norm_stats_cls[0, cls] = np.array([es_u_mean, es_u_stddev])
                    u_norm_stats_cls[1, cls] = np.array([ed_u_mean, ed_u_stddev])
                total_uncert_es_cls = (es_u_stats_cls - es_u_min_cls) / (
                        es_u_max_cls - es_u_min_cls)
                total_uncert_ed_cls = (ed_u_stats_cls - ed_u_min_cls) / (
                        ed_u_max_cls - ed_u_min_cls)
                u_norm_uncty_cls[0, cls] = total_uncert_es_cls
                u_norm_uncty_cls[1, cls] = total_uncert_ed_cls

            norm_uncertainty_cls[img_idx] = u_norm_uncty_cls

            first = False
        self.norm_uncertainty_per_cls = norm_uncertainty_cls
        self.norm_stats_per_cls = u_norm_stats_cls

    def _detect_outliers(self, use_high_threshold=False):
        """
        we pass in a dict of tensors where each dict-entry contains the normalized uncertainty values for the slices
        of an image

        :param :

        :return:
        """

        def determine_class_outliers(measure1, mean1, std1, use_high_threshold=False):
            if use_high_threshold:
                threshold1 = mean1 + std1
            else:
                threshold1 = mean1
            idx1 = np.argwhere(measure1 > threshold1).squeeze()
            if idx1.size != 0:
                set1 = set(idx1) if idx1.size > 1 else set({int(idx1)})
            else:
                # create empty set
                set1 = set()
            return set1

        def make_dict_tuples(mydict, set_outliers, measure1, imgID, cls, phase):
            for sliceID in set_outliers:
                key = tuple((imgID, sliceID))
                # add the imgID/sliceID key to dictionary and append the values of
                # class/uncertainty/#pixels to value-list
                mydict.setdefault(key, []).append(tuple((phase, cls, measure1[sliceID])))
            return mydict

        outliers = OrderedDict()
        # norm_uncertainty_per_cls: dict with tensor [2 (ES/ED), 4 classes, #slices] per image
        for imgID in self.norm_uncertainty_per_cls.keys():
            u_stats = self.norm_uncertainty_per_cls[imgID]
            es_total_uncerty = u_stats[0]  # ES
            ed_total_uncerty = u_stats[1]  # ED
            # loop over classes...only valid for mean stddev, we're ignorning BACKGROUND class
            for cls in np.arange(1, es_total_uncerty.shape[0]):
                # get ES/ED total uncertainties (index 0) and #pixels above threshold (index 2)
                # first normalize statistics for image
                es_total_uncerty_cls = es_total_uncerty[cls]
                ed_total_uncerty_cls = ed_total_uncerty[cls]
                # index 0 = mean, 1 = stddev
                es_mean_cls, es_std_cls = self.norm_stats_per_cls[0, cls, 0], self.norm_stats_per_cls[0, cls, 1]
                ed_mean_cls, ed_std_cls = self.norm_stats_per_cls[1, cls, 0], self.norm_stats_per_cls[1, cls, 1]
                es_set_outliers = determine_class_outliers(es_total_uncerty_cls, es_mean_cls, es_std_cls,
                                                           use_high_threshold)
                ed_set_outliers = determine_class_outliers(ed_total_uncerty_cls, ed_mean_cls, ed_std_cls,
                                                           use_high_threshold)
                if es_set_outliers:
                    # make dict tuples
                    outliers = make_dict_tuples(outliers, es_set_outliers, es_total_uncerty_cls, imgID, cls, phase=0)
                if ed_set_outliers:
                    outliers = make_dict_tuples(outliers, ed_set_outliers, ed_total_uncerty_cls, imgID, cls, phase=1)
        # post-processing: transform the dict key values (a list) into a numpy array
        outlier_stats = OrderedDict()
        self.outliers_per_img_es = OrderedDict([])
        self.outliers_per_img_ed = OrderedDict([])
        self.outliers_per_img_class_es = OrderedDict([])
        self.outliers_per_img_class_ed = OrderedDict([])
        for img_slice_id, u_tuples in outliers.iteritems():
            # img_slice_id is tuple (patientxxx, sliceID)
            # u_tuples is a list of tuples. a tuple consists of (phase, cls, u-value)
            # create tensor for u-values [2 (es/ed), 4 classes]
            u_stats = np.zeros((2, 4))
            for slice_det in u_tuples:
                # slice_det[0]=phase, slice_det[1]=class
                # index phase, class
                u_stats[slice_det[0], slice_det[1]] += slice_det[2]
                # new dictionary key (patientxxx, sliceid, phase), values is list of classes
                new_key = tuple((img_slice_id[0], img_slice_id[1], slice_det[0]))
                if slice_det[0] == 0:
                    # ES
                    self.outliers_per_img_es.setdefault(img_slice_id[0], []).append(img_slice_id[1])
                    self.outliers_per_img_class_es.setdefault(new_key, []).append(slice_det[1])
                else:
                    self.outliers_per_img_ed.setdefault(img_slice_id[0], []).append(img_slice_id[1])
                    self.outliers_per_img_class_ed.setdefault(new_key, []).append(slice_det[1])
            outlier_stats[img_slice_id] = u_stats
        # remove duplicates from the slice list (happens because outliers are detected per class
        for patientid in self.outliers_per_img_es.keys():
            self.outliers_per_img_es[patientid] = list(set(self.outliers_per_img_es[patientid]))
        for patientid in self.outliers_per_img_ed.keys():
            self.outliers_per_img_ed[patientid] = list(set(self.outliers_per_img_ed[patientid]))
        self.outliers = outlier_stats

    def get_outlier_obj(self, use_high_threshold=False):
        self._detect_outliers(use_high_threshold)
        return OutOfDistributionSlices(self.outliers, self.outliers_per_img_es, self.outliers_per_img_ed,
                                       self.outliers_per_img_class_es, self.outliers_per_img_class_ed)

    @staticmethod
    def load_uncertainty_maps(exper_handler=None, full_path=None, u_maps_only=False, verbose=False):
        """
        the numpy "stats" objects contain the following 4 values:
        (1) total_uncertainty (2) #num_pixel_uncertain (3) #num_pixel_uncertain_above_tre (4) num_of_objects
        IMPORTANT: we're currently not loading the complete MAPS but only the statistics

        :param exper_handler:
        :param full_path:
        :param u_maps_only: only load the "u_map" for all the patients/images, aka the uncertainty maps
        of shape [2, 4classes, width, height, #slices]
        :return: An ImageUncertainties object with properties (IMPORTANT FOR THIS OBJECT WE ONLY CONSIDER THE
                 STDDEV uncerainty maps.
        (1) ImageUncertainties.raw_uncertainties:
            An OrderedDictionary. The primary key is the patientID e.g. "patient002". Further the dictionary
            contains the mean stddev stats [2, 4 classes, 4 (measures), #slices], NOTE per class!

        NOTE: we're currently not using the BALD statistics! But the dictionary (patientID keys) has the following
        numpy array:
        [2, 4 (measures), #slices], hence we don't store the measures for the different classes here
        """
        image_umap_stats = OrderedDict()
        # set path in order to save results and figures
        if full_path is None:
            umap_output_dir = os.path.join(exper_handler.exper.config.root_dir,
                                           os.path.join(exper_handler.exper.output_dir, config.u_map_dir))
        else:
            umap_output_dir = full_path
        search_path = os.path.join(umap_output_dir, "*" + UncertaintyMapsGenerator.file_suffix)
        c = 0
        for fname in glob.glob(search_path):
            # get base filename first and then extract patient/name/ID (filename is e.g. patient012_umap.npz)
            file_basename = os.path.splitext(os.path.basename(fname))[0]
            patientID = file_basename[:file_basename.find("_")]
            try:
                data = np.load(fname)
            except IOError:
                print("Unable to load uncertainty maps from {}".format(fname))
            image_umap_stats[patientID] = {"stddev_slice": data['stddev_slice'],
                                           "bald_slice": data['bald_slice'],
                                           "u_threshold": data["u_threshold"],
                                           "u_map": data["u_map"]}
            del data
            c += 1

        if u_maps_only:
            u_maps = OrderedDict()
            for key, value in image_umap_stats.iteritems():
                u_maps[key] = value["u_map"]
            if verbose:
                print("INFO - Loaded {} U-maps from {}.".format(c, search_path))
            return u_maps

        else:
            print("INFO - Loaded {} U-maps from {}. Creating ImageUncertainties object.".format(c, search_path))
            img_uncertainties = ImageUncertainties(image_umap_stats)
            img_uncertainties.stats_per_phase, img_uncertainties.stats_per_cls = \
                UncertaintyMapsGenerator.compute_mean_std_per_class(image_umap_stats)
            img_uncertainties.gen_normalized_stats()
            return img_uncertainties


class UncertaintyMapsGenerator(object):

    file_suffix = "_raw_umaps.npz"

    def __init__(self, exper_handler, test_set=None, mc_samples=10, model=None, checkpoint=None, verbose=False,
                 use_logger=False, u_threshold=0., store_test_results=False, generate_figures=False,
                 aggregate_func="max"):

        self.store_test_results = store_test_results
        self.verbose = verbose
        self.u_threshold = u_threshold
        self.mc_samples = mc_samples
        self.checkpoint = checkpoint
        self.exper_handler = exper_handler
        self.generate_figures = False
        self.aggregate_func = aggregate_func
        if self.aggregate_func == "max":
            self.arr_ref_thresholds = [0.16, 0.18, 0.2, 0.22, 0.24]
        else:
            self.arr_ref_thresholds = [0.12, 0.13, 0.14, 0.16]
        if use_logger and self.exper_handler.logger is None:
            self.exper_handler.logger = create_logger(self.exper_handler.exper, file_handler=True)
        # looks kind of awkward, but originally wanted to use an array of folds, but finally we only process 1
        # hence the index [0]
        self.fold_id = int(self.exper_handler.exper.run_args.fold_ids[0])
        if test_set is None:
            self.info("Loading validation set of fold {} as test "
                      "set.".format(self.exper_handler.exper.run_args.fold_ids[0]))
            self.test_set = ACDC2017TestHandler.get_testset_instance(self.exper_handler.exper.config,
                                                                     self.fold_id,
                                                                     load_train=False, load_val=True,
                                                                     batch_size=None, use_cuda=True)
        else:
            self.test_set = test_set
        self.num_of_images = len(self.test_set.images)
        if model is None:
            self.model = self._get_model()
        else:
            self.model = model
        # set path in order to save results and figures
        self.umap_output_dir = os.path.join(self.exper_handler.exper.config.root_dir,
                                            os.path.join(self.exper_handler.exper.output_dir, config.u_map_dir))
        if not os.path.isdir(self.umap_output_dir):
            os.makedirs(self.umap_output_dir)
        if self.store_test_results:
            self.test_results = TestResults(exper_handler.exper, use_dropout=True, mc_samples=mc_samples)
        else:
            self.test_results = None

    def __call__(self, do_save=True, clean_up=False, save_actual_maps=False, generate_figures=False):

        start_time = time.time()
        message = "INFO - Starting to generate uncertainty maps (agg-func={})" \
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
        # image_range = [0, 1]
        image_range = np.arange(self.num_of_images)
        self.exper_handler.test_results = TestResults(self.exper_handler.exper, use_dropout=True,
                                                      mc_samples=self.mc_samples)
        # make a "pointer" to the test_results object of the exper_handler because we are using both
        self.test_results = self.exper_handler.test_results

        for image_num in tqdm(image_range):
            # get predictions without sampling in order to save the predicted labels we need for the figures
            # get predictions with SAMPLING, we don't need t
            print("Prediction mc=1 without sampling without referral")
            self._test(image_num, mc_samples=1, sample_weights=False, referral_threshold=0.,
                       store_test_results=False, save_pred_labels=True, ref_positives_only=False,
                       use_uncertainty=False, store_details=False)
            print("Prediction mc=10 with sampling without referral")
            self._test(image_num, mc_samples=self.mc_samples, sample_weights=True, referral_threshold=0.,
                       store_test_results=True, save_pred_labels=True, ref_positives_only=False,
                       use_uncertainty=False, store_details=False)
            if do_save:
                saved += 1
                self.save(save_actual_maps=save_actual_maps)

            if len(self.arr_ref_thresholds) != 0:
                original_pred_labels = copy.deepcopy(self.test_set.b_pred_labels)

                for referral_threshold in self.arr_ref_thresholds:
                    # because we pass a single patient_id, the created filtered_u-map will be stored in
                    # exper_handler.referral_umaps object (dict). because we need it in the next step
                    patient_id = self.exper_handler.test_results.image_names[image_num]
                    self.exper_handler.create_filtered_umaps(u_threshold=referral_threshold,
                                                             patient_id=patient_id,
                                                             aggregate_func=self.aggregate_func)
                    # generate prediction with referral OF UNCERTAIN, POSITIVES ONLY
                    ref_u_map = self.exper_handler.referral_umaps[patient_id]
                    print("INFO - Results with referral of all uncertain-POSITIVES using "
                          "threshold {:.2f}".format(referral_threshold))

                    self.test_set.filter_referrals(u_maps=ref_u_map, ref_positives_only=True,
                                                   referral_threshold=referral_threshold)
                    test_accuracy_ref, test_hd_ref, seg_errors_ref = \
                        self.test_set.get_accuracy(compute_hd=True, compute_seg_errors=True, do_filter=True)
                    # save the referred labels
                    self.test_set.save_pred_labels(self.exper_handler.exper.output_dir, u_threshold=referral_threshold,
                                                   ref_positives_only=True, mc_dropout=True)
                    self._show_results(image_num, test_accuracy_ref)
                    # referral with ALL UNCERTAIN PIXELS
                    print("INFO - Results with referral of ALL uncertain pixels"
                          " using threshold {:.2f}".format(referral_threshold))
                    self.test_set.b_pred_labels = copy.deepcopy(original_pred_labels)
                    # generate prediction with referral OF ALL UNCERTAIN PIXELS
                    self.test_set.filter_referrals(u_maps=ref_u_map, ref_positives_only=False,
                                                   referral_threshold=referral_threshold)
                    test_accuracy_ref, test__hd_ref, _ = \
                        self.test_set.get_accuracy(compute_hd=True, compute_seg_errors=True, do_filter=True)
                    # save the referred labels
                    self.test_set.save_pred_labels(self.exper_handler.exper.output_dir, u_threshold=referral_threshold,
                                                   ref_positives_only=False, mc_dropout=True)
                    self._show_results(image_num, test_accuracy_ref)
                    self.test_set.b_pred_labels = copy.deepcopy(original_pred_labels)
                    if self.generate_figures:
                        self.exper_handler.generate_figures(self.test_set, image_range=[image_num],
                                                            referral_threshold=referral_threshold)

        duration = time.time() - start_time
        if do_save:
            self.info("INFO - Successfully saved {} maps to {}".format(saved, self.umap_output_dir))
        if self.store_test_results:
            self.exper_handler.test_results.save_results(fold_ids=self.exper_handler.exper.run_args.fold_ids,
                                                         epoch_id=self.exper_handler.exper.epoch_id)

        self.info("INFO - Total duration of generation process {:.2f} secs".format(duration))

    def save(self, save_actual_maps=False):
        # NOTE: we're currently not saving the actual MAPS, only the uncertainty values per pixels and the stats
        # b_stddev_map: [8 classes, width, height, #slices]. we insert a new axis and split dim1 into 2 x 4
        # new shape [2, 4 classes, width, height, #slices]
        if save_actual_maps:
            stddev_map = np.concatenate((self.test_set.b_stddev_map[np.newaxis, :4, :, :, :],
                                         self.test_set.b_stddev_map[np.newaxis, 4:, :, :, :]))
        else:
            stddev_map = None
        # b_bald_map shape: [2, width, height, #slices]
        try:
            # the b_image_name contains the patientxxx ID! otherwise we have on identification
            filename = self.test_set.b_image_name + UncertaintyMapsGenerator.file_suffix
            filename = os.path.join(self.umap_output_dir, filename)

            np.savez(filename, stddev_slice=self.test_set.b_uncertainty_stats["stddev"],
                     bald_slice=self.test_set.b_uncertainty_stats["bald"],
                     u_threshold=np.array(self.test_set.b_uncertainty_stats["u_threshold"]),
                     u_map=stddev_map)

        except IOError:
            print("Unable to save uncertainty maps to {}".format(filename))

    def clean_up_files(self):
        file_list = glob.glob(os.path.join(self.umap_output_dir, "*" + UncertaintyMapsGenerator.file_suffix))
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

    def _get_model(self):
        print("INFO - loading model {}".format(self.exper_handler.exper.model_name))
        return self.exper_handler.load_checkpoint(verbose=self.verbose,
                                                  drop_prob=self.exper_handler.exper.run_args.drop_prob,
                                                  checkpoint=self.checkpoint)

    def info(self, message):
        if self.exper_handler.logger is None:
            print(message)
        else:
            self.exper_handler.logger.info(message)

    def _test(self, image_num, mc_samples=1, sample_weights=False, referral_threshold=0.,
              store_test_results=False, save_pred_labels=False, ref_positives_only=False,
              use_uncertainty=False, store_details=False):

        self.exper_handler.test(self.model, self.test_set, image_num=image_num,
                                sample_weights=sample_weights,
                                mc_samples=mc_samples, compute_hd=True, discard_outliers=False,
                                referral_threshold=referral_threshold,
                                u_threshold=self.u_threshold,
                                verbose=self.verbose,
                                store_details=store_details,
                                use_seed=False, do_filter=True,
                                repeated_run=False,
                                use_uncertainty=use_uncertainty,
                                ref_positives_only=ref_positives_only,
                                save_pred_labels=save_pred_labels,
                                store_test_results=store_test_results)



