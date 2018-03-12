import numpy as np
from scipy.stats import gaussian_kde
import os
import dill
import sys
from collections import OrderedDict
from config.config import config
if "/home/jogi/.local/lib/python2.7/site-packages" in sys.path:
    sys.path.remove("/home/jogi/.local/lib/python2.7/site-packages")
import matplotlib.pyplot as plt
from matplotlib import cm


class TestResults(object):

    def __init__(self):
        """
            numpy arrays in self.pred_probs have shape [slices, classes, width, height]
                            self.pred_labels [classes, width, height, slices]
                            self.images [classes, width, height, slices]
                            self.pred_probs: [mc_samples, classes, width, height, slices]
        """
        self.images = []
        self.labels = []
        self.pred_labels = []
        self.mc_pred_probs = []
        self.uncertainty_maps = []
        self.test_accuracy = []
        self.test_hd = []
        # for each image we are using during testing we append ONE LIST, which contains for each image slice
        # and ordered dictionary with the following keys a) es_err b) es_corr c) ed_err d) ed_corr
        # this object is used for the detailed analysis of the uncertainties per image-slice, distinguishing
        # the correct and wrongs labeled pixels for the two cardiac phase ES and ED.
        self.image_probs_categorized = []

    def add_results(self, batch_image, batch_labels, pred_labels, b_predictions, uncertainty_map,
                    test_accuracy, test_hd):
        self.images.append(batch_image)
        self.labels.append(batch_labels)
        self.pred_labels.append(pred_labels)
        self.mc_pred_probs.append(b_predictions)
        self.uncertainty_maps.append(uncertainty_map)
        self.test_accuracy.append(test_accuracy)
        self.test_hd.append(test_hd)

    def split_probs_per_slice_class(self, image_num=0):
        """

        For a detailed analysis of the likelihoods p(y* | x*, theta) we split the predicted probabilities
        into (1) ED and ES (2) slices (3) segmentation classes.

        :param image_num: index of the test image we want to process
        :return:
        """
        # Reset object
        if len(self.image_probs_categorized) != 0:
            del self.image_probs_categorized[image_num]

        pred_labels = self.pred_labels[image_num]
        labels = self.labels[image_num]
        # this tensors contains all probs for the samples, but we will work with the means
        mc_pred_probs = self.mc_pred_probs[image_num]
        mean_pred_probs = np.mean(mc_pred_probs, axis=0)
        print(mean_pred_probs.shape)
        std_pred_probs = self.uncertainty_maps[image_num]
        num_slices = labels.shape[3]
        num_classes = labels.shape[0]
        half_classes = num_classes / 2
        probs_per_img_slice = []
        for slice in np.arange(num_slices):
            # object that holds probabilities per image slice
            probs_per_cls = {"es_err_p": OrderedDict(), "es_cor_p": OrderedDict(), "ed_err_p": OrderedDict(),
                             "ed_cor_p": OrderedDict(), "es_err_std": OrderedDict(), "es_cor_std": OrderedDict(),
                             "ed_err_std": OrderedDict(), "ed_cor_std": OrderedDict()}
            for cls in np.arange(num_classes):
                s_labels = labels[cls, :, :, slice].flatten()
                s_pred_labels = pred_labels[cls, :, :, slice].flatten()
                s_slice_cls_probs = mc_pred_probs[:, cls, :, :, slice]
                # we don't need to calculate stddev again, we did that already in experiment.test() method
                s_slice_cls_probs_std = std_pred_probs[cls, :, :, slice].flatten()
                s_slice_cls_probs = np.reshape(s_slice_cls_probs, (s_slice_cls_probs.shape[0],
                                                                   s_slice_cls_probs.shape[1] *
                                                                   s_slice_cls_probs.shape[2]))

                # get the indices for the errors and the correct classified pixels in this slice/per class
                errors = s_labels != s_pred_labels
                correct = np.invert(errors)
                err_probs = s_slice_cls_probs.T[errors].flatten()
                err_std = s_slice_cls_probs_std[errors].flatten()
                pos_probs = s_slice_cls_probs.T[correct].flatten()
                pos_std = s_slice_cls_probs_std[correct].flatten()
                if cls < half_classes:
                    if cls not in probs_per_cls["es_err_p"]:
                        probs_per_cls["es_err_p"][cls] = list(err_probs)
                        probs_per_cls["es_err_std"][cls] = list(err_std)
                    else:
                        probs_per_cls["es_err_p"][cls].extend(list(err_probs))
                        probs_per_cls["es_err_std"][cls].extend(list(err_std))
                    if cls not in probs_per_cls["es_cor_p"]:
                        probs_per_cls["es_cor_p"][cls] = list(pos_probs)
                        probs_per_cls["es_cor_std"][cls] = list(pos_std)
                    else:
                        probs_per_cls["es_cor_p"][cls].extend(list(pos_probs))
                        probs_per_cls["es_cor_std"][cls].extend(list(pos_std))
                else:
                    # print("ED: correct/error {}/{}".format(pos_probs.shape[0], err_probs.shape[0]))
                    if cls - half_classes not in probs_per_cls["ed_err_p"]:
                        probs_per_cls["ed_err_p"][cls - half_classes] = list(err_probs)
                        probs_per_cls["ed_err_std"][cls - half_classes] = list(err_std)
                    else:
                        probs_per_cls["ed_err_p"][cls - half_classes].extend(list(err_probs))
                        probs_per_cls["ed_err_std"][cls - half_classes].extend(list(err_std))
                    if cls - half_classes not in probs_per_cls["ed_cor_p"]:
                        probs_per_cls["ed_cor_p"][cls - half_classes] = list(pos_probs)
                        probs_per_cls["ed_cor_std"][cls - half_classes] = list(pos_std)
                    else:
                        probs_per_cls["ed_cor_p"][cls - half_classes].extend(list(pos_probs))
                        probs_per_cls["ed_cor_std"][cls - half_classes].extend(list(pos_std))
            # finally store probs_per_cls for this slice
            probs_per_img_slice.append(probs_per_cls)

        self.image_probs_categorized.insert(image_num, probs_per_img_slice)

    def visualize_pred_prob_dists(self, image_num=0, width=12, height=12, slice_range=None):

        column_lbls = ["bg", "RV", "MYO", "LV"]
        image_probs = self.image_probs_categorized[image_num]
        image = self.images[image_num]
        label = self.labels[image_num]
        img_pred_labels = self.pred_labels[image_num]
        num_of_classes = label.shape[0]
        half_classes = num_of_classes / 2
        num_of_slices = label.shape[3]
        if slice_range is None:
            slice_range = np.arange(num_of_slices)

        _ = plt.figure(figsize=(width, height))
        counter = 1
        columns = half_classes + 1  # currently only original image and uncertainty map
        rows = 2

        num_of_subplots = rows * 1 * columns  # +1 because the original image is included

        print("Number of subplots {} columns {} rows {}".format(num_of_subplots, columns, rows))
        for idx in slice_range:
            # get the slice and then split ED and ES slices
            img_slice = image[:, :, :, idx]
            true_labels = label[:, :, :, idx]
            img_slice_pred_labels = img_pred_labels[:, :, :, idx]
            # this is a dictionary...see above
            img_slice_probs = image_probs[idx]
            for phase in np.arange(2):
                img = img_slice[phase]  # INDEX 0 = end-systole image
                ax1 = plt.subplot(num_of_subplots, columns, counter)
                if phase == 0:
                    ax1.set_title("End-systole image")
                else:
                    ax1.set_title("End-diastole image")
                offx = config.pad_size
                offy = config.pad_size
                # get rid of the padding that we needed for the image processing
                img = img[offx:-offx, offy:-offy]
                plt.imshow(img, cmap=cm.gray)
                counter += 1
                for cls in np.arange(half_classes):
                    p_err, p_err_std = None, None
                    if phase == 0:
                        if cls in img_slice_probs["es_err_p"]:
                            p_err = np.array(img_slice_probs["es_err_p"][cls])
                            p_err_std = np.array(img_slice_probs["es_err_std"][cls])
                        if cls in img_slice_probs["es_cor_p"]:
                            p_corr = np.array(img_slice_probs["es_cor_p"][cls])
                            p_corr_std = np.array(img_slice_probs["es_cor_std"][cls])
                    else:
                        if cls in img_slice_probs["ed_err_p"]:
                            p_err = np.array(img_slice_probs["ed_err_p"][cls])
                            p_err_std = np.array(img_slice_probs["ed_err_std"][cls])
                        if cls in img_slice_probs["ed_cor_p"]:
                            p_corr = np.array(img_slice_probs["ed_cor_p"][cls])
                            p_corr_std = np.array(img_slice_probs["ed_cor_std"][cls])
                    ax2 = plt.subplot(num_of_subplots, columns, counter)
                    if p_err is not None:
                        density_err = gaussian_kde(p_err)
                        xs_err = np.linspace(0, p_err.max(), 200)
                        density_err.covariance_factor = lambda: .25
                        density_err._compute_covariance()
                        ax2.fill_between(xs_err, density_err(xs_err), label="errors", color="b", alpha=0.2)
                    if p_corr is not None:
                        density_cor = gaussian_kde(p_corr)
                        xs_cor = np.linspace(0, p_corr.max(), 200)
                        density_cor.covariance_factor = lambda: .25
                        density_cor._compute_covariance()
                        ax2.fill_between(xs_cor, density_cor(xs_cor), label="correct", color="g", alpha=0.2)
                    ax2.set_title(column_lbls[cls])
                    ax2.legend(loc="best")
                    counter += 1

    @property
    def N(self):
        return len(self.pred_labels)

    def save_results(self, outfile=None):

        if outfile is None:
            rnd = np.random.randint(0, 1000)
            outfile = os.path.join(os.environ.get("HOME"), "test_results_{}".format(rnd) + ".dll")

        try:
            with open(outfile, 'wb') as f:
                dill.dump(self, f)
            print("INFO - Saved results to {}".format(outfile))
        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
            print("ERROR - can't save results to {}".format(outfile))

    @staticmethod
    def load_results(path_to_exp, full_path=False):

        print("Load from {}".format(path_to_exp))
        try:
            with open(path_to_exp, 'rb') as f:
                test_results = dill.load(f)

        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
            print("Can't open file {}".format(path_to_exp))
            raise IOError

        return test_results
