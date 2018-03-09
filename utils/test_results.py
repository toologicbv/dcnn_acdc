import numpy as np
import os
import dill
import sys
from collections import OrderedDict
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
        self.pred_probs = []
        self.test_accuracy = []
        self.test_hd = []
        self.probs_per_cls = []

    def add_results(self, batch_image, batch_labels, pred_labels, b_predictions, test_accuracy, test_hd):
        self.images.append(batch_image)
        self.labels.append(batch_labels)
        self.pred_labels.append(pred_labels)
        self.pred_probs.append(b_predictions)
        self.test_accuracy.append(test_accuracy)
        self.test_hd.append(test_hd)

    def error_distribution(self, image_num=0):
        """

        :param image_num: index of the test image we want to process
        :return:
        """
        pred_labels = self.pred_labels[image_num]
        labels = self.labels[image_num]
        pred_probs = self.pred_probs[image_num]
        num_slices = labels.shape[3]
        num_classes = labels.shape[0]
        probs_per_cls = {"es_err": OrderedDict(), "es_cor": OrderedDict(), "ed_err": OrderedDict(),
                         "ed_cor": OrderedDict()}
        for slice in np.arange(num_slices):
            for cls in np.arange(num_classes):
                s_labels = labels[cls, :, :, slice].flatten()
                s_pred_labels = pred_labels[cls, :, :, slice].flatten()
                s_slice_cls_probs = pred_probs[:, cls, :, :, slice]
                s_slice_cls_probs = np.reshape(s_slice_cls_probs, (s_slice_cls_probs.shape[0],
                                                                   s_slice_cls_probs.shape[1] *
                                                                   s_slice_cls_probs.shape[2]))
                # get the indices for the errors and the correct classified pixels in this slice/per class
                errors = s_labels != s_pred_labels
                correct = np.invert(errors)
                err_probs = s_slice_cls_probs.T[errors].flatten()
                pos_probs = s_slice_cls_probs.T[correct].flatten()
                if cls <= num_classes // 2:
                    if cls not in probs_per_cls["es_err"]:
                        probs_per_cls["es_err"][cls] = list(err_probs)
                    else:
                        probs_per_cls["es_err"][cls].extend(list(err_probs))
                    if cls not in probs_per_cls["es_cor"]:
                        probs_per_cls["es_cor"][cls] = list(pos_probs)
                    else:
                        probs_per_cls["es_cor"][cls].extend(list(pos_probs))
                else:
                    if cls - num_classes not in probs_per_cls["ed_err"]:
                        probs_per_cls["ed_err"][cls - num_classes] = list(err_probs)
                    else:
                        probs_per_cls["ed_err"][cls - num_classes].extend(list(err_probs))
                    if cls - num_classes not in probs_per_cls["ed_cor"]:
                        probs_per_cls["ed_cor"][cls - num_classes] = list(pos_probs)
                    else:
                        probs_per_cls["ed_cor"][cls - num_classes].extend(list(pos_probs))

        self.probs_per_cls.append(probs_per_cls)

    def visualize_pred_prob_dists(self, image_num=0, width=12, height=12):

        column_lbls = ["bg", "RV", "MYO", "LV"]
        probs_per_cls = self.probs_per_cls[image_num]
        num_of_classes = self.pred_labels[image_num].shape[0]

        fig = plt.figure(figsize=(width, height))
        counter = 1
        columns = num_of_classes + 1  # currently only original image and uncertainty map
        rows = 2

        num_of_subplots = rows * 1 * columns  # +1 because the original image is included

        print("Number of subplots {} columns {} rows {}".format(num_of_subplots, columns, rows))
        for idx in range(1):
            # get the slice and then split ED and ES slices
            image = self.b_image[:, :, :, idx]
            true_labels = self.b_labels[:, :, :, idx]
            pred_labels = self.b_pred_labels[:, :, :, idx]
            pred_probs = self.b_pred_probs[:, :, :, idx]
            uncertainty = self.b_uncertainty_map[:, :, :, idx]
            for phase in np.arange(2):

                img = image[phase]  # INDEX 0 = end-systole image
                ax1 = plt.subplot(num_of_subplots, columns, counter)
                if phase == 0:
                    ax1.set_title("End-systole image, reference and predictions")
                else:
                    ax1.set_title("End-diastole image, reference and predictions")
                offx = self.config.pad_size
                offy = self.config.pad_size
                # get rid of the padding that we needed for the image processing
                img = img[offx:-offx, offy:-offy]
                plt.imshow(img, cmap=cm.gray)
                counter += 1
                # we use the cls_offset to plot ES and ED images in one loop (phase variable)
                cls_offset = phase * self.num_of_classes
                for cls in np.arange(self.num_of_classes):
                    std = uncertainty[cls + cls_offset]
                    ax2 = plt.subplot(num_of_subplots, columns, counter)
                    plt.imshow(std, cmap=cm.coolwarm)
                    ax2.set_title(column_lbls[cls])
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
