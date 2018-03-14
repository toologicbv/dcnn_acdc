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


def get_mean_pred_per_slice(img_slice, img_probs, img_stds, labels, pred_labels, half_classes, stats):
    """

    Note that for detecting the correct and incorrect classified pixels, we only need to look at the background
    class, because the incorrect pixels in this class are the summation of the other 3 classes (for ES and ED
    separately).

    :param img_probs: predicted probabilities for image pixels for the 8 classes [classes, width, height, slices]
            0-3: ES and 4-7 ED
    :param img_stds: standard deviation for image pixels. One for ES and one for ED [classes, width, height, slices]
    :param labels: true labels with [num_of_classes, width, height, slices]
    :param pred_labels: [num_of_classes, width, height, slices]
    :param half_classes: in this case 4
    :param stats: object that holds the dictionaries that capture results of test run
    :return:
    """

    for phase in np.arange(2):
        # start with ES
        bg_class_idx = phase * half_classes
        s_labels_ph = labels[bg_class_idx, :, :, img_slice].flatten()
        s_pred_labels_ph = pred_labels[bg_class_idx, :, :, img_slice].flatten()
        s_labels_ph = np.atleast_1d(s_labels_ph.astype(np.bool))
        s_pred_labels_ph = np.atleast_1d(s_pred_labels_ph.astype(np.bool))
        errors = np.argwhere((~s_pred_labels_ph & s_labels_ph) | (s_pred_labels_ph & ~s_labels_ph))
        correct = np.argwhere((s_pred_labels_ph & s_labels_ph))

        if phase == 0:
            # compute mean stddev (over 4 classes)
            slice_stds_ph = np.mean(img_stds[:half_classes, :, :, img_slice], axis=0).flatten()
            slice_probs_ph = img_probs[:half_classes, :, :, img_slice]
        else:
            # compute mean stddev (over 4 classes)
            slice_stds_ph = np.mean(img_stds[half_classes:, :, :, img_slice], axis=0).flatten()

            slice_probs_ph = img_probs[half_classes:, :, :, img_slice]
        slice_probs_ph = np.reshape(slice_probs_ph, (slice_probs_ph.shape[0],
                                                     slice_probs_ph.shape[1] *
                                                     slice_probs_ph.shape[2]))
        # at this moment slice_probs_ph should be [half_classes, num_of_pixels]
        # and slice_stds_ph contains the pixel stddev for ES or ED [num_of_pixels]
        err_probs = slice_probs_ph.T[errors].flatten()
        err_std = slice_stds_ph[errors].flatten()
        pos_probs = slice_probs_ph.T[correct].flatten()
        pos_std = slice_stds_ph[correct].flatten()
        if phase == 0:
            stats["es_mean_err_p"] = err_probs
            stats["es_mean_cor_p"] = pos_probs
            stats["es_mean_err_std"] = err_std
            stats["es_mean_cor_std"] = pos_std
        else:
            stats["ed_mean_err_p"] = err_probs
            stats["ed_mean_cor_p"] = pos_probs
            stats["ed_mean_err_std"] = err_std
            stats["ed_mean_cor_std"] = pos_std


def get_img_stats_per_slice_class(img_slice_probs, img_slice, phase, half_classes,
                                  p_err_std, p_corr_std, p_err_prob, p_corr_prob):

    for cls in np.arange(half_classes):

        if img_slice == 0:
            if phase == 0:
                p_err_prob = np.array(img_slice_probs["es_err_p"][cls])
                p_corr_prob = np.array(img_slice_probs["es_cor_p"][cls])
                p_err_std = np.array(img_slice_probs["es_err_std"][cls])
                p_corr_std = np.array(img_slice_probs["es_cor_std"][cls])
            else:
                p_err_prob = np.array(img_slice_probs["ed_err_p"][cls])
                p_corr_prob = np.array(img_slice_probs["ed_cor_p"][cls])
                p_err_std = np.array(img_slice_probs["ed_err_std"][cls])
                p_corr_std = np.array(img_slice_probs["ed_cor_std"][cls])
        else:
            if phase == 0:
                p_err_prob = np.concatenate((p_err_prob, np.array(img_slice_probs["es_err_p"][cls])))
                p_corr_prob = np.concatenate((p_corr_prob, np.array(img_slice_probs["es_cor_p"][cls])))
                p_err_std = np.concatenate((p_err_std, np.array(img_slice_probs["es_err_std"][cls])))
                p_corr_std = np.concatenate((p_corr_std, np.array(img_slice_probs["es_cor_std"][cls])))
            else:
                p_err_prob = np.concatenate((p_err_prob, np.array(img_slice_probs["ed_err_p"][cls])))
                p_corr_prob = np.concatenate((p_corr_prob, np.array(img_slice_probs["ed_cor_p"][cls])))
                p_err_std = np.concatenate((p_err_std, np.array(img_slice_probs["ed_err_std"][cls])))
                p_corr_std = np.concatenate((p_corr_std, np.array(img_slice_probs["ed_cor_std"][cls])))

    return p_err_std, p_corr_std, p_err_prob, p_corr_prob


def get_img_stats_per_slice(img_slice_probs, img_slice, phase, p_err_std, p_corr_std, p_err_prob, p_corr_prob):

    if img_slice == 0:
        if phase == 0:
            p_err_prob = img_slice_probs["es_mean_err_p"]
            p_corr_prob = img_slice_probs["es_mean_cor_p"]
            p_err_std = img_slice_probs["es_mean_err_std"]
            p_corr_std = img_slice_probs["es_mean_cor_std"]
        else:
            p_err_prob = img_slice_probs["ed_mean_err_p"]
            p_corr_prob = img_slice_probs["ed_mean_cor_p"]
            p_err_std = img_slice_probs["ed_mean_err_std"]
            p_corr_std = img_slice_probs["ed_mean_cor_std"]
    else:
        if phase == 0:
            p_err_prob = np.concatenate((p_err_prob, np.array(img_slice_probs["es_mean_err_p"])))
            p_corr_prob = np.concatenate((p_corr_prob, np.array(img_slice_probs["es_mean_cor_p"])))
            p_err_std = np.concatenate((p_err_std, np.array(img_slice_probs["es_mean_err_std"])))
            p_corr_std = np.concatenate((p_corr_std, np.array(img_slice_probs["es_mean_cor_std"])))
        else:
            p_err_prob = np.concatenate((p_err_prob, np.array(img_slice_probs["ed_mean_err_p"])))
            p_corr_prob = np.concatenate((p_corr_prob, np.array(img_slice_probs["ed_mean_cor_p"])))
            p_err_std = np.concatenate((p_err_std, np.array(img_slice_probs["ed_mean_err_std"])))
            p_corr_std = np.concatenate((p_corr_std, np.array(img_slice_probs["ed_mean_cor_std"])))

    return p_err_std, p_corr_std, p_err_prob, p_corr_prob


class TestResults(object):

    def __init__(self, exper):
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

        # set path in order to save results and figures
        self.fig_output_dir = os.path.join(exper.config.root_dir,
                                           os.path.join(exper.output_dir, exper.config.figure_path))
        self.save_output_dir = os.path.join(exper.config.root_dir,
                                            os.path.join(exper.output_dir, exper.config.stats_path))

    def add_results(self, batch_image, batch_labels, pred_labels, b_predictions, uncertainty_map,
                    test_accuracy, test_hd):
        """

        :param batch_image: [2, width, height, slices]
        :param batch_labels:
        :param pred_labels:
        :param b_predictions:
        :param uncertainty_map:
        :param test_accuracy:
        :param test_hd:
        :return:
        """
        # get rid off padding around image
        batch_image = batch_image[:, config.pad_size:-config.pad_size, config.pad_size:-config.pad_size, :]

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
            if image_num in self.image_probs_categorized[image_num]:
                del self.image_probs_categorized[image_num]
                insert_idx = image_num
            else:
                del self.image_probs_categorized[0]
                insert_idx = 0
        else:
            insert_idx = image_num

        pred_labels = self.pred_labels[image_num]
        labels = self.labels[image_num]
        # this tensors contains all probs for the samples, but we will work with the means
        mc_pred_probs = self.mc_pred_probs[image_num]
        mean_pred_probs = np.mean(mc_pred_probs, axis=0)
        std_pred_probs = self.uncertainty_maps[image_num]
        num_slices = labels.shape[3]
        num_classes = labels.shape[0]
        half_classes = num_classes / 2
        probs_per_img_slice = []
        for slice in np.arange(num_slices):
            # object that holds probabilities per image slice
            probs_per_cls = {"es_err_p": OrderedDict(), "es_cor_p": OrderedDict(), "ed_err_p": OrderedDict(),
                             "ed_cor_p": OrderedDict(), "es_err_std": OrderedDict(), "es_cor_std": OrderedDict(),
                             "ed_err_std": OrderedDict(), "ed_cor_std": OrderedDict(),
                             "es_mean_err_p": np.zeros(half_classes), "es_mean_cor_p": np.zeros(half_classes),
                             "ed_mean_err_p": np.zeros(half_classes), "ed_mean_cor_p": np.zeros(half_classes),
                             "es_mean_err_std": np.zeros(half_classes), "es_mean_cor_std": np.zeros(half_classes),
                             "ed_mean_err_std": np.zeros(half_classes), "ed_mean_cor_std": np.zeros(half_classes)
                             }

            # mean per class
            get_mean_pred_per_slice(slice, mean_pred_probs, std_pred_probs, labels, pred_labels,
                                    half_classes, probs_per_cls)

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
                s_pred_labels = np.atleast_1d(s_pred_labels.astype(np.bool))
                s_labels = np.atleast_1d(s_labels.astype(np.bool))
                # total_l = s_labels.flatten().shape[0]
                # tp = np.count_nonzero(s_pred_labels & s_labels)
                # fn = np.count_nonzero(~s_pred_labels & s_labels)
                # fp = np.count_nonzero(s_pred_labels & ~s_labels)
                # tn = np.count_nonzero(~s_pred_labels & ~s_labels)
                # errors: fn + fp
                # correct: tp
                errors = np.argwhere((~s_pred_labels & s_labels) | (s_pred_labels & ~s_labels))
                # correct = np.invert(errors)
                correct = np.argwhere((s_pred_labels & s_labels))
                # print("cls {} total/tp/fn/fp/tn {} / {} / {} / {} / {}".format(cls, total_l, tp, fn, fp, tn))
                # print("\t tp / fn+fp {} {}".format(correct.shape[0], errors.shape[0]))
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

        self.image_probs_categorized.insert(insert_idx, probs_per_img_slice)

    def visualize_uncertainty_stats(self, image_num=0, width=16, height=10, info_type="uncertainty",
                                    use_class_stats=False, do_save=False, fig_name=None):

        image_probs = self.image_probs_categorized[image_num]
        label = self.labels[image_num]
        num_of_classes = label.shape[0]
        half_classes = num_of_classes / 2
        num_of_slices = label.shape[3]
        num_of_subplots = 2
        columns = 2
        counter = 1
        kde = True
        if not kde:
            num_of_subplots = 4
            columns = 2

        fig = plt.figure(figsize=(width, height))

        for phase in np.arange(2):
            if phase == 0:
                str_phase = "ES"
            else:
                str_phase = "ED"

            p_err_std, p_corr_std, p_err_prob, p_corr_prob = None, None, None, None
            for img_slice in np.arange(num_of_slices):

                img_slice_probs = image_probs[img_slice]
                if use_class_stats:
                    p_err_std, p_corr_std, p_err_prob, p_corr_prob = \
                        get_img_stats_per_slice_class(img_slice_probs, img_slice, phase, half_classes, p_err_std,
                                                      p_corr_std, p_err_prob, p_corr_prob)
                else:
                    p_err_std, p_corr_std, p_err_prob, p_corr_prob = \
                        get_img_stats_per_slice(img_slice_probs, img_slice, phase, p_err_std, p_corr_std, p_err_prob,
                                            p_corr_prob)

            print("{} correct/error(fp+fn) {} / {}".format(str_phase, p_corr_std.shape, p_err_std.shape))
            ax2 = plt.subplot(num_of_subplots, columns, counter)
            if p_err_std is not None:
                if kde:
                    if info_type == "uncertainty":
                        density_err = gaussian_kde(p_err_std)
                        xs_err = np.linspace(0, p_err_std.max(), 200)
                        density_err.covariance_factor = lambda: .25
                        density_err._compute_covariance()
                        ax2.fill_between(xs_err, density_err(xs_err), label="$\sigma_{pred(fp+fn)}$",
                                         color="b", alpha=0.2)
                    else:
                        density_err = gaussian_kde(p_err_prob)
                        xs_err = np.linspace(0, p_err_prob.max(), 200)
                        density_err.covariance_factor = lambda: .25
                        density_err._compute_covariance()
                        ax2.fill_between(xs_err, density_err(xs_err), label="$p_{pred(fp+fn)}(c|x)$",
                                         color="b", alpha=0.2)
                else:
                    xs_err = np.linspace(0, p_err_std.max(), 200)
                    ax2.hist(p_err_std, bins=xs_err, label=r"$\sigma_{pred(fp+fn)}$", color="b", alpha=0.2)

            if p_corr_std is not None:
                if kde:
                    if info_type == "uncertainty":
                        density_cor = gaussian_kde(p_corr_std)
                        xs_cor = np.linspace(0, p_corr_std.max(), 200)
                        density_cor.covariance_factor = lambda: .25
                        density_cor._compute_covariance()
                        ax2.fill_between(xs_cor, density_cor(xs_cor), label=r"$\sigma_{pred(tp)}$",
                                         color="g", alpha=0.2)
                    else:
                        density_cor = gaussian_kde(p_corr_prob)
                        xs_cor = np.linspace(0, p_corr_prob.max(), 200)
                        density_cor.covariance_factor = lambda: .25
                        density_cor._compute_covariance()
                        ax2.fill_between(xs_cor, density_cor(xs_cor), label="$p_{pred(tp)}(c|x)$",
                                         color="g", alpha=0.2)
                else:
                    counter += 1
                    ax3 = plt.subplot(num_of_subplots, columns, counter)
                    xs_cor = np.linspace(0, p_corr_std.max(), 200)
                    ax3.hist(p_corr_std, bins=xs_cor, label=r"$\sigma_{pred(tp)}$", color="g", alpha=0.2)
                    ax3.set_ylabel("density")
                    ax3.set_xlabel("model uncertainty")
                    ax3.legend(loc="best")

                if info_type == "uncertainty":
                    ax2.set_xlabel("model uncertainty", **config.axis_font)
                else:
                    ax2.set_xlabel(r"softmax $p(y|x)$", **config.axis_font)
                ax2.set_title("All classes ({})".format(str_phase), **config.title_font_medium)
                ax2.legend(loc="best", prop={'size': 16})
                ax2.set_ylabel("density", **config.axis_font)

            counter += 1
        fig.tight_layout()
        if do_save:
            if fig_name is None:
                fig_name = info_type + "_densities_" + str(use_class_stats)
            fig_name = os.path.join(self.fig_output_dir, fig_name + ".png")

            plt.savefig(fig_name, bbox_inches='tight')
            print("INFO - Successfully saved fig %s" % fig_name)

        plt.show()
        plt.close()

    def visualize_test_slices(self, image_num=0, width=8, height=6, slice_range=None, do_save=False,
                              fig_name=None):
        """

        Remember that self.image is a list, containing images with shape [2, height, width, depth]
        NOTE: self.b_pred_labels only contains the image that we just processed and NOT the complete list
        of images as in self.images and self.labels!!!

        NOTE: we only visualize 1 image (given by image_idx)

        """
        column_lbls = ["bg", "RV", "MYO", "LV"]
        image = self.images[image_num]
        img_labels = self.labels[image_num]
        img_pred_labels = self.pred_labels[image_num]
        num_of_classes = img_labels.shape[0]
        half_classes = num_of_classes / 2
        num_of_slices = img_labels.shape[3]

        if slice_range is None:
            slice_range = np.arange(0, num_of_slices // 2)

        fig = plt.figure(figsize=(width, height))
        counter = 1
        columns = half_classes + 1
        if img_pred_labels is not None:
            rows = 4
            plot_preds = True
        else:
            rows = 2
            plot_preds = False

        num_of_subplots = rows * 1 * columns  # +1 because the original image is included
        if len(slice_range) * num_of_subplots > 100:
            print("WARNING: need to limit number of subplots")
            slice_range = slice_range[:5]
        str_slice_range = [str(i) for i in slice_range]
        print("Number of subplots {} columns {} rows {} slices {}".format(num_of_subplots, columns, rows,
                                                                          ",".join(str_slice_range)))
        for idx in slice_range:
            # get the slice and then split ED and ES slices
            img = image[:, :, :, idx]
            labels = img_labels[:, :, :, idx]

            if plot_preds:
                pred_labels = img_pred_labels[:, :, :, idx]
            img_ed = img[0]  # INDEX 0 = end-diastole image
            img_es = img[1]  # INDEX 1 = end-systole image

            ax1 = plt.subplot(num_of_subplots, columns, counter)
            ax1.set_title("End-systole image", **config.title_font_medium)
            plt.imshow(img_ed, cmap=cm.gray)
            plt.axis('off')
            counter += 1
            for cls1 in np.arange(half_classes):
                ax2 = plt.subplot(num_of_subplots, columns, counter)
                plt.imshow(labels[cls1], cmap=cm.gray)
                ax2.set_title(column_lbls[cls1] + " (true labels)", **config.title_font_medium)
                plt.axis('off')
                if plot_preds:
                    ax3 = plt.subplot(num_of_subplots, columns, counter + columns)
                    plt.imshow(pred_labels[cls1], cmap=cm.gray)
                    plt.axis('off')
                    ax3.set_title(column_lbls[cls1] + " (pred labels)", **config.title_font_medium)
                counter += 1

            cls1 += 1
            counter += columns
            ax2 = plt.subplot(num_of_subplots, columns, counter)
            ax2.set_title("End-diastole image", **config.title_font_medium)
            plt.imshow(img_es, cmap=cm.gray)
            plt.axis('off')
            counter += 1
            for cls2 in np.arange(half_classes):
                ax4 = plt.subplot(num_of_subplots, columns, counter)
                plt.imshow(labels[cls1 + cls2], cmap=cm.gray)
                ax4.set_title(column_lbls[cls2] + " (true labels)", **config.title_font_medium)
                plt.axis('off')
                if plot_preds:
                    ax5 = plt.subplot(num_of_subplots, columns, counter + columns)
                    plt.imshow(pred_labels[cls1 + cls2], cmap=cm.gray)
                    ax5.set_title(column_lbls[cls2] + " (pred labels)", **config.title_font_medium)
                    plt.axis('off')
                counter += 1

            counter += columns

        fig.tight_layout()
        if do_save:
            file_suffix = "_".join(str_slice_range)
            if fig_name is None:
                fig_name = "test_img{}".format(image_num) + "_vis_pred_" + file_suffix
            fig_name = os.path.join(self.fig_output_dir, fig_name + ".png")

            plt.savefig(fig_name, bbox_inches='tight')
            print("INFO - Successfully saved fig %s" % fig_name)
        plt.show()

    def visualize_prediction_uncertainty(self, image_num=0, width=12, height=12, slice_range=None, do_save=False,
                                         fig_name=None):

        column_lbls = ["bg", "RV", "MYO", "LV"]
        image = self.images[image_num]
        labels = self.labels[image_num]
        img_pred_labels = self.pred_labels[image_num]
        uncertainty_map = self.uncertainty_maps[image_num]
        num_of_classes = labels.shape[0]
        half_classes = num_of_classes / 2
        num_of_slices = labels.shape[3]

        if slice_range is None:
            slice_range = np.arange(0, num_of_slices // 2)

        fig = plt.figure(figsize=(width, height))
        counter = 1
        columns = half_classes + 1  # currently only original image and uncertainty map
        rows = 2
        num_of_slices = len(slice_range)
        num_of_subplots = rows * num_of_slices * columns  # +1 because the original image is included
        str_slice_range = [str(i) for i in slice_range]
        print("Number of subplots {} columns {} rows {} slices {}".format(num_of_subplots, columns, rows,
                                                                          ",".join(str_slice_range)))
        for idx in slice_range:
            # get the slice and then split ED and ES slices
            image_slice = image[:, :, :, idx]
            true_labels = labels[:, :, :, idx]
            pred_labels = img_pred_labels[:, :, :, idx]
            uncertainty = uncertainty_map[:, :, :, idx]
            for phase in np.arange(2):

                img = image_slice[phase]  # INDEX 0 = end-systole image
                ax1 = plt.subplot(num_of_subplots, columns, counter)
                if phase == 0:
                    ax1.set_title("End-systole image", **config.title_font_medium)
                else:
                    ax1.set_title("End-diastole image", **config.title_font_medium)

                plt.imshow(img, cmap=cm.gray)
                plt.axis('off')
                counter += 1
                # we use the cls_offset to plot ES and ED images in one loop (phase variable)
                cls_offset = phase * half_classes
                mean_stddev = 0.
                for cls in np.arange(half_classes):
                    std = uncertainty[cls + cls_offset]
                    mean_stddev += std
                    ax2 = plt.subplot(num_of_subplots, columns, counter)
                    x2_plot = ax2.imshow(std, cmap=cm.coolwarm, vmin=0.0, vmax=0.6)
                    plt.colorbar(x2_plot, cax=ax2)
                    ax2.set_title(r"$\sigma_{{pred}}$ {}".format(column_lbls[cls]), **config.title_font_medium)
                    plt.axis('off')
                    counter += 1
                    true_cls_labels = true_labels[cls + cls_offset]
                    pred_cls_labels = pred_labels[cls + cls_offset]
                    errors = true_cls_labels != pred_cls_labels
                    ax3 = plt.subplot(num_of_subplots, columns, counter + half_classes)
                    ax3.set_title("Errors {}".format(column_lbls[cls]), **config.title_font_medium)
                    plt.imshow(errors, cmap=cm.gray)
                    plt.axis('off')

                # plot the average uncertainty per pixel (over classes)
                mean_stddev = 1./float(half_classes) * mean_stddev
                ax4 = plt.subplot(num_of_subplots, columns, counter )
                ax4_plot = ax4.imshow(mean_stddev, cmap=cm.coolwarm, vmin=0.0, vmax=0.6)
                # plt.colorbar(ax4_plot, cax=ax4_plot)
                ax4.set_title(r"$\sigma_{{pred}}$ {}".format("mean"), **config.title_font_medium)
                plt.axis('off')
                # move to the correct subplot space for the next phase (ES/ED)
                counter += half_classes + 1  # move counter forward in subplot
        fig.tight_layout()
        if do_save:
            file_suffix = "_".join(str_slice_range)
            if fig_name is None:
                fig_name = "test_img{}".format(image_num) + "_vis_pred_uncertainty_" + file_suffix
            fig_name = os.path.join(self.fig_output_dir, fig_name + ".png")

            plt.savefig(fig_name, bbox_inches='tight')
            print("INFO - Successfully saved fig %s" % fig_name)

    @property
    def N(self):
        return len(self.pred_labels)

    def save_results(self, outfile=None):

        if outfile is None:
            rnd = np.random.randint(0, 1000)
            outfile = "test_results_{}".format(rnd)

        outfile = os.path.join(self.save_output_dir, outfile + ".dll")

        try:
            with open(outfile, 'wb') as f:
                dill.dump(self, f)
            print("INFO - Saved results to {}".format(outfile))
        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
            print("ERROR - can't save results to {}".format(outfile))

    @property
    def mean_accuracy(self):

        columns = self.test_accuracy[0].shape[0]
        mean_acc = np.empty((0, columns))
        for run in np.arange(self.N):
            acc = self.test_accuracy[run]
            mean_acc = np.vstack([mean_acc, acc]) if mean_acc.size else acc

        return np.mean(mean_acc, axis=0)

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
