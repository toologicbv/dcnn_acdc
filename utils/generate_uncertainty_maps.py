import os
import argparse
import time
from tqdm import tqdm

import torch
import numpy as np
from utils.test_handler import ACDC2017TestHandler
from utils.experiment import ExperimentHandler
from common.acquisition_functions import bald_function
from common.common import create_logger
from config.config import config


ROOT_DIR = os.getenv("REPO_PATH", "/home/jorg/repo/dcnn_acdc/")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
EXPERS = {"MC005_F2": "20180328_10_54_36_dcnn_mcv1_150000E_lr2e02",
          "MC005_F0": "20180330_09_56_01_dcnn_mcv1_150000E_lr2e02"}


def do_parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Generate Uncertainty Maps')

    parser.add_argument('--exper_id', default="MC005_F2")
    parser.add_argument('--model_name', default="MC dropout p={}")
    parser.add_argument('--mc_samples', type=int, default=10, help="# of MC samples")
    parser.add_argument('--cuda', action='store_true', default=False, help='use GPU')
    parser.add_argument('--verbose', action='store_true', default=False, help='show debug messages')
    args = parser.parse_args()

    return args


def _print_flags(args, logger=None):
    """
    Prints all entries in argument parser.
    """
    for key, value in vars(args).items():
        if logger is not None:
            logger.info(key + ' : ' + str(value))
        else:
            print(key + ' : ' + str(value))

    if args.cuda:
        if logger is not None:
            logger.info(" *** RUNNING ON GPU *** ")
        else:
            print(" *** RUNNING ON GPU *** ")


def get_exper_handler(args):
    exp_model_path = os.path.join(LOG_DIR, EXPERS[args.exper_id])
    exper = ExperimentHandler.load_experiment(exp_model_path)
    exper_hdl = ExperimentHandler(exper, use_logfile=False)
    exper_hdl.set_root_dir(ROOT_DIR)
    exper_hdl.set_model_name(args.model_name.format(exper.run_args.drop_prob))
    return exper_hdl


class UncertaintyMapsGenerator(object):

    def __init__(self, exper_handler, test_set=None, mc_samples=10, model=None, checkpoint=None, verbose=False,
                 use_logger=False):

        self.verbose = verbose
        self.mc_samples = mc_samples
        self.checkpoint = checkpoint
        self.exper_handler = exper_handler
        if use_logger and self.exper_handler.logger is None:
            self.exper_handler.logger = create_logger(self.exper_handler.exper, file_handler=True)
        self.fold_id = int(self.exper_handler.exper.run_args.fold_ids[0])
        if test_set is None:
            self.test_set = self._get_test_set()
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

    def __call__(self,):

        start_time = time.time()
        message = "INFO - Starting to generate uncertainty maps " \
                  "for {} images using {} samples".format(self.num_of_images, self.mc_samples)
        self.info(message)
        for image_num in tqdm(np.arange(self.num_of_images)):
            self._generate(image_num=image_num)
            self.save_maps()

        duration = time.time() - start_time
        self.info("INFO - Total duration of generation process {:.2f} secs".format(duration))

    def _generate(self, image_num):

        # correct the divisor for calculation of stdev when low number of samples (biased), used in np.std
        if self.mc_samples <= 25 and self.mc_samples != 1:
            ddof = 1
        else:
            ddof = 0
        # b_predictions has shape [mc_samples, classes, width, height, slices]
        b_predictions = np.zeros(tuple([self.mc_samples] + list(self.test_set.labels[image_num].shape)))
        slice_idx = 0
        # batch_generator iterates over the slices of a particular image
        for batch_image, batch_labels in self.test_set.batch_generator(image_num):

            b_test_losses = np.zeros(self.mc_samples)
            bald_values = np.zeros((2, batch_labels.shape[2], batch_labels.shape[3]))
            for s in np.arange(self.mc_samples):
                test_loss, test_pred = self.model.do_test(batch_image, batch_labels,
                                                          voxel_spacing=self.test_set.new_voxel_spacing,
                                                          compute_hd=False, test_mode=False)
                b_predictions[s, :, :, :, self.test_set.slice_counter] = test_pred.data.cpu().numpy()

                b_test_losses[s] = test_loss.data.cpu().numpy()
                # dice_loss_es, dice_loss_ed = model.get_dice_losses(average=True)
            # mean/std for each pixel for each class
            mc_probs = b_predictions[:, :, :, :, self.test_set.slice_counter]
            mean_test_pred, std_test_pred = np.mean(mc_probs, axis=0, keepdims=True), \
                                            np.std(mc_probs, axis=0, ddof=ddof)

            bald_values[0] = bald_function(b_predictions[:, 0:4, :, :, self.test_set.slice_counter])
            bald_values[1] = bald_function(b_predictions[:, 4:, :, :, self.test_set.slice_counter])

            means_test_loss = np.mean(b_test_losses)
            # NOTE: we set do_filter (connected components post-processing) to FALSE here because we will do
            # this at the end for the complete 3D label object, but not for the individual slices
            self.test_set.set_pred_labels(mean_test_pred, u_threshold=None, verbose=self.verbose,
                                          do_filter=False)

            self.test_set.set_stddev_map(std_test_pred)
            self.test_set.set_bald_map(bald_values)
            slice_acc, slice_hd = self.test_set.compute_slice_accuracy(compute_hd=False)

            # NOTE: currently only displaying the BALD uncertainty stats but we also capture the stddev stats
            es_total_uncert, es_num_of_pixel_uncert, es_num_pixel_uncert_above_tre, num_of_conn_commponents = \
                self.test_set.b_uncertainty_stats["bald"][0, :, slice_idx]  # ES
            ed_total_uncert, ed_num_of_pixel_uncert, ed_num_pixel_uncert_above_tre, num_of_conn_commponents = \
                self.test_set.b_uncertainty_stats["bald"][1, :, slice_idx]  # ED
            es_seg_errors = np.sum(self.test_set.b_seg_errors[slice_idx, :4])
            ed_seg_errors = np.sum(self.test_set.b_seg_errors[slice_idx, 4:])

            if self.verbose:
                print("Test img/slice {}/{}".format(image_num, slice_idx))
                print("ES: Total BALD/seg-errors/#pixel/#pixel(tre) \tDice (RV/Myo/LV)\tHD (RV/Myo/LV)")
                print("  \t{:.2f}/{}/{}/{} \t\t\t{:.2f}/{:.2f}/{:.2f}"
                      "\t\t{:.2f}/{:.2f}/{:.2f}".format(es_total_uncert, es_seg_errors, es_num_of_pixel_uncert,
                                                        es_num_pixel_uncert_above_tre,
                                                        slice_acc[1], slice_acc[2], slice_acc[3],
                                                        slice_hd[1], slice_hd[2], slice_hd[3]))
                # print(np.array_str(np.array(es_region_mean_uncert), precision=3))
                print("ED: Total BALD/seg-errors/#pixel/#pixel(tre)\tDice (RV/Myo/LV)\tHD (RV/Myo/LV)")
                print("  \t{:.2f}/{}/{}/{} \t\t\t{:.2f}/{:.2f}/{:.2f}"
                      "\t\t{:.2f}/{:.2f}/{:.2f}".format(ed_total_uncert, ed_seg_errors, ed_num_of_pixel_uncert,
                                                        ed_num_pixel_uncert_above_tre,
                                                        slice_acc[5], slice_acc[6], slice_acc[7],
                                                        slice_hd[5], slice_hd[6], slice_hd[7]))
                # print(np.array_str(np.array(ed_region_mean_uncert), precision=3))
                print("------------------------------------------------------------------------")
            slice_idx += 1

    def save_maps(self):
        # b_stddev_map: [8 classes, width, height, #slices]. we insert a new axis and split dim1 into 2 x 4
        # new shape [2, 4, width, height, #slices]
        stddev_map = np.concatenate((self.test_set.b_stddev_map[np.newaxis, :4, :, :, :],
                                     self.test_set.b_stddev_map[np.newaxis, 4:, :, :, :]))
        # b_bald_map shape: [2, width, height, #slices]
        try:
            filename = self.test_set.b_image_name + "_umaps.npz"
            filename = os.path.join(self.umap_output_dir, filename)
            np.savez(filename, stddev_map=stddev_map, bald_map=self.test_set.b_bald_map,
                     stddev_stats=self.test_set.b_uncertainty_stats["stddev"],
                     bald_map_stats=self.test_set.b_uncertainty_stats["bald"])
            self.info("INFO - Succesfully saved maps to {}".format(filename))
        except IOError:
            print("Unable to save uncertainty maps for image {}")

    @staticmethod
    def load_uncertainty_maps(image_name):

        data = np.load('mat.npz')
        print data['name1']
        print data['name2']

    def _get_test_set(self):
        # Important note:
        # we set batch_size to None, which means all images will be loaded
        # we set val_only parameter to False, which means images from "train" and "validation" directory will be
        # loaded.
        return ACDC2017TestHandler(exper_config=self.exper_handler.exper.config,
                                   search_mask=self.exper_handler.exper.config.dflt_image_name + ".mhd",
                                   fold_ids=[self.fold_id],
                                   debug=False, batch_size=1,
                                   use_cuda=self.exper_handler.exper.run_args.cuda,
                                   val_only=False, use_iso_path=True)

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


def main():
    args = do_parse_args()
    SEED = 4325
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    if args.cuda:
        torch.backends.cudnn.enabled = True

    np.random.seed(SEED)
    exper_handler = get_exper_handler(args)
    _print_flags(args)
    maps_generator = UncertaintyMapsGenerator(exper_handler, verbose=args.verbose, mc_samples=args.mc_samples)
    maps_generator()


if __name__ == '__main__':
    main()

