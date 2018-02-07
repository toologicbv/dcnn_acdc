import time
import os
import glob
import torch
from torch.autograd import Variable
import numpy as np

from common.parsing import do_parse_args
from common.common import OPTIMIZER_DICT
from utils.experiment import Experiment, ExperimentHandler
from utils.config import config
from utils.batch_handlers import TwoDimBatchHandler
from in_out.load_data import HVSMR2016CardiacMRI
from models.model_handler import load_model, save_checkpoint


def training(exper_hdl):
    """

    :param args:
    :return:
    """

    dataset = HVSMR2016CardiacMRI(data_dir=exper_hdl.exper.config.data_dir,
                                  search_mask=exper_hdl.exper.config.dflt_image_name + ".nii",
                                  norm_scale=exper_hdl.exper.config.norm_method, load_type="numpy",
                                  val_fold=exper_hdl.exper.run_args.val_fold_id)

    dcnn_model = load_model(exper_hdl, simple=False)
    exper_hdl.exper.optimizer = OPTIMIZER_DICT[exper_hdl.exper.config.optimizer](dcnn_model.parameters(),
                                                                                 lr=exper_hdl.exper.run_args.lr)

    exper_hdl.exper.batches_per_epoch = 2
    exper_hdl.logger.info("Size data-set {} // number of epochs {} // batch-size {} // batches/epoch {}".format(
        dataset.__len__(), exper_hdl.exper.run_args.epochs, exper_hdl.exper.run_args.batch_size,
        exper_hdl.exper.batches_per_epoch))
    num_val_runs = 0
    for epoch_id in range(exper_hdl.exper.run_args.epochs):
        exper_hdl.next_epoch()
        start_time = time.time()
        exper_hdl.logger.info("Start epoch {}".format(exper_hdl.exper.epoch_id))
        for batch_id in range(exper_hdl.exper.batches_per_epoch):
            new_batch = TwoDimBatchHandler(exper_hdl.exper)
            new_batch.generate_batch_2d(dataset.images, dataset.labels)

            # print("new_batch.b_images", new_batch.b_images.shape)
            b_out = dcnn_model(new_batch.b_images)
            b_loss = dcnn_model.get_loss(b_out, new_batch.b_labels)
            # compute gradients w.r.t. model parameters
            b_loss.backward(retain_graph=False)
            exper_hdl.exper.optimizer.step()
            # sum_grads = dcnn_model.sum_grads()
            exper_hdl.exper.epoch_stats["mean_loss"][epoch_id] += b_loss.data.cpu().squeeze().numpy()
            dcnn_model.zero_grad()

        exper_hdl.exper.epoch_stats["mean_loss"][epoch_id] *= 1./float(exper_hdl.exper.batches_per_epoch)
        if exper_hdl.exper.run_args.val_freq != 0 and (exper_hdl.exper.epoch_id % exper_hdl.exper.run_args.val_freq == 0
                                                       or
                                                       exper_hdl.exper.epoch_id == exper_hdl.exper.run_args.epochs):
            # validate model
            num_val_runs += 1
            val_batch = TwoDimBatchHandler(exper_hdl.exper)
            val_batch.generate_batch_2d(dataset.val_images, dataset.val_labels)
            dcnn_model.eval()
            b_predictions = dcnn_model(val_batch.b_images)
            val_loss = dcnn_model.get_loss(b_predictions, val_batch.b_labels)
            val_loss = val_loss.data.cpu().squeeze().numpy()[0]
            # compute dice score for both classes (myocardium and bloodpool)
            dice = HVSMR2016CardiacMRI.compute_accuracy(b_predictions, val_batch.b_labels)
            # store epochID and validation loss
            exper_hdl.exper.val_stats["mean_loss"][num_val_runs-1] = np.array([exper_hdl.exper.epoch_id, val_loss])
            exper_hdl.exper.val_stats["dice_coeff"][num_val_runs - 1] = np.array([exper_hdl.exper.epoch_id, dice[0],
                                                                                  dice[1]])
            exper_hdl.logger.info("Model validation in epoch {}: current loss {:.3f}\t "
                                  "dice-coeff(myo/blood) {:.3f}/{:.3f}".format(exper_hdl.exper.epoch_id, val_loss,
                                                                           dice[0], dice[1]))
            dcnn_model.train()

        if exper_hdl.exper.run_args.chkpnt and (exper_hdl.exper.epoch_id % exper_hdl.exper.run_args.chkpnt_freq == 0 or
                                      exper_hdl.exper.epoch_id == exper_hdl.exper.run_args.epochs):
            save_checkpoint(exper_hdl, {'epoch': exper_hdl.exper.epoch_id,
                                        'state_dict': dcnn_model.state_dict(),
                                        'best_prec1': 0.},
                            False, dcnn_model.__class__.__name__)
            # save exper statistics
            exper_hdl.save_experiment()
        end_time = time.time()
        total_time = end_time - start_time
        exper_hdl.logger.info("End epoch {}: mean loss: {:.3f} / duration {:.2f} seconds".format(
            exper_hdl.exper.epoch_id,
            exper_hdl.exper.epoch_stats["mean_loss"][
            epoch_id],
            total_time))
    exper_hdl.save_experiment()
    del dataset
    del dcnn_model


def main():
    args = do_parse_args()

    exper = Experiment(config, run_args=args, set_seed=True)
    exper_hdl = ExperimentHandler(exper)
    exper.start()
    exper_hdl.print_flags()

    if args.cmd == 'train':
        training(exper_hdl)
    elif args.cmd == 'test':
        raise NotImplementedError("test mode is not yet implemented")


if __name__ == '__main__':
    main()
