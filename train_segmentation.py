import time
import os
import glob
import torch
from torch.autograd import Variable
import numpy as np

from common.parsing import do_parse_args
from config.config import config
from utils.experiment import Experiment, ExperimentHandler
from utils.batch_handlers import TwoDimBatchHandler
from in_out.load_data import ACDC2017DataSet
from models.model_handler import load_model, save_checkpoint


def training(exper_hdl):
    """

    :param args:
    :return:
    """

    dataset = ACDC2017DataSet(exper_hdl.exper.config, search_mask=config.dflt_image_name + ".mhd",
                              fold_ids=exper_hdl.exper.run_args.fold_ids, preprocess=False,
                              debug=exper_hdl.exper.run_args.quick_run)

    dcnn_model = load_model(exper_hdl)
    # IMPORTANT: I AM CURRENTLY NOT USING THE FUNCTIONALITY TO RUN MULTIPLE BATCHES PER EPOCH!!!
    exper_hdl.exper.batches_per_epoch = 1
    exper_hdl.logger.info("Size train/val-data-set {}/{} :: number of epochs {} "
                          ":: batch-size {} "
                          ":: batches/epoch {}".format(
                            dataset.__len__()[0], dataset.__len__()[1], exper_hdl.exper.run_args.epochs,
                            exper_hdl.exper.run_args.batch_size,
                            exper_hdl.exper.batches_per_epoch))

    num_val_runs = 0
    for epoch_id in range(exper_hdl.exper.run_args.epochs):
        exper_hdl.next_epoch()
        dices = np.zeros((exper_hdl.exper.batches_per_epoch, 2))
        # in order to store the 6 dice coefficients
        accuracy = np.zeros((exper_hdl.exper.batches_per_epoch, 6))
        # in order to store the 2 mean dice losses (ES/ED)
        losses = np.zeros(exper_hdl.exper.batches_per_epoch)
        start_time = time.time()
        # exper_hdl.logger.info("Start epoch {}".format(exper_hdl.exper.epoch_id))
        for batch_id in range(exper_hdl.exper.batches_per_epoch):
            new_batch = TwoDimBatchHandler(exper_hdl.exper)
            new_batch.generate_batch_2d(dataset.train_images, dataset.train_labels)
            b_loss = dcnn_model.do_train(new_batch)
            exper_hdl.set_lr(dcnn_model.get_lr())
            # get the soft dice loss for ES and ED classes (average over each four classes)
            dices[batch_id] = dcnn_model.get_dice_losses(average=True)
            accuracy[batch_id] = dcnn_model.get_accuracy()
            losses[batch_id] = b_loss

        losses = np.mean(losses)
        accuracy = np.mean(accuracy, axis=0)
        dices = np.mean(dices, axis=0)
        exper_hdl.set_batch_loss(losses)
        exper_hdl.set_accuracy(accuracy)
        exper_hdl.set_dice_losses(dices)

        if exper_hdl.exper.run_args.val_freq != 0 and (exper_hdl.exper.epoch_id % exper_hdl.exper.run_args.val_freq == 0
                                                       or
                                                       exper_hdl.exper.epoch_id == exper_hdl.exper.run_args.epochs):
            # validate model
            exper_hdl.next_val_run()
            num_val_runs += 1
            val_batch = TwoDimBatchHandler(exper_hdl.exper, batch_size=exper_hdl.exper.config.val_batch_size,
                                           test_run=True)
            val_batch.generate_batch_2d(dataset.images(train=False), dataset.labels(train=False))
            val_loss, _ = dcnn_model.do_test(val_batch.get_images(), val_batch.get_labels())
            val_accuracy = dcnn_model.get_accuracy()
            val_dice_losses = dcnn_model.get_dice_losses(average=True)
            # store epochID and validation loss
            val_loss = val_loss.data.cpu().numpy()[0]
            exper_hdl.exper.val_stats["mean_loss"][num_val_runs-1] = val_loss
            exper_hdl.exper.val_stats["dice_coeff"][num_val_runs - 1] = val_accuracy
            exper_hdl.set_accuracy(val_accuracy, val_run_id=num_val_runs)
            exper_hdl.set_dice_losses(val_dice_losses, val_run_id=num_val_runs)
            exper_hdl.logger.info("---> VALIDATION of model in epoch {}: current loss {:.3f}\t "
                                  "dice-coeff:: ES {:.3f}/{:.3f}/{:.3f} --- "
                                  "ED {:.3f}/{:.3f}/{:.3f}".format(exper_hdl.exper.epoch_id, val_loss,
                                                                   val_accuracy[0], val_accuracy[1],
                                                                   val_accuracy[2], val_accuracy[3],
                                                                   val_accuracy[4], val_accuracy[5]))

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
        if exper_hdl.exper.epoch_id % exper_hdl.exper.run_args.print_freq == 0:
            lr = dcnn_model.get_lr()
            exper_hdl.logger.info("End epoch {}: mean loss: {:.3f} / mean dice-loss (ES/ED) {}"
                                  " / duration {:.2f} seconds "
                                  "lr={:.5f}".format(exper_hdl.exper.epoch_id, exper_hdl.get_epoch_loss(),
                                                                      np.array_str(dices, precision=3),
                                                                      # exper_hdl.get_epoch_accuracy(),
                                                                      total_time,
                                                                      lr))
            exper_hdl.logger.info("Current dice accuracies ES {:.3f}/{:.3f}/{:.3f} \t"
                                  "ED {:.3f}/{:.3f}/{:.3f} ".format(accuracy[0], accuracy[1],
                                                                    accuracy[2], accuracy[3],
                                                                    accuracy[4], accuracy[5]))
    exper_hdl.save_experiment(final_run=True)
    del dataset
    del dcnn_model


def main():
    args = do_parse_args()
    SEED = 4325
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    if args.cuda:
        torch.backends.cudnn.enabled = True

    np.random.seed(SEED)

    exper = Experiment(config, run_args=args)
    exper_hdl = ExperimentHandler(exper)
    exper.start()
    exper_hdl.print_flags()
    training(exper_hdl)


if __name__ == '__main__':
    main()
