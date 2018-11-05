import time
import torch
import numpy as np

from common.parsing import do_parse_args
from config.config import config
from utils.experiment import Experiment, ExperimentHandler
from utils.batch_handlers import TwoDimBatchHandler
from utils.test_handler import ACDC2017TestHandler
from in_out.load_data import ACDC2017DataSet
from models.model_handler import load_model, save_checkpoint


def check_dependencies_run_args(run_args):

    if run_args.train_outlier_only and not run_args.guided_train:
        raise ValueError("if train_outlier_only then guided_train needs to be true!")

    if run_args.retrain_chkpnt and not run_args.retrain_chkpnt:
        raise ValueError("if retrain_chkpnt then retrain_chkpnt needs to be true!")


def training(exper_hdl):
    """

    :param args:
    :return:
    """
    # create dataset
    dataset = ACDC2017DataSet(exper_hdl.exper.config, search_mask=config.dflt_image_name + ".mhd",
                              fold_ids=exper_hdl.exper.run_args.fold_ids, preprocess=False,
                              debug=exper_hdl.exper.run_args.quick_run, do_flip= True,
                              incomplete_only=exper_hdl.exper.run_args.incomplete_slices)
    # Load model
    if exper_hdl.exper.run_args.retrain_exper is not None:
        dcnn_model = exper_hdl.load_checkpoint(verbose=False, drop_prob=exper_hdl.exper.run_args.drop_prob,
                                               checkpoint=exper_hdl.exper.run_args.retrain_chkpnt,
                                               retrain=True,
                                               exper_dir=exper_hdl.exper.run_args.retrain_exper)
    else:
        dcnn_model = load_model(exper_hdl)

    # assign model to CPU or GPU if available
    device = torch.device("cuda" if exper_hdl.exper.run_args.cuda else "cpu")
    dcnn_model = dcnn_model.to(device)

    if exper_hdl.exper.run_args.guided_train:
        test_set = ACDC2017TestHandler(exper_config=exper_hdl.exper.config,
                                       search_mask=exper_hdl.exper.config.dflt_image_name + ".mhd", fold_ids=[0],
                                       debug=False, batch_size=None, use_cuda=exper_hdl.exper.run_args.cuda,
                                       load_train=True, load_val=False, use_iso_path=True)
        # when retraining a model and we're training on outlier slices, we first create an outlier set
        if exper_hdl.exper.run_args.retrain_exper:
            exper_hdl.info("NOTE: Before we start training we create an outlier (slice) dataset. This may"
                           " take a while.")
            outlier_dataset = exper_hdl.create_outlier_dataset(dataset, model=dcnn_model, test_set=test_set,
                                                               checkpoint=None, mc_samples=5,
                                                               u_threshold=0.1, use_train_set=True,
                                                               do_save_u_stats=True, use_high_threshold=True,
                                                               do_save_outlier_stats=True)
            if outlier_dataset.__len__() != 0:
                # outlier_freq = compute_batch_freq_outliers(exper_hdl.exper.run_args, outlier_dataset, dataset)
                outlier_freq = 2
                exper_hdl.info("NOTE: using outlier dataset every {} epoch".format(outlier_freq))
            else:
                # No outliers detected!
                outlier_dataset = None
                outlier_freq = None
        else:
            outlier_dataset = None
            outlier_freq = None
    else:
        test_set = None
        outlier_dataset = None
        outlier_freq = None

    exper_hdl.init_batch_statistics(dataset.trans_dict)

    # IMPORTANT: I AM CURRENTLY NOT USING THE FUNCTIONALITY TO RUN MULTIPLE BATCHES PER EPOCH!!!
    exper_hdl.exper.batches_per_epoch = 1
    exper_hdl.logger.info("Size train/val-data-set {}/{} :: number of epochs {} "
                          ":: batch-size {} "
                          ":: batches/epoch {}".format(
                            dataset.__len__()[0], dataset.__len__()[1], exper_hdl.exper.run_args.epochs,
                            exper_hdl.exper.run_args.batch_size,
                            exper_hdl.exper.batches_per_epoch))

    for epoch_id in range(exper_hdl.exper.run_args.epochs):
        exper_hdl.next_epoch()
        dices = np.zeros((exper_hdl.exper.batches_per_epoch, 2))
        # in order to store the 6 dice coefficients
        accuracy = np.zeros((exper_hdl.exper.batches_per_epoch, 6))
        # in order to store the 2 mean dice losses (ES/ED)
        losses = np.zeros(exper_hdl.exper.batches_per_epoch)
        reg_losses = np.zeros(exper_hdl.exper.batches_per_epoch)
        start_time = time.time()
        used_outliers = False
        for batch_id in range(exper_hdl.exper.batches_per_epoch):
            new_batch = TwoDimBatchHandler(exper_hdl.exper)
            if (exper_hdl.exper.run_args.guided_train and outlier_dataset is not None and \
               (exper_hdl.exper.epoch_id % outlier_freq == 0)) or exper_hdl.exper.run_args.train_outlier_only:
                # In this case we're only training on outlier image slices!
                # exper_hdl.logger.info("Epoch {} using outlier dataset".format(exper_hdl.exper.epoch_id))
                new_batch.generate_batch_2d(outlier_dataset.images, outlier_dataset.labels,
                                            num_of_slices=outlier_dataset.num_of_slices,
                                            img_slice_ids=outlier_dataset.img_slice_ids)
                used_outliers = True
            else:
                new_batch.generate_batch_2d(dataset.train_images, dataset.train_labels,
                                            num_of_slices=dataset.train_num_slices,
                                            img_slice_ids=dataset.train_img_slice_ids)

            b_loss = dcnn_model.do_train(new_batch)
            exper_hdl.set_lr(dcnn_model.get_lr())
            # get the soft dice loss for ES and ED classes (average over each four classes)
            dices[batch_id] = dcnn_model.get_dice_losses(average=True)
            accuracy[batch_id] = dcnn_model.get_accuracy()
            losses[batch_id] = b_loss
            reg_losses[batch_id] = dcnn_model.get_reg_loss()
            exper_hdl.exper.batch_stats.update_stats(new_batch.batch_stats)

        losses = np.mean(losses)
        reg_losses = np.mean(reg_losses)
        accuracy = np.mean(accuracy, axis=0)
        dices = np.mean(dices, axis=0)
        exper_hdl.set_batch_loss(losses, used_outliers=used_outliers, reg_loss=reg_losses)
        exper_hdl.set_accuracy(accuracy, used_outliers=used_outliers)
        exper_hdl.set_dice_losses(dices)

        if exper_hdl.exper.run_args.val_freq != 0 and (exper_hdl.exper.epoch_id % exper_hdl.exper.run_args.val_freq == 0
                                                       or
                                                       exper_hdl.exper.epoch_id == exper_hdl.exper.run_args.epochs):
            # validate model
            exper_hdl.eval(dataset, dcnn_model, val_set_size=exper_hdl.exper.config.val_set_size)

        if exper_hdl.exper.run_args.chkpnt and (exper_hdl.exper.epoch_id % exper_hdl.exper.run_args.chkpnt_freq == 0 or
                                                exper_hdl.exper.epoch_id == exper_hdl.exper.run_args.epochs):
            save_checkpoint(exper_hdl, {'epoch': exper_hdl.exper.epoch_id,
                                        'state_dict': dcnn_model.state_dict(),
                                        'best_prec1': 0.},
                            False, dcnn_model.__class__.__name__)
            if exper_hdl.exper.run_args.guided_train:
                outlier_dataset = exper_hdl.create_outlier_dataset(dataset, model=dcnn_model, test_set=test_set,
                                                                   checkpoint=None, mc_samples=5,
                                                                   u_threshold=0.1, use_train_set=True,
                                                                   do_save_u_stats=True, use_high_threshold=True,
                                                                   do_save_outlier_stats=True)
                # outlier_freq = compute_batch_freq_outliers(exper_hdl.exper.run_args, outlier_dataset, dataset)
                outlier_freq = 2
                exper_hdl.info("NOTE: using outlier dataset every {} epoch".format(outlier_freq))
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
                                                                      total_time,
                                                                      lr))
            if exper_hdl.exper.run_args.use_reg_loss:
                loss_wo = losses - reg_losses
                exper_hdl.logger.info("\t\tLoss + reg-loss {:.2f} + {:.2f}".format(loss_wo, reg_losses))
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
        torch.backends.cudnn.deterministic = True

    np.random.seed(SEED)
    exper_hdl = ExperimentHandler()
    exper_hdl.set_exper(Experiment(config, run_args=args), use_logfile=True)
    exper_hdl.print_flags()
    check_dependencies_run_args(exper_hdl.exper.run_args)
    training(exper_hdl)


if __name__ == '__main__':
    main()

"""
python train_segmentation.py --lr=0.0002 --batch_size=4 --val_freq=10 --epochs=15 --use_cuda --fold_ids=0 
--print_freq=5 --weight_decay=0.0001 --model="dcnn_mc" --drop_prob=0.05 --cycle_length=10000 --loss_function=softdice 
--quick_run
"""