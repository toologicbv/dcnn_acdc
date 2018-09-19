import time
import torch
import numpy as np

from common.parsing import do_parse_args
from common.hvsmr.config import config_hvsmr
from utils.hvsmr.experiment import HVSMRExperiment as Experiment
from utils.hvsmr.exper_handler import HVSMRExperimentHandler as ExperimentHandler
from utils.hvsmr.batch_handler import HVSMRTwoDimBatchHandler
from in_out.hvsmr.load_data import HVSMR2016DataSet
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
    dataset = HVSMR2016DataSet(exper_hdl.exper.config, search_mask=config_hvsmr.dflt_image_name + ".nii",
                               fold_id=exper_hdl.exper.run_args.fold_ids[0],
                               debug=exper_hdl.exper.run_args.quick_run, preprocess="rescale",
                               logger=exper_hdl.logger)
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

    # IMPORTANT: I AM CURRENTLY NOT USING THE FUNCTIONALITY TO RUN MULTIPLE BATCHES PER EPOCH!!!
    exper_hdl.exper.batches_per_epoch = 1
    exper_hdl.logger.info("INFO - #slices train/val-data-set {}/{} :: number of epochs {} "
                          ":: batch-size {} "
                          ":: batches/epoch {}".format(dataset.train_num_slices, dataset.val_num_slices,
                                                       exper_hdl.exper.run_args.epochs,
                                                       exper_hdl.exper.run_args.batch_size,
                                                       exper_hdl.exper.batches_per_epoch))

    for epoch_id in range(exper_hdl.exper.run_args.epochs):
        exper_hdl.next_epoch()
        dices = np.zeros((exper_hdl.exper.batches_per_epoch, 2))
        # in order to store the 2 dice coefficients for myocardium and bloodpool
        accuracy = np.zeros((exper_hdl.exper.batches_per_epoch, 2))
        losses = np.zeros(exper_hdl.exper.batches_per_epoch)
        start_time = time.time()
        for batch_id in range(exper_hdl.exper.batches_per_epoch):
            new_batch = HVSMRTwoDimBatchHandler(exper_hdl.exper)
            new_batch.generate_batch_2d(dataset.train_images, dataset.train_labels,
                                        num_of_slices=dataset.num_images_train)

            b_loss = dcnn_model.do_train(new_batch)
            exper_hdl.set_lr(dcnn_model.get_lr())
            # get the soft dice loss for ES and ED classes (average over each four classes)
            accuracy[batch_id] = dcnn_model.get_accuracy()
            losses[batch_id] = b_loss

        accuracy = np.mean(accuracy, axis=0)
        exper_hdl.set_batch_loss(np.mean(losses))
        total_time = time.time() - start_time
        if exper_hdl.exper.run_args.val_freq != 0 and (exper_hdl.exper.epoch_id % exper_hdl.exper.run_args.val_freq == 0
                                                       or
                                                       exper_hdl.exper.epoch_id == exper_hdl.exper.run_args.epochs):
            # validate model
            exper_hdl.eval(dataset, dcnn_model, val_set_size=exper_hdl.exper.config.val_set_size)

        if exper_hdl.exper.epoch_id % exper_hdl.exper.run_args.print_freq == 0:
            lr = dcnn_model.get_lr()
            exper_hdl.logger.info("End epoch {}: mean loss: {:.3f}, dice (Myo/LV): {:.3f}/{:.3f}"
                                  " / duration {:.2f} seconds "
                                  "lr={:.5f}".format(exper_hdl.exper.epoch_id, exper_hdl.get_epoch_loss(),
                                                     accuracy[0], accuracy[1],
                                                     total_time, lr))

    exper_hdl.save_experiment(final_run=True)

    
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
    exper_hdl.set_exper(Experiment(config_hvsmr, run_args=args), use_logfile=True)
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