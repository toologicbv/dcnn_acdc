import time
import torch
import numpy as np

from common.dslices.parsing import do_parse_args, config

from utils.dslices.experiment import Experiment as ExperimentSD
from utils.dslices.exper_handler import ExperimentHandler as ExperHandlerSD
from utils.dslices.exper_handler import ExperHandlerEnsemble
from utils.dslices.batch_handler import BatchHandler as BatchHandlerSD
from in_out.load_data import ACDC2017DataSet
from models.model_handler import save_checkpoint, load_slice_detector_model


def training(args):
    """

    :param exper_hdl:
    :return:
    """
    # first get experiment from segmentation run
    seg_exper_dict = ExperHandlerEnsemble(args.exper_dict)
    seg_exper_hdl = seg_exper_dict.seg_exper_handlers[args.fold_id]
    exper_hdl = ExperHandlerSD()
    exper_hdl.set_exper(ExperimentSD(config, seg_exper=seg_exper_hdl.exper, run_args=args), use_logfile=True)
    exper_hdl.print_flags()

    # create dataset, complementary to segmentation experiments, we load images without augmentation and padding
    dataset = ACDC2017DataSet(seg_exper_hdl.exper.config, search_mask=config.dflt_image_name + ".mhd",
                              fold_ids=seg_exper_hdl.exper.run_args.fold_ids, preprocess=False,
                              debug=exper_hdl.exper.run_args.quick_run, do_augment=False,
                              incomplete_only=False)
    # Load model. In the same procedure the model is assigned to the CPU or GPU
    sd_vgg_model = load_slice_detector_model(exper_hdl)

    # IMPORTANT: I AM CURRENTLY NOT USING THE FUNCTIONALITY TO RUN MULTIPLE BATCHES PER EPOCH!!!
    exper_hdl.exper.batches_per_epoch = 1
    exper_hdl.logger.info("Size train/val-data-set {}/{} :: number of epochs {} "
                          ":: batch-size {} "
                          ":: batches/epoch {}".format(
                            dataset.__len__()[0], dataset.__len__()[1], exper_hdl.exper.run_args.epochs,
                            exper_hdl.exper.run_args.batch_size,
                            exper_hdl.exper.batches_per_epoch))

    new_batch = BatchHandlerSD(fold_id=args.fold_id, exper_ensemble=seg_exper_dict,
                               data_set=dataset, cuda=args.cuda)
    for epoch_id in range(exper_hdl.exper.run_args.epochs):
        exper_hdl.next_epoch()
        # in order to store the 2 mean dice losses (ES/ED)
        losses = np.zeros(exper_hdl.exper.batches_per_epoch)
        start_time = time.time()
        for x_input, y_lbl in new_batch(batch_size=args.batch_size, backward_freq=1):
            loss = sd_vgg_model.do_train(x_input, y_lbl, new_batch)
            print("Epoch ID: {} loss {:.3f}".format(exper_hdl.exper.epoch_id,
                                                    loss.item()))


    #         new_batch.generate_batch_2d(dataset.train_images, dataset.train_labels,
    #                                         num_of_slices=dataset.train_num_slices,
    #                                         img_slice_ids=dataset.train_img_slice_ids)
    #
    #         b_loss = sd_vgg_model.do_train(new_batch)
    #         losses[batch_id] = b_loss
    #
    #     losses = np.mean(losses)

        # if exper_hdl.exper.run_args.val_freq != 0 and (exper_hdl.exper.epoch_id % exper_hdl.exper.run_args.val_freq == 0
        #                                                or
        #                                                exper_hdl.exper.epoch_id == exper_hdl.exper.run_args.epochs):
        #     # validate model
        #     exper_hdl.eval(dataset, dcnn_model, val_set_size=exper_hdl.exper.config.val_set_size)
        #
        # if exper_hdl.exper.run_args.chkpnt and (exper_hdl.exper.epoch_id % exper_hdl.exper.run_args.chkpnt_freq == 0 or
        #                                         exper_hdl.exper.epoch_id == exper_hdl.exper.run_args.epochs):
        #     save_checkpoint(exper_hdl, {'epoch': exper_hdl.exper.epoch_id,
        #                                 'state_dict': dcnn_model.state_dict(),
        #                                 'best_prec1': 0.},
        #                     False, dcnn_model.__class__.__name__)
        #
        #     # save exper statistics
        #     exper_hdl.save_experiment()
        # end_time = time.time()
        # total_time = end_time - start_time
        # if exper_hdl.exper.epoch_id % exper_hdl.exper.run_args.print_freq == 0:
        #     exper_hdl.logger.info("End epoch {}: mean loss: {:.3f} "
        #                           " / duration {:.2f} seconds "
        #                           "".format(exper_hdl.exper.epoch_id, exper_hdl.get_epoch_loss(),
        #                                                               total_time))

    exper_hdl.save_experiment(final_run=True)
    del dataset
    del sd_vgg_model


def main():
    args = do_parse_args()
    SEED = 4325
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    if args.cuda:
        torch.backends.cudnn.enabled = True

    np.random.seed(SEED)
    training(args)


if __name__ == '__main__':
    main()

"""
CUDA_VISIBLE_DEVICES=1 python train_slice_detector.py --use_cuda --batch_size=10  --val_freq=10 --print_freq=5 
--fold_id=0 --quick_run --epochs=100 --model=sdvgg11_bn --lr=0.00005

"""