import time
import torch
import numpy as np

from common.dslices.parsing import do_parse_args, config

from utils.dslices.experiment import Experiment as ExperimentSD
from utils.dslices.exper_handler import ExperimentHandler as ExperHandlerSD
from utils.dslices.exper_handler import ExperHandlerEnsemble
from utils.dslices.batch_handler import BatchHandler as BatchHandlerSD
from utils.dslices.accuracies import compute_eval_metrics
from in_out.load_data import ACDC2017DataSet
from models.model_handler import save_checkpoint, load_slice_detector_model
from in_out.dataset_slice_detector import create_dataset


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
    exper_hdl.logger.info("INFO - Creating dataset for slice detection. This may take a while, be patient!")
    sd_dataset = create_dataset(dataset, seg_exper_dict, type_of_map=exper_hdl.exper.run_args.type_of_map,
                                degenerate_type="mean")
    exper_hdl.logger.info("INFO - Ready")
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

    train_batch = BatchHandlerSD(data_set=sd_dataset, is_train=True, cuda=args.cuda)
    for epoch_id in range(exper_hdl.exper.run_args.epochs):
        exper_hdl.next_epoch()
        start_time = time.time()
        x_input, y_lbl = train_batch(batch_size=args.batch_size, backward_freq=1)
        # returns cross-entropy loss (binary) and predicted probabilities [batch-size, 2] for this batch
        loss, pred_probs = sd_vgg_model.do_forward_pass(x_input, y_lbl)
        train_batch.add_loss(loss)
        if train_batch.do_backward:
            sd_vgg_model.zero_grad()
            train_batch.mean_loss()
            train_batch.loss.backward(retain_graph=False)
            sd_vgg_model.optimizer.step()
            train_batch.reset()
            exper_hdl.set_loss(loss)
        if exper_hdl.exper.run_args.val_freq != 0 and (exper_hdl.exper.epoch_id % exper_hdl.exper.run_args.val_freq == 0
                                                       or
                                                       exper_hdl.exper.epoch_id == exper_hdl.exper.run_args.epochs):
            # validate model
            exper_hdl.eval(sd_dataset, sd_vgg_model)

        if exper_hdl.exper.run_args.chkpnt and (exper_hdl.exper.epoch_id % exper_hdl.exper.run_args.chkpnt_freq == 0 or
                                                exper_hdl.exper.epoch_id == exper_hdl.exper.run_args.epochs):
            save_checkpoint(exper_hdl, {'epoch': exper_hdl.exper.epoch_id,
                                        'state_dict': sd_vgg_model.state_dict(),
                                        'best_prec1': 0.},
                            False, sd_vgg_model.__class__.__name__)

            # save exper statistics
            exper_hdl.save_experiment()
        end_time = time.time()
        total_time = end_time - start_time
        if exper_hdl.exper.epoch_id % exper_hdl.exper.run_args.print_freq == 0 or \
                exper_hdl.exper.epoch_id == exper_hdl.exper.run_args.epochs:
            f1, roc_auc, pr_auc, acc = compute_eval_metrics(y_lbl.data.cpu().numpy(),
                                                                  np.argmax(pred_probs.data.cpu().numpy(), axis=1))
            print("End epoch ID: {} loss {:.3f} / f1={:.3f} / roc_auc={:.3f} / "
                  "pr_auc={:.3f} / acc={:.3f} "
                  "duration {:.2f} seconds".format(exper_hdl.exper.epoch_id, loss.item(), f1, roc_auc,
                                                   pr_auc, acc, total_time))

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