import time
import torch
import numpy as np

from common.dslices.parsing import do_parse_args, config

from utils.dslices.experiment import Experiment as ExperimentSD
from utils.dslices.exper_handler import ExperimentHandler as ExperHandlerSD
from utils.exper_hdl_ensemble import ExperHandlerEnsemble
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
    seg_exper_ensemble = ExperHandlerEnsemble(args.exper_dict)
    seg_exper_hdl = seg_exper_ensemble.seg_exper_handlers[args.fold_id]
    exper_hdl = ExperHandlerSD()
    exper_hdl.set_exper(ExperimentSD(config, seg_exper=seg_exper_hdl.exper, run_args=args), use_logfile=True)
    exper_hdl.print_flags()
    exper_hdl.logger.info("INFO - Creating dataset for slice detection. This may take a while, be patient!")
    sd_dataset = create_dataset(exper_hdl, seg_exper_ensemble,
                                num_of_input_chnls=exper_hdl.exper.run_args.num_input_chnls,
                                type_of_map=exper_hdl.exper.run_args.type_of_map,
                                degenerate_type="mean", pos_label=1, logger=exper_hdl.logger)

    # Load model. In the same procedure the model is assigned to the CPU or GPU
    sd_vgg_model = load_slice_detector_model(exper_hdl)
    decayed_lr = False
    # IMPORTANT: I AM CURRENTLY NOT USING THE FUNCTIONALITY TO RUN MULTIPLE BATCHES PER EPOCH!!!
    exper_hdl.exper.batches_per_epoch = 1
    train_batch = BatchHandlerSD(data_set=sd_dataset, is_train=True, cuda=args.cuda)
    mean_epoch_loss = []
    # mean_epoch_stats = np.zeros((exper_hdl.exper.run_args.print_freq, ))
    for epoch_id in range(exper_hdl.exper.run_args.epochs):
        exper_hdl.next_epoch()
        start_time = time.time()
        x_input, y_lbl, _ = train_batch(batch_size=args.batch_size, backward_freq=sd_vgg_model.backward_freq,
                                        do_balance=True)
        # returns cross-entropy loss (binary) and predicted probabilities [batch-size, 2] for this batch
        loss, pred_probs = sd_vgg_model.do_forward_pass(x_input, y_lbl)
        train_batch.add_loss(loss)
        exper_hdl.set_loss(loss.item())
        mean_epoch_loss.append(loss.item())
        if train_batch.do_backward:
            # exper_hdl.logger.info("Backpropagation @epoch:{}".format(exper_hdl.exper.epoch_id))
            sd_vgg_model.zero_grad()
            train_batch.mean_loss()
            train_batch.loss.backward(retain_graph=False)
            if train_batch.loss.item() < 1. and not decayed_lr:
                new_lr = sd_vgg_model.lr * 0.5
                exper_hdl.info("*** LEARNING-RATE - setting new lr={} old-lr={} ***".format(new_lr, sd_vgg_model.lr))
                sd_vgg_model.set_learning_rate(lr=new_lr)
                # only lower lr once
                decayed_lr = True

            sd_vgg_model.optimizer.step()
            # grads = sd_vgg_model.sum_grads()
            # print("---> Sum-grads {:.3f}".format(grads))
            train_batch.reset()

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
            np_pred_probs = pred_probs.data.cpu().numpy()
            mean_epoch_loss = np.mean(mean_epoch_loss)
            f1, roc_auc, pr_auc, prec, rec, fpr, tpr, precision, recall = \
                compute_eval_metrics(y_lbl.data.cpu().numpy(), np.argmax(pred_probs.data.cpu().numpy(), axis=1),
                                     np_pred_probs[:, 1])

            exper_hdl.logger.info("End epoch ID: {} mean-loss {:.3f} / f1={:.3f} / roc_auc={:.3f} / "
                                  "pr_auc={:.3f} / prec={:.3f} / rec={:.3f} "
                                  "duration {:.2f} seconds".format(exper_hdl.exper.epoch_id, mean_epoch_loss, f1, roc_auc,
                                                   pr_auc, prec, rec, total_time))
            mean_epoch_loss = []
        if exper_hdl.exper.run_args.val_freq != 0 and (exper_hdl.exper.epoch_id % exper_hdl.exper.run_args.val_freq == 0
                                                       or
                                                       exper_hdl.exper.epoch_id == exper_hdl.exper.run_args.epochs):
            # validate model
            exper_hdl.eval(sd_dataset, sd_vgg_model, verbose=False)

    exper_hdl.save_experiment(final_run=True)
    del sd_dataset
    del sd_vgg_model


def main():
    args = do_parse_args()
    SEED = 4325
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    if args.cuda:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True

    np.random.seed(SEED)
    training(args)


if __name__ == '__main__':
    main()

"""
CUDA_VISIBLE_DEVICES=0 python train_slice_detector.py --use_cuda --batch_size=16  --val_freq=200 --print_freq=25 
--epochs=2000 --model=sdvgg11_bn --lr=0.00001 --fold_id=1 --type_of_map=u_map --chkpnt


CUDA_VISIBLE_DEVICES=0 nohup python train_slice_detector.py --use_cuda --batch_size=4 --val_freq=100 --print_freq=25
--epochs=5000 --model=sdvgg11_bn --lr=0.00005 --fold_id=0 --type_of_map=e_map --chkpnt --chkpnt_freq=1500
> /home/jorg/tmp/tr_sdvgg_f3_6000_bs8.log 2>&1&
"""