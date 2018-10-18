import time
import torch
import numpy as np

from common.detector.parsing import do_parse_args
from common.detector.config import config_detector
from utils.dslices.accuracies import compute_eval_metrics
from utils.detector.experiment import Experiment as ExperimentRD
from utils.detector.exper_handler import ExperimentHandler as ExperHandlerRD
from utils.exper_hdl_ensemble import ExperHandlerEnsemble
from utils.detector.batch_handler import BatchHandler as BatchHandler
from models.model_handler import save_checkpoint, load_region_detector_model
from in_out.detector.detector_dataset import create_dataset


def training(args):
    """

    :param exper_hdl:
    :return:
    """
    # first get experiment from segmentation run
    seg_exper_ensemble = ExperHandlerEnsemble(args.exper_dict)
    seg_exper_hdl = seg_exper_ensemble.seg_exper_handlers[args.fold_id]
    exper_hdl = ExperHandlerRD()
    exper_hdl.set_exper(ExperimentRD(config_detector, seg_exper=seg_exper_hdl.exper, run_args=args), use_logfile=True)
    exper_hdl.print_flags()
    exper_hdl.logger.info("INFO - Creating dataset for slice detection. This may take a while, be patient!")

    dataset = create_dataset(seg_exper_ensemble, train_fold_id=exper_hdl.exper.run_args.fold_id,
                             quick_run=exper_hdl.exper.run_args.quick_run,
                             type_of_map=exper_hdl.exper.run_args.type_of_map,
                             num_of_input_chnls=exper_hdl.exper.run_args.num_input_chnls)

    # Load model. In the same procedure the model is assigned to the CPU or GPU
    rd_model = load_region_detector_model(exper_hdl)
    decayed_lr = False
    # IMPORTANT: I AM CURRENTLY NOT USING THE FUNCTIONALITY TO RUN MULTIPLE BATCHES PER EPOCH!!!
    exper_hdl.exper.batches_per_epoch = 1
    train_batch = BatchHandler(data_set=dataset, is_train=True, cuda=args.cuda, verbose=False,
                               keep_bounding_boxes=False, backward_freq=rd_model.backward_freq)
    mean_epoch_loss = []
    # mean_epoch_stats = np.zeros((exper_hdl.exper.run_args.print_freq, ))
    for epoch_id in range(exper_hdl.exper.run_args.epochs):
        exper_hdl.next_epoch()
        start_time = time.time()
        x_input, y_lbl_dict = train_batch(batch_size=args.batch_size, do_balance=True)
        # returns cross-entropy loss (binary) and predicted probabilities [batch-size, 2] for this batch
        y_lbl_grid8 = y_lbl_dict[8]
        y_lbl_grid4 = y_lbl_dict[4]
        loss, pred_probs = rd_model.do_forward_pass(x_input, y_lbl_grid8, y_labels_extra=y_lbl_grid4)
        train_batch.add_loss(loss)
        exper_hdl.set_loss(loss.item())
        mean_epoch_loss.append(loss.item())
        if train_batch.do_backward:
            # exper_hdl.logger.info("Backpropagation @epoch:{}".format(exper_hdl.exper.epoch_id))
            rd_model.zero_grad()
            train_batch.mean_loss()
            train_batch.loss.backward(retain_graph=False)
            if train_batch.loss.item() < 1. and not decayed_lr:
                new_lr = rd_model.lr * 0.5
                exper_hdl.info("*** LEARNING-RATE - setting new lr={} old-lr={} ***".format(new_lr, rd_model.lr))
                rd_model.set_learning_rate(lr=new_lr)
                # only lower lr once
                decayed_lr = True

            rd_model.optimizer.step()
            # grads = rd_model.sum_grads()
            # print("---> Sum-grads {:.3f}".format(grads))
            train_batch.reset()

        if exper_hdl.exper.run_args.chkpnt and (exper_hdl.exper.epoch_id % exper_hdl.exper.run_args.chkpnt_freq == 0 or
                                                exper_hdl.exper.epoch_id == exper_hdl.exper.run_args.epochs):
            save_checkpoint(exper_hdl, {'epoch': exper_hdl.exper.epoch_id,
                                        'state_dict': rd_model.state_dict(),
                                        'best_prec1': 0.},
                            False, rd_model.__class__.__name__)

            # save exper statistics
            exper_hdl.save_experiment()
        end_time = time.time()
        total_time = end_time - start_time
        if exper_hdl.exper.epoch_id % exper_hdl.exper.run_args.print_freq == 0 or \
                exper_hdl.exper.epoch_id == exper_hdl.exper.run_args.epochs:
            np_pred_probs = pred_probs.data.cpu().numpy()
            mean_epoch_loss = np.mean(mean_epoch_loss)
            f1, roc_auc, pr_auc, prec, rec, fpr, tpr, precision, recall = \
                compute_eval_metrics(y_lbl_grid8.data.cpu().numpy(), np.argmax(pred_probs.data.cpu().numpy(), axis=1),
                                     np_pred_probs[:, 1])

            exper_hdl.logger.info("End epoch ID: {} mean-loss {:.3f} / f1={:.3f} / roc_auc={:.3f} / "
                                  "pr_auc={:.3f} / prec={:.3f} / rec={:.3f} "
                                  "duration {:.2f} seconds".format(exper_hdl.exper.epoch_id, mean_epoch_loss, f1,
                                                                   roc_auc,
                                                                   pr_auc, prec, rec, total_time))
            mean_epoch_loss = []
        if exper_hdl.exper.run_args.val_freq != 0 and (exper_hdl.exper.epoch_id % exper_hdl.exper.run_args.val_freq == 0
                                                       or
                                                       exper_hdl.exper.epoch_id == exper_hdl.exper.run_args.epochs):
            # validate model
            exper_hdl.eval(dataset, rd_model, verbose=False)

    exper_hdl.save_experiment(final_run=True)
    del dataset
    del rd_model


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
CUDA_VISIBLE_DEVICES=0 python train_region_detector.py --use_cuda --batch_size=16  --print_freq=10 
--epochs=2000 --model=rd1 --lr=0.0001 --fold_id=0 --type_of_map=e_map --quick_run


CUDA_VISIBLE_DEVICES=0 nohup python train_slice_detector.py --use_cuda --batch_size=4 --val_freq=100 --print_freq=25
--epochs=5000 --model=sdvgg11_bn --lr=0.00005 --fold_id=0 --type_of_map=e_map --chkpnt --chkpnt_freq=1500
> /home/jorg/tmp/tr_sdvgg_f3_6000_bs8.log 2>&1&
"""