import torch
import numpy as np


class BatchHandler(object):

    def __init__(self, fold_id, exper_ensemble, data_set, cuda=False):
        self.fold_id = fold_id
        self.cuda = cuda
        self.patient_id = "patient095"
        self.exper_ensemble = exper_ensemble
        self.all_folds = self.exper_ensemble.seg_exper_handlers.keys()
        self.u_maps = None
        self.pred_labels = None
        self.data_set = data_set
        self.loss = torch.zeros(1)
        self.num_sub_batches = torch.zeros(1)
        self.backward_freq = None
        if self.cuda:
            self.loss = self.loss.cuda()
            self.num_sub_batches = self.num_sub_batches.cuda()
        # {key: y[key] if key in y else x[key]
        # for key in set(x) + set(y)

    def _merge_input_set(self):
        pass

    def add_loss(self, loss):
        self.num_sub_batches += 1
        self.loss += loss

    def mean_loss(self):
        f = 1./self.num_sub_batches
        self.loss = 1./self.num_sub_batches * self.loss

    def reset(self):
        self.loss = torch.zeros(1)
        self.num_sub_batches = torch.zeros(1)
        if self.cuda:
            self.loss = self.loss.cuda()
            self.num_sub_batches = self.num_sub_batches.cuda()

    @property
    def do_backward(self):
        if self.backward_freq is not None:
            return self.backward_freq == self.num_sub_batches
        else:
            return True

    def __call__(self, batch_size=2, backward_freq=None):
        """
        Construct a batch of shape [batch_size, w, h, 3channels]

        :param batch_size:
        :return:
        """
        self.backward_freq = backward_freq
        # item_num = torch.randint(low=0, high=len(self.u_maps), size=(batch_size,))
        item_num = [self.data_set.trans_dict[self.patient_id]]
        pfold_id = self.exper_ensemble.get_patient_fold_id(self.patient_id)
        seg_exper_handlers = self.exper_ensemble.seg_exper_handlers
        seg_exper_handlers[pfold_id].get_pred_labels(patient_id=self.patient_id)
        self.pred_labels = seg_exper_handlers[pfold_id].pred_labels
        seg_exper_handlers[pfold_id].get_referral_maps(u_threshold=0.001, per_class=False,
                                                            patient_id=self.patient_id,
                                                            aggregate_func="max", use_raw_maps=True,
                                                            load_ref_map_blobs=False)
        self.u_maps = seg_exper_handlers[pfold_id].referral_umaps
        # for idx in item_num.numpy().astype(dtype=np.int):
        for idx in item_num:
            patient_id = self.data_set.image_names[idx]
            img = self.data_set.train_images[idx]
            p_lbl = self.pred_labels[patient_id]
            u_map = self.u_maps[patient_id]
            batch = []
            if batch_size > img.shape[3]:
                batch_size = img.shape[3]
            for b in np.arange(batch_size):
                x_in_es = np.concatenate((np.expand_dims(img[0, :, :, b], axis=0),
                                          np.expand_dims(p_lbl[0, :, :, b], axis=0),
                                          np.expand_dims(u_map[0, :, :, b], axis=0)), axis=0)
                x_in_es = np.expand_dims(x_in_es, axis=0)
                x_in_es = torch.FloatTensor(torch.from_numpy(x_in_es).float())
                batch.append(x_in_es)
            x_in_es = torch.cat(batch, dim=0)
            if batch_size == 10:
                y_lbl = torch.tensor([1, 0, 0, 1, 0, 1, 0, 1, 0, 1])
            elif batch_size == 4:
                y_lbl = torch.tensor([1, 0, 0, 1])
            else:
                y_lbl = torch.tensor([1])
            if self.cuda:
                x_in_es = x_in_es.cuda()
                y_lbl = y_lbl.cuda()
            yield x_in_es, y_lbl


