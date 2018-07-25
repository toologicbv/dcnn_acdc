import torch
import numpy as np


class BatchHandler(object):

    def __init__(self, fold_id, seg_exper_handlers, data_set, cuda=False):
        self.fold_id = fold_id
        self.cuda = cuda
        self.patient_id = "patient095"
        self.all_folds = seg_exper_handlers.keys()
        self.seg_exper_handlers = seg_exper_handlers
        self.seg_exper_handlers[1].get_pred_labels(patient_id=self.patient_id)
        self.pred_labels = self.seg_exper_handlers[1].pred_labels
        self.seg_exper_handlers[1].get_referral_maps(u_threshold=0.001, per_class=False,
                                                     patient_id=self.patient_id,
                                                     aggregate_func="max", use_raw_maps=True)
        self.u_maps = self.seg_exper_handlers[1].referral_umaps
        self.data_set = data_set
        # {key: y[key] if key in y else x[key]
        # for key in set(x) + set(y)

    def _merge_input_set(self):
        pass

    def __call__(self, batch_size=2):
        """
        Construct a batch of shape [batch_size, w, h, 3channels]

        :param batch_size:
        :return:
        """
        item_num = torch.randint(low=0, high=len(self.u_maps), size=(batch_size,))
        item_num = [self.data_set.trans_dict[self.patient_id]]
        # for idx in item_num.numpy().astype(dtype=np.int):
        for idx in item_num:
            patient_id = self.data_set.image_names[idx]
            img = self.data_set.train_images[idx]
            p_lbl = self.pred_labels[patient_id]
            u_map = self.u_maps[patient_id]
            for i in np.arange(img.shape[3]):
                x_in_es = np.concatenate((np.expand_dims(img[0, :, :, i], axis=-1),
                                          np.expand_dims(p_lbl[0, :, :, i], axis=-1),
                                          np.expand_dims(u_map[0, :, :, i], axis=-1)), axis=-1)
                x_in_es = torch.FloatTensor(torch.from_numpy(x_in_es).float())
                if self.cuda:
                    x_in_es = x_in_es.cuda()
                yield x_in_es


