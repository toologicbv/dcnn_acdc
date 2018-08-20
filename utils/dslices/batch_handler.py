import torch
import numpy as np


class BatchHandler(object):

    def __init__(self, data_set, is_train=True, cuda=False):

        self.cuda = cuda
        self.is_train = is_train
        self.images = None
        self.labels = None
        self.patient_ids = None
        self.number_of_images = 0
        self._prepare(data_set)
        self.loss = torch.zeros(1)
        self.num_sub_batches = torch.zeros(1)
        self.backward_freq = None
        if self.cuda:
            self.loss = self.loss.cuda()
            self.num_sub_batches = self.num_sub_batches.cuda()

    def _prepare(self, data_set):
        if self.is_train:
            self.images = data_set.train_images
            self.labels = data_set.train_labels

        else:
            self.images = self.data_set.test_images
            self.labels = self.data_set.test_labels
        self.patient_ids = self.images.keys()
        self.number_of_images = len(self.patient_ids)

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
                                   3channels: (1) input image
                                              (2) predicted segmentation mask
                                              (3) generated (raw/unfiltered) u-map/entropy map

        :param batch_size:
        :return:
        """
        if batch_size % 2 != 0:
            raise ValueError("ERROR - Batch size must be multiple of 2. Got {}".format(batch_size))
        self.backward_freq = backward_freq
        half_batch = batch_size / 2
        item_num = np.random.randint(low=0, high=self.number_of_images, size=1)[0]
        patient_id = self.patient_ids[item_num]
        image = self.images[patient_id]
        print(image.shape)
        labels = self.labels[patient_id]
        num_of_slices = image.shape[4]
        if half_batch > num_of_slices:
            batch_size = image.shape[4]
            half_batch = batch_size / 2
        # get half of batch size from ES and the other from ED
        slice_idx_es = np.random.randint(low=0, high=num_of_slices, size=(half_batch,))
        slice_idx_ed = np.random.randint(low=0, high=num_of_slices, size=(half_batch,))
        es_img_slices = image[0, :, :, :, slice_idx_es]
        es_label_slices = labels[0, slice_idx_es]
        ed_img_slices = image[1, :, :, :, slice_idx_ed]
        ed_label_slices = labels[1, slice_idx_ed]
        print("ed_img_slices.shape ", ed_img_slices.shape)
        if self.cuda:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
        return x_batch, y_batch


