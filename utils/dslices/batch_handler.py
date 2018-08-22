import torch
import numpy as np


class BatchHandler(object):

    def __init__(self, data_set, is_train=True, cuda=False):
        """
            data_set  of object type SliceDetectorDataSet

        """
        self.cuda = cuda
        self.is_train = is_train
        self.number_of_patients = data_set.get_size(is_train=is_train)
        self.data_set = data_set
        self.loss = torch.zeros(1)
        self.num_sub_batches = torch.zeros(1)
        self.backward_freq = None
        self.current_patient_id = None
        if not is_train:
            self.patient_ids = data_set.get_patient_ids(is_train=False)
            self.item_num = 0
        else:
            self.patient_ids = None
            self.item_num = None

        if self.cuda:
            self.loss = self.loss.cuda()
            self.num_sub_batches = self.num_sub_batches.cuda()

    def next_patient_id(self):
        if self.item_num == self.number_of_patients:
            self.item_num = 0
        patient_id = self.patient_ids[self.item_num]
        self.item_num += 1
        return patient_id

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

    def __call__(self, batch_size=None, backward_freq=None, patient_id=None):
        """
        Construct a batch of shape [batch_size, 3channels, w, h]
                                   3channels: (1) input image
                                              (2) predicted segmentation mask
                                              (3) generated (raw/unfiltered) u-map/entropy map

        :param batch_size:
        :param patient_id: if not None than we use all slices
        :return: input batch and corresponding references
        """
        self.backward_freq = backward_freq
        if self.is_train:
            # (1) Sample ONE patient ID
            if patient_id is None:
                item_num = np.random.randint(low=0, high=self.number_of_patients, size=1)[0]
                patient_id = self.data_set.get_patient_ids(is_train=self.is_train)[item_num]
        else:
            # testing or validation
            if patient_id is None:
                patient_id = self.next_patient_id()
        self.current_patient_id = patient_id
        # IMPORTANT:
        # REMEMBER. the dataset contains input tensors of shape [2, 3channels, w, h, #slices]
        #           WHEREAS the label object of dataset has shape [2, #slices] (binary encoding)
        input_3c, label, extra_label = self.data_set.get(patient_id, self.is_train)
        num_of_slices = int(input_3c.shape[0])
        # if batch_size is None we set it to the total number of slices for ES/ED
        # This can happen during testing/validation
        if batch_size is None:
            batch_size = num_of_slices
        if num_of_slices < batch_size:
            # reduce batch size
            batch_size = num_of_slices
            # print("WARNING - #slices {} - new batch-size {}".format(num_of_slices, batch_size))

        # (2) Sample half of batch size from ES and the other from ED image slices
        if self.is_train:
            # randomly select slices
            slice_idx = np.random.randint(low=0, high=num_of_slices, size=(batch_size,))
            # randomly determine rotation: 90 * num_rotations (k-factor for np.rot90)
            num_rotations = np.random.randint(low=0, high=3, size=(1,))[0]
        else:
            # testing or validation: we take all slices for this patient
            slice_idx = np.arange(int(num_of_slices))
            num_rotations = 0
        # image shape: [#slices, 3channels, w, h] and we sample a couple of slices (batch-size)
        #              The result is [batch-size, 3channels, w, h]
        img_slices = input_3c[slice_idx, :, :, :]
        if num_rotations != 0:
            img_slices = np.rot90(img_slices, k=num_rotations, axes=(2, 3)).copy()
        # we need to reshape the input from [3, w, h, batch-size] to [batch-size, 3, w, h]
        # img_slices = np.reshape(img_slices, (img_slices.shape[3], img_slices.shape[0], img_slices.shape[1],
        #                                     img_slices.shape[2]))
        label_slices = label[slice_idx]
        y_extra_labels = extra_label[slice_idx]
        # concatenate along dim0 = batch dimension
        # x_batch = np.concatenate((es_img_slices, ed_img_slices), axis=0)
        x_batch = torch.FloatTensor(torch.from_numpy(img_slices).float())
        # y_batch = np.concatenate((es_label_slices, ed_label_slices), axis=0)
        y_batch = torch.LongTensor(torch.from_numpy(label_slices).long())
        # print("x_batch.shape ", x_batch.shape)
        if self.cuda:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
        return x_batch, y_batch, y_extra_labels


