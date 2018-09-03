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
        self.current_slice_ids = None
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
        # we use this method during testing/validation to get all patient ids in sequence. If last item is reached
        # counter is reset and first patient id is returned again
        if self.item_num == self.number_of_patients:
            self.item_num = 0
        patient_id = self.patient_ids[self.item_num]
        self.item_num += 1
        return patient_id

    def add_loss(self, loss):
        self.num_sub_batches += 1
        self.loss += loss

    def mean_loss(self):
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

    def __call__(self, batch_size=None, backward_freq=None, patient_id=None, do_balance=True, do_permute=True):
        """
        Construct a batch of shape [batch_size, 3channels, w, h]
                                   3channels: (1) input image
                                              (2) predicted segmentation mask
                                              (3) generated (raw/unfiltered) u-map/entropy map

        :param batch_size:
        :param patient_id: if not None than we use all slices
        :param do_balance: if True batch is balanced w.r.t. normal and degenerate slices
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
        # REMEMBER. the dataset contains input tensors of shape [3channels, w, h, #slices]
        #           WHEREAS the label object of dataset has shape [#slices] (binary encoding)
        img_slices, label_slices, additional_labels, batch_size = self.determine_slices(patient_id,
                                                                                        do_balance, batch_size,
                                                                                        do_permute=do_permute)
        # concatenate along dim0 = batch dimension
        # x_batch = np.concatenate((es_img_slices, ed_img_slices), axis=0)
        x_batch = torch.FloatTensor(torch.from_numpy(img_slices).float())
        # y_batch = np.concatenate((es_label_slices, ed_label_slices), axis=0)
        y_batch = torch.LongTensor(torch.from_numpy(label_slices).long())
        # print("x_batch.shape ", x_batch.shape)
        if self.cuda:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
        return x_batch, y_batch, additional_labels

    def determine_slices(self, patient_id, do_balance, batch_size, do_permute=True):
        input_3c, label, extra_label = self.data_set.get(patient_id, self.is_train, do_merge_sets=not do_balance)
        # if do_balance is TRUE the previous method returns a tuple per variable (normal-slices, degenerate slices)
        # otherwise the method returns a concatenated numpy tensor in which the first dim is #slices per object
        if do_balance:

            num_of_slices = int(input_3c[0].shape[0])
            deg_num_of_slices = int(input_3c[1].shape[0])
            # print("INFO - #slices {}/{}".format(num_of_slices, deg_num_of_slices))
            if self.is_train:
                # we assert that batch_size is given during TRAINING (not necessarily the case for testing)
                half_batch_size = batch_size / 2
                # currently assuming that batch_size = 2
                slice_ids = np.random.randint(low=0, high=num_of_slices, size=half_batch_size)
                deg_slice_ids = np.random.randint(low=0, high=deg_num_of_slices, size=half_batch_size)
            else:
                # during testing we make sure all degenrate slices are selected and the batch is adjusted on this size
                # furthermore we select the "normal" slices randomly.
                batch_size = int(2 * deg_num_of_slices)
                slice_ids = np.random.randint(low=0, high=num_of_slices, size=deg_num_of_slices)
                self.current_slice_ids = slice_ids
                deg_slice_ids = np.arange(deg_num_of_slices)
            # image shape: [#slices, 3channels, w, h] and we sample a couple of slices (batch-size)
            #              The result is [batch-size, 3channels, w, h]
            img_slices = np.concatenate((input_3c[0][slice_ids, :, :, :], input_3c[1][deg_slice_ids, :, :, :]), axis=0)
            label_slices = np.concatenate((label[0][slice_ids], label[1][deg_slice_ids]), axis=0)
            additional_labels = np.concatenate((extra_label[0][slice_ids], extra_label[1][deg_slice_ids]), axis=0)
        else:
            # We don't balance the batch. Hence for training we just randomly pick the slices
            # and for testing we take all slices of this patient id
            # input3c is of shape [#slices, 3, w, h]
            num_of_slices = int(input_3c.shape[0])
            # if batch_size is None we set it to the total number of slices for ES/ED
            # This can happen during testing/validation
            if batch_size is None:
                batch_size = num_of_slices
            if num_of_slices < batch_size:
                # reduce batch size
                batch_size = num_of_slices
            if self.is_train:
                slice_ids = np.random.randint(low=0, high=num_of_slices, size=batch_size)
            else:
                # during testing we take the complete volume
                slice_ids = np.arange(int(num_of_slices))
            self.current_slice_ids = slice_ids
            img_slices = input_3c[slice_ids]
            label_slices = label[slice_ids]
            additional_labels = extra_label[slice_ids]

        if self.is_train:
            # randomly determine rotation: 90 * num_rotations (k-factor for np.rot90)
            num_rotations = np.random.randint(low=0, high=3, size=(1,))[0]
        else:
            # testing or validation: we take all slices for this patient
            slice_idx = np.arange(int(num_of_slices))
            num_rotations = 0

        if num_rotations != 0:
            img_slices = np.rot90(img_slices, k=num_rotations, axes=(2, 3)).copy()
        if do_permute:
            # shuffle the indices of arrays
            perm_idx = np.random.permutation(np.arange(int(batch_size)))
            img_slices = img_slices[perm_idx]
            label_slices = label_slices[perm_idx]
            additional_labels = additional_labels[perm_idx]

        return img_slices, label_slices, additional_labels, batch_size
