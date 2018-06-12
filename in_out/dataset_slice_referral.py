import numpy as np


class SliceReferralDataSet(object):

    dataset_size = 100

    def __init__(self, referral_results, slice_features, referral_threshold, verbose=False):
        """


        :param referral_results: ReferralResults object
        :param slice_features: SliceFeatureGenerator object
        :param referral_threshold:
        """
        self.verbose = verbose
        self.referral_threshold = referral_threshold
        self.features = slice_features
        self.targets = referral_results.img_slice_improvements[referral_threshold]
        self.num_of_features = slice_features.num_of_features

    def split_train_test_set(self, fold_id, cardiac_phase=0):
        """

        :param fold_id: possible values 0-3, determines the split between train/test sets
        :param cardiac_phase:
        :return:
        """
        if cardiac_phase not in [0, 1]:
            raise ValueError("ERROR - cardiac_phase must be equal to 0=ES or 1=ED and not {}".format(cardiac_phase))
        test_set_ids = self.features.fold_test_patient_ids[fold_id]
        test_set_size = len(test_set_ids)
        train_set_size = SliceReferralDataSet.dataset_size - test_set_size
        # prepare numpy arrays
        x_train = np.empty((0, self.num_of_features - 1))
        x_test = np.empty((0, 1))
        train_ids = np.empty(0)
        test_ids = np.empty(0)
        y_train = np.empty((0, self.num_of_features - 1))
        y_test = np.empty((0, 1))

        for patient_id, features in self.features.patient_features.iteritems():
            features_phase = features[cardiac_phase]
            y_targets = self.targets[patient_id][cardiac_phase]

            # features_phase has shape [#slices, num_of_features]
            # Note: num_of_features is +1 for the patient_id, so we cut off that column and store is separately
            ids = features_phase[:, -1]
            if patient_id not in test_set_ids:
                x_train = np.vstack([x_train, features_phase[:, :-1]]) if x_train.size else features_phase[:, :-1]
                y_train = np.hstack([y_train, y_targets]) if y_train.size else y_targets
                train_ids = np.hstack((train_ids, ids)) if train_ids.size else ids

            else:
                x_test = np.vstack([x_test, features_phase[:, :-1]]) if x_test.size else features_phase[:, :-1]
                y_test = np.hstack([y_test, y_targets]) if y_test.size else y_targets
                test_ids = np.hstack([test_ids, ids]) if test_ids.size else ids

        if self.verbose:
            print("x_train, x_test ", x_train.shape, x_test.shape)
            print("y_train, y_test ", y_train.shape, y_test.shape)

        return x_train, y_train, x_test, y_test, train_ids, test_ids

