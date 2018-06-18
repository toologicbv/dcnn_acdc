import os
import dill


class Patients(object):

    """
        should be situated in repository/dcnn_acdc/data/Folds/

        example to create dill file and then later load the object
        from in_out.patient_classification import Patient

        patient = Patient.create_dict(exper_handler.exper.config.data_dir)
        patient = Patient()
        patient.load(exper_handler.exper.config.data_dir)
        print(patient.category.keys())

    """
    file_name = "patient_cardiac_disease_classes.txt"

    def __init__(self):
        self.category = {}

    def load(self, path_to_fold_root_dir):
        file_name = os.path.join(path_to_fold_root_dir, Patients.file_name.replace("txt", "dll"))
        if not os.path.isfile(file_name):
            p = Patients.create_dict(path_to_fold_root_dir)
            self.category = p.category
        try:
            with open(file_name, 'rb') as f:
                self.category = dill.load(f)
        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
            print("ERROR - Can't open file {}".format(file_name))
            raise IOError

    @staticmethod
    def create_dict(root_fold_dir):
        patient_disease_dict = {}
        file_name = os.path.join(root_fold_dir, Patients.file_name)
        with open(file_name) as fp:
            for line in fp:
                line = line.rstrip()
                patient_id, pclass = line.split(",")
                patient_id = patient_id.replace('"', '')
                pclass = pclass.replace('"', '').replace(" ", "")
                patient_disease_dict[patient_id] = pclass

            patient = Patients()
            patient.category = patient_disease_dict
            outfile = os.path.join(root_fold_dir, file_name.replace("txt", "dll"))
            try:
                with open(outfile, 'wb') as f:
                    dill.dump(patient_disease_dict, f)
                print("INFO - Saved results to {}".format(outfile))
            except IOError as e:
                print "I/O error({0}): {1}".format(e.errno, e.strerror)
                print("ERROR - can't save results to {}".format(outfile))

        return patient
