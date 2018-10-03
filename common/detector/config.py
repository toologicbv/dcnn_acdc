class BaseConfig(object):

    def __init__(self):

        # default data directory
        # remember to ADD env variable REPO_PATH on machine. REPO_PATH=<absolute path to repository >
        self.dt_map_dir = "dt_maps"
        # ES: 1 = RV; 2 = MYO; 3 = LV; tuple(non-apex-basal inter-observ-var, apex-basal inter-observ-var)
        # ED: 5 = RV; 6 = MYO; 7 = LV
        self.acdc_inter_observ_var = {1: [14.05, 9.05], 2: [7.8, 5.8], 3: [8.3, 5.65],  # ES
                                      5: [12.35, 8.15], 6: [6.95, 5.25], 7: [5.9, 4.65]}  # ED
        self.acdc_background_classes = [0, 4]
        # TODO We don't know size of padding yet. Depends on model architecture!
        self.acdc_pad_size = 20


config_detector = BaseConfig()
