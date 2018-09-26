class BaseConfig(object):

    def __init__(self):

        # default data directory
        # remember to ADD env variable REPO_PATH on machine. REPO_PATH=<absolute path to repository >
        self.dt_map_dir = "dt_maps"
        # 1 = RV; 2 = MYO; 3 = LV; tuple(non-apex-basal inter-observ-var, apex-basal inter-observ-var)
        self.acdc_inter_observ_var = {1: [12.35, 8.15], 2: [6.95, 5.25], 3: [5.9, 4.65],
                                      5: [14.05, 9.05], 6: [7.8, 5.8], 7: [8.3, 5.65]}
        self.acdc_background_classes = [0, 4]


config_detector = BaseConfig()
