papto_knee = {
        "min_duration":0.05,
        "power_threshold":2,
        "min_frequency_width":1,
        "bp_filter_order":3,
        "n_cycles":7,
        "prominence_factor":0.05,
        "normalization_method":"papto",
        "mode":"knee",
        "peak_width_limits":(1, 20),
        "max_n_peaks":6,
        "frequency_band":(1, 50), # for the aperiodic fit
}

papto_fixed = {
        "min_duration":0.1,
        "power_threshold":8,
        "min_frequency_width":1,
        "bp_filter_order":3,
        "n_cycles":7,
        "prominence_factor":0.1,
        "normalization_method":"papto",
        "mode":"fixed",
        "peak_width_limits":(1, 20),
        "max_n_peaks":6,
        "frequency_band":(1, 50), # for the aperiodic fit
}

stppto_fixed = {
        "min_duration":0.01,
        "power_threshold":0,
        "min_frequency_width":1,
        "bp_filter_order":3,
        "n_cycles":7,
        "prominence_factor":0.05,
        "normalization_method":"stppto",
        "mode":"fixed",
        "peak_width_limits":(2, 10),
        "max_n_peaks":4,
        "frequency_band":(1, 50), # for the aperiodic fit
}

stppto_knee = {
        "min_duration":0.05,
        "power_threshold":0,
        "min_frequency_width":1,
        "bp_filter_order":3,
        "n_cycles":7,
        "prominence_factor":0.05,
        "normalization_method":"stppto",
        "mode":"knee",
        "peak_width_limits":(1, 20),
        "max_n_peaks":6,
        "frequency_band":(1, 50), # for the aperiodic fit
}

median = {
        "min_duration":0.05,
        "power_threshold":8,
        "min_frequency_width":1,
        "bp_filter_order":3,
        "n_cycles":7,
        "prominence_factor":0.02,
        "normalization_method":"median",
}


def get_parameters(config):
        if config == "papto_knee":
                return papto_knee
        elif config == "papto_fixed":
                return papto_fixed
        elif config == "median":
                return median
        elif config == "stppto_fixed":
                return stppto_fixed
        elif config == "stppto_knee":
                return stppto_knee
        else:
                raise ValueError("Invalid config")
