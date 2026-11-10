from burst_detection.io_utils import load_mat_file
# import matplotlib.pyplot as plt
from burst_detection.enums import BoscType, FrequencyBand, TFRMethod
from burst_detection.bosc import Bosc

if __name__ == "__main__":
    path = r"data/sub-0002_SomMotA_4_R.mat"
    data, fs = load_mat_file(path)
    bosc = Bosc(frequency_band=FrequencyBand.BETA, bosc_type=BoscType.FOOOF, log_freqs=False, step=1)
    bosc.detect_bursts(
        data,
        fs,
        tfr_method=TFRMethod.MORLET,
        power_threshold=95,
        min_duration=3,
        bp_filter_order=3,
        n_cycles=7,
        frequency_band=(1, 50),
        freq_to_exclude=[(8, 15)]
    )
    bosc.plot_bursts(data_range=(66500, 68500), show=True, colorbar=False)
