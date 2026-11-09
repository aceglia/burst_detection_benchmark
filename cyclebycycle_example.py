from burst_detection.waveform import CycleByCycleDetection, CycleByCycleConfig
import numpy as np
from burst_detection.io_utils import load_mat_file
from burst_detection.enums import FrequencyBand, BurstMethod


if __name__ == "__main__":
    config = CycleByCycleConfig(burst_method=BurstMethod.CYCLES)
    config.amp_fraction = 0.3
    config.amp_consistency = 0.1
    config.period_consistency = 0.5
    config.monotonicity = 0.8
    config.min_n_cycles = 3
    config.center_extrema = "peak"
    config.burst_fraction_threshold = 1  # Used only for BurstMethod.AMP


    path = r"data/sub-0002_SomMotA_4_R.mat"
    signal, fs = load_mat_file(path)
    signal = signal[66000:70_000]  # Select a specific segment of the data for analysis
    cycle_by_cycle_detector = CycleByCycleDetection(frequency_band=FrequencyBand.BETA)
    cycle_by_cycle_detector.detect_bursts(signal, fs, config)
    cycle_by_cycle_detector.plot_bursts(range=(500, 2500), show=True)