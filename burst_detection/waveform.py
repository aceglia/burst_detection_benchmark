from bycycle import Bycycle
from matplotlib import pyplot as plt
import numpy as np
from burst_detection.enums import FrequencyBand, BurstMethod
from burst_detection.burst import Burst
from burst_detection.plot_utils import plot_bursts
from burst_detection.process_utils import bandpass_filter


class CycleByCycleConfig:
    def __init__(self, burst_method: BurstMethod = BurstMethod.CYCLES):
        self.amp_fraction = 0.3
        self.amp_consistency = 0.1
        self.period_consistency = 0.5
        self.monotonicity = 0.8
        self.min_n_cycles = 3
        self.center_extrema = "trough"
        self.burst_method = burst_method
        self.burst_fraction_threshold = 0.5  # Used only for BurstMethod.AMP

    def get_thresholds(self):
        if self.burst_method == BurstMethod.CYCLES:
            return {
                'amp_fraction': self.amp_fraction,
                'amp_consistency': self.amp_consistency,
                'period_consistency': self.period_consistency,
                'monotonicity': self.monotonicity,
                'min_n_cycles': self.min_n_cycles
            }
        elif self.burst_method == BurstMethod.AMP:
            return {
                'burst_fraction_threshold': self.burst_fraction_threshold,
                'min_n_cycles': self.min_n_cycles
            }
        else:
            raise ValueError(f"Unsupported burst method: {self.burst_method}")

    def __call__(self):
        return self.get_thresholds()



class CycleByCycleDetection:
    def __init__(self, frequency_band: tuple | FrequencyBand = None):
        self.frequency_band = frequency_band
        if isinstance(frequency_band, FrequencyBand):
            self.frequency_band = frequency_band.frequency_range()
        elif frequency_band is not None and not isinstance(frequency_band, tuple):
            raise ValueError("frequency_band must be a tuple or a FrequencyBand enum")
        self.bursts = []
        self.is_bursts_computed = False
        self.fs = None
        self.signal = None

    def detect_bursts(self, signal: np.ndarray, fs: float, config: CycleByCycleConfig) -> np.ndarray:
        self.fs = fs
        self.signal = signal
        self.filtered_signal = bandpass_filter(
            signal, lowcut=self.frequency_band[0], highcut=self.frequency_band[1], fs=fs, order=3
        )

        bm = Bycycle(
            center_extrema=config.center_extrema,
            burst_method=config.burst_method.value,
            thresholds=config(),
        )

        bm.fit(signal, fs, self.frequency_band)
        bm.df_features[bm.df_features["is_burst"]]
        side_e = "peak" if "sample_trough" in bm.df_features.columns else "trough"
        features = bm.df_features[bm.df_features["is_burst"]]
        if features.empty:
            self.bursts = []
            self.is_bursts_computed = True
            return self.bursts
        self._merge_burst_intervals(features, fs, side_e)
        self.is_bursts_computed = True
        return self.bursts

    def _merge_burst_intervals(
        self,
        df_features,
        fs,
        side_e,
    ):
        start, end = df_features[f"sample_last_{side_e}"].values, df_features[f"sample_next_{side_e}"].values
        diff = end[:-1] - start[1:]
        merge = diff == 0
        group_starts = start[np.r_[0, np.where(~merge)[0] + 1]]
        group_ends = end[np.r_[np.r_[0, np.where(~merge)[0] + 1][1:] - 1, len(start) - 1]]
        self.bursts = [self._burst_from_intervals(s, e, fs) for s, e in zip(group_starts, group_ends)]

    def _burst_from_intervals(self, starts, ends, fs):
        burst = Burst(start_time=starts / fs, end_time=ends / fs, duration=(ends - starts) / fs)
        return burst

    def plot_bursts(self, range=None, show=True):
        if not self.is_bursts_computed:
            raise RuntimeError("Bursts have not been computed yet. Call detect_bursts() first.")
        return plot_bursts(
            signal=self.filtered_signal,
            bursts=self.bursts,
            fs=self.fs,
            time=np.arange(len(self.filtered_signal)) / self.fs,
            frequency_vector=np.linspace(self.frequency_band[0], self.frequency_band[1], int((self.frequency_band[1] - self.frequency_band[0]) / 1) + 1),
            envelope=None,
            range=range,
            tfr_plot=False,
            show=show,
        )