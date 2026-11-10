from .enums import FrequencyBand, ThresholdMethod, TFRMethod
from .process_utils import bandpass_filter, amplitude_envelope, compute_tfr, check_signal_dims
from .burst_detection_utils import detect_bursts_tfr, detect_bursts_timeseries
import numpy as np
from .plot_utils import plot_bursts


class Thresholding:
    def __init__(self, frequency_band: tuple | FrequencyBand = None):
        self.frequency_band = frequency_band
        if isinstance(frequency_band, FrequencyBand):
            self.frequency_band = frequency_band.frequency_range()
        elif frequency_band is not None and not isinstance(frequency_band, tuple):
            raise ValueError("frequency_band must be a tuple or a FrequencyBand enum")
        self.bursts = []
        self.is_bursts_computed = False


class TimeDomainThresholding(Thresholding):
    def __init__(self, frequency_band: tuple | FrequencyBand = None):
        super().__init__(frequency_band)
        self.filtered_signal = None
        self.amplitude_envelope = None
        self.fs = None

    def detect_bursts(
        self,
        signal: np.ndarray,
        fs: float,
        threshold_method: ThresholdMethod = ThresholdMethod.SD,
        high_thresh=3.0,
        low_thresh=None,
        min_duration=0.05,
        bp_filter_order=3,
    ) -> np.ndarray:
        """
        Detects beta bursts using a dual-amplitude threshold.

        Parameters:
        -----------
        signal: np.ndarray
          1D 2D or 3Dnumpy array (n_epochs, n_channels, n_times) of the raw EEG/MEG time series.
        fs: float
          Sampling frequency (Hz).
        threshold_method: ThresholdMethod
          Method for thresholding ('sd' for standard deviation, 'percentile' for percentile-based).
        high_thresh: float | int
          Higher threshold (in units of median amplitude or percentiles in percent) to trigger a burst.
        low_thresh: float | int
          Lower threshold (in units of median amplitude or percentiles in percent) to end a burst.
        min_duration: float
          Minimum burst duration in seconds to filter out transient noise.
        bp_filter_order: int
            Order of the bandpass filter.

        Returns:
        --------
        - self.amplitude_envelope: np.ndarray
            The raw amplitude envelope.
        """
        signal = check_signal_dims(signal)
        self.is_bursts_computed = True
        self.filtered_signal = bandpass_filter(
            signal, lowcut=self.frequency_band[0], highcut=self.frequency_band[1], fs=fs, order=bp_filter_order
        )
        self.amplitude_envelope = amplitude_envelope(self.filtered_signal)
        self.fs = fs
        median_amp = np.median(self.amplitude_envelope, axis=-1, keepdims=True)
        low_thresh = low_thresh if low_thresh is not None else high_thresh

        high_thresh = (
            high_thresh * median_amp
            if threshold_method == ThresholdMethod.SD
            else np.percentile(self.amplitude_envelope, high_thresh, axis=-1, keepdims=True)
        )
        low_thresh = low_thresh * median_amp if threshold_method == ThresholdMethod.SD else high_thresh
        self.threshold = (low_thresh, high_thresh)
        self.bursts = detect_bursts_timeseries(self.amplitude_envelope, fs, high_thresh, low_thresh, min_duration)
        return self.amplitude_envelope

    def get_bursts(self):
        return self.bursts

    def plot_bursts(self, data_range=None, show=True):
        if not self.is_bursts_computed:
            raise RuntimeError("Bursts have not been computed yet. Call detect_beta_bursts() first.")

        return plot_bursts(
            signal=self.filtered_signal,
            bursts=self.bursts,
            fs=self.fs,
            time=np.arange(self.filtered_signal.shape[-1]) / self.fs,
            frequency_vector=np.linspace(self.frequency_band[0], self.frequency_band[1], 1),
            envelope=self.amplitude_envelope,
            data_range=data_range,
            tfr_plot=False,
            show=show,
        )


class FrequencyDomainThresholding(Thresholding):
    def __init__(self, frequency_band: tuple, step: float = 0.25):
        super().__init__(frequency_band)
        self.tfr = None
        self.signal = None
        self.bursts = None
        self.step = step

    def detect_bursts(
        self,
        signal: np.ndarray,
        fs: float,
        tfr_method: TFRMethod = TFRMethod.MORLET,
        power_threshold=5,
        min_duration=0.05,
        min_frequency_width=2,
        bp_filter_order=3,
        n_cycles=7,
        prominence_factor=0.05,
        normalization_method=None,
        **normalization_kwargs,
    ) -> np.ndarray:
        """
        Detects bursts in the frequency domain based on a specified threshold.

        Parameters:
        -----------
        signal: np.ndarray
          1D numpy array of the raw EEG/MEG time series.
        fs: float
          Sampling frequency (Hz).
        min_duration: float
          Minimum burst duration in seconds to filter out transient noise.
        min_frequency_width: float
          Minimum frequency width in Hz to filter out narrowband noise.
        bp_filter_order: int
          Order of the bandpass filter.
        n_cycles: int
          Number of cycles for the Morlet wavelet.
        prominence_factor: float
            Factor to determine the prominence of peaks in the TFR for burst detection. Based on the maximum value of the TFR, the prominence threshold is set to prominence_factor * max(TFR).
        normalization_method: str
            Method to normalize TFR. Can be None (no normalization), 'median' (normalize by median), or 'papto' (normalize by the aperiodic component). Default is False.
        Returns:
        --------
        self.spectrum: np.ndarray
            The input spectrum.
        """
        self.fs = fs
        signal = check_signal_dims(signal)
        self.signal = signal
        self.filtered_signal = bandpass_filter(
            signal, lowcut=self.frequency_band[0], highcut=self.frequency_band[1], fs=fs, order=bp_filter_order
        )
        if "frequency_band" in normalization_kwargs and normalization_kwargs["frequency_band"] != self.frequency_band:
            freq_band_tmp = normalization_kwargs["frequency_band"]
        else:
            freq_band_tmp = self.frequency_band
        self.frequency_vector = np.linspace(
            freq_band_tmp[0],
            freq_band_tmp[1],
            int((freq_band_tmp[1] - freq_band_tmp[0]) / self.step) + 1,
        )

        self.tfr = compute_tfr(
            self.signal,
            tfr_method,
            fs,
            fmin=freq_band_tmp[0],
            fmax=freq_band_tmp[1],
            n_cycles=n_cycles,
            normalization_method=normalization_method,
            frequency_vector=self.frequency_vector,
            **normalization_kwargs,
        )
        # recompute vector using the origin processing frequency band 
        common_idx = (self.frequency_vector >= self.frequency_band[0]) & (self.frequency_vector <= self.frequency_band[-1])
        self.tfr = self.tfr[..., common_idx, :]
        self.frequency_vector = self.frequency_vector[common_idx]

        self.bursts = detect_bursts_tfr(
            tfr=self.tfr,
            prominence_factor=prominence_factor,
            threshold=power_threshold,
            fs=fs,
            min_duration=min_duration,
            min_frequency_width=min_frequency_width,
            freqs=self.frequency_vector,
        )
        return self.tfr

    def plot_bursts(self, data_range=None, show=True, colorbar=False):
        if self.tfr is None:
            raise RuntimeError("Bursts have not been computed yet. Call detect_bursts() first.")
        return plot_bursts(
            signal=self.filtered_signal,
            bursts=self.bursts,
            fs=self.fs,
            time=np.arange(self.filtered_signal.shape[-1]) / self.fs,
            frequency_vector=self.frequency_vector,
            tfr=self.tfr,
            tfr_plot=True,
            data_range=data_range,
            show=show,
            colorbar=colorbar,
        )
