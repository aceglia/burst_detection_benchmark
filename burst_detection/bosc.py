from .enums import FrequencyBand, BoscType, TFRMethod
from .process_utils import compute_tfr, bandpass_filter, bosc_bg_fit, exclude_rhythmic_peaks, check_signal_dims
from .plot_utils import plot_bursts
import numpy as np
from .burst_detection_utils import bosc_burst_det


class Bosc:
    def __init__(
        self,
        frequency_band: tuple | FrequencyBand = None,
        n_freqs_steps: int = None,
        step: float = None,
        bosc_type=BoscType.CLASSIC,
        log_freqs: bool = False,
    ):
        self.frequency_band = frequency_band
        if isinstance(frequency_band, FrequencyBand):
            self.frequency_band = frequency_band.frequency_range()
        elif frequency_band is not None and not isinstance(frequency_band, tuple):
            raise ValueError("frequency_band must be a tuple or a FrequencyBand enum")
        self.bursts = []
        self.is_bursts_computed = False
        self.bosc_type = bosc_type
        self.log_freqs = log_freqs
        self.n_freqs_steps = n_freqs_steps
        self.step = step
        if self.n_freqs_steps is None and self.step is None:
            raise ValueError("Either n_freqs_steps or step must be provided.")
        elif self.n_freqs_steps is not None and self.step is not None:
            raise ValueError("Only one of n_freqs_steps or step should be provided.")
        if self.step is not None and log_freqs is False:
            self.n_freqs_steps = int((self.frequency_band[1] - self.frequency_band[0]) / self.step) + 1
        if log_freqs and bosc_type == BoscType.FOOF:
            raise ValueError("log_freqs cannot be True when using BoscType.FOOF.")
        
        # init other attributes
        self.fs = None
        self.n_cycles = None
        self.tfr_padding = None
        self.detection_padding = None
        self.reference_padding = None
        self.background = None
        self.signal = None

    def detect_bursts(
        self,
        signal: np.ndarray,
        fs: float,
        tfr_method: TFRMethod = TFRMethod.MORLET,
        bp_filter_order=3,
        power_threshold=95,  # percentile threshold for power in percent
        min_duration=3,  # in number of cycle
        n_cycles=7,
        normalization_method=None,
        freq_to_exclude=(),
        **normalization_kwargs,
    ) -> np.ndarray:
        signal = check_signal_dims(signal)
        if not isinstance(min_duration, int) or min_duration <= 0:
            raise ValueError("min_duration must be a positive integer representing the number of cycles.")

        self.fs = fs
        self.n_cycles = n_cycles
        if "frequency_band" in normalization_kwargs:
            fmin, fmax = normalization_kwargs["frequency_band"]
            n_freqs_steps = int((fmax - fmin) / self.step) + 1

        else:
            fmin, fmax = self.frequency_band
            n_freqs_steps = self.n_freqs_steps

        freq_funct = np.logspace if self.log_freqs else np.linspace
        freqs = (
            freq_funct(np.log10(fmin), np.log10(fmax), n_freqs_steps)
            if self.log_freqs
            else np.linspace(fmin, fmax, n_freqs_steps)
        )
        self.signal = signal
        self.filtered_signal = bandpass_filter(
            signal, lowcut=self.frequency_band[0], highcut=self.frequency_band[1], fs=fs, order=bp_filter_order
        )

        self.frequency_vector = freqs
        self.tfr = compute_tfr(
            self.signal,
            tfr_method,
            fs,
            fmin=fmin,
            fmax=fmax,
            n_cycles=n_cycles,
            normalization_method=normalization_method,
            frequency_vector=self.frequency_vector,
            **normalization_kwargs,
        )
        excluded = np.zeros_like(self.frequency_vector, dtype=bool)
        if len(freq_to_exclude) > 0 and self.bosc_type == BoscType.EXTENDED:
            # freqs, tfr, excluded = exclude_rhythmic_peaks(self.frequency_vector, self.tfr, freq_to_exclude, wavelet_cycles=n_cycles)
            print("NOT IMPLEMENTED: Excluding frequency ranges is not yet implemented for EXTENDED BoscType.")
            freqs, tfr = self.frequency_vector, self.tfr
        elif len(freq_to_exclude) > 0 and self.bosc_type != BoscType.EXTENDED:
            print(
                "WARNING: You specified frequency range to exclude but didn't choose the EXTENDED bosc type. It will be ignored."
            )
            freqs, tfr = self.frequency_vector, self.tfr
        else:
            freqs, tfr = self.frequency_vector, self.tfr

        background_power = bosc_bg_fit(tfr.transpose(2, 3, 0, 1).reshape(tfr.shape[2], -1), freqs, self.bosc_type.value)
        common_idx = (self.frequency_vector >= self.frequency_band[0]) & (
            self.frequency_vector <= self.frequency_band[-1]
        )
        self.tfr = self.tfr[..., common_idx, :]
        self.frequency_vector = self.frequency_vector[common_idx]
        background_power = background_power[common_idx[excluded == False]]
        self.bursts = bosc_burst_det(
            self.tfr,
            freqs=self.frequency_vector,
            fs=self.fs,
            power_threshold=power_threshold,
            n_cycles=min_duration,
            background_power=background_power,
        )
        self.is_bursts_computed = True
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
