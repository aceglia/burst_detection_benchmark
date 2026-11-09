import mne
import numpy as np
from scipy.signal import butter, filtfilt, hilbert, welch
from .superlet import superlets
from .enums import TFRMethod
import matplotlib.pyplot as plt
import statsmodels.api as sm
from fooof import FOOOF


def compute_tfr(
    data: np.ndarray,
    method: TFRMethod = TFRMethod.MORLET,
    fs: float = 1000,
    frequency_vector: np.ndarray = None,
    fmin: float = 1,
    fmax: float = 40,
    n_cycles: int = 7,
    normalization_method: str = None,
    superlet_order=(1, 5),
    **normalization_kwargs,
):
    """
    Compute the Time-Frequency Representation (TFR) of the input data.

    Parameters:
    -----------
    - data: np.ndarray
        2D array of shape (n_channels, n_times)
    - method: TFRMethod
        method for TFR computation ('morlet' or 'multitaper')
    - fs: float
        sampling frequency of the data
    - fmin: float
        minimum frequency of interest
    - fmax: float
        maximum frequency of interest
    - n_cycles: int
        number of cycles for Morlet wavelet
    - normalization_method: str
        method to normalize TFR. Can be None (no normalization), 'median' (normalize by median), or 'papto' (normalize by the aperiodic component). Default is None.

    Returns:
    --------
    - tfr: np.ndarray
        Power of the Time-Frequency Representation (TFR) as a 3D array of shape (n_channels, n_frequencies, n_times)
    """
    if np.ndim(data) == 1:
        data = data[np.newaxis, :]
    if np.ndim(data) == 2:
        data = data[np.newaxis, :, :]
    if np.ndim(data) != 3:
        raise ValueError("Input data must be 1D, 2D, or 3D array.")

    if method == TFRMethod.MORLET:
        frequencies = (
            np.linspace(fmin, fmax + 1, int((fmax - fmin) / 0.25) + 1) if frequency_vector is None else frequency_vector
        )
        tfr = mne.time_frequency.tfr_array_morlet(
            data, sfreq=fs, freqs=frequencies, n_cycles=n_cycles, output="complex"
        )
        tfr = np.abs(tfr) ** 2  # Power of the TFR
    elif method == TFRMethod.SUPERLET:
        frequencies = (
            np.linspace(fmin, fmax + 1, int((fmax - fmin) / 0.25) + 1) if frequency_vector is None else frequency_vector
        )
        tfr = superlets(data.squeeze(), fs, frequencies, n_cycles, ord=superlet_order)
    else:
        raise ValueError("Unsupported method. Use 'morlet' or 'superlet'.")
    tfr = tfr.squeeze()  # Remove singleton dimensions if present

    # normalize as a factor of the median
    if normalization_method == "median":
        median = np.median(np.abs(tfr), axis=-1, keepdims=True)
        tfr = tfr / median

    elif normalization_method == "papto":
        # Fit the aperiodic component of the power spectrum
        fvec = np.linspace(fmin, fmax, tfr.shape[0])
        exponent, offset = aperiodic_fit(data.squeeze(), fs=fs, **normalization_kwargs, show_fit=False)
        nc = (fvec ** (exponent)) / (10**offset)
        # Normalize TFR by the aperiodic fit
        tfr = tfr * nc[..., np.newaxis]

    elif normalization_method == "stppto":
        fvec = np.linspace(fmin, fmax, tfr.shape[0])
        # apply aperiodic fit for all single time
        foof = FOOOF(
            peak_width_limits=normalization_kwargs.get("peak_width_limits", (1, 20)),
            max_n_peaks=normalization_kwargs.get("max_n_peaks", 6),
            min_peak_height=normalization_kwargs.get("min_peak_height", 0),
            peak_threshold=normalization_kwargs.get("peak_threshold", 2.0),
            aperiodic_mode=normalization_kwargs.get("mode", "knee"),
            verbose=False,
        )
        tfr = get_rp_from_ap_fit(fvec, foof, tfr).T

    elif normalization_method is not None:
        raise ValueError("Unsupported normalization method. Use None, 'median', or 'papto'.")
    return tfr


def get_rp_from_ap_fit(frequencies, foof, tfr):
    """
    Compute the expected aperiodic power (EAP) from the aperiodic fit parameters.

    Parameters:
    - frequencies: 1D array of frequency values
    - foof: FOOOF object
    - tfr: 2D array of time-frequency representation

    Returns:
    - eap: 1D array of expected aperiodic power values
    """
    def get_rp(frequencies, exponent, offset, tfr_tmp):
        return (frequencies**exponent) / (10**offset) * tfr_tmp

    def get_params(foof, tfr_tmp):
        foof.fit(frequencies, tfr_tmp)
        params = foof.get_params("aperiodic")
        return params[0], params[-1]  # exponent, offset

    def run_fit(frequencies, foof, tfr_tmp):
        try:
            return get_rp(frequencies, *get_params(foof, tfr_tmp), tfr_tmp)
        except Exception as e:
            if foof.aperiodic_mode == "knee":
                print(f"Error occurred while fitting aperiodic component: {e}.\
                    trying againt with a fixed aperiodic mode.")
                foof.aperiodic_mode = "fixed"
                rp = get_rp(frequencies, *get_params(foof, tfr_tmp), tfr_tmp)
                foof.aperiodic_mode = "knee"  # Reset to original mode
                return rp

    return np.array([run_fit(frequencies, foof, tfr[:, i]) for i in range(tfr.shape[1])])


def amplitude_envelope(data):
    """
    Compute the amplitude envelope of the input data using the Hilbert transform.

    Parameters:
    - data: 1D array of shape (n_times,)

    Returns:
    - envelope: 1D array of shape (n_times,), amplitude envelope of the input data
    """
    analytic_signal = hilbert(data)
    envelope = np.abs(analytic_signal)
    return envelope


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply a Butterworth bandpass filter to the input data.

    Parameters:
    - data: 1D array of shape (n_times,)
    - lowcut: float, low cutoff frequency
    - highcut: float, high cutoff frequency
    - fs: float, sampling frequency of the data
    - order: int, order of the Butterworth filter

    Returns:
    - filtered_data: 1D array of shape (n_times,), bandpass filtered data
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def compute_spectrum(signal: np.ndarray, fs: float, frequency_band: tuple):
    """
    Computes the power spectrum of the input signal within the specified frequency band.

    Parameters:
    -----------
    - signal: np.ndarray
        1D numpy array of the raw EEG/MEG time series.
    - fs: float
        Sampling frequency (Hz).

    - frequency_band: tuple
        The frequency band of interest as a tuple (fmin, fmax).

    Returns:
    --------
    - f: np.ndarray
        The frequency values.
    - spectrum: np.ndarray
        The computed power spectrum.
    """
    f, Pxx = welch(signal, fs=fs, nperseg=1024)
    band_mask = (f >= frequency_band[0]) & (f <= frequency_band[1])
    spectrum = Pxx[band_mask]
    return f[band_mask], spectrum


def calculate_fwhm(signal, peak_idx, coordinates):
    """
    Calculate the full width at half maximum (FWHM).

    Parameters
    ----------
    signal : ndarray
        1D signal (frequency or time).
    peak_idx : int
        Index of the local maximum.
    coordinates : ndarray
        Frequency or time coordinates.

    Returns
    -------
    ndarray
        [lower_bound, upper_bound, FWHM]
    """

    peak_power = signal[peak_idx]
    under_threshold = np.flatnonzero(signal < peak_power / 2)

    step = np.min(np.diff(coordinates))

    below = under_threshold[under_threshold < peak_idx]
    above = under_threshold[under_threshold > peak_idx]

    # Threshold crossings on both sides
    if len(below) > 0 and len(above) > 0:

        lower = coordinates[below[-1] + 1]
        upper = coordinates[above[0] - 1]
        width = upper - lower + step

    # No threshold crossing below the peak
    elif len(below) == 0 and len(above) > 0:

        lower = coordinates[0]
        upper = coordinates[above[0] - 1]
        width = 2 * (upper - coordinates[peak_idx]) + step

    # No threshold crossing above the peak
    elif len(below) > 0 and len(above) == 0:

        lower = coordinates[below[-1] + 1]
        upper = coordinates[-1]
        width = 2 * (coordinates[peak_idx] - lower) + step

    # No threshold crossing on either side
    else:

        lower = coordinates[0]
        upper = coordinates[-1]
        width = 2 * (coordinates[-1] - coordinates[0] + step)

    return np.array([lower, upper, width])


def aperiodic_fit(
    signal,
    fs,
    frequency_band=(1, 40),
    peak_width_limits=[0.5, 12],
    max_n_peaks=np.inf,
    min_peak_height=0,
    peak_threshold=2.0,
    mode="fixed",
    plot_fit=True,
    show_fit=True,
):
    """
    Fit the aperiodic component of the power spectrum using the FOOOF model.

    Parameters:
    -----------
    - signal: np.ndarray
        1D numpy array of the raw EEG/MEG time series.
    - fs: float
        Sampling frequency (Hz).
    - fmin: float
        Minimum frequency for fitting (Hz).
    - fmax: float
        Maximum frequency for fitting (Hz).

    Returns:
    --------
    - aperiodic_params: dict
        Dictionary containing the aperiodic parameters (offset, exponent).
    """
    f, Pxx = compute_spectrum(signal, fs, frequency_band=(frequency_band[0], frequency_band[1]))
    fm = FOOOF(
        peak_width_limits=peak_width_limits,
        max_n_peaks=max_n_peaks,
        min_peak_height=min_peak_height,
        peak_threshold=peak_threshold,
        aperiodic_mode=mode,
        verbose=True,
    )
    fm.report(f, Pxx, plt_log=plot_fit)
    if show_fit:
        plt.show()
    exponent = fm.get_params("aperiodic_params", "exponent")
    offset = fm.get_params("aperiodic_params", "offset")
    return exponent, offset


def bosc_bg_fit(tfr, freqs, bg_fit_method="extended"):
    """
    Fit the aperiodic component of the power spectrum using the BOSC method.

    Parameters:
    -----------
    - tfr: np.ndarray
        3D array of shape (n_channels, n_frequencies, n_times) representing the time-frequency representation.
    - freqs: np.ndarray
        1D array of frequency values corresponding to the TFR.

    Returns:
    --------
    - background_power: np.ndarray
        Estimated background power spectrum.
    """
    if bg_fit_method == "classic":
        background_power = bg_fit_linear(tfr, freqs)
    elif bg_fit_method == "extended":
        background_power = bg_fit_extended(tfr, freqs)
    elif bg_fit_method == "fooof":
        background_power = bg_fit_foof(tfr, freqs)
    else:
        raise ValueError("Unsupported background fit method. Use 'linear' or 'extended'.")
    return background_power


def bg_fit_extended(tfr, freqs):
    log_freqs = np.log10(freqs)
    log_freqs = sm.add_constant(log_freqs)
    dataForBG = np.log10(tfr.squeeze()).mean(-1)
    rlm_model = sm.RLM(dataForBG, log_freqs, M=sm.robust.norms.TukeyBiweight())
    rlm_results = rlm_model.fit()
    values = 10 ** rlm_results.predict(log_freqs)
    plt.figure("Linear Fit of Background Power Spectrum")
    plt.plot(freqs, values, label="Linear Fit", color="red")
    plt.plot(freqs, np.mean(tfr.squeeze(), axis=-1), label="Mean Power Spectrum", color="blue")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.legend()
    return values


def bg_fit_linear(tfr, freqs):
    log_freqs = np.log10(freqs)
    log_power = np.mean(np.log10(tfr.squeeze()), axis=-1)
    coefs = np.polyfit(log_freqs, log_power, 1)
    values = 10 ** np.polyval(coefs, log_freqs)
    plt.figure("Linear Fit of Background Power Spectrum")
    plt.plot(freqs, values, label="Linear Fit", color="red")
    plt.plot(freqs, np.mean(tfr.squeeze(), axis=-1), label="Mean Power Spectrum", color="blue")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.legend()
    return values


def bg_fit_foof(tfr, freqs):
    fm = FOOOF(
        aperiodic_mode="knee",
        verbose=True,
    )
    spectrum = np.mean(tfr.squeeze(), axis=-1)
    fm.report(freqs, spectrum, plt_log=True)
    background_log = fm.get_model("aperiodic")
    return 10**background_log


def exclude_rhythmic_peaks(freqs, tfr, peak_range, wavelet_cycles=7):
    to_keep = np.ones(freqs.shape, dtype=bool)
    if not isinstance(peak_range, list):
        peak_range = [peak_range]
    tfr = tfr.squeeze()
    if len(peak_range) == 0:
        return freqs, tfr
    else:
        for freq_tmp in peak_range:
            low, high = freq_tmp
            # Indices within the specified exclusion range
            freq_indices = np.where((freqs >= low) & (freqs <= high))[0]
            if len(freq_indices) == 0:
                continue
            # Find empirical peak within the range
            mean_bg_within_range = tfr[freq_indices, :].mean(axis=1)
            peak_idx = freq_indices[np.argmax(mean_bg_within_range)]
            # Approximate wavelet extension (FWHM)
            peak_freq = freqs[peak_idx]

            low_freq = peak_freq - peak_freq / wavelet_cycles
            up_freq = peak_freq + peak_freq / wavelet_cycles

            # Exclude expanded frequency range
            to_keep[(freqs >= low_freq) & (freqs <= up_freq)] = False
        excluded = ~to_keep
    return freqs[to_keep], tfr[to_keep, :], excluded
