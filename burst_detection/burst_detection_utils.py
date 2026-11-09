from .process_utils import calculate_fwhm, aperiodic_fit, bosc_bg_fit
import numpy as np
from .burst import Burst
from skimage.morphology import h_maxima
from scipy.stats import chi2
from scipy.ndimage import label
import statsmodels as sm


def detect_bursts_tfr(
    tfr, prominence_factor=0.05, fs=250, min_duration=0.05, min_frequency_width=2, threshold=None, freqs=None
):
    """
    Detect beta events using thresholded time-frequency regions.

    Parameters
    ----------
    tfr : ndarray
        Complex TFR, shape (frequencies, times),
        or power TFR with the same shape.

    prominence_factor : float
        Factor to multiply by the maximum TFR value to determine the minimum prominence.
    fs : float
        Sampling frequency of the signal (Hz).
    min_duration : float
        Minimum duration of a burst in seconds.
    min_frequency_width : float
        Minimum frequency width of a burst in Hz.
    freqs : ndarray
        Frequencies corresponding to the TFR, shape (frequencies,).

    Returns
    -------
    burst_list : list of burst class
        List of detected bursts, each represented as a burst class
    """

    tfr = np.asarray(tfr).squeeze()
    minimum_prominence = prominence_factor * np.max(tfr)
    mask_h = h_maxima(tfr, h=minimum_prominence)
    peakf, peakt = np.where(mask_h.squeeze())

    power = tfr[peakf, peakt]

    freqs = np.asarray(freqs).squeeze() if freqs is not None else np.arange(tfr.shape[0])
    bursts = []
    for f, t, p in zip(peakf, peakt, power):
        if threshold is not None and p < threshold:
            continue
        lowerf, upperf, widthf = calculate_fwhm(tfr[:, t], f, freqs)
        lowert, uppert, widtht = calculate_fwhm(tfr[f, :], t, np.arange(tfr.shape[1]))
        lowert = lowert / fs
        uppert = uppert / fs
        widtht = widtht / fs
        if (
            widthf <= min_frequency_width
            or widtht <= min_duration
            or widthf <= min_frequency_width
            or widtht <= min_duration
        ):
            continue
        bursts.append(
            Burst(
                low_freq=lowerf,
                high_freq=upperf,
                start_time=lowert,
                end_time=uppert,
                freq_width=widthf,
                duration=widtht,
                peak_freq=freqs[f],
                peak_time=t / fs,
                peak_amplitude=p,
            )
        )
    return bursts


def detect_bursts_timeseries(signal, fs, high_thresh, low_thresh, min_duration):
    # Find the start and end indices of bursts based on the high and low thresholds
    start_idxs = np.flatnonzero((signal[1:] >= high_thresh) & (signal[:-1] < high_thresh)) + 1
    end_idxs = np.flatnonzero((signal[1:] < low_thresh) & (signal[:-1] >= low_thresh)) + 1

    if signal[0] >= high_thresh and 0 not in start_idxs:
        start_idxs = np.insert(start_idxs, 0, 0)

    if signal[-1] >= low_thresh and len(signal) - 1 not in end_idxs:
        end_idxs = np.append(end_idxs, len(signal) - 1)

    # Ensure that each start index has a corresponding end index
    end_positions = np.searchsorted(end_idxs, start_idxs, side="right")
    _, idx = np.unique(end_positions, return_index=True)
    keep = np.zeros(len(end_positions), dtype=bool)
    keep[idx] = True

    # Remove duplicate end positions
    valid = keep & (end_positions < len(end_idxs))
    end_idxs = end_idxs[end_positions[valid]]
    start_idxs = start_idxs[valid]

    # backfill start indices with the last index where the signal was below the low threshold
    idxs_under = np.flatnonzero(signal < low_thresh)
    if len(idxs_under):
        previous_under = np.maximum.accumulate(np.where(signal < low_thresh, np.arange(len(signal)), -1))
        start_idxs = np.where(
            previous_under[start_idxs] >= 0,
            previous_under[start_idxs] + 1,
            start_idxs,
        )

    # interpaolate the start and end indices to get a more accurate estimate of the time of the burst
    # Interpolate burst start (crossing low_thresh upwards)
    start_idxs_float = (
        start_idxs - 1 + (low_thresh - signal[start_idxs - 1]) / (signal[start_idxs] - signal[start_idxs - 1])
    )

    # Interpolate burst end (crossing low_thresh downwards)
    end_idxs_float = end_idxs - 1 + (low_thresh - signal[end_idxs - 1]) / (signal[end_idxs] - signal[end_idxs - 1])
    start_times = start_idxs_float / fs
    end_times = end_idxs_float / fs
    duration_sec = end_times - start_times

    # compute the duration of each burst in seconds and filter out bursts that are shorter than the minimum duration
    valid = duration_sec >= min_duration
    start_idxs = start_idxs[valid]
    end_idxs = end_idxs[valid]
    start_times = start_times[valid]
    end_times = end_times[valid]
    duration_sec = duration_sec[valid]
    time_vector = np.arange(len(signal)) / fs

    # Fill in the burst list with the detected bursts
    bursts = [
        Burst(
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            peak_freq=None,
            peak_time=time_vector[signal == np.max(signal[start_idx:end_idx])],
            peak_amplitude=np.max(signal[start_idx:end_idx]),
        )
        for start_idx, end_idx, start_time, end_time, duration in zip(
            start_idxs, end_idxs, start_times, end_times, duration_sec
        )
    ]
    return bursts


def bosc_burst_det(tfr, freqs, fs, power_threshold=95, n_cycles=3, background_power=None):
    if background_power is None:
        raise ValueError("background_power must be provided for burst detection.")
    chi2_value = chi2.ppf(power_threshold / 100, 2)
    power_threshold = background_power * chi2_value / 2

    significant_power = tfr > power_threshold[None, None, :, None]
    duration_threshold = np.ceil(n_cycles * fs / freqs).astype(int)
    padded = np.pad(significant_power.astype(int), ((0, 0), (0, 0), (0, 0), (1, 1)), mode="constant", constant_values=0)
    padded = padded.squeeze()
    tfr = tfr.squeeze()
    bursts = []
    for f, freq in enumerate(freqs):
        minimum_duration = duration_threshold[f]
        x = significant_power[0, 0, f, :].astype(int)
        padded = np.pad(x, (1, 1), constant_values=0)
        dx = np.diff(padded)
        pos = np.where(dx == 1)[0]  # starts, 0-based
        neg = np.where(dx == -1)[0] # ends, exclusive
        for start, end in zip(pos, neg):
            episode_span = end - start - 1
            if episode_span >= minimum_duration:        
                peak_amplitude = np.max(tfr[f, start:end])
                peak_time = np.argmax(tfr[f, start:end]) + start
                bursts.append(
                    Burst(
                        start_time=start / fs,
                        end_time=end / fs,
                        duration=episode_span / fs,
                        peak_freq=freq,
                        peak_time=peak_time / fs,
                        peak_amplitude=peak_amplitude,
                    )
                )
    return bursts
