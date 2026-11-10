from .process_utils import calculate_fwhm
import numpy as np
from .burst import Burst
from skimage.morphology import h_maxima
from scipy.stats import chi2


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
    bursts = {}
    for i in range(tfr.shape[0]):
        bursts[i] = {}
        for j in range(tfr.shape[1]):
            bursts[i][j] = []
            tfr_1d = tfr[i, j]
            minimum_prominence = prominence_factor * np.max(tfr_1d)
            mask_h = h_maxima(tfr_1d, h=minimum_prominence)
            peakf, peakt = np.where(mask_h.squeeze())

            power = tfr_1d[peakf, peakt]

            freqs = np.asarray(freqs).squeeze() if freqs is not None else np.arange(tfr_1d.shape[0])
            for f, t, p in zip(peakf, peakt, power):
                if threshold is not None and p < threshold:
                    continue
                lowerf, upperf, widthf = calculate_fwhm(tfr_1d[:, t], f, freqs)
                lowert, uppert, widtht = calculate_fwhm(tfr_1d[f, :], t, np.arange(tfr_1d.shape[1]))
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
                bursts[i][j].append(
                    Burst(
                        low_freq=lowerf,
                        high_freq=upperf,
                        start_time=lowert,
                        end_time=uppert,
                        peak_freq=freqs[f],
                        peak_time=t / fs,
                        peak_amplitude=p,
                    )
                )
    return bursts


def detect_bursts_timeseries(signal, fs, high_thresh, low_thresh, min_duration):
    # Find the start and end indices of bursts based on the high and low thresholds
    # signal shape: (n_epochs, n_channels, n_samples)

    starts = (signal[..., 1:] >= high_thresh) & (signal[..., :-1] < high_thresh)
    ends = (signal[..., 1:] < low_thresh) & (signal[..., :-1] >= low_thresh)

    # Add transitions at the first and last samples
    starts = np.concatenate([
        (signal[..., :1] >= high_thresh),
        starts
    ], axis=-1)

    ends = np.concatenate([
        ends,
        (signal[..., -1:] >= low_thresh)
    ], axis=-1)

    # Extract indices along the last axis
    start_idxs = np.where(starts)
    end_idxs = np.where(ends)
    bursts = {}
    for i in range(signal.shape[0]):
        bursts[i] = {}
        for j in range(signal.shape[1]):
            bursts[i][j] = []
            signal_1d = signal[i, j, :]
            starts_1d = start_idxs[-1][(start_idxs[0] == i) & (start_idxs[1] == j)]
            ends_1d = end_idxs[-1][(end_idxs[0] == i) & (end_idxs[1] == j)]
            low_thresh_1d = low_thresh[i, j]
            # Ensure that each start index has a corresponding end index
            end_positions = np.searchsorted(ends_1d, starts_1d, side="right")
            _, idx = np.unique(end_positions, return_index=True)
            keep = np.zeros(len(end_positions), dtype=bool)
            keep[idx] = True

            # Remove duplicate end positions
            valid = keep & (end_positions < len(ends_1d))
            ends_1d = ends_1d[end_positions[valid]]
            starts_1d = starts_1d[valid]

            # backfill start indices with the last index where the signal was below the low threshold
            idxs_under = np.flatnonzero(signal_1d < low_thresh_1d)
            if len(idxs_under):
                previous_under = np.maximum.accumulate(np.where(signal_1d < low_thresh_1d, np.arange(len(signal_1d)), -1))
                starts_1d = np.where(
                    previous_under[starts_1d] >= 0,
                    previous_under[starts_1d] + 1,
                    starts_1d,
                )

            # interpaolate the start and end indices to get a more accurate estimate of the time of the burst
            # Interpolate burst start (crossing low_thresh upwards)
            start_1d_float = (
                starts_1d - 1 + (low_thresh_1d - signal_1d[starts_1d - 1]) / (signal_1d[starts_1d] - signal_1d[starts_1d - 1])
            )

            # Interpolate burst end (crossing low_thresh downwards)
            end_1d_float = ends_1d - 1 + (low_thresh_1d - signal_1d[ends_1d - 1]) / (signal_1d[ends_1d] - signal_1d[ends_1d - 1])
            start_times = start_1d_float / fs
            end_times = end_1d_float / fs
            duration_sec = end_times - start_times

            # compute the duration of each burst in seconds and filter out bursts that are shorter than the minimum duration
            valid = duration_sec >= min_duration
            starts_1d = starts_1d[valid]
            ends_1d = ends_1d[valid]
            start_times = start_times[valid]
            end_times = end_times[valid]
            duration_sec = duration_sec[valid]
            time_vector = np.arange(len(signal_1d)) / fs

            # Fill in the burst list with the detected bursts
            bursts[i][j] = [Burst(
                    start_time=start_time,
                    end_time=end_time,
                    peak_time=time_vector[signal_1d == np.max(signal_1d[start_idx:end_idx])],
                    peak_amplitude=np.max(signal_1d[start_idx:end_idx]),
                ) for start_idx, end_idx, start_time, end_time in zip(starts_1d, ends_1d, start_times, end_times)]
    return bursts


def bosc_burst_det(tfr, freqs, fs, power_threshold=95, n_cycles=3, background_power=None):
    if background_power is None:
        raise ValueError("background_power must be provided for burst detection.")
    chi2_value = chi2.ppf(power_threshold / 100, 2)
    power_threshold = background_power * chi2_value / 2

    significant_power = tfr > power_threshold[None, None, :, None]
    duration_threshold = np.ceil(n_cycles * fs / freqs).astype(int)
    padded = np.pad(significant_power.astype(int), ((0, 0), (0, 0), (0, 0), (1, 1)), mode="constant", constant_values=0)
    bursts = {}
    for i in range(tfr.shape[0]):
        bursts[i] = {}
        for j in range(tfr.shape[1]):
            bursts[i][j] = []
            for f, freq in enumerate(freqs):
                minimum_duration = duration_threshold[f]
                x = significant_power[i, j, f, :].astype(int)
                padded = np.pad(x, (1, 1), constant_values=0)
                dx = np.diff(padded)
                pos = np.where(dx == 1)[0]  # starts, 0-based
                neg = np.where(dx == -1)[0]  # ends, exclusive
                for start, end in zip(pos, neg):
                    episode_span = end - start - 1
                    if episode_span >= minimum_duration:
                        peak_amplitude = np.max(tfr[i, j, f, start:end])
                        peak_time = np.argmax(tfr[i, j, f, start:end]) + start
                        bursts[i][j].append(
                            Burst(
                                start_time=start / fs,
                                end_time=end / fs,
                                peak_freq=freq,
                                peak_time=peak_time / fs,
                                peak_amplitude=peak_amplitude,
                            )
                        )
    return bursts

def freq_span_computation(tfr, time_span, fs, freqs, bursts):
    """
    Compute the frequency span of each burst based on the TFR and the time span.

    Parameters
    ----------
    tfr : ndarray
        Complex TFR, shape (frequencies, times),
        or power TFR with the same shape.
    time_span : tuple
        Tuple containing the start and end times of the burst (in seconds).
    fs : float
        Sampling frequency of the signal (Hz).
    freqs : ndarray
        Frequencies corresponding to the TFR, shape (frequencies,).
    bursts : list of Burst
        List of detected bursts.

    Returns
    -------
    bursts : list of Burst
        List of detected bursts with updated frequency spans.
    """
    for burst in bursts:
        start_idx = int(burst.start_time * fs)
        end_idx = int(burst.end_time * fs)
        tfr_segment = tfr[:, start_idx:end_idx]
        threshold = tfr[:, start_idx] + 2 * tfr[:, start_idx:end_idx].std()
        freq_indices = np.where(tfr_segment > 0)[0]
        if len(freq_indices) > 0:
            burst.low_freq = freqs[freq_indices[0]]
            burst.high_freq = freqs[freq_indices[-1]]
        else:
            burst.low_freq = None
            burst.high_freq = None

    return bursts