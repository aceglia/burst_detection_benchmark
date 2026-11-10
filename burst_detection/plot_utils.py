import numpy as np
import matplotlib.pyplot as plt
from typing import List
from .burst import Burst


def plot_bursts_1d(
    signal,
    bursts,
    fs,
    time,
    frequency_vector,
    tfr=None,
    envelope=None,
    data_range=None,
    show=True,
    colorbar=True,
    tfr_plot=True,
    epoch_idx=None,
    channel_idx=None,
):
    n_rows = 2 if tfr_plot else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 6), sharex=True)
    axes = axes if n_rows > 1 else [axes]
    time_ranged = time[data_range[0]:data_range[1]] if data_range is not None else time
    signal = signal[data_range[0]:data_range[1]] if data_range is not None else signal
    envelope = envelope[data_range[0]:data_range[1]] if envelope is not None and data_range is not None else envelope
    tfr = tfr[..., data_range[0] : data_range[1]] if tfr is not None and data_range is not None else tfr
    axes[0].plot(time_ranged, signal, label="Signal", color="blue")
    if envelope is not None:
        axes[0].plot(time_ranged, envelope, label="Amplitude Envelope", color="orange")
    [
        axes[0].axvspan(burst.start_time, burst.end_time, color="red", alpha=0.3)
        for burst in bursts
        if data_range is None or (burst.start_time * fs >= data_range[0] and burst.end_time * fs <= data_range[1])
    ]
    axes[0].set_xlim(time_ranged[0], time_ranged[-1])
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Signal")
    axes[0].legend()

    if tfr_plot and tfr is not None:
        tfr = tfr.squeeze()
        frequencies = frequency_vector
        axes[1].pcolormesh(time_ranged, frequencies, tfr, shading="auto", cmap="jet")
        for burst in bursts:
            if burst.peak_time is None or burst.peak_freq is None:
                continue
            if data_range is not None:
                if burst.peak_time * fs < data_range[0] or burst.peak_time * fs > data_range[1]:
                    continue
            axes[1].scatter(burst.peak_time, burst.peak_freq, color="k", s=50, label="Detected Burst")
            axes[1].plot([burst.start_time, burst.end_time], [burst.peak_freq, burst.peak_freq], color="k", linewidth=2)
            axes[1].vlines([burst.peak_time, burst.peak_time], burst.low_freq, burst.high_freq, color="k", linewidth=2)
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Frequency (Hz)")
        axes[1].set_title("TFR")
        if colorbar:
            cbar = plt.colorbar(
                axes[1].imshow(
                    tfr,
                    extent=[time_ranged[0], time_ranged[-1], frequency_vector[0], frequency_vector[-1]],
                    aspect="auto",
                    cmap="jet",
                ),
                ax=axes[1],
            )
            cbar.set_label("Power")

    fig.suptitle(f"Detected Bursts for epoch {epoch_idx}, channel {channel_idx} in the frequency band: {frequency_vector[0]}-{frequency_vector[-1]} Hz")
    if show:
        plt.show()
    return fig, axes


def plot_bursts(
    signal,
    bursts: List[Burst],
    fs,
    time,
    frequency_vector,
    tfr=None,
    envelope=None,
    data_range=None,
    show=True,
    colorbar=True,
    tfr_plot=True,
):
    for i in range(len(signal)):
        for j in range(len(signal[i])):
            plot_bursts_1d(
                signal=signal[i, j],
                bursts=bursts[i][j],
                fs=fs,
                time=time,
                frequency_vector=frequency_vector,
                tfr=tfr[i, j] if tfr is not None else None,
                envelope=envelope[i, j] if envelope is not None else None,
                data_range=data_range,
                show=False,
                colorbar=colorbar,
                tfr_plot=tfr_plot,
                epoch_idx=i,
                channel_idx=j,
            )
    if show:
        plt.show()
