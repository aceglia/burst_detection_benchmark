import numpy as np


class Burst:
    def __init__(self, start_time, end_time, peak_freq=None, peak_time=None, peak_amplitude=None, low_freq=None, high_freq=None):
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.start_time = start_time
        self.end_time = end_time
        self.peak_freq = peak_freq
        self.peak_time = peak_time
        self.peak_amplitude = peak_amplitude
        self.amplitude_vector = None  # Initialize amplitude_vector to None

    def to_dict(self):
        return {
            "low_freq": self.low_freq,
            "high_freq": self.high_freq,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "freq_span": self.freq_span,
            "duration": self.duration,
            "peak_freq": self.peak_freq,
            "peak_time": self.peak_time,
            "peak_amplitude": self.peak_amplitude,
            "amplitude_vector": self.amplitude_vector,
        }

    @property
    def freq_span(self):
        if self.low_freq is not None and self.high_freq is not None:
            return self.high_freq - self.low_freq
        else:
            return None

    @property
    def duration(self):
        return self.end_time - self.start_time
    
    def is_bursting(self, time_range):
        if self.peak_time is None:
            return False
        return time_range[0] <= self.peak_time <= time_range[1]

    @property
    def volume(self):
        if self.freq_span is not None and self.duration is not None and self.amplitude_vector is not None:
            mean_centered_signal = self.amplitude_vector - np.mean(self.amplitude_vector)
            return np.trapz(mean_centered_signal, dx=1.0) * self.freq_span * self.duration

        else:
            return None

    def __repr__(self):
        return (
            f"FrequencyBurst(low_freq={self.low_freq:.3f}, high_freq={self.high_freq:.3f}, "
            f"start_time={self.start_time:.3f}, end_time={self.end_time:.3f}, freq_span={self.freq_span:.3f}, duration={self.duration:.3f}, peak_freq={self.peak_freq:.3f}, peak_time={self.peak_time:.3f}, peak_amplitude={self.peak_amplitude:.3f})"
        )

    @classmethod
    def from_dict(cls, burst_dict):
        return cls(
            low_freq=burst_dict["low_freq"],
            high_freq=burst_dict["high_freq"],
            start_time=burst_dict["start_time"],
            end_time=burst_dict["end_time"],
            peak_freq=burst_dict["peak_freq"],
            peak_time=burst_dict["peak_time"],
            peak_amplitude=burst_dict["peak_amplitude"],
        )
