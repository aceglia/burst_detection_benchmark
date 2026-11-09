class Burst:
    def __init__(self, start_time, end_time, duration, peak_freq=None, peak_time=None, peak_amplitude=None, freq_width=None, low_freq=None, high_freq=None):
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.start_time = start_time
        self.end_time = end_time
        self.freq_width = freq_width
        self.duration = duration
        self.peak_freq = peak_freq
        self.peak_time = peak_time
        self.peak_amplitude = peak_amplitude

    def to_dict(self):
        return {
            "low_freq": self.low_freq,
            "high_freq": self.high_freq,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "freq_width": self.freq_width,
            "duration": self.duration,
            "peak_freq": self.peak_freq,
            "peak_time": self.peak_time,
            "peak_amplitude": self.peak_amplitude,
        }

    def __repr__(self):
        return (
            f"FrequencyBurst(low_freq={self.low_freq:.3f}, high_freq={self.high_freq:.3f}, "
            f"start_time={self.start_time:.3f}, end_time={self.end_time:.3f}, freq_width={self.freq_width:.3f}, duration={self.duration:.3f}, peak_freq={self.peak_freq:.3f}, peak_time={self.peak_time:.3f}, peak_amplitude={self.peak_amplitude:.3f})"
        )

    @classmethod
    def from_dict(cls, burst_dict):
        return cls(
            low_freq=burst_dict["low_freq"],
            high_freq=burst_dict["high_freq"],
            start_time=burst_dict["start_time"],
            end_time=burst_dict["end_time"],
            freq_width=burst_dict["freq_width"],
            duration=burst_dict["duration"],
            peak_freq=burst_dict["peak_freq"],
            peak_time=burst_dict["peak_time"],
            peak_amplitude=burst_dict["peak_amplitude"],
        )
