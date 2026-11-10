import numpy as np
from burst_detection.enums import FrequencyBand, BurstMethod

class BurstFeatures:
    def __init__(self, signal: np.ndarray, fs: float, frequency_band: FrequencyBand = None):
        self.tfr = None
        if signal.ndim == 1:
            self.signal = signal[np.newaxis, :]
        self.signal = signal
        self.fs = fs
        self.frequency_band = frequency_band.frequency_range() if isinstance(frequency_band, FrequencyBand) else frequency_band

    def split_epochs(self, epoch_length: float):
        """
        Splits the signal into epochs of specified length.

        Parameters:
        -----------
        epoch_length: float
            Length of each epoch in seconds.

        Returns:
        --------
        epochs: list of np.ndarray
            List containing the epochs.
        """
        if self.signal.ndim == 2 and self.signal.shape[0] > 1:
            raise ValueError("Signal should be a 2D array (channels x samples) or a 3D array with a signle epoch to be processed.")
        
        num_samples_per_epoch = int(epoch_length * self.fs)
        num_epochs = len(self.signal) // num_samples_per_epoch
        epochs = [
            self.signal[:, i * num_samples_per_epoch : (i + 1) * num_samples_per_epoch]
            for i in range(num_epochs)
        ]
        return epochs

    def get_burst_rate(self, bursts: list) -> float:
        """
        Computes the burst rate (bursts per second) based on the detected bursts.

        Parameters:
        -----------
        bursts: list
            List of detected bursts.

        Returns:
        --------
        burst_rate: float
            Burst rate in bursts per second.
        """
        if not bursts:
            return 0.0

        total_duration = len(self.signal) / self.fs  # Total duration of the signal in seconds
        burst_rate = len(bursts) / total_duration  # Burst rate in bursts per second
        return burst_rate

    def get_burst_volume(self, bursts: list) -> float:
        """
        Computes the total burst volume based on the detected bursts.

        Parameters:
        -----------
        bursts: list
            List of detected bursts.

        Returns:
        --------
        total_volume: float
            Total burst volume.
        """
        total_volume = 0.0
        for burst in bursts:
            if burst.volume is not None:
                total_volume += burst.volume
        return total_volume