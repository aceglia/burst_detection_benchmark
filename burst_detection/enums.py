from enum import Enum

class FrequencyBand(Enum):
    """
    Enum for frequency types.
    """
    BETA = "beta"
    GAMMA = "gamma"
    DELTA = "delta"
    THETA = "theta"

    def __str__(self):
        return self.value

    def frequency_range(self):
        """
        Returns the frequency range for the given frequency type.
        """
        if self == FrequencyBand.BETA:
            return (13, 30)
        elif self == FrequencyBand.GAMMA:
            return (30, 100)
        elif self == FrequencyBand.DELTA:
            return (0.5, 4)
        elif self == FrequencyBand.THETA:
            return (4, 8)
        else:
            raise ValueError(f"Unknown frequency band: {self}")

class ThresholdMethod(Enum):
    """
    Enum for thresholding methods.
    """
    SD = "sd"
    PERCENTILE = "percentile"

class TFRMethod(Enum):
    """
    Enum for Time-Frequency Representation (TFR) methods.
    """
    MORLET = "morlet"
    SUPERLET = "superlet"

    def __str__(self):
        return self.value

class BoscType(Enum):
    """
    Enum for BOSC (Better Oscillation Detection) types.
    """
    CLASSIC = "classic"
    EXTENDED = "extended"
    FOOOF = "fooof"

    def __str__(self):
        return self.value

class BurstMethod(Enum):
    """
    Enum for burst detection methods.
    """
    CYCLES = "cycles"
    AMP = "amp"

    def __str__(self):
        return self.value