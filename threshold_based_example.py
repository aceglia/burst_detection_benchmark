from burst_detection.enums import FrequencyBand, ThresholdMethod, TFRMethod
from burst_detection.io_utils import load_mat_file
from burst_detection.threshold import TimeDomainThresholding, FrequencyDomainThresholding
from config import get_parameters

if __name__ == "__main__":
    path = r"data/sub-0002_SomMotA_4_R.mat"
    data, fs = load_mat_file(path)
    data = data[66000:70_000]  # Select a specific segment of the data for analysis
    freq_band = FrequencyBand.BETA

    threshold_meth = TimeDomainThresholding(frequency_band=freq_band)
    threshold_meth.detect_bursts(data, fs, threshold_method=ThresholdMethod.SD, high_thresh=3, low_thresh=1)
    threshold_meth.plot_bursts(range=(500, 2500), show=False)

    thresh_freq = FrequencyDomainThresholding(frequency_band=freq_band, step=0.25)
    thresh_freq.detect_bursts(data, fs, tfr_method=TFRMethod.SUPERLET, **get_parameters("stppto_fixed"), superlet_order=(1, 5))
    thresh_freq.plot_bursts(range=(500, 2500), show=True, colorbar=False)
