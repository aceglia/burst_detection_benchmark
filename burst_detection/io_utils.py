from scipy.io import loadmat, savemat
import numpy as np


def load_mat_file(file_path):
    """Load a .mat file and return the data."""
    scout_mean_timecourses = loadmat(file_path)
    data = np.squeeze(scout_mean_timecourses["Value"])
    fs = 250
    return data, fs
