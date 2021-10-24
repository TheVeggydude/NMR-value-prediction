import reader
import matplotlib.pylab as plt
import numpy as np
import scipy.fft


def center_fft(x):
    return np.roll(np.abs(x), int(len(x) / 2))


def normalize_experiment(x):
    norm = np.linalg.norm(x)
    return x/norm


# DATA ACQUISITION
directory = 'simulation/Batch Sun 24 Oct 2021 13:42:22'
data, ground_truth = reader.read_dat_directory(directory)

# PREPROCESSING
data = np.apply_along_axis(scipy.fft.fft, 2, data)  # perform fast Fourier transform
data = np.apply_along_axis(np.flip, 2, data)  # flip data so ATP is on the right side
data = np.apply_along_axis(center_fft, 2, data)  # center PCR peak in the middle

data = [normalize_experiment(experiment) for experiment in data]  # normalize within a single experiment
