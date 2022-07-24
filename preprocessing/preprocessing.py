import numpy as np


def center_fft(x):
    return np.roll(np.abs(x), int(len(x) / 2))


def as_absolute(x):
    return np.asarray([np.absolute(experiment) for experiment in x])


def preprocess(data, ground):

    # Get max value of data
    norm_scalar = np.amax(data)

    # For each experiment in the dataset
    for i, experiment in enumerate(data):

        data[i] = np.asarray([experiment / norm_scalar])  # scale data
        ground[i] = np.asarray([ground[i] / norm_scalar])  # scale ground truth

    return data, ground
