import numpy as np
import scipy.fft


def center_fft(x):
    return np.roll(np.abs(x), int(len(x) / 2))


def as_absolute(x):
    return np.asarray([np.absolute(experiment) for experiment in x])


def preprocess(data, ground, v=2, transpose=True):
    if v == 2:

        if transpose:
            data = np.asarray([
                as_absolute(  # make complex values absolute.
                    np.transpose(exp)  # transpose experiments.
                ) for exp in data
            ])

        # Get max value of data
        norm_scalar = np.amax(data)

        # For each experiment in the dataset
        for i, experiment in enumerate(data):

            data[i] = np.asarray([experiment / norm_scalar])  # scale data
            ground[i] = np.asarray([ground[i] / norm_scalar])  # scale ground truth

        # Shuffle axes to have the same order as the data axes.
        ground = ground.transpose((0, 2, 1))

    else:
        print("Preprocess functionality not implemented for given version number!")

    return data, ground
