import numpy as np
import scipy.fft


def center_fft(x):
    return np.roll(np.abs(x), int(len(x) / 2))


def normalize_experiment(x):
    norm = np.linalg.norm(x)
    return x/norm


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

        # For each experiment in the dataset
        for i, experiment in enumerate(data):

            # Get maximal value
            norm_scalar = np.amax(experiment)

            data[i] = np.asarray([experiment / norm_scalar])  # scale data
            ground[i] = np.asarray([ground[i] / norm_scalar])  # scale ground truth

        # Shuffle axes to have the same order as the data axes.
        ground = ground.transpose((0, 2, 1))

    elif v == 1:
        data = np.apply_along_axis(scipy.fft.fft, 2, data)  # perform fast Fourier transform
        data = np.apply_along_axis(np.flip, 2, data)  # flip data so ATP is on the right side
        data = np.apply_along_axis(center_fft, 2, data)  # center PCR peak in the middle
        data = np.asarray([normalize_experiment(experiment) for experiment in data])  # normalize per experiment

    else:
        print("Preprocess functionality not implemented for given version number!")

    return data, ground
