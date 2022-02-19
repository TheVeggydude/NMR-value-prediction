import numpy as np
import os


def read_csv(dir_str):
    """
    Reads .csv contents of a particular directory, if it exists.
    :param dir_str: String of the absolute path to the directory.
    :returns : Numpy array containing csv data.
    """

    ground_truth = np.genfromtxt(dir_str + "/" + "ground_truth.csv", delimiter=',')
    return np.asarray(ground_truth)


def read_dat_directory(dir_str, r_shape=(-1, 1000)):
    """
    Reads the .dat file contents of a particular directory, if it exists.
    :param dir_str: String of the absolute path to the directory.
    :param r_shape: the desired reshape dimensions, defaults for usage with fid files.
    :returns : Numpy array containing all .dat data.
    """

    directory = os.fsencode(dir_str)
    data = []

    dat_files = [file for file in os.listdir(directory) if os.fsdecode(file).endswith(".dat")]
    for file in dat_files:
        data.append(
            read_dat(dir_str + "/" + os.fsdecode(file), r_shape)
        )

    return np.asarray(data)


def read_dat(filename, r_shape):
    """
    Reads a Numpy array from a given .dat file.
    :param filename: String with absolute path to file.
    :param r_shape: the desired reshape dimensions for the array.
    :returns : Numpy array for a single .dat file.
    """

    raw = np.fromfile(filename, np.complex64)
    return raw.reshape(r_shape)
