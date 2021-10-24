import numpy as np
import os


def read_dat_directory(dir_str):
    """
    Reads the .dat file and ground_truth.csv contents of a particular directory, if it exists.
    :param dir_str: String of the absolute path to the directory.
    :returns : Tuple of two numpy arrays -> (data, ground_truth)
    """
    directory = os.fsencode(dir_str)

    data = []
    ground_truth = np.genfromtxt(dir_str + "/" + "ground_truth.csv", delimiter=',')

    dat_files = [file for file in os.listdir(directory) if os.fsdecode(file).endswith(".dat")]
    for file in dat_files:
        data.append(
            read_fid(dir_str + "/" + os.fsdecode(file))
        )

    return np.asarray(data), np.asarray(ground_truth)


def read_fid(filename):
    """
    Reads a numpy array from a given .dat file.
    :param filename: String with absolute path to file.

    """

    raw = np.fromfile(filename, np.complex64)
    return raw.reshape(-1, 1000)
