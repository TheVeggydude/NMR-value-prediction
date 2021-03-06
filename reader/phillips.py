import numpy as np
from ast import literal_eval


class PhillipsData:
    def __init__(self, params, data):
        """
        Creates a PhillipsData object from experiment parameters and data points.
        :param params: SPAR file data, converted to a dictionary.
        :param data: SDAT file data, converted to a 2D array of imaginary numbers.
        """
        self.params = params
        self.data = data

    @classmethod
    def from_file_pair(cls, spar_file, sdat_file):
        """
        Creates a data object directly from data source files.
        :param spar_file: String path to the SPAR file.
        :param sdat_file: String path to the SDAT file.
        :return: PhillipsData object.
        """
        spar_params = read_spar(spar_file)
        data = read_sdat(sdat_file,
                         spar_params['samples'],
                         spar_params['rows'])

        return cls(spar_params, data)

    def real(self):
        """
        Returns only the real parts of the imaginary numbers in the data.
        :return: 2D NumPy array of real numbers.
        """
        return self.data.real

    def imaginary(self):
        """
        Returns only the imaginary parts of the imaginary numbers in the data.
        :return: 2D NumPy array of imaginary numbers.
        """
        return self.data.imag

    def absolute(self):
        """
        Returns the (computed) absolutes of the imaginary numbers in the data.
        :return: 2D NumPy array of computed absolute numbers.
        """
        return np.absolute(self.data)


def read_spar(filename):
    """Read the .spar file.
    :param filename: file path

    :return: dict of parameters read from spar file
    :rtype: dict
    """

    parameter_dict = {}
    with open(filename, 'r') as f:
        for line in f:
            # ignore comments (!) and empty lines
            if line == "\n" or line.startswith("!"):
                continue

            # Handle
            key, value = map(str.strip, line.split(":", 1))
            try:
                val = literal_eval(value)
            except (ValueError, SyntaxError):
                if value == '':
                    val = None
                else:
                    val = value

            parameter_dict.update({key: val})

    return parameter_dict


def read_sdat(filename, samples, rows):
    """Read the .sdat file.
    :param filename: File path
    :param samples: Number of spectral points
    :param rows: Number of rows of data
    """
    with open(filename, 'rb') as f:
        raw = f.read()

    floats = _vax_to_ieee_single_float(raw)
    data_iter = iter(floats)
    complex_iter = (complex(r, i) for r, i in zip(data_iter, data_iter))
    raw_data = np.fromiter(complex_iter, "complex64")
    raw_data = np.reshape(raw_data, (rows, samples)).squeeze()

    return raw_data


# From VESPA - BSD license.
def _vax_to_ieee_single_float(data):
    """Converts a float in Vax format to IEEE format.

    data should be a single string of chars that have been read in from
    a binary file. These will be processed 4 at a time into float values.
    Thus the total number of byte/chars in the string should be divisible
    by 4.

    Based on VAX data organization in a byte file, we need to do a bunch of
    bitwise operations to separate out the numbers that correspond to the
    sign, the exponent and the fraction portions of this floating point
    number

    role :      S        EEEEEEEE      FFFFFFF      FFFFFFFF      FFFFFFFF
    bits :      1        2      9      10                               32
    bytes :     byte2           byte1               byte4         byte3

    This is taken from the VESPA project source code under a BSD licence.
    """
    f = []
    n_float = int(len(data) / 4)
    for i in range(n_float):

        byte2 = data[0 + i * 4]
        byte1 = data[1 + i * 4]
        byte4 = data[2 + i * 4]
        byte3 = data[3 + i * 4]

        # hex 0x80 = binary mask 10000000
        # hex 0x7f = binary mask 01111111

        sign = (byte1 & 0x80) >> 7
        exponent = ((byte1 & 0x7f) << 1) + ((byte2 & 0x80) >> 7)
        fracture = ((byte2 & 0x7f) << 16) + (byte3 << 8) + byte4

        if sign == 0:
            sign_mult = 1.0
        else:
            sign_mult = -1.0

        if 0 < exponent:
            # note 16777216.0 == 2^24
            val = sign_mult * (0.5 + (fracture / 16777216.0)) * pow(2.0, exponent - 128.0)
            f.append(val)
        elif exponent == 0 and sign == 0:
            f.append(0)
        else:
            f.append(0)
            # may want to raise an exception here ...

    return f
