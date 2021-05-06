import numpy as np
import scipy.fft


def fourier(a, part=None, roll=True, abs=False):
    """

    :param a:
    :param part:
    :param roll:
    :param abs:
    :return:
    """

    # Fourier transform
    result = None
    if part == "all":                   # standard fourier
        result = scipy.fft.fft(a)
    else:                               # real-only fourier results
        result = scipy.fft.rfft(a)
        if part == "imag":
            result = np.flip(result)    # reverse for imaginary fourier results

    

    # Check for rolling
    np.roll(fft_real, int(len(fft_real) / 2))