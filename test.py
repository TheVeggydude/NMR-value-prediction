import reader
import matplotlib.pylab as plt
import numpy as np
import scipy.fft

experiment = reader.PhillipsData.from_file_pair('Data/batch_2_clean/set_3A/n8951_5_1_raw_act.SPAR',
                                                'Data/batch_2_clean/set_3A/n8951_5_1_raw_act.SDAT')

pulse_1 = experiment.data[0, :500]

# Plot the pulse data.
plt.plot(pulse_1.real)
plt.plot(pulse_1.imag)
plt.plot(np.absolute(pulse_1))

plt.title('FID')
plt.legend(['Real', 'Imaginary', 'Absolute'])
plt.show()

# Perform fourier transform (real)
fft_real = scipy.fft.rfft(pulse_1.real)
fft_imag = scipy.fft.rfft(pulse_1.imag)
fft_abs = scipy.fft.rfft(np.absolute(pulse_1))

# # Roll fourier data
fft_real = np.roll(fft_real, int(len(fft_real)/2))
fft_imag = np.roll(fft_imag, int(len(fft_real)/2))
fft_abs = np.roll(fft_abs, int(len(fft_real)/2))

plt.plot(fft_real)
plt.plot(np.abs(fft_real))

plt.title('Fourier transform')
plt.legend(['rfft of real', 'abs rfft of real'])
plt.show()
