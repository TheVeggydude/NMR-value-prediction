import phillips_reader
import matplotlib.pylab as plt
import numpy as np
import scipy.fft

experiment = phillips_reader.PhillipsData.from_file_pair('set_3A/n8951_5_1_raw_act.SPAR',
                                                'set_3A/n8951_5_1_raw_act.SDAT')

pulse_1 = experiment.data[0, :512]

# Plot the pulse data.
plt.plot(pulse_1.real)
plt.plot(pulse_1.imag)
plt.plot(np.absolute(pulse_1))

plt.title('FID')
plt.legend(['Real', 'Imaginary', 'Absolute'])
plt.show()

# # Perform fourier transform (real)
# fft_real = scipy.fft.rfft(pulse_1.real)
# fft_imag = scipy.fft.rfft(pulse_1.imag)
# fft_abs = scipy.fft.rfft(np.absolute(pulse_1))
#
# # # Roll fourier data
# fft_real = np.roll(fft_real, int(len(fft_real)/2))
# fft_imag = np.roll(fft_imag, int(len(fft_real)/2))
# fft_abs = np.roll(fft_abs, int(len(fft_real)/2))
#
# plt.plot(fft_real)
# plt.plot(np.abs(fft_real))
#
# plt.title('Fourier transform')
# plt.legend(['rfft of real', 'abs rfft of real'])
# plt.show()

# The actual FFT as used by NMR folk.
fft_complete = scipy.fft.fft(pulse_1)

# Roll the data so the Phosphorus-  131 peak is in the center
fft_complete = np.roll(fft_complete, int(len(fft_complete)/2))

plt.plot(np.abs(fft_complete))
plt.title("FFT complex values pulse_1")
plt.show()
