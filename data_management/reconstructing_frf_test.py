import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')

freqs = [670, 893, 1068, 1130, 1305, 1456]
omega_rs = np.array(freqs) * 2 * np.pi
amplitudes = np.array([0.613, -0.117, -1.130, 0.593, 0.266, 0.252])
phases = np.array([23.7, -2.4, -13.8, -8.4, -35, -51])
phases_rad = np.deg2rad(phases)
As = amplitudes * np.exp(1j * phases_rad)
Q = [274, 456, 111, 214, 991, 271]
etas = 1/np.array(Q)
freq_range = [600, 1600]
frequencies = np.linspace(freq_range[0], freq_range[1],
                                  (freq_range[1] - freq_range[0]) * 10)
omegas = frequencies * 2 * np.pi

file_path = 'C:/Users/noahs/Documents/ceeo/modal stuff/Siemens Plate Test/point1_data.tsv'
data = pd.read_csv(file_path, delimiter='\t')
filtered_data = data[(data['freq (Hz)'] >= freq_range[0]) & (data['freq (Hz)'] <= freq_range[1])]
data_freqs = filtered_data['freq (Hz)'].values
data_real = filtered_data['real'].values
data_imag = filtered_data['complex'].values
data_mag = np.abs(data_real + 1j * data_imag)

accelerance = np.zeros(len(frequencies)) + 0j
for A, omega_r, eta in zip(As, omega_rs, etas):
    accelerance += (-omegas**2 * A) / (omega_r ** 2 - omegas ** 2 + 1j * eta * omega_r**2)

accelerance2 = np.zeros(len(frequencies)) + 0j
for A, omega_r, eta in zip(As, omega_rs, etas):
    accelerance2 += (-omegas**2 * A) / (omega_r**2 - omegas ** 2 + 1j * eta * omega_r * omegas)

plt.figure(figsize=(12, 6))
plt.plot(data_freqs, data_mag, label='Experimental')
plt.plot(frequencies, np.abs(accelerance), label='omega_r**2')
plt.plot(frequencies, np.abs(accelerance2), '--', label='omega_r*omegas')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()