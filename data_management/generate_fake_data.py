import pandas as pd
import numpy as np

import plotting

frequencies = np.linspace(0, 2000, 2000)
omega = frequencies * 2 * np.pi
alpha = np.zeros(len(frequencies)) + 0j

A_mag = 0.0873
A_phase = np.pi
A_real = A_mag * np.cos(A_phase)
A_imag = A_mag * np.sin(A_phase)

B_mag = 0.0873
B_phase = 0.5716
B_real = B_mag * np.cos(B_phase)
B_imag = B_mag * np.sin(B_phase)

# enter [freq, A  , n    ]
modes = [[500, A_real + 1j*A_imag, 0.02],
         [800, B_real + 1j*B_imag, 0.02]]

for mode in modes:
    w_r = mode[0] * 2 * np.pi
    alpha += mode[1] / (w_r ** 2 - omega ** 2 + 1j * mode[2] * w_r ** 2)


real = np.real(alpha)
imag = np.imag(alpha)

magnitude = np.sqrt(real**2 + imag**2)
phase = np.arctan2(imag, real)
plotting.plot_mag_and_phase(frequencies, magnitude, phase)

# Create a new dataframe to store the results
fake_data = pd.DataFrame({
    'freq (Hz)': frequencies,
    'real': real,
    'complex': imag
})


#plotting.plot_real_vs_imag(real, imag)

# Save the receptance data to a new TSV file
fake_data.to_csv('c:/Users/noahs/Documents/ceeo/modal stuff/Code/data/Plate/Plate 03/csv/fake_data_2.tsv', sep='\t', index=False)