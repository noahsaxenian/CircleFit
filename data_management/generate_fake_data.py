import pandas as pd
import numpy as np

import plotting

frequencies = np.linspace(0, 60000, 10000)
omega = frequencies * 2 * np.pi
alpha = np.zeros(len(frequencies)) + 0j

# enter [freq, A  , n    ]
modes = [[30000, 200, 0.02]]


for mode in modes:
    w_r = mode[0] * 2 * np.pi
    alpha += mode[1] / (w_r ** 2 - omega ** 2 + 1j * mode[2] * w_r ** 2)


real = np.real(alpha)
imag = np.imag(alpha)

magnitude = np.sqrt(real**2 + imag**2)
phase = np.arctan2(imag, real)

# Create a new dataframe to store the results
fake_data = pd.DataFrame({
    'freq (Hz)': frequencies,
    'real': real,
    'complex': imag
})

plotting.plot_mag_and_phase(frequencies, magnitude, phase)
#plotting.plot_real_vs_imag(real, imag)

# Save the receptance data to a new TSV file
fake_data.to_csv('c:/Users/noahs/Documents/ceeo/modal stuff/Code/data/Plate/Plate 03/csv/fake_data_2.tsv', sep='\t', index=False)