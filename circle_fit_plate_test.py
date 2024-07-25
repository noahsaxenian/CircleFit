import pandas as pd
from circle_fit import CircleFit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import peak_finder
from interactive_circle_fit import InteractiveCircleFit
from interactive_peak_finder import InteractivePeakFinder

# Load your FRF data from a CSV file
file_path = 'C:/Users/noahs/Documents/ceeo/modal stuff/Siemens Plate Test/point1_data.tsv'
#file_path = 'C:/Users/noahs/Documents/ceeo/modal stuff/Code/data/Plate/Plate 04/csv/Plate 04 H_001_trf.tsv'
data = pd.read_csv(file_path, delimiter='\t')

# Filter data to desired frequency range
freq_range = [610, 1510]
filtered_data = data[(data['freq (Hz)'] >= freq_range[0]) & (data['freq (Hz)'] <= freq_range[1])]

# Find peaks and ranges
#peaks, peak_ranges = peak_finder.peak_ranges(filtered_data, distance=10, prominence=10)
p = InteractivePeakFinder(filtered_data)
peaks = p.peaks
peak_ranges = p.ranges


# Create array of circle fits for each peak
#modes = [CircleFit(data, peak, freq_range=freq_range) for peak, freq_range in zip(peaks, peak_ranges)]
modes = [CircleFit(data, peak) for peak in peaks]

# Interactive plot to help choose best range of points for each peak
interactive_fit = InteractiveCircleFit(modes)

# Generate simulated plot
frequencies = np.linspace(freq_range[0], freq_range[1], (freq_range[1]-freq_range[0])*10)
omega = frequencies * 2 * np.pi
alpha = np.zeros(len(frequencies)) + 0j

for mode in modes:
    alpha += mode.A / (mode.omega_r ** 2 - omega ** 2 + 1j * mode.damping * mode.omega_r ** 2)

# ### mass and stiffness residuals calculation
# data_comparison = filtered_data['real'].values + 1j * filtered_data['complex'].values
# M_residual = 1 / ((omega[0]**2) * (alpha[0] - data_comparison[0]))
# alpha_mass_corrected = alpha - (1/(M_residual * omega**2))
#
# K_residual = 1 / (data_comparison[-1] - alpha_mass_corrected[-1])
# print(M_residual, K_residual)
#
# alpha_corrected = alpha_mass_corrected #+ (1/K_residual)

### residuals as pseudo modes
data_comparison = filtered_data['real'].values + 1j * filtered_data['complex'].values
#choice of natural frequencies of pseudo modes
# omega_r1 = (freq_range[0] - 100) * 2 * np.pi
# omega_r2 = (freq_range[-1] + 100) * 2 * np.pi
omega_r1 = 532*2*np.pi
omega_r2 = 1718*2*np.pi

k1 = 1 / ((1 - (omega[0]**2/omega_r1**2)) * (data_comparison[0] - alpha[0]))
k1_average = np.mean([1 / ((1 - (omega[i]**2 / omega_r1**2)) * (data_comparison[i] - alpha[i])) for i in range(3)])

mode1 = ((1/k1_average) / (1 - (omega**2 / omega_r1**2)))
alpha_cor_1 = alpha + mode1

k2 = 1 / ((1 - (omega[-1]**2/omega_r2**2)) * (data_comparison[-1] - alpha_cor_1[-1]))
k2_average = np.mean([1 / ((1 - (omega[-i]**2 / omega_r2**2)) * (data_comparison[-i] - alpha_cor_1[-i])) for i in range(1, 4)])

mode2 = ((1/k2_average) / (1 - (omega**2 / omega_r2**2)))

alpha_corrected = alpha + mode1 + mode2

# recorrected
alpha_cor_2 = alpha + mode2
k1_average = np.mean([1 / ((1 - (omega[i]**2 / omega_r1**2)) * (data_comparison[i] - alpha_cor_2[i])) for i in range(3)])
mode1 = ((1/k1_average) / (1 - (omega**2 / omega_r1**2)))
alpha_cor_1 = alpha + mode1
k2_average = np.mean([1 / ((1 - (omega[-i]**2 / omega_r2**2)) * (data_comparison[-i] - alpha_cor_1[-i])) for i in range(1, 4)])
mode2 = ((1/k2_average) / (1 - (omega**2 / omega_r2**2)))

alpha_corrected = alpha + mode1 + mode2


real = np.real(alpha)
imag = np.imag(alpha)
magnitude = np.sqrt(real**2 + imag**2)
phase = np.arctan2(imag, real)

real_cor = np.real(alpha_corrected)
imag_cor = np.imag(alpha_corrected)
magnitude_cor = np.sqrt(real_cor**2 + imag_cor**2)
phase_cor = np.arctan2(imag_cor, real_cor)

data_freqs = filtered_data['freq (Hz)'].values
data_real = filtered_data['real'].values
data_imag = filtered_data['complex'].values
data_mag = np.sqrt(data_real**2 + data_imag**2)
data_phase = np.arctan2(data_imag, data_real)


def plot_mag_and_phase():
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(4, 1)
    # Create the first subplot (3/4 of the area)
    ax1 = fig.add_subplot(gs[0:3, 0])
    ax1.plot(data_freqs, data_mag, label='Experimental')
    ax1.plot(frequencies, magnitude, label='Simulated')
    ax1.plot(frequencies, magnitude_cor, label='With Residuals')
    #ax1.plot(siemens_freqs, siemens_mag, label='Siemens Fit')
    ax1.set_xlabel('Frequency')
    ax1.set_ylabel('Magnitude')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True)

    # Create the second subplot (1/4 of the area)
    ax2 = fig.add_subplot(gs[3, 0])
    ax2.plot(data_freqs, data_phase, label='Experimental')
    ax2.plot(frequencies, phase, label='Simulated')
    ax2.plot(frequencies, phase_cor, label='With Residuals')
    #ax2.plot(siemens_freqs, siemens_phase, label='Siemens Fit')
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Phase')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()

def plot_real_and_imag():
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(4, 1)
    # Create the first subplot (3/4 of the area)
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax1.plot(frequencies, real, color='#ff7f0e', label='Simulated')
    ax1.plot(data_freqs, data_real, color='#1f77b4', label='Experimental')
    #ax1.plot(siemens_freqs, siemens_mag, label='Siemens Fit')
    ax1.set_xlabel('Frequency')
    ax1.set_ylabel('Real')
    #ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True)

    # Create the second subplot (1/4 of the area)
    ax2 = fig.add_subplot(gs[2:4, 0])
    ax2.plot(frequencies, imag, color='#ff7f0e', label='Simulated')
    ax2.plot(data_freqs, data_imag, color='#1f77b4', label='Experimental')
    #ax2.plot(siemens_freqs, siemens_phase, label='Siemens Fit')
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Imaginary')
    #ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()


plot_mag_and_phase()
#plot_real_and_imag()