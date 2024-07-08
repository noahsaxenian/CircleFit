import pandas as pd
from circle_fit import CircleFit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import peak_finder
from interactive_circle_fit import InteractiveCircleFit
from interactive_peak_finder import InteractivePeakFinder

# Load your FRF data from a CSV file
file_path = 'C:/Users/noahs/Documents/ceeo/modal stuff/Siemens Plate Test/point1_data_receptance.tsv'
data = pd.read_csv(file_path, delimiter='\t')

regenerated_file_path = 'C:/Users/noahs/Documents/ceeo/modal stuff/Siemens Plate Test/point6_regenerated_receptance.tsv'
siemens_fit = pd.read_csv(file_path, delimiter='\t')

# Filter data to desired frequency range
freq_range = [1000, 1800]
filtered_data = data[(data['freq (Hz)'] >= freq_range[0]) & (data['freq (Hz)'] <= freq_range[1])]
siemens_fit = siemens_fit[(siemens_fit['freq (Hz)'] >= freq_range[0]) & (siemens_fit['freq (Hz)'] <= freq_range[1])]

# Find peaks and ranges
#peaks, peak_ranges = peak_finder.peak_ranges(filtered_data, distance=10, prominence=10)
p = InteractivePeakFinder(filtered_data)
peaks = p.peaks
peak_ranges = p.ranges


# Create array of circle fits for each peak
modes = [CircleFit(data, peak, freq_range=freq_range) for peak, freq_range in zip(peaks, peak_ranges)]
for mode in modes:
    mode.run()

# Interactive plot to help choose best range of points for each peak
interactive_fit = InteractiveCircleFit(modes)

# Generate simulated plot
frequencies = np.linspace(freq_range[0], freq_range[1], (freq_range[1]-freq_range[0])*10)
omega = frequencies * 2 * np.pi
alpha = np.zeros(len(frequencies)) + 0j

for mode in modes:
    alpha += mode.A / (mode.omega ** 2 - omega ** 2 + 1j * mode.damping * mode.omega ** 2)


real = np.real(alpha)
imag = np.imag(alpha)
magnitude = np.sqrt(real**2 + imag**2)
phase = np.arctan2(imag, real)

data_freqs = filtered_data['freq (Hz)'].values
data_real = filtered_data['real'].values
data_imag = filtered_data['complex'].values
data_mag = np.sqrt(data_real**2 + data_imag**2)
data_phase = np.arctan2(data_imag, data_real)

siemens_freqs = siemens_fit['freq (Hz)']
siemens_real = siemens_fit['real']
siemens_imag = siemens_fit['complex']
siemens_mag = np.sqrt(siemens_real**2 + siemens_imag**2)
siemens_phase = np.arctan2(siemens_imag, siemens_real)



def plot_mag_and_phase():
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(4, 1)
    # Create the first subplot (3/4 of the area)
    ax1 = fig.add_subplot(gs[0:3, 0])
    ax1.plot(frequencies, magnitude, label='Simulated')
    ax1.plot(data_freqs, data_mag, 'x', label='Experimental')
    #ax1.plot(siemens_freqs, siemens_mag, label='Siemens Fit')
    ax1.set_xlabel('Frequency')
    ax1.set_ylabel('Magnitude')
    ax1.legend()
    ax1.grid(True)

    # Create the second subplot (1/4 of the area)
    ax2 = fig.add_subplot(gs[3, 0])
    ax2.plot(frequencies, phase, label='Simulated')
    ax2.plot(data_freqs, data_phase, 'x', label='Experimental')
    #ax2.plot(siemens_freqs, siemens_phase, label='Siemens Fit')
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Phase')
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
    ax1.plot(frequencies, real, label='Simulated')
    ax1.plot(data_freqs, data_real, 'x', label='Experimental')
    #ax1.plot(siemens_freqs, siemens_mag, label='Siemens Fit')
    ax1.set_xlabel('Frequency')
    ax1.set_ylabel('Real')
    ax1.legend()
    ax1.grid(True)

    # Create the second subplot (1/4 of the area)
    ax2 = fig.add_subplot(gs[2:4, 0])
    ax2.plot(frequencies, imag, label='Simulated')
    ax2.plot(data_freqs, data_imag, 'x', label='Experimental')
    #ax2.plot(siemens_freqs, siemens_phase, label='Siemens Fit')
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Imaginary')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()


plot_mag_and_phase()
#plot_real_and_imag()