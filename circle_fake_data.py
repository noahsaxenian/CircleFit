import pandas as pd
from circle_fit import CircleFit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from peak_finder import peak_ranges
from interactive_plot import InteractiveCircleFit

# Load your FRF data from a CSV file
file_path = 'c:/Users/noahs/Documents/ceeo/modal stuff/Code/data/Plate/Plate 03/csv/fake_data_1.tsv'
data = pd.read_csv(file_path, delimiter='\t')


#fake data
# freqs = [500]
# points = [10]

freq_range = [0, 1000]

filtered_data = data[(data['freq (Hz)'] >= freq_range[0]) & (data['freq (Hz)'] <= freq_range[1])]

peaks, peak_ranges = peak_ranges(filtered_data, prominence = 0.0001)


modes = []

for i in range(len(peaks)):
    mode = CircleFit(data, peaks[i], freq_range=peak_ranges[i])
    mode.run()
    interactive_fit = InteractiveCircleFit(mode)
    interactive_fit.show()
    modes.append(mode)

frequencies = np.linspace(freq_range[0], freq_range[1], freq_range[1]-freq_range[0])
omega = frequencies * 2 * np.pi
alpha = np.zeros(len(frequencies)) + 0j

for mode in modes:
    alpha += mode.A / (mode.omega**2 - omega**2 + 1j * mode.damping * mode.omega**2)

real = np.real(alpha)
imag = np.imag(alpha)
magnitude = np.sqrt(real**2 + imag**2)
phase = np.arctan2(imag, real)

data_freqs = filtered_data['freq (Hz)']
data_real = filtered_data['real']
data_imag = filtered_data['complex']
data_mag = np.sqrt(data_real**2 + data_imag**2)
data_phase = np.arctan2(data_imag, data_real)


#plot
def plot_mag_and_phase():
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(4, 1)
    # Create the first subplot (3/4 of the area)
    ax1 = fig.add_subplot(gs[0:3, 0])
    ax1.plot(frequencies, magnitude, label='Simulated')
    ax1.plot(data_freqs, data_mag, 'x', label='Experimental')
    ax1.set_xlabel('Frequency (ω)')
    ax1.set_ylabel('Magnitude')
    ax1.set_title('Magnitude')
    ax1.legend()
    ax1.grid(True)

    # Create the second subplot (1/4 of the area)
    ax2 = fig.add_subplot(gs[3, 0])
    ax2.plot(frequencies, phase, label='Simulated')
    ax2.plot(data_freqs, data_phase, 'x', label='Experimental')
    ax2.set_xlabel('Frequency (ω)')
    ax2.set_ylabel('Phase')
    ax2.set_title('Phase')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_real_and_imag():
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(4, 1)
    # Create the first subplot (3/4 of the area)
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax1.plot(frequencies, real, label='Simulated')
    ax1.plot(data_freqs, data_real, 'x', label='Experimental')
    #ax1.plot(siemens_freqs, siemens_mag, label='Siemens Fit')
    ax1.set_xlabel('Frequency (ω)')
    ax1.set_ylabel('Magnitude')
    ax1.set_title('Magnitude')
    ax1.legend()
    ax1.grid(True)

    # Create the second subplot (1/4 of the area)
    ax2 = fig.add_subplot(gs[2:4, 0])
    ax2.plot(frequencies, imag, label='Simulated')
    ax2.plot(data_freqs, data_imag, 'x', label='Experimental')
    #ax2.plot(siemens_freqs, siemens_phase, label='Siemens Fit')
    ax2.set_xlabel('Frequency (ω)')
    ax2.set_ylabel('Phase')
    ax2.set_title('Phase')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()

plot_mag_and_phase()
#plot_real_and_imag()
