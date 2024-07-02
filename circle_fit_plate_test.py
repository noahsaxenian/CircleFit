import pandas as pd
from circle_fit import CircleFit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import peak_finder
from interactive_plot import InteractiveCircleFit

# Load your FRF data from a CSV file
file_path = 'C:/Users/noahs/Documents/ceeo/modal stuff/Siemens Plate Test/point1_data_receptance.tsv'
data = pd.read_csv(file_path, delimiter='\t')

regenerated_file_path = 'C:/Users/noahs/Documents/ceeo/modal stuff/Siemens Plate Test/point1_regenerated_receptance.tsv'
siemens_fit = pd.read_csv(file_path, delimiter='\t')

#identified peaks
freqs = [229, 287, 533, 671, 893, 1068, 1131, 1305, 1457, 1717, 1857]
points = [5, 3, 2, 5, 3, 5, 4, 3, 4, 4, 4]

freq_range = [200, 2000]

filtered_data = data[(data['freq (Hz)'] >= freq_range[0]) & (data['freq (Hz)'] <= freq_range[1])]
siemens_fit = siemens_fit[(siemens_fit['freq (Hz)'] >= freq_range[0]) & (siemens_fit['freq (Hz)'] <= freq_range[1])]

peaks, peak_ranges = peak_finder.peak_ranges(filtered_data, distance = 10, prominence=0.001)

modes = []

# for i in range(len(freqs)):
#     mode = CircleFit(data, peaks[i], freq_range=peak_ranges[i])
#     mode.choose_points_interactive()
#     modes.append(mode)

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
    #print(mode.A, mode.omega, mode.damping)
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
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(4, 1)
    # Create the first subplot (3/4 of the area)
    ax1 = fig.add_subplot(gs[0:3, 0])
    ax1.plot(frequencies, magnitude, label='Simulated')
    ax1.plot(data_freqs, data_mag, 'x', label='Experimental')
    #ax1.plot(siemens_freqs, siemens_mag, label='Siemens Fit')
    ax1.set_xlabel('Frequency (ω)')
    ax1.set_ylabel('Magnitude')
    ax1.legend()
    ax1.grid(True)

    # Create the second subplot (1/4 of the area)
    ax2 = fig.add_subplot(gs[3, 0])
    ax2.plot(frequencies, phase, label='Simulated')
    ax2.plot(data_freqs, data_phase, 'x', label='Experimental')
    #ax2.plot(siemens_freqs, siemens_phase, label='Siemens Fit')
    ax2.set_xlabel('Frequency (ω)')
    ax2.set_ylabel('Phase')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()

def plot_real_and_imag():
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(4, 1)
    # Create the first subplot (3/4 of the area)
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax1.plot(frequencies, real, label='Simulated')
    ax1.plot(data_freqs, data_real, 'x', label='Experimental')
    #ax1.plot(siemens_freqs, siemens_mag, label='Siemens Fit')
    ax1.set_xlabel('Frequency (ω)')
    ax1.set_ylabel('Real')
    ax1.legend()
    ax1.grid(True)

    # Create the second subplot (1/4 of the area)
    ax2 = fig.add_subplot(gs[2:4, 0])
    ax2.plot(frequencies, imag, label='Simulated')
    ax2.plot(data_freqs, data_imag, 'x', label='Experimental')
    #ax2.plot(siemens_freqs, siemens_phase, label='Siemens Fit')
    ax2.set_xlabel('Frequency (ω)')
    ax2.set_ylabel('Imaginary')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()


plot_mag_and_phase()
#plot_real_and_imag()