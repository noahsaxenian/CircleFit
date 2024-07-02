import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

def peak_ranges(data, distance = 1, prominence = 0.001, plot=True):
    freqs = data['freq (Hz)'].values
    real = data['real'].values
    imag = data['complex'].values
    mag = np.sqrt(real**2 + imag**2)
    phase = np.arctan2(imag, real)

    # Detect peaks
    peak_indices, properties = find_peaks(mag, height=0, distance=distance, prominence=prominence)

    #print(peak_indices)
    peak_freqs = freqs[peak_indices]
    peak_mags = mag[peak_indices]
    prominences = properties['prominences']
    #print(peak_freqs)

    # Determine frequency ranges around each peak
    freq_ranges = []
    for i in range(len(peak_freqs)):
        prominence = prominences[i]
        mag_min = peak_mags[i] - prominence*0.8

        higher_index = peak_indices[i]
        higher_mag = mag[higher_index]
        while (higher_mag > mag_min):
            if higher_index == len(freqs) - 1:
                break
            higher_index += 1
            higher_mag = mag[higher_index]
            if mag[higher_index] > mag[higher_index-1]:
                higher_index -= 1
                break

        lower_index = peak_indices[i]
        lower_mag = mag[lower_index]
        while lower_mag > mag_min:
            lower_index -= 1
            lower_mag = mag[lower_index]
            if mag[lower_index] > mag[lower_index+1]:
                lower_index += 1
                break

        freq_min, freq_max = freqs[lower_index], freqs[higher_index]
        freq_ranges.append((freq_min, freq_max))

    if plot:
        plt.figure()
        plt.plot(freqs, mag, label = "mag")
        colors = ['g', 'r', 'c', 'm', 'y']
        # Plotting frequency ranges
        for i, (low, high) in enumerate(freq_ranges):
            color = colors[i % len(colors)]
            plt.plot(peak_freqs[i], peak_mags[i], 'x', color=color)
            plt.axvline(x=low, color=color, linestyle='--', linewidth=1)
            plt.axvline(x=high, color=color, linestyle='--', linewidth=1)
            #plt.fill_betweenx([0, max(mag)], low, high, alpha=0.1, color='r')
        plt.legend()
        plt.show()

    print("Peaks found: " + str(peak_freqs))

    return peak_freqs, freq_ranges