# experimenting with detecting peaks across multiple datasets

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib as mpl
mpl.use('TkAgg')

# freq_range = (100, 1000)
#
# data = []
# for i in range(1, 17):
#     if i < 10:
#         file_path = f'C:/Users/noahs/Documents/ceeo/modal stuff/Code/data/Plate/PlateFull1/csv/PlateFull1 H_00{i}_trf.tsv'
#     else:
#         file_path = f'C:/Users/noahs/Documents/ceeo/modal stuff/Code/data/Plate/PlateFull1/csv/PlateFull1 H_0{i}_trf.tsv'
#     raw_data = pd.read_csv(file_path, delimiter='\t')
#     filtered = raw_data[(raw_data['freq (Hz)'] >= freq_range[0]) & (raw_data['freq (Hz)'] <= freq_range[1])]
#     data.append(filtered)
#
# frequencies = np.array([df['freq (Hz)'].values for df in data])
# real = [df['real'].values for df in data]
# imag = [df['complex'].values for df in data]
# magnitudes = [np.sqrt(re**2 + im**2) for (re, im) in zip(real, imag)]


def multi_peak_detect(all_frequencies, all_magnitudes):
    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(left=0.1, bottom=0.25)

    # Initial plot
    line, = ax.plot([], [], alpha=0.5, label='Magnitude vs Frequency')
    # Store manual peaks
    manual_peaks = []
    common_peaks = []

    # Update function for the slider
    def update(val):
        nonlocal common_peaks
        prominence = slider.val
        all_peaks = []
        factor = 2  # rounding factor for peaks

        for freq, mag in zip(all_frequencies, all_magnitudes):
            peaks, _ = find_peaks(mag, prominence=prominence)
            detected_peaks = freq[peaks]
            rounded_peaks = np.round(detected_peaks / factor) * factor
            all_peaks.append(rounded_peaks)

        all_peaks = np.concatenate(all_peaks)
        threshold = len(all_frequencies) * 0.5
        unique_peaks, counts = np.unique(all_peaks, return_counts=True)
        common_peaks = unique_peaks[counts >= threshold]

        # Clear previous lines
        ax.clear()

        # Plot each FRF
        for freq, mag in zip(all_frequencies, all_magnitudes):
            ax.plot(freq, mag, alpha=0.5, label='Magnitude vs Frequency')

        # Plot common peaks
        for peak in common_peaks:
            ax.axvline(x=peak, color='r', linestyle='--', label='Common Peak')

        # Plot manual peaks
        for peak in manual_peaks:
            ax.axvline(x=peak, color='b', linestyle='--', label='Manual Peak')

        ax.set_yscale('log')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.set_ylim(bottom=100)
        ax.set_title('Magnitude vs Frequency for Multiple Arrays')

        fig.canvas.draw_idle()

    def close(event):
        plt.close(fig)

    # Function to add a peak manually
    def add_peak(event):
        if event.inaxes == ax:
            x_data = event.xdata
            if x_data is not None:
                new_value = True
                for p in manual_peaks:
                    if abs(x_data - p) < 5:
                        manual_peaks.remove(p)
                        new_value = False
                if new_value:
                    manual_peaks.append(x_data)
                update(slider.val)  # Update plot with new manual peaks


    # Slider for prominence
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Prominence', 0, 20000, valinit=3000)

    # Button for accepting and closing the window
    ax_button = plt.axes([0.45, 0.02, 0.1, 0.05], facecolor='lightgoldenrodyellow')
    button = Button(ax_button, 'Accept')

    # Connect the click event to the add_peak function
    fig.canvas.mpl_connect('button_press_event', add_peak)

    button.on_clicked(close)
    slider.on_changed(update)

    update(slider.val)
    plt.show()

    final_peaks = np.sort(np.append(common_peaks, manual_peaks))

    return(final_peaks)

#peaks = multi_peak_detect(frequencies, magnitudes)
#print(peaks)