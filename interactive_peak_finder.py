from circle_fit import CircleFit
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.widgets import SpanSelector, Button
import numpy as np
from scipy.signal import find_peaks

class InteractivePeakFinder:
    def __init__(self, data):
        self.data = data

        self.peaks = None
        self.ranges = None

        self.distance = 10
        self.prominence = 0.8
        self.width_multiplier = 0.5

        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        plt.subplots_adjust(bottom=0.3)

        # Add accept button
        self.button_ax = plt.axes([0.45, 0.02, 0.1, 0.05])
        self.accept_button = Button(self.button_ax, 'Accept')
        self.accept_button.on_clicked(self.accept_peaks)

        #self.prom_slider_ax = plt.axes([0.92, 0.2, 0.03, 0.65])
        self.prom_slider_ax = plt.axes([0.2, 0.15, 0.65, 0.03])
        self.prom_slider = Slider(self.prom_slider_ax, 'Prominence', 0, 1, valinit=self.prominence, valstep=0.0001, orientation='horizontal')
        self.prom_slider.on_changed(self.update_prom)

        #self.width_slider_ax = plt.axes([0.96, 0.2, 0.03, 0.65])
        self.width_slider_ax = plt.axes([0.2, 0.1, 0.65, 0.03])
        self.width_slider = Slider(self.width_slider_ax, 'Width', 0, 1, valinit=self.width_multiplier, valstep=0.01, orientation='horizontal')
        self.width_slider.on_changed(self.update_width)

        # initialize data
        self.initialize_plot()


    def initialize_plot(self):
        peak_freqs, peak_mags, freq_ranges = self.peak_ranges(prominence=self.prominence, width_multiplier=self.width_multiplier)

        self.peak_marks = []
        self.lower_lines = []
        self.upper_lines = []

        freqs = self.data['freq (Hz)'].values
        real = self.data['real'].values
        imag = self.data['complex'].values
        mag = np.sqrt(real ** 2 + imag ** 2)
        self.ax.plot(freqs, mag, label="mag")

        colors = ['g', 'r', 'c', 'm', 'y']
        # Plotting frequency ranges
        for i, (low, high) in enumerate(freq_ranges):
            color = colors[i % len(colors)]
            self.peak_marks.append(self.ax.plot(peak_freqs[i], peak_mags[i], 'x', color=color))
            self.lower_lines.append(self.ax.axvline(x=low, color=color, linestyle='--', linewidth=1))
            self.upper_lines.append(self.ax.axvline(x=high, color=color, linestyle='--', linewidth=1))

        self.ax.set_xlabel('Frequency')
        self.ax.set_ylabel('Magnitude')
        self.ax.set_title('Identified Peaks')
        self.ax.relim()
        self.ax.autoscale_view()

        self.show()

    def peak_ranges(self, distance=1, prominence=0.1, width_multiplier=0.8):
        self.prominence = prominence
        self.width_multiplier = width_multiplier
        freqs = self.data['freq (Hz)'].values
        real = self.data['real'].values
        imag = self.data['complex'].values
        mag = np.sqrt(real ** 2 + imag ** 2)
        phase = np.arctan2(imag, real)

        # Detect peaks
        peak_indices, properties = find_peaks(mag, height=0, distance=distance, prominence=prominence)

        # print(peak_indices)
        peak_freqs = freqs[peak_indices]
        peak_mags = mag[peak_indices]
        prominences = properties['prominences']
        # print(peak_freqs)

        # Determine frequency ranges around each peak
        freq_ranges = []
        for i in range(len(peak_freqs)):
            prominence = prominences[i]
            mag_min = peak_mags[i] - prominence * width_multiplier

            higher_index = peak_indices[i]
            higher_mag = mag[higher_index]
            while (higher_mag > mag_min):
                if higher_index == len(freqs) - 1:
                    break
                higher_index += 1
                higher_mag = mag[higher_index]
                if mag[higher_index] > mag[higher_index - 1]:
                    higher_index -= 1
                    break

            lower_index = peak_indices[i]
            lower_mag = mag[lower_index]
            while lower_mag > mag_min:
                lower_index -= 1
                lower_mag = mag[lower_index]
                if mag[lower_index] > mag[lower_index + 1]:
                    lower_index += 1
                    break

            freq_min, freq_max = freqs[lower_index], freqs[higher_index]
            freq_ranges.append((freq_min, freq_max))

        self.peaks = peak_freqs
        self.ranges = freq_ranges

        return peak_freqs, peak_mags, freq_ranges


    def update_prom(self, val):
        self.prominence = self.prom_slider.val

        peak_freqs, peak_mags, freq_ranges = self.peak_ranges(prominence=self.prominence)

        # colors = ['g', 'r', 'c', 'm', 'y']
        # # Plotting frequency ranges
        # for i, (low, high) in enumerate(freq_ranges):
        #     color = colors[i % len(colors)]
        #
        #     self.peak_marks[i].set_data([peak_freqs[i]], [peak_mags[i]], color=color)
        #     self.lower_lines[i].set_xdata([low, low], color=color)
        #     self.upper_lines[i].set_xdata([high, high], color=color)

        self.ax.clear()
        self.peak_marks = []
        self.lower_lines = []
        self.upper_lines = []

        self.ax.set_xlabel('Frequency')
        self.ax.set_ylabel('Magnitude')
        self.ax.set_title('Identified Peaks')

        freqs = self.data['freq (Hz)'].values
        real = self.data['real'].values
        imag = self.data['complex'].values
        mag = np.sqrt(real ** 2 + imag ** 2)
        self.ax.plot(freqs, mag, label="mag")

        colors = ['g', 'r', 'c', 'm', 'y']
        # Plotting frequency ranges
        for i, (low, high) in enumerate(freq_ranges):
            color = colors[i % len(colors)]
            self.peak_marks.append(self.ax.plot(peak_freqs[i], peak_mags[i], 'x', color=color))
            self.lower_lines.append(self.ax.axvline(x=low, color=color, linestyle='--', linewidth=1))
            self.upper_lines.append(self.ax.axvline(x=high, color=color, linestyle='--', linewidth=1))

        self.fig.canvas.draw_idle()

    def update_width(self, val):
        self.width_multiplier = self.width_slider.val
        peak_freqs, peak_mags, freq_ranges = self.peak_ranges(prominence=self.prominence, width_multiplier=self.width_multiplier)

        # Plotting frequency ranges
        for i, (low, high) in enumerate(freq_ranges):
            self.lower_lines[i].set_xdata([low, low])
            self.upper_lines[i].set_xdata([high, high])

        self.fig.canvas.draw_idle()


    def show(self):
        plt.show()

    def accept_peaks(self, event):
        plt.close(self.fig)