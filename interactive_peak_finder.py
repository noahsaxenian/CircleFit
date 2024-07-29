from circle_fit_receptance import CircleFit
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
        self.prominence = 316
        self.width_multiplier = 0.5

        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        plt.subplots_adjust(bottom=0.3)

        # Add accept button
        self.button_ax = plt.axes([0.45, 0.02, 0.1, 0.05])
        self.accept_button = Button(self.button_ax, 'Accept')
        self.accept_button.on_clicked(self.accept_peaks)

        #self.prom_slider_ax = plt.axes([0.92, 0.2, 0.03, 0.65])
        self.prom_slider_ax = plt.axes([0.2, 0.15, 0.65, 0.03])
        slider_value = np.log10(self.prominence) + 5
        self.prom_slider = Slider(self.prom_slider_ax, 'Prominence', 0, 15, valinit=slider_value, valstep=0.0001, orientation='horizontal')
        self.prom_slider.on_changed(self.update_prom)

        # initialize data
        self.initialize_plot()


    def initialize_plot(self):
        peak_freqs, peak_mags = self.peak_ranges(prominence=self.prominence, width_multiplier=self.width_multiplier)

        self.peak_marks = []

        freqs = self.data['freq (Hz)'].values
        real = self.data['real'].values
        imag = self.data['complex'].values
        if 'magnitude' in self.data.columns:
            mag = self.data['magnitude'].values
        else:
            mag = np.sqrt(real ** 2 + imag ** 2)
        self.ax.plot(freqs, mag)
        self.ax.set_yscale('log')

        # Plotting frequency ranges
        for peak_freq, peak_mag in zip(peak_freqs, peak_mags):
            self.peak_marks.append(self.ax.plot(peak_freq, peak_mag, 'x', color='r'))

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
        if 'magnitude' in self.data.columns:
            mag = self.data['magnitude'].values
        else:
            mag = np.sqrt(real ** 2 + imag ** 2)

        # Detect peaks
        peak_indices, properties = find_peaks(mag, height=0, distance=distance, prominence=prominence)

        # print(peak_indices)
        peak_freqs = freqs[peak_indices]
        peak_mags = mag[peak_indices]
        prominences = properties['prominences']

        self.peaks = peak_freqs

        return peak_freqs, peak_mags


    def update_prom(self, val):
        self.prominence = 10 ** (self.prom_slider.val - 5)

        peak_freqs, peak_mags = self.peak_ranges(prominence=self.prominence)

        self.ax.clear()
        self.peak_marks = []

        self.ax.set_xlabel('Frequency')
        self.ax.set_ylabel('Magnitude')
        self.ax.set_title('Identified Peaks')
        self.ax.set_yscale('log')

        freqs = self.data['freq (Hz)'].values
        real = self.data['real'].values
        imag = self.data['complex'].values
        if 'magnitude' in self.data.columns:
            mag = self.data['magnitude'].values
        else:
            mag = np.sqrt(real ** 2 + imag ** 2)
        self.ax.plot(freqs, mag, label="mag")

        # Plotting frequency ranges
        for peak_freq, peak_mag in zip(peak_freqs, peak_mags):
            self.peak_marks.append(self.ax.plot(peak_freq, peak_mag, 'x', color='r'))

        self.fig.canvas.draw_idle()


    def show(self):
        plt.show()

    def accept_peaks(self, event):
        plt.close(self.fig)