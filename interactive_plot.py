from circle_fit import CircleFit
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.widgets import SpanSelector, Button
import numpy as np

class InteractiveCircleFit:
    def __init__(self, circle_fit):
        self.circle_fit = circle_fit
        self.fig, self.axs = plt.subplots(1, 2, figsize=(15, 6))
        plt.subplots_adjust(bottom=0.2)

        # initialize slider
        # plt.subplots_adjust(bottom=0.25)
        # self.slider_ax = plt.axes([0.2, 0.1, 0.6, 0.03])
        # self.slider = Slider(self.slider_ax, 'Number of Points', 3, 50, valinit=self.circle_fit.points, valstep=1)
        # self.slider.on_changed(self.update)

        # Add span selector
        self.span = SpanSelector(
            self.axs[0],
            self.on_select,
            'horizontal',
            useblit=True,
            props=dict(alpha=0.5, facecolor='lightblue'),
            interactive=True,
            drag_from_anywhere=True
        )

        # Add accept button
        self.button_ax = plt.axes([0.45, 0.05, 0.1, 0.05])
        self.accept_button = Button(self.button_ax, 'Accept')
        self.accept_button.on_clicked(self.accept_range)

        # initialize data
        self.initialize_plot()

    def initialize_plot(self):
        freq = self.circle_fit.wide_freq
        magnitudes = self.circle_fit.wide_magnitudes
        freq_min = self.circle_fit.freq_min
        freq_max = self.circle_fit.freq_max
        resonant_frequency = self.circle_fit.resonant_frequency
        real = self.circle_fit.real
        cplx = self.circle_fit.cplx
        h = self.circle_fit.h
        k = self.circle_fit.k
        r = self.circle_fit.r
        x_pos = h + r * np.cos(self.circle_fit.theta)
        y_pos = k + r * np.sin(self.circle_fit.theta)
        circle_points = np.linspace(0, 2 * np.pi, 100)
        x_fit = h + r * np.cos(circle_points)
        y_fit = k + r * np.sin(circle_points)

        self.span.extents = (freq_min, freq_max)

        self.magnitude_plot, = self.axs[0].plot(freq, magnitudes)
        #self.freq_min_line = self.axs[0].axvline(x=freq_min, color='r', linestyle='--', linewidth=1)
        #self.freq_max_line = self.axs[0].axvline(x=freq_max, color='r', linestyle='--', linewidth=1)
        self.resonant_freq_point, = self.axs[0].plot(resonant_frequency, max(magnitudes), 'x', color='g')
        self.axs[0].set_xlabel('Frequency')
        self.axs[0].set_ylabel('Magnitude')
        self.axs[0].set_title('Use Cursors to Select Range Around Peak')

        self.data_scatter = self.axs[1].scatter(real, cplx, label='Data')
        self.circle_plot, = self.axs[1].plot(x_fit, y_fit, label='Fitted Circle', color='red')
        self.resonant_point, = self.axs[1].plot([x_pos], [y_pos], 'o', label=f'Estimate = {resonant_frequency} Hz',
                                                color='green')
        self.center_line, = self.axs[1].plot([x_pos, h], [y_pos, k], 'k--', alpha=0.5)
        self.axs[1].set_xlabel('Real')
        self.axs[1].set_ylabel('Complex')
        self.axs[1].legend()
        self.axs[1].grid(True)
        self.axs[1].axis('equal')
        self.axs[1].set_title('Circle Fit to Mode Data')

        for ax in self.axs:
            ax.relim()
            ax.autoscale_view()

    def on_select(self, freq_min, freq_max):
        self.circle_fit.filter_data_range(freq_min, freq_max)
        self.update()


    def update(self):
        #self.circle_fit.points = int(self.slider.val)
        self.circle_fit.run()

        # get data
        freq = self.circle_fit.wide_freq
        magnitudes = self.circle_fit.wide_magnitudes
        freq_min = self.circle_fit.freq_min
        freq_max = self.circle_fit.freq_max
        resonant_frequency = self.circle_fit.resonant_frequency
        real = self.circle_fit.real
        cplx = self.circle_fit.cplx
        h = self.circle_fit.h
        k = self.circle_fit.k
        r = self.circle_fit.r
        x_pos = h + r * np.cos(self.circle_fit.theta)
        y_pos = k + r * np.sin(self.circle_fit.theta)
        circle_points = np.linspace(0, 2 * np.pi, 100)
        x_fit = h + r * np.cos(circle_points)
        y_fit = k + r * np.sin(circle_points)

        # update plot
        self.magnitude_plot.set_data(freq, magnitudes)
        #self.freq_min_line.set_xdata([freq_min, freq_min])
        #self.freq_max_line.set_xdata([freq_max, freq_max])
        self.resonant_freq_point.set_data([resonant_frequency], [max(magnitudes)])

        self.data_scatter.set_offsets(np.c_[real, cplx])
        self.circle_plot.set_data(x_fit, y_fit)
        self.resonant_point.set_data([x_pos], [y_pos])
        self.resonant_point.set_label(f'Estimate = {resonant_frequency:.1f} Hz')
        self.center_line.set_data([x_pos, h], [y_pos, k])

        self.axs[1].legend()
        for ax in self.axs:
            ax.relim()
            ax.autoscale_view()
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()

    def accept_range(self, event):
        plt.close(self.fig)