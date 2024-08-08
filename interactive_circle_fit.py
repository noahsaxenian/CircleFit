import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector, Button
import numpy as np
from sdof import circle_fit, filter_data

def interactive_circle_fit(frequencies, real, imaginary, peak):
    magnitudes = np.abs(real**2 + imaginary**2)
    params = None

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.2)

    start, end = filter_data(frequencies, magnitudes, peak)
    params = circle_fit(frequencies[start:end], real[start:end], imaginary[start:end])
    if params['quality_factor'] > 0.999:
        return params

    def update(freq_min, freq_max):
        nonlocal params

        freq_range = None
        if freq_min != freq_max:
            freq_range = [freq_min, freq_max]

        start, end = filter_data(frequencies, magnitudes, peak, freq_range=freq_range)

        params = circle_fit(frequencies[start:end], real[start:end], imaginary[start:end])
        h, k, r = params['h'], params['k'], params['r']
        theta_r = params['theta_r']
        resonant_frequency = params['resonant_frequency']

        points = np.linspace(0, 2 * np.pi, 100)
        x_fit = h + r * np.cos(points)
        y_fit = k + r * np.sin(points)
        x_pos = h + r * np.cos(theta_r)
        y_pos = k + r * np.sin(theta_r)

        # Update magnitude plot
        axs[0].cla()
        axs[0].plot(frequencies[start - 10:end + 10], magnitudes[start - 10:end + 10], label='Magnitude')
        axs[0].plot(resonant_frequency, max(magnitudes[start:end]), 'x', color='g')
        axs[0].set_xlabel('Frequency')
        axs[0].set_ylabel('Magnitude')
        axs[0].set_yscale('log')
        axs[0].set_title('Adjust Range Around Peak')
        axs[0].legend()

        # Update circle fit plot
        axs[1].cla()
        axs[1].scatter(real[start:end], imaginary[start:end], label='Data')
        axs[1].plot(x_fit, y_fit, label='Fitted Circle', color='red')
        axs[1].plot([x_pos], [y_pos], 'o', label=f'Estimate = {resonant_frequency} Hz', color='green')
        axs[1].set_xlabel('Real')
        axs[1].set_ylabel('Complex')
        axs[1].legend()
        axs[1].grid(True)
        axs[1].axis('equal')
        axs[1].set_title('Circle Fit')

        # Update span extents
        span.extents = (frequencies[start], frequencies[end])

        for ax in axs:
            ax.relim()
            ax.autoscale_view()

        fig.canvas.draw_idle()

    def close(event):
        plt.close(fig)

    def remove(event):
        nonlocal params
        params['A'] = 0 + 0j
        params['eta_r'] = 0
        params['theta_r'] = 0
        params['quality_factor'] = 0

        plt.close(fig)

    # Add span selector
    span = SpanSelector(
        axs[0],
        update,
        'horizontal',
        useblit=True,
        props=dict(alpha=0.3, facecolor='lightblue'),
        interactive=True,
        drag_from_anywhere=True
    )

    # Add accept button
    button_ax_accept = plt.axes((0.35, 0.05, 0.1, 0.05))
    accept_button = Button(button_ax_accept, 'Accept')
    accept_button.on_clicked(close)

    # Add ignore button
    button_ax_ignore = plt.axes((0.50, 0.05, 0.1, 0.05))
    ignore_button = Button(button_ax_ignore, 'Ignore')
    ignore_button.on_clicked(remove)

    update(0,0)
    plt.show()

    return params


