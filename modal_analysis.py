import numpy as np
from circle_fit import CircleFit
import pandas as pd
from interactive_peak_finder import InteractivePeakFinder
from interactive_circle_fit import InteractiveCircleFit
from simulated_frf import SimulatedFRF
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import peak_finder


class ModalAnalysis():

    def __init__(self, combined_data, freq_range, locations, res_freqs=None):
        self.freq_range = freq_range
        self.m = len(locations)          # number of locations
        self.locations = np.array(locations)
        self.residual_frequencies = res_freqs

        filtered_data = combined_data[(combined_data['freq (Hz)'] >= self.freq_range[0]) & (combined_data['freq (Hz)'] <= self.freq_range[1])]
        p = InteractivePeakFinder(filtered_data)
        self.peaks = p.peaks
        self.prominence = p.prominence
        self.n = len(self.peaks)             # number of modes

        self.data = np.empty((self.m, self.m), dtype=object)  # matrix to store inputted FRF data
        self.H = np.full((self.m, self.m), None, dtype=object)  # matrix to store simulated FRFs
        self.mode_shapes = np.zeros((self.m, self.n), dtype=np.complex128)  # self.mode_shapes[k,r] is mode r at point k

        # Final Parameters
        self.A = np.zeros((self.n, self.m, self.m), dtype=np.complex128)    # self.A[mode_r, impulse_point, response_point]
        self.omega = np.zeros(self.n)
        self.eta = np.zeros(self.n)

    def define_locations(self, locations):
        if len(locations) == self.m:
            self.locations = np.array(locations)
        else:
            print('locations must have equal length number of points')


    def curve_fit(self, data, impulse_point, response_point, interactive=True):
        filtered_data = data[(data['freq (Hz)'] >= self.freq_range[0]) & (data['freq (Hz)'] <= self.freq_range[1])]
        self.data[impulse_point, response_point] = filtered_data

        peaks = peak_finder.get_peaks(filtered_data, distance=10, prominence=self.prominence, plot=interactive)

        while len(peaks) != len(self.peaks):
            print('different number of peaks found')
            # find peaks and ranges
            p = InteractivePeakFinder(filtered_data)
            peaks = p.peaks


        # Create array of circle fits for each peak
        modes = [CircleFit(data, peak) for peak in peaks]

        if interactive:
            # Interactive plot to help choose best range of points for each peak
            InteractiveCircleFit(modes)

        # store FRF
        omega_rs = []
        As = []
        etas = []
        quals = []
        for mode in modes:
            omega_rs.append(mode.omega_r)
            As.append(mode.A)
            etas.append(mode.damping)
            quals.append(mode.quality_factor)

        frf = SimulatedFRF(omega_rs, As, etas, self.freq_range, quality_factors=quals, res_freqs=self.residual_frequencies)
        frf.calculate_residuals(data)
        if interactive:
            frf.plot_mag_and_phase(data)

        self.H[impulse_point, response_point] = frf


    def correct_modal_properties(self):
        sum_omega = np.zeros(self.n)
        sum_eta = np.zeros(self.n)
        total_weight = np.zeros(self.n)
        for row in self.H:
            for frf in row:
                if frf != None:
                    quality_factors = frf.quality_factors
                    sum_omega += frf.omega * quality_factors
                    sum_eta += frf.eta * quality_factors
                    total_weight += quality_factors

        self.omega = sum_omega / total_weight
        self.eta = sum_eta / total_weight



    def calculate_mode_shapes(self, driving_point):
        driving_point_frf = self.H[driving_point, driving_point]
        if driving_point_frf == 0:
            print('no frf at this location')
            return

        A = driving_point_frf.A
        for r in range(self.n):
            self.mode_shapes[driving_point, r] = np.sqrt(A[r])
            for j in range(self.m):
                frf = self.H[j, driving_point]
                self.mode_shapes[j,r] = frf.A[r] / self.mode_shapes[driving_point, r]

    def plot_mode_shape(self, mode):
        x = self.locations[:, 0]
        y = self.locations[:, 1]

        # Initialize the z values
        z = []
        for i in range(self.m):
            shape = self.mode_shapes[i, mode]
            mag = np.abs(shape)
            phase = np.angle(shape)
            if abs(phase) > np.pi / 2:
                mag = -mag
            z.append(mag)

        # Normalize z values
        largest = max(z)
        z = np.array(z) / largest

        # Create a grid for plotting
        X, Y = np.meshgrid(np.unique(x), np.unique(y))
        Z = griddata((x, y), z, (X, Y), method='linear')

        # Create the figure and axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the initial wireframe
        wireframe = ax.plot_wireframe(X, Y, Z, color='blue')

        ax.set_zlim(-1, 1)

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Mode Shape')

        def update(frame):
            # Update z values by scaling them
            scale = np.sin(frame / 10.0)  # Scale factor ranges from -1 to 1
            Z = griddata((x, y), z * scale, (X, Y), method='linear')
            ax.clear()  # Clear the previous plot
            ax.plot_wireframe(X, Y, Z, color='blue')
            ax.set_zlim(-1,1)
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Mode Shape')
            return wireframe,

        # Create animation
        ani = FuncAnimation(fig, update, frames=np.arange(0, 200), interval=50, blit=False)

        plt.show()

