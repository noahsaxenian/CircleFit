import numpy as np
from circle_fit_mobility import CircleFit
import pandas as pd
from interactive_peak_finder import InteractivePeakFinder
from interactive_circle_fit import InteractiveCircleFit
from reconstructed_frf import ReconstructedFRF
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import peak_finder

class ModalAnalysis():
    """
    Class to perform modal analysis, including:
    curve fitting each FRF with circle fit at each mode
    stores FRFs and raw data
    compute and plot mode shapes
    """

    def __init__(self, combined_data, freq_range, locations, res_freqs=None):
        """
        combined_data: a data set used to identify modes (often a combination of the FRFs at each point)
        freq_range: frequency range of interest
        locations: coordinates of each location for mode shape mesh
        res_freqs: user identifies frequencies to place residuals for FRF regeneration
        """
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


    def curve_fit(self, data, impulse_point, response_point, interactive=True):
        """
        Fits a single FRF dataset with regenerated curve using CircleFit
        """
        filtered_data = data[(data['freq (Hz)'] >= self.freq_range[0]) & (data['freq (Hz)'] <= self.freq_range[1])]
        self.data[impulse_point, response_point] = filtered_data

        peaks = peak_finder.get_peaks(filtered_data, distance=10, prominence=self.prominence, plot=False)
        #peaks = self.peaks

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
            if mode.quality_factor < 0.95:
                InteractiveCircleFit([mode])
            omega_rs.append(mode.omega_r)
            As.append(mode.A)
            etas.append(mode.damping)
            quals.append(mode.quality_factor)

        frf = ReconstructedFRF(omega_rs, As, etas, self.freq_range, quality_factors=quals, res_freqs=self.residual_frequencies)
        frf.calculate_residuals(data)
        frf.generate_mobility()
        frf.plot_mag_and_phase(data, impulse_point, response_point)
        #frf.results()

        self.H[impulse_point, response_point] = frf


    def correct_modal_properties(self):
        """
        Performs a weighted average based on quality factors to obtain better estimates of mode properties
        note: unfinished/not fully tested
        """
        sum_omega = np.zeros(self.n)
        sum_eta = np.zeros(self.n)
        total_weight = np.zeros(self.n)
        for row in self.H:
            for frf in row:
                if frf != None:
                    quality_factors = frf.quality_factors
                    sum_omega += frf.omega_r * quality_factors
                    sum_eta += frf.eta * quality_factors
                    total_weight += quality_factors

        self.omega = sum_omega / total_weight
        self.eta = sum_eta / total_weight
        print(f'resonant frequencies: {self.omega / (2*np.pi)}')



    def calculate_mode_shapes(self, driving_point):
        """
        Calculates mode shapes given the index of the driving point (excitement and response at same location)
        Driving point response is necessary, all other mode shapes calculated based on this
        """
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
        """
        Generates a wireframe animation of specified mode
        """
        x = self.locations[:, 0]
        y = self.locations[:, 1]

        # Calculate z values
        shape = self.mode_shapes[:, mode]
        mag = np.abs(shape)
        phase = np.angle(shape)
        z = np.where(abs(phase) > np.pi / 2, -mag, mag)

        # Normalize z values
        z = z / np.max(np.abs(z))

        # Create a grid for plotting
        X, Y = np.meshgrid(np.unique(x), np.unique(y))
        Z = griddata((x, y), z, (X, Y), method='linear')

        # Create the figure and axis
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the initial surface
        surface = ax.plot_surface(X, Y, Z, color='blue')

        ax.set_zlim(-2, 2)
        ax.set_title(f'Mode {mode+1}')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Mode Shape')

        # Precompute sin values for animation
        frames = 200
        sin_values = np.sin(np.arange(frames) / 10.0)

        def update(frame):
            # Scale Z values
            new_Z = Z * sin_values[frame]

            # # Remove previous wireframe
            for artist in ax.collections:
                artist.remove()

            ax.plot_surface(X, Y, new_Z, color='blue')

            return surface,

        # Create animation
        ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)

        plt.show()