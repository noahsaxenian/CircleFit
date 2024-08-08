from interactive_circle_fit import interactive_circle_fit
from reconstructed_frf import ReconstructedFRF
from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata
from sdof import *
from multi_peak_detect import *

class ModalAnalysis():
    """
    Class to perform modal analysis, including:
    curve fitting each FRF with circle fit at each mode
    stores FRFs and raw data
    compute and plot mode shapes
    """

    def __init__(self, all_data, locations, res_freqs=None, peaks=None):
        """
        all_data: matrix of pandas data frames.
            all_data[i, j] is data for impulse point i and response point j
            columns "freq (Hz)", "real", "complex"
        locations: coordinates of each location for mode shape mesh
        res_freqs: user identifies frequencies to place residuals for FRF regeneration
        peaks: user identifies modes of interest
        """
        self.m = len(locations)          # number of locations
        self.locations = np.array(locations)
        self.residual_frequencies = res_freqs

        self.data = all_data  # matrix to store inputted FRF data

        if peaks is not None:
            self.peaks = peaks
        else:
            self.peaks = self.detect_peaks()
            print(self.peaks)

        self.n = len(self.peaks)             # number of modes

        self.H = np.full((self.m, self.m), None, dtype=object)  # matrix to store simulated FRFs
        self.mode_shapes = np.zeros((self.m, self.n), dtype=np.complex128)  # self.mode_shapes[k,r] is mode r at point k

        # Final Parameters
        self.A = np.zeros((self.n, self.m, self.m), dtype=np.complex128)    # self.A[mode_r, impulse_point, response_point]
        self.omega = np.zeros(self.n)
        self.eta = np.zeros(self.n)



    def detect_peaks(self):
        data_list = [x for x in self.data.ravel() if x is not None]
        frequencies = np.array([data['freq (Hz)'] for data in data_list])
        real = [data['real'] for data in data_list]
        imaginary = [data['complex'] for data in data_list]
        magnitudes = np.array([np.sqrt(re**2 + im**2) for (re, im) in zip(real, imaginary)])

        peaks = multi_peak_detect(frequencies, magnitudes)

        return peaks


    def fit_all(self):
        """fits all inputted FRFs"""
        for i in range(self.data.shape[0]):  # Loop over rows
            for j in range(self.data.shape[1]):  # Loop over columns
                if self.data[i, j] is not None:
                    self.H[i, j] = self.curve_fit(self.data[i,j], interactive=True)

    def fit_frf(self, impulse_point, response_point):
        self.H[i, j] = self.curve_fit(self.data[impulse_point, response_point], interactive=True)


    def curve_fit(self, data, interactive=True):
        """
        Fits a single FRF dataset with regenerated curve using CircleFit
        """
        frequencies = data['freq (Hz)'].values
        real = data['real'].values
        imaginary = data['complex'].values
        magnitudes = np.abs(real**2 + imaginary**2)

        # store FRF
        omega_rs = []
        As = []
        etas = []
        quals = []
        for peak in self.peaks:
            start, end = filter_data(frequencies, magnitudes, peak)

            params = circle_fit(frequencies[start:end], real[start:end], imaginary[start:end])

            if params['quality_factor'] < 0.999:
                params = interactive_circle_fit(frequencies, real, imaginary, peak)

            omega_rs.append(params['omega_r'])
            As.append(params['A'])
            etas.append(params['eta_r'])
            quals.append(params['quality_factor'])



        freq_range = (frequencies[0], frequencies[-1])
        frf = ReconstructedFRF(omega_rs, As, etas, freq_range, quality_factors=quals)
        frf.calculate_residuals(frequencies, real, imaginary)

        if interactive:
            frf.results()
            #frf.plot_mag_and_phase(data)

        return frf


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

        freq = round(self.omega[mode] / (2*np.pi), 1)

        # Calculate z values
        shape = self.mode_shapes[:, mode]
        mag = np.abs(shape)
        phase = np.angle(shape)
        z = np.where(abs(phase) > np.pi / 2, -mag, mag)

        # Normalize z values
        z = z / np.max(np.abs(z))

        # # Create a grid for plotting
        # X, Y = np.meshgrid(np.unique(x), np.unique(y))
        # Z = griddata((x, y), z, (X, Y), method='linear')

        # Create a finer grid for plotting
        x_range = np.linspace(np.min(x), np.max(x), 20)  # Increase the number of points for more density
        y_range = np.linspace(np.min(y), np.max(y), 20)  # Increase the number of points for more density
        X, Y = np.meshgrid(x_range, y_range)

        Z = griddata((x, y), z, (X, Y), method='cubic')

        # Check if Z has NaN values in the gap area
        if np.any(np.isnan(Z)):
            print("Warning: Z contains NaN values in the gap area.")

        # Create the figure and axis
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the initial surface
        surface = ax.plot_surface(X, Y, Z, color='blue')

        ax.set_zlim(-2, 2)
        ax.set_title(f'Mode {mode+1}: {freq} Hz')
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