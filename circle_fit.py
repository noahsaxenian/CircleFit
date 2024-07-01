import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

class CircleFit:
    '''Circle fitting class for a single mode
    use run() if number of points surrounding each mode is already selected
    use choose_points() to test out how number of points affects fit'''
    def __init__(self, data, freq_est, points=10, freq_range=None):
        self.data = data
        self.freq_est = freq_est
        self.points = points
        self.freq_min = None
        self.freq_max = None
        if freq_range:
            self.freq_min = freq_range[0]
            self.freq_max = freq_range[1]
        self.freq = None
        self.real = None
        self.cplx = None
        self.h = None
        self.k = None
        self.r = None
        self.frequencies = None
        self.omegas = None
        self.angles = None
        self.resonant_frequency = None
        self.omega = None
        self.theta = None
        self.damping = None
        self.damping_std = None
        self.A = None
        self.phase = None
        self.B = 0

        self.filter_data_range()

    def run(self):
        # Runs through necessary functions to find all parameters
        self.filter_data(self.points)
        self.fit_circle()
        self.calculate_resonant_frequency()
        self.calculate_damping()
        self.calculate_modal_parameters()

    def filter_data(self, num_points):
        ''' Filters data input to the inputted number of points
        surrounding frequency estimate '''

        # Find the index of the closest frequency to self.freq_est
        closest_index = np.abs(self.data['freq (Hz)'] - self.freq_est).idxmin()
        # Calculate the start and end indices
        start_index = max(0, closest_index - num_points)
        end_index = min(len(self.data) - 1, closest_index + num_points)

        # Filter the data
        filtered_data = self.data.iloc[start_index:end_index + 1]
        self.freq = filtered_data['freq (Hz)']
        self.real = filtered_data['real']
        self.cplx = filtered_data['complex']
        self.frequencies = self.freq.values
        self.omegas = self.frequencies * 2 * np.pi
        self.freq_min = min(self.freq)
        self.freq_max = max(self.freq)

    def filter_data_range(self):
        ''' Filters data input to the inputted frequency range '''

        # Find the index of the closest frequency to self.freq_est
        closest_index = np.abs(self.data['freq (Hz)'] - self.freq_est).idxmin()
        # Calculate the start and end indices
        start_index = np.abs(self.data['freq (Hz)'] - self.freq_min).idxmin()
        end_index = np.abs(self.data['freq (Hz)'] - self.freq_max).idxmin()

        # Filter the data
        filtered_data = self.data.iloc[start_index:end_index + 1]
        self.freq = filtered_data['freq (Hz)']
        self.real = filtered_data['real']
        self.cplx = filtered_data['complex']
        self.frequencies = self.freq.values
        self.omegas = self.frequencies * 2 * np.pi
        self.freq_min = min(self.freq)
        self.freq_max = max(self.freq)


    def fit_circle(self):
        ''' Function to fit circle
        Kasa method gets initial guess
        Least squares for final fit '''

        def residuals(params, x, y):
            # Used for circle fitting
            h, k, r = params
            return (x - h) ** 2 + (y - k) ** 2 - r ** 2

        x = self.real
        y = self.cplx

        # Kasa method for initial guess
        A = np.column_stack((x, y, np.ones_like(x)))
        b = x ** 2 + y ** 2
        c = np.linalg.lstsq(A, b, rcond=None)[0]
        h_kasa = 0.5 * c[0]
        k_kasa = 0.5 * c[1]
        r_kasa = np.sqrt(c[2] + h_kasa ** 2 + k_kasa ** 2)
        initial_guess = [h_kasa, k_kasa, r_kasa]

        # Perform least squares fitting
        result = least_squares(residuals, initial_guess, args=(x, y))
        self.h, self.k, self.r = result.x

    def calculate_resonant_frequency_quadratic(self):
        ''' doesn't work yet '''
        raw_angles = np.arctan2(self.cplx - self.k, self.real - self.h) % (2 * np.pi)
        self.angles = np.unwrap(raw_angles)
        omegas_2 = self.omegas ** 2

        domega2_dtheta = np.gradient(omegas_2, self.angles)

        coefficients = np.polyfit(omegas_2, domega2_dtheta,2)
        # Get the polynomial function from the coefficients
        quadratic_fit = np.poly1d(coefficients)
        # Evaluate the polynomial at the desired points
        omegas_2_dense = np.linspace(omegas_2.min(), omegas_2.max(), 1000)
        deriv_dense = quadratic_fit(omegas_2_dense)

        max_index = np.argmax(deriv_dense)
        self.omega = np.sqrt(omegas_2_dense[max_index])
        self.resonant_frequency = self.omega / (np.pi * 2)

        # Plot the original data and the fitted curve
        plt.plot(omegas_2, domega2_dtheta, 'o', color='green', label='data')
        plt.plot(omegas_2_dense, deriv_dense, color='blue', label='quadratic fit')
        plt.xlabel('Frequency (ω^2)')
        plt.ylabel('dω^2/dtheta')
        plt.legend()
        plt.show()

    def calculate_resonant_frequency(self):
        ''' Evaluates angles from center of circle to each point
        Fits a spline to the angles against frequencies
        Take derivative, location of max rate of angle change is natural frequency '''
        raw_angles = np.arctan2(self.cplx - self.k, self.real - self.h) % (2 * np.pi)
        self.angles = np.unwrap(raw_angles)
        spline = UnivariateSpline(self.frequencies, self.angles, s=0)
        frequencies_dense = np.linspace(self.freq_min, self.freq_max, 1000)
        angles_dense = spline(frequencies_dense)
        dtheta_df = np.gradient(angles_dense, frequencies_dense)
        resonant_frequency_index = np.argmin(dtheta_df)
        self.resonant_frequency = round(frequencies_dense[resonant_frequency_index], 1)
        self.omega = self.resonant_frequency * 2 * np.pi
        self.theta = spline(self.resonant_frequency)

        #self.plot_angles()

    def calculate_damping(self):
        '''Function to calculate damping
        Calculates for each pair of points above and below resonant frequency
        Finds average, standard dev'''

        # Split frequencies and angles into lower and higher groups
        split = self.resonant_frequency
        lower_frequencies = self.frequencies[self.frequencies < split]
        higher_frequencies = self.frequencies[self.frequencies >= split]
        lower_angles = self.angles[self.frequencies < split]
        higher_angles = self.angles[self.frequencies >= split]

        print("lower frequencies: " + str(lower_frequencies))
        print("higher frequencies: " + str(higher_frequencies))

        # Initialize array for damping coefficients
        damping_coeffs = []
        theta = self.theta

        # Calculate coefficient for each pair
        for i in range(min(len(lower_angles), len(higher_angles))):
            w_a = higher_frequencies[i] * 2 * np.pi
            w_b = lower_frequencies[-i-1] * 2 * np.pi
            theta_a = abs(higher_angles[i] - theta)
            theta_b = abs(lower_angles[-i-1] - theta)
            # Equation for damping coefficient
            n = (w_a ** 2 - w_b ** 2) / (self.omega ** 2 * (np.tan(theta_a / 2) + np.tan(theta_b / 2)))
            damping_coeffs.append(n)

        self.damping = np.mean(damping_coeffs)
        self.damping_std = np.std(damping_coeffs)
        return self.damping, self.damping_std

    def calculate_modal_parameters(self):
        """ Calculates modal constant magnitude and phase
        Calculates residual """
        magA = (self.omega ** 2) * self.damping * (2 * self.r)
        x_pos = self.h + self.r * np.cos(self.theta)
        y_pos = self.k + self.r * np.sin(self.theta)
        self.phase = np.atan2(y_pos, x_pos) + np.pi / 2
        self.A = magA * np.cos(self.phase) + 1j * magA * np.sin(self.phase)

        B_x = self.h + self.r * np.cos(self.theta + np.pi)
        B_y = self.k + self.r * np.sin(self.theta + np.pi)
        B_mag = np.sqrt(B_x ** 2 + B_y ** 2)
        B_phase = np.atan2(B_y, B_x)
        self.B = B_mag + 1j * B_phase
        return self.A, self.phase


    def plot_circle(self):
        '''Create a nice plot of the fitted circle'''
        points = np.linspace(0, 2 * np.pi, 100)
        x_fit = self.h + self.r * np.cos(points)
        y_fit = self.k + self.r * np.sin(points)
        x_pos = self.h + self.r * np.cos(self.theta)
        y_pos = self.k + self.r * np.sin(self.theta)
        plt.figure(figsize=(6, 6))
        plt.plot(self.real, self.cplx, 'o', label='Data')
        plt.plot(x_pos, y_pos, 'o', label='Estimate', color='green')
        plt.plot(x_fit, y_fit, label='Fitted Circle', color='red')
        plt.xlabel('Real')
        plt.ylabel('Complex')
        plt.legend()
        plt.grid(True)
        plt.plot([x_pos, self.h], [y_pos, self.k], 'k--', alpha=0.5)
        plt.axis('equal')
        plt.title('Circle Fit to Mode Data')
        plt.show()

    def choose_points(self):
        '''Allows the user to test out numbers of points to see how they look'''
        print('Performing circle fit at ' + str(self.freq_est) + ' Hz. Frequency range: ' + str(self.freq_min) + '-' + str(self.freq_max))
        while True:
            self.fit_circle()
            self.calculate_resonant_frequency()
            self.plot_circle()
            accept = str(input("Do you accept this fit? [y/n]"))
            if accept == 'y':
                break
            self.points = int(input("Enter the number of points: "))
            self.filter_data(self.points)

        self.calculate_damping()
        self.calculate_modal_parameters()

        self.summarize_results()
        self.plot_comparison()


    def plot_angles(self):
        spline = UnivariateSpline(self.frequencies, self.angles, s=0)
        frequencies_dense = np.linspace(self.freq_min, self.freq_max, 1000)
        angles_dense = spline(frequencies_dense)

        plt.figure(figsize=(6, 6))
        plt.plot(self.frequencies, self.angles, 'o', label='Angles', color='b')
        plt.plot(frequencies_dense, angles_dense, label='Spline Fit', color='r')
        plt.plot(self.resonant_frequency, self.theta, 'o', label='Estimate', color='g')
        plt.xlabel('Frequency')
        plt.ylabel('Angle')
        plt.title('Angle vs Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()

    def summarize_results(self):
        #print(f'Fitted circle center: ({self.h:.2f}, {self.k:.2f})')
        #print(f'Fitted circle radius: {self.r:.2f}')
        print(f'Resonant Frequency: {self.resonant_frequency:.1f}')
        print(f'Damping: {self.damping:.5f}')
        print(f'Damping standard dev: {self.damping_std:.5f}')
        print(f'Modal Constant: {self.A:.5f}')
        print(f'Mod Const Phase: {np.degrees(self.phase):.5f}')
        print(f'B: {self.B:.5f}')

    def plot_comparison(self):
        #filtered_data = self.data[(self.data['freq (Hz)'] >= freq_min) & (self.data['freq (Hz)'] <= freq_max)]
        #freq = filtered_data['freq (Hz)']
        #real = filtered_data['real']
        #cplx = filtered_data['complex']
        freq = self.freq
        real = self.real
        cplx = self.cplx

        freq_sim = np.linspace(min(freq), max(freq), 500)
        omega = freq_sim * 2 * np.pi
        w_r = self.resonant_frequency * 2 * np.pi
        # Compute complex response function
        alpha = self.A / (w_r ** 2 - omega ** 2 + 1j * self.damping * w_r ** 2) #+ self.B
        # Separate real and imaginary parts
        alpha_real = np.real(alpha)
        alpha_imag = np.imag(alpha)

        #plot real
        plt.figure(figsize=(12, 12))
        plt.subplot(2, 1, 1)
        plt.plot(freq_sim, alpha_real, label='Simulated Real Part')
        plt.scatter(freq, real, color='red', label='Experimental Real Part')
        plt.xlabel('Frequency')
        plt.ylabel('Real Part')
        plt.title('Real Part of Frequency Response')
        plt.legend()
        plt.grid(True)

        #plot imaginary
        plt.subplot(2,1,2)
        plt.plot(freq_sim, alpha_imag, label='Simulated Imaginary Part')
        plt.scatter(freq, cplx, color='red', label='Experimental Imaginary Part')
        plt.xlabel('Frequency')
        plt.ylabel('Imaginary Part')
        plt.title('Imaginary Part of Frequency Response')
        plt.legend()
        plt.grid(True)
        plt.show()


