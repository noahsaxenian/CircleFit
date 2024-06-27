import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

class CircleFit:
    def __init__(self, data, freq_est, points=10):
        self.data = data
        self.freq_est = freq_est
        self.points = points
        self.freq_min = None
        self.freq_max = None
        self.freq = None
        self.real = None
        self.cplx = None
        self.h = None
        self.k = None
        self.r = None
        self.frequencies = None
        self.angles = None
        self.resonant_frequency = None
        self.omega = None
        self.theta = None
        self.damping = None
        self.damping_std = None
        self.A = None
        self.phase = None
        self.B = 0

    def run(self):
        self.filter_by_points(self.points)
        self.fit_circle()
        self.calculate_resonant_frequency()
        self.calculate_damping()
        self.calculate_modal_parameters()

    def filter_data(self):
        filtered_data = self.data[(self.data['freq (Hz)'] >= self.freq_min) & (self.data['freq (Hz)'] <= self.freq_max)]
        self.freq = filtered_data['freq (Hz)']
        self.real = filtered_data['real']
        self.cplx = filtered_data['complex']
        self.frequencies = self.freq.values

    def filter_by_points(self, num_points):
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

        self.freq_min = min(self.freq)
        self.freq_max = max(self.freq)

    @staticmethod
    def residuals(params, x, y):
        h, k, r = params
        return (x - h) ** 2 + (y - k) ** 2 - r ** 2


    def fit_circle(self):
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
        result = least_squares(self.residuals, initial_guess, args=(x, y))

        # Update circle parameters
        self.h, self.k, self.r = result.x

    def calculate_resonant_frequency(self):
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

    def calculate_damping(self):
        # Split frequencies and angles into lower and higher arrays
        split = self.resonant_frequency
        lower_frequencies = self.frequencies[self.frequencies < split]
        higher_frequencies = self.frequencies[self.frequencies >= split]
        lower_angles = self.angles[self.frequencies < split]
        higher_angles = self.angles[self.frequencies >= split]

        def damping(w_r, w_a, w_b, theta_a, theta_b):
            return float((w_a ** 2 - w_b ** 2) / (w_r ** 2 * (np.tan(theta_a / 2) + np.tan(theta_b / 2))))

        damping_coeffs = []
        theta = self.theta

        for i in range(min(len(lower_angles), len(higher_angles))):
            w_a = higher_frequencies[i] * 2 * np.pi
            w_b = lower_frequencies[-i-1] * 2 * np.pi
            theta_a = abs(higher_angles[i] - theta)
            theta_b = abs(lower_angles[-i-1] - theta)
            #n = damping(self.resonant_frequency, w_a, w_b, theta_a, theta_b)
            n = (w_a ** 2 - w_b ** 2) / (self.omega ** 2 * (np.tan(theta_a / 2) + np.tan(theta_b / 2)))
            damping_coeffs.append(n)
        #print(damping_coeffs)

        self.damping = np.mean(damping_coeffs)
        self.damping_std = np.std(damping_coeffs)
        return self.damping, self.damping_std

    def calculate_modal_parameters(self):
        self.A = (self.omega ** 2) * self.damping * (2 * self.r)
        x_pos = self.h + self.r * np.cos(self.theta)
        y_pos = self.k + self.r * np.sin(self.theta)
        self.phase = np.atan2(y_pos, x_pos)

        B_x = self.h + self.r * np.cos(self.theta + np.pi)
        B_y = self.k + self.r * np.sin(self.theta + np.pi)
        self.B = np.sqrt(B_x ** 2 + B_y ** 2)
        return self.A, self.phase


    def plot_circle(self):
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
        while True:
            self.points = int(input("Enter the number of points: "))
            self.run()
            self.plot_circle()
            accept = str(input("Do you accept this fit? [y/n]"))
            if accept == 'y':
                break
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
        alpha = self.A / (w_r ** 2 - omega ** 2 + 1j * self.damping * w_r ** 2) + self.B
        # Separate real and imaginary parts
        alpha_real = np.real(alpha)
        alpha_imag = np.imag(alpha)

        #plot real
        plt.figure(figsize=(12, 12))
        plt.subplot(2, 1, 1)
        plt.plot(freq_sim, alpha_real, label='Simulated Real Part')
        plt.scatter(freq, real, color='red', label='Experimental Real Part')
        plt.xlabel('Frequency (ω)')
        plt.ylabel('Real Part')
        plt.title('Real Part of Frequency Response')
        plt.legend()
        plt.grid(True)

        #plot imaginary
        plt.subplot(2,1,2)
        plt.plot(freq_sim, alpha_imag, label='Simulated Imaginary Part')
        plt.scatter(freq, cplx, color='red', label='Experimental Imaginary Part')
        plt.xlabel('Frequency (ω)')
        plt.ylabel('Imaginary Part')
        plt.title('Imaginary Part of Frequency Response')
        plt.legend()
        plt.grid(True)
        plt.show()


