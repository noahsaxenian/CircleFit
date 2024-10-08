from scipy.optimize import least_squares
from scipy.interpolate import UnivariateSpline
import numpy as np
import matplotlib.pyplot as plt


def filter_data(frequencies, magnitudes, freq_est, db_threshold=3, freq_range=None):
    ''' Filters data input to the inputted number of points
    surrounding frequency estimate '''

    # Find the index of the closest frequency to self.freq_est

    closest_index = np.abs(frequencies - freq_est).argmin()

    if freq_range is not None:
        # Calculate the start and end indices
        start_index = np.abs(frequencies - freq_range[0]).argmin()
        end_index = np.abs(frequencies - freq_range[1]).argmin()
    else:
        peak_amplitude = magnitudes[closest_index]
        # Calculate the threshold amplitude in linear scale (assuming amplitude is in linear scale)
        threshold_amplitude = peak_amplitude / (10 ** (db_threshold / 20))
        # Create a boolean mask where amplitude is above the threshold
        above_threshold = magnitudes >= threshold_amplitude
        below_peak = magnitudes <= peak_amplitude * 2
        # Find the start and end indices of the continuous segment around the peak
        start_index = closest_index
        end_index = closest_index


        on_hump = magnitudes == -1
        last_mag = magnitudes[closest_index]
        decreasing = False
        for i in range(closest_index, -1, -1):
            if magnitudes[i] > last_mag:
                if decreasing:
                    break
                decreasing = False
            if magnitudes[i] < last_mag:
                decreasing = True
            last_mag = magnitudes[i]
            on_hump[i] = True

        last_mag = magnitudes[closest_index]
        decreasing = False
        for i in range(closest_index, len(frequencies)):
            if magnitudes[i] > last_mag:
                if decreasing:
                    on_hump[i-1] = False
                    break
                decreasing = False
            if magnitudes[i] < last_mag:
                decreasing = True
            last_mag = magnitudes[i]
            on_hump[i] = True

        # Expand the range to the left
        for i in range(closest_index - 1, -1, -1):
            if above_threshold[i] and on_hump[i]:
                start_index = i
            else:
                break
        # Expand the range to the right
        for i in range(closest_index + 1, len(frequencies)):
            if above_threshold[i] and on_hump[i]:
                end_index = i
            else:
                break

        end_index += 1

        # Ensure indices are within the valid range
        start_index = max(0, start_index)
        end_index = min(len(frequencies) - 1, end_index)

    if end_index - start_index < 4:
        print('Range includes too few points, increasing to 5 points')
        # Calculate the start and end indices
        start_index = max(0, closest_index - 2)
        end_index = min(len(frequencies) - 1, closest_index + 2)

    # # Filter the data
    # freq = frequencies[start_index:end_index]
    # real = real[start_index:end_index]
    # imag = imaginary[start_index:end_index]
    # mag = magnitudes[start_index:end_index]

    return start_index, end_index

def circle_fit(frequencies, real, imag, plot=False):

    cplx = real + 1j * imag
    mag = np.abs(cplx)

    ### FIT CIRCLE ###
    def residuals(params, x, y):
        # Used for circle fitting
        h, k, r = params
        distances = np.sqrt((x - h) ** 2 + (y - k) ** 2)
        return distances - r

    # Kasa method for initial guess
    A = np.column_stack((real, imag, np.ones_like(real)))
    b = real ** 2 + imag ** 2
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    h_kasa = 0.5 * c[0]
    k_kasa = 0.5 * c[1]
    r_kasa = np.sqrt(c[2] + h_kasa ** 2 + k_kasa ** 2)
    initial_guess = [h_kasa, k_kasa, r_kasa]

    # Perform least squares fitting
    result = least_squares(residuals, initial_guess, args=(real, imag))
    h, k, r = result.x

    # Calculate the mean square deviation (quality factor)
    residuals_final = residuals(result.x, real, imag)
    msd = np.mean(residuals_final ** 2)
    quality_factor = 1 - 10 * (msd / r ** 2)
    if quality_factor < 0: quality_factor = 0

    ### CALCULATE RESONANT FREQUENCY ###
    angles = np.unwrap(np.arctan2(imag-k, real-h))
    try:
        # Attempt to fit the UnivariateSpline
        spline = UnivariateSpline(frequencies, angles, s=0)
        frequencies_dense = np.linspace(frequencies[0], frequencies[-1], len(frequencies)*10)
        angles_dense = spline(frequencies_dense)
        dtheta_df = np.gradient(angles_dense, frequencies_dense)
        index = np.argmin(dtheta_df)
        resonant_frequency = frequencies_dense[index]
        theta_r = spline(resonant_frequency)

        # plt.figure()
        # plt.plot(frequencies, angles)
        # plt.plot(frequencies_dense, angles_dense)
        # plt.show()

    except Exception as e:
        print(f"Spline fitting failed: {e}")
        print("Using max point")
        resonant_frequency = frequencies[np.argmax(mag)]
        theta_r = angles[np.argmax(mag)]

    omega_r = resonant_frequency * 2 * np.pi

    ### CALCULATE DAMPING ###
    # Split frequencies and angles into lower and higher groups
    lower_frequencies = frequencies[frequencies < resonant_frequency]
    higher_frequencies = frequencies[frequencies > resonant_frequency]
    lower_angles = angles[frequencies < resonant_frequency]
    higher_angles = angles[frequencies > resonant_frequency]

    # Calculate coefficient for each pair
    damping_coeffs = []
    for i in range(min(len(lower_angles), len(higher_angles))):
        omega_a = higher_frequencies[i] * 2 * np.pi
        omega_b = lower_frequencies[-i - 1] * 2 * np.pi
        theta_a = abs(higher_angles[i] - theta_r)
        theta_b = abs(lower_angles[-i - 1] - theta_r)
        # Equation for damping coefficient
        n = (omega_a ** 2 - omega_b ** 2) / (omega_r ** 2 * (np.tan(theta_a / 2) + np.tan(theta_b / 2)))
        damping_coeffs.append(n)

    eta_r = np.mean(damping_coeffs)
    damping_std = np.std(damping_coeffs)
    damping_spread = (damping_std / eta_r) * 100
    if np.isnan(eta_r):
        print(f"failed to find eta, setting to 0")
        eta_r = 0


    ### MODAL CONSTANT ###
    magA = omega_r * eta_r * 2 * r
    A = magA * np.cos(theta_r) + 1j * magA * np.sin(theta_r)

    if plot:
        points = np.linspace(0, 2 * np.pi, 100)
        x_fit = h + r * np.cos(points)
        y_fit = k + r * np.sin(points)
        x_pos = h + r * np.cos(theta_r)
        y_pos = k + r * np.sin(theta_r)

        plt.figure(figsize=(12, 6))
        title = f'Resonant Freq: {resonant_frequency} | Damping: {eta_r} | Magnitude: {magA} | phase: {np.angle(A)}'
        plt.suptitle(title)

        plt.subplot(1, 2, 1)
        plt.plot(frequencies, mag)
        plt.plot(resonant_frequency, max(mag), 'x', color='g')
        plt.xlabel('Frequency')
        plt.ylabel('Magnitude')

        plt.subplot(1, 2, 2)
        plt.plot(real, imag, 'o', label='Data')
        label_string = 'Estimate = ' + str(resonant_frequency) + ' Hz'
        plt.plot(x_pos, y_pos, 'o', label=label_string, color='green')
        plt.plot(x_fit, y_fit, label='Fitted Circle', color='red')
        plt.xlabel('Real')
        plt.ylabel('Imaginary')
        plt.legend()
        plt.grid(True)
        plt.plot([x_pos, h], [y_pos, k], 'k--', alpha=0.5)
        plt.axis('equal')
        plt.title('Circle Fit to Mode Data')
        plt.show()

    ### RESULTS ###
    results = {
        "resonant_frequency": resonant_frequency,
        "omega_r": omega_r,
        "eta_r": eta_r,
        "A": A,
        "theta_r": theta_r,
        "quality_factor": quality_factor,
        "h": h,
        "k": k,
        "r": r
    }

    return results


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def just_fit(frequencies, real, imag):
    # Example complex function to fit
    def complex_func(x, A_real, A_imag, omega_r, eta_r):
        # A is split into real and imaginary components
        A = A_real + 1j * A_imag
        return (A * x * 1j) / (omega_r ** 2 - x ** 2 + 1j * eta_r * omega_r ** 2)

    # Residual function that combines real and imaginary parts
    def residuals(params, x_data, y_data):
        A_real, A_imag, omega_r, eta_r = params
        model = complex_func(x_data, A_real, A_imag, omega_r, eta_r)
        residual = np.concatenate([np.real(model - y_data), np.imag(model - y_data)])
        return residual

    x_data = frequencies * 2 * np.pi
    y_data = real + imag * 1j

    # Use an initial guess from circle fit or any other source
    circle = circle_fit(frequencies, real, imag)
    # Split A into real and imaginary parts if necessary
    A_real = np.real(circle["A"])
    A_imag = np.imag(circle["A"])
    initial_guess = [A_real, A_imag, circle["omega_r"], circle["eta_r"]]
    print(f"Initial guess: {initial_guess}")

    # Reconstruct using the initial guess
    initial_model = complex_func(x_data, *initial_guess)

    # Fit the data using least_squares
    result = least_squares(residuals, initial_guess, args=(x_data, y_data))

    # Extract the fitted parameters
    fitted_params = result.x
    A_fitted = fitted_params[0] + 1j * fitted_params[1]
    omega_r_fitted = fitted_params[2]
    eta_r_fitted = fitted_params[3]

    print(f"Fitted A: {A_fitted}, omega_r: {omega_r_fitted}, eta_r: {eta_r_fitted}")

    # Reconstruct using the fitted parameters
    final_model = complex_func(x_data, fitted_params[0], fitted_params[1], fitted_params[2], fitted_params[3])

    # Compute magnitude and phase for data, initial guess, and final fit
    magnitude_data = np.abs(y_data)
    phase_data = np.angle(y_data)

    magnitude_initial = np.abs(initial_model)
    phase_initial = np.angle(initial_model)

    magnitude_final = np.abs(final_model)
    phase_final = np.angle(final_model)

    # Plot magnitude and phase on one figure
    plt.figure(figsize=(10, 5))

    # Plot magnitude
    plt.subplot(1, 2, 1)
    plt.plot(frequencies, magnitude_data, label='Data (Magnitude)', marker='o', linestyle='None')
    plt.plot(frequencies, magnitude_initial, label='Initial Guess from Circle Fit (Magnitude)', linestyle='--')
    plt.plot(frequencies, magnitude_final, label='Final Fit (Magnitude)', linestyle='-')
    plt.title('Magnitude')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()

    # Plot phase
    plt.subplot(1, 2, 2)
    plt.plot(frequencies, phase_data, label='Data (Phase)', marker='o', linestyle='None')
    plt.plot(frequencies, phase_initial, label='Initial Guess from Circle Fit (Phase)', linestyle='--')
    plt.plot(frequencies, phase_final, label='Final Fit (Phase)', linestyle='-')
    plt.title('Phase')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (radians)')
    plt.legend()

    plt.suptitle('Reconstructed Data with Initial Guess and Final Fit (Magnitude and Phase)')
    plt.tight_layout()
    plt.show()

    return fitted_params
