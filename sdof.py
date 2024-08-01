from scipy.optimize import least_squares
from scipy.interpolate import UnivariateSpline
import numpy as np

def circle_fit(frequencies, complex_points):
    real = np.real(complex_points)
    imag = np.imag(complex_points)
    mag = np.abs(complex_points)

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

    ### MODAL CONSTANT ###
    magA = omega_r * eta_r * 2 * r
    A = magA * np.cos(theta_r) + 1j * magA * np.sin(theta_r)

    ### RESULTS ###
    #print(f'fitted circle h, k, r: {h}, {k}, {r}')
    print(f'resonant frequency: {resonant_frequency}')
    print(f'damping coefficient: {eta_r}, percent spread: {damping_spread}')
    print(f'modal constant mag: {magA} and phase: {theta_r}')
    results = [omega_r, eta_r, A]

    return omega_r, eta_r, A