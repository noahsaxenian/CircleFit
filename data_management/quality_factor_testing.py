import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def fit_circle(x_points, y_points):
    ''' Function to fit circle
    Kasa method gets initial guess
    Least squares for final fit '''

    def residuals(params, x, y):
        # Used for circle fitting
        h, k, r = params
        distances = np.sqrt((x - h) ** 2 + (y - k) ** 2)
        return distances - r

    x = x_points
    y = y_points

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
    h, k, r = result.x

    # Calculate the mean square deviation (quality factor)
    residuals_final = residuals(result.x, x, y)
    msd = np.mean(residuals_final**2)
    nmsd = msd / r**2
    quality_factor = 1 - 10*nmsd
    print(f'quality factor: {quality_factor}')

    # # calculate r^2 value
    # radial_distances = np.sqrt((x-h)**2 + (y-k)**2)
    # mean_distance = np.mean(radial_distances)
    # tss = np.sum((radial_distances - mean_distance) ** 2)
    #
    # #residuals = radial_distances - r
    # ssr = np.sum(residuals_final ** 2)
    # print(ssr, tss)
    #
    # r_squared = 1 - (ssr / tss)
    #
    # print(f'R squared: {r_squared}')

    # plotting
    points = np.linspace(0, 2 * np.pi, 100)
    x_fit = h + r * np.cos(points)
    y_fit = k + r * np.sin(points)
    plt.figure()
    plt.plot(x, y, 'o', label='Data')
    plt.plot(x_fit, y_fit, label='Fitted Circle', color='red')
    plt.axis('equal')
    plt.show()

x = np.array([0, 0, 1, 1])
y = np.array([0, 1, 0, 2])

fit_circle(x, y)
fit_circle(10000*x, 10000*y)
fit_circle(np.array([0,0,10]), np.array([0,10,0]))