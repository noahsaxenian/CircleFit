import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.interpolate import UnivariateSpline


# Function to define the circle model
def residuals(params, x, y):
    h, k, r = params
    return (x - h) ** 2 + (y - k) ** 2 - r ** 2


# Function to fit a circle to the data
def fit_circle(x, y):
    # Initial guess for the circle parameters (h, k, r)
    x_m = np.mean(x)
    y_m = np.mean(y)
    initial_guess = [x_m, y_m, np.mean(np.sqrt((x - x_m) ** 2 + (y - y_m) ** 2))]

    # Perform the least squares optimization
    result = least_squares(residuals, initial_guess, args=(x, y))

    h, k, r = result.x
    return h, k, r


w_guess = 286
points = 4

freq_min = w_guess - points
freq_max = w_guess + points

# Load your FRF data from a CSV file
file_path = 'c:/Users/noahs/Documents/ceeo/modal stuff/Code/data/Plate/Plate 03/csv/Plate 03 receptance.tsv'
data = pd.read_csv(file_path, delimiter='\t')

# Filter data to get desired frequencies
filtered_data = data[(data['freq (Hz)'] >= freq_min) & (data['freq (Hz)'] <= freq_max)]
freq = filtered_data['freq (Hz)']
real = filtered_data['real']
cplx = filtered_data['complex']
frequencies = freq.values

# Fit a circle to the real vs complex data
h, k, r = fit_circle(real, cplx)
points = np.linspace(0, 2 * np.pi, 100)
x_fit = h + r * np.cos(points)
y_fit = k + r * np.sin(points)

print(f'Fitted circle center: ({h:.2f}, {k:.2f})')
print(f'Fitted circle radius: {r:.2f}')

# Find angles to each point
raw_angles = np.arctan2(cplx - k, real - h) % (2 * np.pi)
angles = np.unwrap(raw_angles)

# Interpolate the angle data
spline = UnivariateSpline(frequencies, angles, s=0)
frequencies_dense = np.linspace(freq_min, freq_max, 1000)
angles_dense = spline(frequencies_dense)

# Compute the derivative of the angle with respect to frequency
dtheta_df = np.gradient(angles_dense, frequencies_dense)

# Find the frequency at which the derivative is maximum
resonant_frequency_index = np.argmin(dtheta_df)
resonant_frequency = round(frequencies_dense[resonant_frequency_index], 1)
theta = resonant_angle = spline(resonant_frequency)

print(f"Resonant Frequency: {resonant_frequency}")


# find damping
def damping(w_r, w_a, w_b, theta_a, theta_b):
    return float((w_a ** 2 - w_b ** 2) / (w_r ** 2 * (np.tan(theta_a / 2) + np.tan(theta_b / 2))))


lower = {}
higher = {}
for i in range(len(frequencies)):
    if frequencies[i] < resonant_frequency:
        lower[frequencies[i]] = angles[i]
    else:
        higher[frequencies[i]] = angles[i]

damping_coeffs = []
for w_a in higher:
    theta_a = abs(higher[w_a] - theta)
    for w_b in lower:
        theta_b = abs(lower[w_b] - theta)
        n = damping(resonant_frequency, w_a, w_b, theta_a, theta_b)
        damping_coeffs.append(n)

damping_avg = np.mean(damping_coeffs)
damping_std = np.std(damping_coeffs)

print('Damping:', damping_avg)
print('Damping standard dev: ', damping_std)

# estimate modal constant
A = (resonant_frequency ** 2) * damping_avg * (2 * r)
print('Modal Constant: ', A)

x_pos = h + r * np.cos(theta)
y_pos = k + r * np.sin(theta)

phase = np.atan2(y_pos, x_pos)
print('Mod Const Phase:', np.degrees(phase))

# Plot the real vs complex data and the fitted circle
plt.figure(figsize=(5, 10))
plt.subplot(2, 1, 1)
plt.plot(real, cplx, 'o', label='Data')
plt.plot(x_pos, y_pos, 'o', label='Estimate', color='green')
plt.plot(x_fit, y_fit, label='Fitted Circle', color='red')
plt.xlabel('Real')
plt.ylabel('Complex')
plt.legend()
plt.grid(True)

# plot line to resonant freq
plt.plot([x_pos, h], [y_pos, k], 'k--', alpha=0.5)

plt.axis('equal')
plt.title('Circle Fit to Mode Data')

plt.subplot(2,1, 2)
plt.plot(frequencies, angles, 'o', label='Angles', color='b')
plt.plot(frequencies_dense, angles_dense, label='Spline Fit', color='r')
plt.plot(resonant_frequency, theta, 'o', label='Estimate', color='g')
plt.xlabel('Frequency')
plt.ylabel('Angle')
plt.title('Angle vs Frequency')
plt.legend()
plt.grid(True)
plt.show()
