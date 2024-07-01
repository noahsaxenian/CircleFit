import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
import plotting
import pandas as pd

# System parameters
m = np.array([0.1, 0.1, 0.1])  # Masses of each degree of freedom
k = np.array([30000, 20000, 30000])  # Spring constants
c = np.array([1, 1, 1])  # Damping coefficients
DOF = len(m)

# Mass matrix
M = np.diag(m)
# Damping matrix
C = np.diag(c)

# Stiffness matrix (assuming a chain with each mass connected by a spring)
K = np.zeros((DOF, DOF))
for i in range(DOF - 1):
    K[i, i] += k[i] + k[i + 1]
    K[i, i + 1] += -k[i + 1]
    K[i + 1, i] += -k[i + 1]

K[DOF-1, DOF-1] = k[DOF-1]
print(K)

# Initial conditions (displacement and velocity)
u0 = np.zeros(DOF)
v0 = np.zeros(DOF)
v0[0] = 1.0

# Time parameters
t_start = 0.0
t_end = 10.0
dt = 0.001
timesteps = int((t_end - t_start) / dt) + 1

# Initialize arrays to store results
u = np.zeros((DOF, timesteps))
v = np.zeros((DOF, timesteps))
a = np.zeros((DOF, timesteps))
t = np.linspace(t_start, t_end, timesteps)

# Initial conditions
u[:, 0] = u0
v[:, 0] = v0

# Simulation loop
for i in range(1, timesteps):
    # Calculate acceleration
    a[:, i - 1] = np.linalg.solve(M, -C @ v[:, i - 1] - K @ u[:, i - 1])

    # Update velocity and displacement using Euler's method
    v[:, i] = v[:, i - 1] + a[:, i - 1] * dt
    u[:, i] = u[:, i - 1] + v[:, i] * dt


plt.figure(figsize=(10, 4))
# Plotting results
for i in range(DOF):

    plt.subplot(3, 1, 1)
    plt.plot(t, u[i, :], label=f'DOF {i+1}')
    plt.ylabel('Displacement')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(t, v[i, :], label=f'DOF {i+1}')
    plt.ylabel('Velocity')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(t, a[i, :], label=f'DOF {i+1}')
    plt.ylabel('Acceleration')
    plt.xlabel('Time')
    plt.legend()

plt.tight_layout()
plt.show()

# Compute FFT
fft_result = np.fft.fft(u[0])

# Calculate frequencies
freq = np.fft.fftfreq(timesteps, d=dt)
positive_freq_indices = np.where(freq >= 0)
freq_positive = freq[positive_freq_indices]
fft_positive = fft_result[positive_freq_indices]

real = np.real(fft_positive)
imag = np.imag(fft_positive)
mag = np.sqrt(real**2 + imag**2)
phase = np.atan2(imag, real)

plotting.plot_mag_and_phase(freq_positive, mag, phase)

# Create a new dataframe to store the results
fake_data = pd.DataFrame({
    'freq (Hz)': freq_positive,
    'real': real,
    'complex': imag
})

# Save the receptance data to a new TSV file
fake_data.to_csv('c:/Users/noahs/Documents/ceeo/modal stuff/Code/data/Plate/Plate 03/csv/fake_data_2.tsv', sep='\t', index=False)