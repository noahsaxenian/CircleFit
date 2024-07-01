import numpy as np
import matplotlib.pyplot as plt
import plotting


def simulate_sdof_system(mass, spring_constant, damping, t_max=10, dt=0.0001, impulse_force=1.0):
    # Generate time array
    t = np.arange(0, t_max, dt)
    num_points = len(t)

    # Placeholder for the system's response (position over time)
    positions = np.zeros(num_points)

    # Placeholder for velocities and accelerations
    velocities = np.zeros(num_points)
    accelerations = np.zeros(num_points)

    # Initial conditions
    velocities[0] = impulse_force / mass  # Instantaneous force converted to initial velocity

    # Perform simulation using numerical integration (Euler method for demonstration)
    for i in range(1, num_points):
        # Calculate acceleration using the equation: m*a + c*v + k*x = 0
        accelerations[i - 1] = (-damping * velocities[i - 1] - spring_constant * positions[i - 1]) / mass

        # Update velocities and positions using Euler method
        velocities[i] = velocities[i - 1] + accelerations[i - 1] * dt
        positions[i] = positions[i - 1] + velocities[i - 1] * dt

    # Plotting the system's response over time
    plt.figure()
    plt.plot(t, positions, label="Displacement")
    plt.title('System Response Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement')
    plt.legend()
    plt.grid()
    plt.show()

    # Compute FFT
    fft_result = np.fft.fft(positions)

    # Calculate frequencies
    freq = np.fft.fftfreq(len(positions), d=dt)
    positive_freq_indices = np.where(freq >= 0)
    freq_positive = freq[positive_freq_indices]
    fft_positive = fft_result[positive_freq_indices]

    real = np.real(fft_positive)
    imag = np.imag(fft_positive)
    mag = np.sqrt(real ** 2 + imag ** 2)
    phase = np.atan2(imag, real)

    plotting.plot_mag_and_phase(freq_positive, mag, phase)

    return t, positions



# Example usage
mass = 1  # Mass
spring_constant = 1000  # Spring constant
damping = 0.5  # Damping coefficient

# Simulate the system
t, positions = simulate_sdof_system(mass, spring_constant, damping, impulse_force=1.0)

