import matplotlib.pyplot as plt

def plot_real_and_imag(frequencies, real, imag):
    #plot real
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 1, 1)
    plt.plot(frequencies, real, label='Simulated Real Part')
    plt.xlabel('Frequency')
    plt.ylabel('Real Part')
    plt.title('Real Part of Frequency Response')
    plt.legend()
    plt.grid(True)

    #plot imaginary
    plt.subplot(2,1,2)
    plt.plot(frequencies, imag, label='Simulated Imaginary Part')
    plt.xlabel('Frequency')
    plt.ylabel('Imaginary Part')
    plt.title('Imaginary Part of Frequency Response')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_mag_and_phase(frequencies, magnitude, phase):
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 1, 1)
    plt.plot(frequencies, magnitude, label='Magnitude')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.title('Magnitude')
    plt.yscale('log')
    #plt.xscale('log')
    plt.legend()
    plt.grid(True)

    #plot phase
    plt.subplot(2,1,2)
    plt.plot(frequencies, phase, label='Phase')
    plt.xlabel('Frequency')
    plt.ylabel('Phase')
    plt.title('Phase')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_real_vs_imag(real, imag):
    plt.figure(figsize=(12, 12))
    plt.plot(real, imag, label='Phase')
    plt.xlabel('Real')
    plt.ylabel('Imag')
    plt.title('Real vs Imaginary')
    plt.legend()
    plt.grid(True)
    plt.show()