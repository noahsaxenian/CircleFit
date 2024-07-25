import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class SimulatedFRF():

    def __init__(self, omega_r, A, eta, freq_range, quality_factors=None, res_freqs=None):

        # arrays of length N (number of modes)
        self.omega_r = np.array(omega_r)
        self.A = np.array(A)
        self.eta = np.array(eta)
        self.n = len(omega_r)
        if quality_factors:
            self.quality_factors = np.array(quality_factors)
        else: self.quality_factors = np.ones(self.n)

        # parameters for residuals
        self.residual_frequencies = res_freqs
        self.residuals = False
        self.r_w1 = None
        self.r_w2 = None
        self.r_k1 = None
        self.r_k2 = None

        # arrays to hold simulated function
        self.freq_range = freq_range
        self.frequencies = np.linspace(self.freq_range[0], self.freq_range[1],
                                  (self.freq_range[1] - self.freq_range[0]) * 10)
        self.omegas = self.frequencies * 2 * np.pi
        self.alpha = np.zeros(len(self.frequencies)) + 0j
        self.alpha_corrected = np.zeros(len(self.frequencies)) + 0j
        self.accelerance = np.zeros(len(self.frequencies)) + 0j
        self.generate_alpha()

    def define_residual_parameters(self, r_w1, r_w2, r_k1, r_k2):
        self.r_w1 = r_w1
        self.r_w2 = r_w2
        self.r_k1 = r_k1
        self.r_k2 = r_k2
        self.residuals = True
        self.generate_alpha()

    def calculate_residuals(self, data):
        filtered_data = data[(data['freq (Hz)'] >= self.freq_range[0]) & (data['freq (Hz)'] <= self.freq_range[1])]
        data_cplx = filtered_data['real'].values + 1j * filtered_data['complex'].values

        ### calculate residuals as pseudo-modes
        # choice of natural frequencies of pseudo modes
        if self.residual_frequencies:
            r_f1, r_f2 = self.residual_frequencies
            self.r_w1 = r_f1 * 2 * np.pi
            self.r_w2 = r_f2 * 2 * np.pi
        else:
            self.r_w1 = (self.freq_range[0] - 100) * 2 * np.pi
            self.r_w2 = (self.freq_range[-1] + 100) * 2 * np.pi

        # stiffness estimated as average of difference over five points
        self.r_k1 = np.mean(
            [1 / ((1 - (self.omegas[i] ** 2 / self.r_w1 ** 2)) * (data_cplx[i] - self.alpha[i])) for i in range(5)])

        mode1 = ((1 / self.r_k1) / (1 - (self.omegas ** 2 / self.r_w1 ** 2)))  # lower residual mode
        alpha_cor_1 = self.alpha + mode1

        self.r_k2 = np.mean(
            [1 / ((1 - (self.omegas[-i] ** 2 / self.r_w2 ** 2)) * (data_cplx[-i] - alpha_cor_1[-i])) for i in
             range(1, 6)])

        mode2 = ((1 / self.r_k2) / (1 - (self.omegas ** 2 / self.r_w2 ** 2)))  # higher residual mode

        # re-correct mode1 in case mode2 messes up mode1
        alpha_cor_2 = self.alpha + mode2
        self._k1 = np.mean(
            [1 / ((1 - (self.omegas[i] ** 2 / self.r_w1 ** 2)) * (data_cplx[i] - alpha_cor_2[i])) for i in range(5)])
        mode1 = ((1 / self.r_k1) / (1 - (self.omegas ** 2 / self.r_w1 ** 2)))

        self.alpha_corrected = self.alpha + mode1 + mode2

    def generate_alpha(self):
        for A, omega_r, eta in zip(self.A, self.omega_r, self.eta):
            self.alpha += A / (omega_r ** 2 - self.omegas ** 2 + 1j * eta * omega_r ** 2)

        if self.residuals:
            residual_1 = ((1 / self.r_k1) / (1 - (self.omegas ** 2 / self.r_w1 ** 2)))
            residual_2 = ((1 / self.r_k2) / (1 - (self.omegas ** 2 / self.r_w2 ** 2)))
            self.alpha_corrected = self.alpha + residual_1 + residual_2

        return self.alpha

    def generate_accelerance(self):
        self.accelerance = -self.alpha * self.omegas**2


    def plot_mag_and_phase(self, data):
        filtered_data = data[(data['freq (Hz)'] >= self.freq_range[0]) & (data['freq (Hz)'] <= self.freq_range[1])]

        real = np.real(self.alpha)
        imag = np.imag(self.alpha)
        magnitude = np.sqrt(real ** 2 + imag ** 2)
        phase = np.arctan2(imag, real)

        real_cor = np.real(self.alpha_corrected)
        imag_cor = np.imag(self.alpha_corrected)
        magnitude_cor = np.sqrt(real_cor ** 2 + imag_cor ** 2)
        phase_cor = np.arctan2(imag_cor, real_cor)

        data_freqs = filtered_data['freq (Hz)'].values
        data_real = filtered_data['real'].values
        data_imag = filtered_data['complex'].values
        data_mag = np.sqrt(data_real ** 2 + data_imag ** 2)
        data_phase = np.arctan2(data_imag, data_real)


        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(4, 1)
        # Create the first subplot (3/4 of the area)
        ax1 = fig.add_subplot(gs[0:3, 0])
        ax1.plot(data_freqs, data_mag, label='Experimental')
        ax1.plot(self.frequencies, magnitude, label='Reconstructed')
        ax1.plot(self.frequencies, magnitude_cor, label='With Residuals')
        ax1.set_xlabel('Frequency')
        ax1.set_ylabel('Magnitude')
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.legend()
        ax1.grid(True)

        # Create the second subplot (1/4 of the area)
        ax2 = fig.add_subplot(gs[3, 0])
        ax2.plot(data_freqs, data_phase, label='Experimental')
        ax2.plot(self.frequencies, phase, label='Simulated')
        ax2.plot(self.frequencies, phase_cor, label='With Residuals')
        ax2.set_xlabel('Frequency')
        ax2.set_ylabel('Phase')
        ax2.set_xscale('log')
        ax2.legend()
        ax2.grid(True)

        # Adjust layout
        plt.tight_layout()
        plt.show()
