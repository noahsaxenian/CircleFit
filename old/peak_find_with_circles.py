import pandas as pd
import numpy as np
from sdof import circle_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')

file_path = 'C:/Users/noahs/Documents/ceeo/modal stuff/Code/data/Plate/PlateFull1/csv/PlateFull1 H_007_trf.tsv'
data = pd.read_csv(file_path, delimiter='\t')

freq_range = [100, 2200]
start_index = np.abs(data['freq (Hz)'] - freq_range[0]).idxmin()
end_index = np.abs(data['freq (Hz)'] - freq_range[1]).idxmin()
filtered_data = data.iloc[start_index:end_index + 1]

freq = np.array(filtered_data['freq (Hz)'].values)
real = np.array(filtered_data['real'].values)
imag = np.array(filtered_data['complex'].values)
cplx = real + 1j * imag

num = 100
for i in range(0, len(freq) - num, 30):
    fr = freq[i:i+num]
    re = real[i:i+num]
    im = imag[i:i+num]
    results = circle_fit(fr, re, im)
    if results['quality_factor'] > 0.999:
        resonant_frequency = results['resonant_frequency']
        h = results['h']
        k = results['k']
        r = results['r']
        theta_r = results['theta_r']
        print(resonant_frequency)

        '''Create a nice plot of the fitted circle'''
        magnitudes = np.sqrt(re ** 2 + im ** 2)
        max_mag = np.max(magnitudes)

        points = np.linspace(0, 2 * np.pi, 100)
        x_fit = h + r * np.cos(points)
        y_fit = k + r * np.sin(points)
        x_pos = h + r * np.cos(theta_r)
        y_pos = k + r * np.sin(theta_r)

        plt.figure(figsize=(12, 6))
        plt.subplot(1,2,1)
        plt.plot(fr, magnitudes)
        plt.plot(resonant_frequency, max_mag, 'x', color='g')
        plt.xlabel('Frequency')
        plt.ylabel('Magnitude')

        plt.subplot(1,2,2)
        plt.plot(re, im, 'o', label='Data')
        label_string = 'Estimate = ' + str(resonant_frequency) + ' Hz'
        plt.plot(x_pos, y_pos, 'o', label=label_string, color='green')
        plt.plot(x_fit, y_fit, label='Fitted Circle', color='red')
        plt.xlabel('Real')
        plt.ylabel('Complex')
        plt.legend()
        plt.grid(True)
        plt.plot([x_pos, h], [y_pos, k], 'k--', alpha=0.5)
        plt.axis('equal')
        plt.title('Circle Fit to Mode Data')
        plt.show()