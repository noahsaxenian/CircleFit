from modal_analysis import ModalAnalysis
import pandas as pd
import numpy as np

data = []
file_path = 'C:/Users/noahs/Documents/ceeo/modal stuff/Code/data/Plate/PlateDrivingPointTest4/csv/PlateDrivingPointTest4 H_001_trf.tsv'
#file_path = 'C:/Users/noahs/Documents/ceeo/modal stuff/Siemens Plate Test/point13_data.tsv'
data.append(pd.read_csv(file_path, delimiter='\t'))

for i in range(len(data)):
    data[i]['magnitude'] = np.abs(data[i]['real'] + 1j*data[i]['complex'])

combined_data = pd.concat(data).groupby(level=0).max()


freq_range = [650, 1200]
residual_frequencies = (550, 1300)
#freq_range = [200, 350]
locations = [(0,0)]

plate = ModalAnalysis(combined_data, freq_range, locations, res_freqs=residual_frequencies)

plate.curve_fit(data[0], 0, 0, interactive=True)

plate.calculate_mode_shapes(0)

plate.plot_mode_shape(0)