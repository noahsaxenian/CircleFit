from modal_analysis import ModalAnalysis
import pandas as pd
import numpy as np

data = []
for i in range(1,5):
    file_path = f'C:/Users/noahs/Documents/ceeo/modal stuff/Siemens Plate Test/point{i}_data.tsv'
    data.append(pd.read_csv(file_path, delimiter='\t'))
for i in range(6,10):
    file_path = f'C:/Users/noahs/Documents/ceeo/modal stuff/Siemens Plate Test/point{i}_data.tsv'
    data.append(pd.read_csv(file_path, delimiter='\t'))
for i in range(11,15):
    file_path = f'C:/Users/noahs/Documents/ceeo/modal stuff/Siemens Plate Test/point{i}_data.tsv'
    data.append(pd.read_csv(file_path, delimiter='\t'))
for i in range(16,20):
    file_path = f'C:/Users/noahs/Documents/ceeo/modal stuff/Siemens Plate Test/point{i}_data.tsv'
    data.append(pd.read_csv(file_path, delimiter='\t'))

for i in range(len(data)):
    data[i]['magnitude'] = np.abs(data[i]['real'] + 1j*data[i]['complex'])
combined_data = pd.concat(data).groupby(level=0).max()


#freq_range = [1290, 1320]
freq_range = [200, 250]

locations = [
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 3), (1, 2), (1, 1), (1, 0),
    (2, 0), (2, 1), (2, 2), (2, 3),
    (3, 3), (3, 2), (3, 1), (3, 0)
]

plate = ModalAnalysis(combined_data, freq_range, locations)

for i in range(16):
    plate.curve_fit(data[i], i, 10, interactive=False)

plate.calculate_mode_shapes(10)

plate.plot_mode_shape(0)