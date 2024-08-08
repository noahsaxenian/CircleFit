from modal_analysis import ModalAnalysis
import pandas as pd
import numpy as np

freq_range = (200, 250)
num_locations = 16
data = np.empty((num_locations, num_locations), dtype=object)
for i in range(1, 17):
    if i < 10:
        file_path = f'C:/Users/noahs/Documents/ceeo/modal stuff/Code/data/Plate/PlateFull1/csv/PlateFull1 H_00{i}_trf.tsv'
    else:
        file_path = f'C:/Users/noahs/Documents/ceeo/modal stuff/Code/data/Plate/PlateFull1/csv/PlateFull1 H_0{i}_trf.tsv'
    raw_data = pd.read_csv(file_path, delimiter='\t')
    filtered_data = raw_data[(raw_data['freq (Hz)'] >= freq_range[0]) & (raw_data['freq (Hz)'] <= freq_range[1])]
    data[i-1, 6] = filtered_data


# # example with pre identified peaks that "work"
# freq_range = [100, 11000]
# residuals = None

locations = [
    (0, 3), (1, 3), (2, 3), (3, 3),
    (0, 2), (1, 2), (2, 2), (3, 2),
    (0, 1), (1, 1), (2, 1), (3, 1),
    (0, 0), (1, 0), (2, 0), (3, 0)
]

plate = ModalAnalysis(data, locations)

plate.fit_all()
plate.correct_modal_properties()
plate.calculate_mode_shapes(6)

for i in range(plate.n):
    plate.plot_mode_shape(i)
