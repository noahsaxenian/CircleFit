from modal_analysis import ModalAnalysis
import pandas as pd
import numpy as np

data = []
for i in range(1,17):
    if i<10:
        file_path = f'C:/Users/noahs/Documents/ceeo/modal stuff/Code/data/Plate/PlateFull1/csv/PlateFull1 H_00{i}_trf.tsv'
    else:
        file_path = f'C:/Users/noahs/Documents/ceeo/modal stuff/Code/data/Plate/PlateFull1/csv/PlateFull1 H_0{i}_trf.tsv'
    data.append(pd.read_csv(file_path, delimiter='\t'))

for i in range(len(data)):
    data[i]['magnitude'] = np.abs(data[i]['real'] + 1j*data[i]['complex'])
combined_data = pd.concat(data).groupby(level=0).max()

peaks = None

# freq_range = [4000, 5000]
# residuals = None
# peaks = [4463]

# freq_range = [200, 300]
# residuals = (150, 400)
# peaks = None

freq_range = [100, 11000]
residuals = None
peaks = [230, 286, 1070, 1717, 2580, 2987, 3713, 10461]

locations = [
    (0,3), (1,3), (2,3), (3,3),
    (0,2), (1,2), (2,2), (3,2),
    (0,1), (1,1), (2,1), (3,1),
    (0,0), (1,0), (2,0), (3,0)
]

plate = ModalAnalysis(combined_data, freq_range, locations, res_freqs=residuals, peaks=peaks)

for i in range(16):
    plate.curve_fit(data[i], i, 6, interactive=False)

plate.correct_modal_properties()
plate.calculate_mode_shapes(6)

for i in range(plate.n):
    plate.plot_mode_shape(i)