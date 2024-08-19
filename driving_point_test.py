from modal_model import ModalModel
import pandas as pd
import numpy as np

# file_path = 'C:/Users/noahs/Documents/ceeo/modal stuff/Code/data/Plate/PlateFull1/csv/PlateFull1 H_007_trf.tsv'
file_path = f'C:/Users/noahs/Documents/ceeo/modal stuff/data/Plate/Plate_full_26/csv/Plate_full_26 H_026_trf.tsv'

raw_data = pd.read_csv(file_path, delimiter='\t')

freq_range = [100, 2500]
filtered_data = raw_data[(raw_data['freq (Hz)'] >= freq_range[0]) & (raw_data['freq (Hz)'] <= freq_range[1])]

num_locations = 1
data = np.empty((num_locations, num_locations), dtype=object)
data[0,0] = filtered_data

locations = [(0,0)]

plate = ModalModel(data, locations)

plate.fit_frf(0, 0, interactive=True)