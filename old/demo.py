from modal_model import ModalAnalysis
import pandas as pd
import numpy as np

demo_num = 2

file_path = f'C:/Users/noahs/Documents/ceeo/modal stuff/Code/data/demo/demo 0{demo_num}/csv/demo 0{demo_num} H_001_trf.tsv'
data = pd.read_csv(file_path, delimiter='\t')


freq_range = [800, 2200]
residual_frequencies = (700, 2300)

locations = [(0,0)]

plate = ModalAnalysis(data, freq_range, locations, res_freqs=residual_frequencies)

plate.curve_fit(data, 0, 0, interactive=True)

plate.calculate_mode_shapes(0)

plate.correct_modal_properties()

#plate.plot_mode_shape(0)