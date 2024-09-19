from modal_model import ModalModel
import pandas as pd
import numpy as np
from stl import mesh
from animation_pyvista import *
import pickle

freq_range = (100, 3000)
num_locations = 25
driving_point_location = 7
data = np.empty((num_locations, num_locations), dtype=object)

for i in range(1, num_locations + 1):
    if i < 10:
        file_path = f'C:/Users/noahs/Documents/ceeo/modal stuff/data/wood_plate/wood_plate 01/csv/wood_plate 01 H_00{i}_trf.tsv'
    else:
        file_path = f'C:/Users/noahs/Documents/ceeo/modal stuff/data/wood_plate/wood_plate 01/csv/wood_plate 01 H_0{i}_trf.tsv'
    raw_data = pd.read_csv(file_path, delimiter='\t')
    filtered_data = raw_data[(raw_data['freq (Hz)'] >= freq_range[0]) & (raw_data['freq (Hz)'] <= freq_range[1])]
    data[i-1, driving_point_location-1] = filtered_data

locations = [(-55.0, 60.0, 0.0),
 (-25.0, 60.0, 0.0),
 (0.0, 60.0, 0.0),
 (25.0, 60.0, 0.0),
 (55.0, 60.0, 0.0),
 (-55.0, 30.0, 0.0),
 (-25.0, 30.0, 0.0),
 (0.0, 30.0, 0.0),
 (25.0, 30.0, 0.0),
 (55.0, 30.0, 0.0),
 (-55.0, 0.0, 0.0),
 (-25.0, 0.0, 0.0),
 (0.0, 0.0, 0.0),
 (25.0, 0.0, 0.0),
 (55.0, 0.0, 0.0),
 (-55.0, -30.0, 0.0),
 (-25.0, -30.0, 0.0),
 (0.0, -30.0, 0.0),
 (25.0, -30.0, 0.0),
 (55.0, -30.0, 0.0),
 (-55.0, -60.0, 0.0),
 (-25.0, -60.0, 0.0),
 (0.0, -60.0, 0.0),
 (25.0, -60.0, 0.0),
 (55.0, -60.0, 0.0)]

plate = ModalModel(data, locations)

mesh_path = 'C:/Users/noahs/Downloads/wood_plate_final.stl'
your_mesh = mesh.Mesh.from_file(mesh_path)
plate.set_mesh(your_mesh)

plate.fit_all(interactive=False)
plate.correct_modal_properties()
plate.calculate_mode_shapes(driving_point=driving_point_location-1)

plate.set_landmark_vertices(100, [0, 70, -70, 70])
plate.calculate_distance_matrix()
