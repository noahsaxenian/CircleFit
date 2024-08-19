from modal_model import ModalModel
import pandas as pd
import numpy as np
from stl import mesh
from animation_pyvista import *

freq_range = (120, 1200)
num_locations = 26
driving_point_location = 26
data = np.empty((num_locations, num_locations), dtype=object)

for i in range(1, num_locations + 1):
    if i < 10:
        file_path = f'C:/Users/noahs/Documents/ceeo/modal stuff/data/Plate/Plate_full_26/csv/Plate_full_26 H_00{i}_trf.tsv'
    else:
        file_path = f'C:/Users/noahs/Documents/ceeo/modal stuff/data/Plate/Plate_full_26/csv/Plate_full_26 H_0{i}_trf.tsv'
    raw_data = pd.read_csv(file_path, delimiter='\t')
    filtered_data = raw_data[(raw_data['freq (Hz)'] >= freq_range[0]) & (raw_data['freq (Hz)'] <= freq_range[1])]
    data[i-1, driving_point_location-1] = filtered_data

locations = [(-60.0, 60.0, 0.0), (-30.0, 60.0, 0.0), (0.0, 60.0, 0.0),
             (30.0, 60.0, 0.0), (60.0, 60.0, 0.0), (-60.0, 30.0, 0.0),
             (-30.0, 30.0, 0.0), (0.0, 30.0, 0.0), (30.0, 30.0, 0.0),
             (60.0, 30.0, 0.0), (-60.0, 0.0, 0.0), (-30.0, 0.0, 0.0),
             (0.0, 0.0, 0.0), (30.0, 0.0, 0.0), (60.0, 0.0, 0.0),
             (-60.0, -30.0, 0.0), (-30.0, -30.0, 0.0), (0.0, -30.0, 0.0),
             (30.0, -30.0, 0.0), (60.0, -30.0, 0.0), (-60.0, -60.0, 0.0),
             (-30.0, -60.0, 0.0), (0.0, -60.0, 0.0), (30.0, -60.0, 0.0),
             (60.0, -60.0, 0.0), (0.0, 15.0, 0.0)]

plate = ModalModel(data, locations)

mesh_path = 'C:/Users/noahs/Downloads/fine_square_mesh.stl'
stl_mesh = mesh.Mesh.from_file(mesh_path)
plate.set_mesh(stl_mesh)

plate.fit_all(interactive=True)
plate.correct_modal_properties()
plate.calculate_mode_shapes(driving_point=driving_point_location-1)


#plate.save_to_file('plate')


