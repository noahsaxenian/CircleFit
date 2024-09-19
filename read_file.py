import pickle
from modal_model import ModalModel
import numpy as np
from stl import mesh

def read_file(filename):
    with (open(f'{filename}.pkl', 'rb') as file):
        model = pickle.load(file)
    return model

plate = read_file('plate_cutout')
plate.fit_frf(6, 6)

# plate.auto_select_landmarks(spacing=3, grid_points=15**2)
# plate.calculate_distance_matrix()
#
# plate.save_to_file('plate_cutout')
#
# plate.plot_mode_shape(1, 'spline')

# if input('Do you want to save the model? (y/n) ') == 'y':
#     plate.save_to_file('plate_cutout')