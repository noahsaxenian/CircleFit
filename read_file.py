import pickle
from modal_model import ModalModel
import numpy as np
from stl import mesh

def read_file(filename):
    with (open(f'{filename}.pkl', 'rb') as file):
        model = pickle.load(file)
    return model

plate = read_file('plate')
plate.plot_frf(0,25)