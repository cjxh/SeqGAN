import numpy as np

def save_as_npy(file_name):
    data = np.loadtxt(file_name + '.txt')
    np.save(file_name, data)

save_as_npy('data/glove_vectors')
