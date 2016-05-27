import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from msslib.utils import *
import matplotlib.pyplot as plt
import functools as f

data_dir = "/home/finlay/HMT/data/VenetusA/block_samples/"
all_file_paths = listpaths(data_dir)
data_paths = sorted(list(filter(lambda x: '_data.npy' in x, all_file_paths)))
label_paths = sorted(list(filter(lambda x: '_label.npy' in x, all_file_paths)))

def load_data(p):
    d = np.load(p)
    w = np.reshape(d, (np.product(d.shape[:2]), d.shape[2]))
    return w

def load_label(p):
    l = np.load(p)
    return l.flatten() / 10

d1 = load_data(data_paths[0])
l1 = load_label(label_paths[0])


labels, data = f.reduce(lambda x, y: (
        np.append(x[0], load_label(y[0])),
        np.concatenate((x[1], load_data(y[1])), axis=0), 
    ), zip(label_paths[1:5], data_paths[1:5]), 
    (l1, d1))

np.save('labels.npy', labels)
np.save('data.npy', data)
