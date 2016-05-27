import click
import os
import random

import functools as f
import itertools as it

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from msslib.utils import *

def load_flat_data(p):
    d = np.load(p)
    w = np.reshape(d, (np.product(d.shape[:2]), d.shape[2]))
    return w

def load_flat_label(p):
    l = np.load(p)
    return l.flatten() / 10

def load_flat(paths):
    label_paths, data_paths = paths
    d1 = load_flat_data(data_paths[0])
    l1 = load_flat_label(label_paths[0])
    return f.reduce(lambda x, y: (
        np.append(x[0], load_flat_label(y[0])),
        np.concatenate((x[1], load_flat_data(y[1])), axis=0), 
    ), zip(label_paths[1:], data_paths[1:]), 
    (l1, d1))

def produce_reducer(paths, technique, components):
    labels, data = load_flat(paths)
    if technique == 'PCA':
        reducer = PCA(n_components=components)
        reducer.fit(data)
    elif technique == 'LDA':
        reducer = LinearDiscriminantAnalysis(n_components=components)
        reducer.fit(data, labels)
    return reducer

def random_selection(l_paths, d_paths, num):
    labels = []
    data = []
    for _ in range(num):
        i = random.randrange(0,len(l_paths))
        labels.append(l_paths[i])
        data.append(d_paths[i])
    return labels, data

@click.command()
@click.argument('technique')
@click.argument('samples', default=20)
@click.argument('components', default=100)
@click.argument('data_dir', type=click.Path(exists=True, resolve_path=True))
@click.argument('output_dir', type=click.Path(exists=True, resolve_path=True))
def reduce_dataset(technique, samples, components, data_dir, output_dir):
    all_file_paths = listpaths(data_dir)
    data_paths = sorted(list(filter(lambda x: '_data.npy' in x, all_file_paths)))
    label_paths = sorted(list(filter(lambda x: '_label.npy' in x, all_file_paths)))
    reducer = produce_reducer(random_selection(label_paths, data_paths, samples), technique, components) 
    for path in data_paths:
        d = np.load(path)
        flat_d = np.reshape(d, (np.product(d.shape[:2]), d.shape[2]))
        reduced_d = reducer.transform(flat_d)
        out_path = format_path(output_dir, 'npy', only_basename(path))
        #click.echo(out_path)
        np.save(out_path, reduced_d.reshape(d.shape[0], d.shape[1], reduced_d.shape[1]))

if __name__ == '__main__':
    reduce_dataset()

