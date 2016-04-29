# Imports required for CLI Script
import click
import os

from msslib.utils import *
from msslib.prepare import *
from msslib.page_prima.page import PrimaPage
from msslib.evaluate import *
from scipy import misc
import numpy as np
import functools as f

def open_and_flatten(page_paths:[str]):
    page_path, label_path, result_path = page_paths
    page_opener = page_img_opener(PrimaPage(page_path))
    scaler = img_scaler((0.2,0.2))
    label = scaler(page_opener(label_path))[0]
    results = misc.imread(result_path)
    return label.flatten(), results.flatten()

@click.command()
@click.argument('page_dir', type=click.Path(exists=True, resolve_path=True))
@click.argument('label_dir', type=click.Path(exists=True, resolve_path=True))
@click.argument('results_dir', type=click.Path(exists=True, resolve_path=True))
def evaluate_results(page_dir, label_dir, results_dir):
    # Define a couple of functions to generate and group all the paths. 
    get_page_path = compose(f.partial(format_path, page_dir, 'xml'), only_basename)
    get_label_path = compose(f.partial(format_path, label_dir, 'png'), only_basename)
    get_results_path = compose(f.partial(format_path, results_dir, 'png'), only_basename)

    output_name = "%s_confusion_matrices.npy" % only_basename(results_dir)
    # One of these could really be identity
    path_formatter = applier(get_page_path, get_label_path, get_results_path)
    paths = map(path_formatter, listpaths(results_dir))
    label_results_pairs = map(open_and_flatten, paths) 
    label, results = f.reduce(lambda x,y: np.append(x[0], y[0]), np.append(x[1], y[1]), label_results_pairs)
    c_matrix = normalised_confusion_matrix(label, results)
    np.save(output_name, c_matrix)

if __name__ == '__main__':
    evaluate_results()
  
