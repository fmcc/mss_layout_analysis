#! /bin

# Imports required for CLI Script
import click
import os

import functools as f
import itertools as it
import numpy as np

from scipy.cluster import vq

# msslib imports 
from msslib.prepare import *
from msslib.sample import *
from msslib.features import *
from msslib.cluster import *

def centroids_from_page_samples(no_of_samples:int, window_size:int, page_scale:int or tuple, page_paths:[str]):
    # Initialise a weighting function for this window size
    weighter = gaussian_weighter(window_size)
    # Initialise a function to get a window of this size from an image
    windower = f.partial(win_centred_on, window=window_size)
    # Initialise a function to shift a point to accomodate a border from this window size. 
    shifter = f.partial(point_shift, window=window_size)
    # Initialise a scaling function for images. 
    scaler = img_scaler(page_scale)
    # Define the sampling function
    make_observations = compose(f.partial(prepare_features_slice, np.s_[:,:10]), real_fft, weighter, std_dev_contrast_stretch)
    # Open the two images 
    img, label = open_image_label(*page_paths)
    # Scale both images down
    img, label = scaler(img, label)
    # Create an image for sampling with FFT 
    f_img = prepare_fft_image(img, window_size)
    #define methods to access images.
    access_img = img_accessor(img, identity)
    access_label = img_accessor(label, identity)
    access_f_img = img_accessor(f_img, compose(windower, shifter))

    print(page_paths[0])
    # Create a random coordinate generator
    random_coord = random_xy_coord_gen(0,img.shape[0],0,img.shape[1])
    # Take samples
    samples = take_n_samples(no_of_samples, random_coord, applier(access_label, compose(make_observations, access_f_img)))   
    # Group samples by labels
    collected_samples = to_dict_list(samples)
    # Calculate and return centroids 
    return labelled_centroids(collected_samples)

def centroids_from_pages(paths):
    _get_centroids = f.partial(centroids_from_page_samples, 10000, 41, (0.2,0.2))
    return labelled_centroids(to_dict_list(it.chain.from_iterable(map(_get_centroids, paths))))

def img_slices(img_shape: tuple, step:int):
    height, width = img_shape
    height_steps = list(range(0, height, step)) + [height]
    width_steps = list(range(0, width, step)) + [width]
    slices = []
    for i in pairwise(height_steps):
        for j in pairwise(width_steps):
            slices.append(np.s_[i[0]:i[1],j[0]:j[1]])
    return slices

def new_labelled_page(no_of_samples:int, window_size:int, page_scale:int or tuple, labelled_centroids:[tuple], page_paths:[str]):
    ### Duplication from above
    weighter = gaussian_weighter(window_size)
    windower = f.partial(win_centred_on, window=window_size)
    shifter = f.partial(point_shift, window=window_size)
    scaler = img_scaler(page_scale)
    make_observations = compose(f.partial(prepare_features_slice, np.s_[:,:10]), real_fft, weighter, std_dev_contrast_stretch)
    img, label = open_image_label(*page_paths)
    img, label = scaler(img, label)
    f_img = prepare_fft_image(img, window_size)
    access_img = img_accessor(img, identity)
    access_label = img_accessor(label, identity)
    access_f_img = img_accessor(f_img, compose(windower, shifter))
    ### End of duplication
    labels = [a[0] for a in labelled_centroids]
    centroids = np.asarray([a[1] for a in labelled_centroids])
    new_label = np.zeros_like(label)
    for s in img_slices(new_label.shape, 80):
        unlabelled_samples = sample_all_in_area(s, applier(identity, compose(make_observations, access_f_img)))   
        coords = [a[0] for a in unlabelled_samples]
        observations = np.asarray([a[1] for a in unlabelled_samples])
        codes, dist = vq.vq(observations, centroids)
        for i, code in zip(coords, codes):
            new_label[i] = labels[code]
    return new_label

@click.command()
@click.argument('page_dir', type=click.Path(exists=True, resolve_path=True))
@click.argument('img_dir', type=click.Path(exists=True, resolve_path=True))
@click.argument('label_dir', type=click.Path(exists=True, resolve_path=True))
@click.argument('output_dir', type=click.Path(exists=True, resolve_path=True))
def average_clustering(page_dir, img_dir, label_dir, output_dir):
    # Define a couple of functions to generate and group all the paths. 
    get_page_path = compose(f.partial(format_path, page_dir, 'xml'), only_basename)
    get_img_path = compose(f.partial(format_path, img_dir, 'jpg'), only_basename)
    get_label_path = compose(f.partial(format_path, label_dir, 'png'), only_basename)
    
    # One of these could really be identity
    path_formatter = applier(get_page_path, get_img_path, get_label_path)

    paths = list(filter(lambda x: 'RN' in x, listpaths(page_dir)))
    random.shuffle(paths) 
    # We take 50 images to average over as a training set.
    training_paths = paths[:50] 
    
    trained_centroids = centroids_from_pages(map(path_formatter, training_paths))
    label_page = compose(f.partial(new_labelled_page, 10000, 41, (0.2,0.2), trained_centroids), path_formatter)
    # And keep the rest to process with the resulting data. 
    process_paths = paths[50:]

    create_output_path = compose(f.partial(format_path, output_dir, 'png'), only_basename)
    for p in process_paths:
        path_out = create_output_path(p)
        print(path_out)
        page_out = label_page(p)
        misc.imsave(path_out, page_out)

if __name__ =='__main__':
    average_clustering()
