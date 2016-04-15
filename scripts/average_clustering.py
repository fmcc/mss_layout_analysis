#! /bin

# Imports required for CLI Script
import click
import os

import functools as f
import itertools as it

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
    make_observations = compose(prepare_features, real_fft, weighter, std_dev_contrast_stretch)
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
    # Create a random coordinate generator
    random_coord = random_xy_coord_gen(0,img.shape[0],0,img.shape[1])
    # Take samples
    samples = take_n_samples(no_of_samples, random_coord, applier(access_label, compose(make_observations, access_f_img)))   
    # Group samples by labels
    collected_samples = to_dict_list(samples)
    # Calculate and return centroids 
    return labelled_centroids(collected_samples)

def centroids_from_pages(paths):
    _get_centroids = f.partial(centroids_from_page_samples, 10000, 41  (0.2,0.2))
    return labelled_centroids(to_dict_list(it.chain.from_iterable(map(_get_centroids, paths))))

@click.command()
@click.argument('page_dir', type=click.Path(exists=True, resolve_path=True))
@click.argument('img_dir', type=click.Path(exists=True, resolve_path=True))
@click.argument('label_dir', type=click.Path(exists=True, resolve_path=True))
@click.argument('output_dir', type=click.Path(exists=True, resolve_path=True))
def average_clustering(page_dir, img_dir, label_dir, output_dir):

    print(os.listdir(img_dir))

    return 
    recto_paths = list(filter(lambda x: 'RN' in x, listpaths(page_dir)))
    random.shuffle(recto_paths) 
    # We take 50 images to average over as a training set.
    training_paths = paths[:50] 
    # And keep the rest to process with the resulting data. 
    process_paths = paths[50:]

if __name__ =='__main__':
    average_clustering()
