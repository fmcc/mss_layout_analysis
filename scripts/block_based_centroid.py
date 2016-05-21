# Imports required for CLI Script
import click
import os

import functools as f
import itertools as it
from collections import Counter

from scipy.cluster import vq
# msslib imports 
from msslib.prepare import *
from msslib.sample import *
from msslib.features import *
from msslib.cluster import *

def _train_centroids(no_of_samples:int, window_size:int, block_size: int, page_size:tuple, page_paths:[str]):
    # Initialise a function to get a window of this size from an image
    windower = f.partial(win_centred_on, window=window_size)
    # Initialise a function to shift a point to accomodate a border from this window size. 
    shifter = f.partial(point_shift, window=window_size)
    # Initialise a scaling function for images. 
    resizer = img_resizer(page_size)
    # Open the two images 
    img, label = open_image_label(*page_paths)
    # Scale both images down
    img, label = resizer(img, label)
    # Create an image for sampling with FFT 
    f_img = prepare_fft_image(img, window_size)
    #define methods to access images.
    access_img = img_accessor(img, identity)
    access_label = img_accessor(label, identity)
    access_f_img = img_accessor(f_img, compose(windower, shifter))
    # Define the sampling function
    make_observations = compose(prepare_features, real_fft, std_dev_contrast_stretch, access_f_img)
    samples = []

    for w in win_iter(page_size, block_size):
        labelled_obs = take_n_samples(no_of_samples, random_point_in_window(w), applier(access_label, make_observations))
        block_labels = [a[0] for a in labelled_obs]
        block_obs = [a[1] for a in labelled_obs]
        most_common_label = Counter(block_labels).most_common(1)[0][0] 
        avg_block_obs = calculate_centroid(block_obs)
        samples.append((most_common_label, avg_block_obs))

    # Group samples by labels
    collected_samples = to_dict_list(samples)
    # Calculate and return centroids 
    return labelled_centroids(collected_samples)

def centroids_from_pages(paths):
    _get_centroids = f.partial(_train_centroids, 10, 41, 20, (1200, 900))
    return labelled_centroids(to_dict_list(it.chain.from_iterable(map(_get_centroids, paths))))

def new_labelled_page(no_of_samples:int, window_size:int, block_size:int, page_size: tuple, labelled_centroids:[tuple], page_paths:[str]):
    ### Duplication from above
    windower = f.partial(win_centred_on, window=window_size)
    shifter = f.partial(point_shift, window=window_size)
    resizer = img_resizer(page_size)
    img, label = open_image_label(*page_paths)
    img, label = resizer(img, label)
    f_img = prepare_fft_image(img, window_size)
    access_img = img_accessor(img, identity)
    access_label = img_accessor(label, identity)
    access_f_img = img_accessor(f_img, compose(windower, shifter))
    make_observations = compose(prepare_features, real_fft, std_dev_contrast_stretch, access_f_img)
    ### End of duplication
    labels = [a[0] for a in labelled_centroids]
    centroids = np.asarray([a[1] for a in labelled_centroids])
    new_label = np.zeros_like(label)
    for w in win_iter(page_size, block_size):
        block_obs = take_n_samples(no_of_samples, random_point_in_window(w), make_observations)
        # Need to expand dims since it'll only be one item
        avg_block_obs = np.expand_dims(calculate_centroid(block_obs), axis=0)
        codes, dist = vq.vq(avg_block_obs, centroids)
        # Only one code being matched here. 
        new_label[w] = labels[codes[0]]
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

    paths = list(filter(lambda x: 'RN' in x, listpaths(label_dir)))
    random.shuffle(paths) 
    # We take 50 images to average over as a training set.
    training_paths = paths[:50] 
    
    trained_centroids = centroids_from_pages(map(path_formatter, training_paths))
    label_page = compose(f.partial(new_labelled_page, 10, 41, 20, (1200, 900), trained_centroids), path_formatter)
    # And keep the rest to process with the resulting data. 
    process_paths = paths[50:]

    create_output_path = compose(f.partial(format_path, output_dir, 'png'), only_basename)
    for p in process_paths:
        page_out = label_page(p)
        path_out = create_output_path(p)
        misc.imsave(path_out, page_out)

if __name__ =='__main__':
    average_clustering()
