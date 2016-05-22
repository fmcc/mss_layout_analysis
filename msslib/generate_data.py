import functools as f

from msslib.sample import *
from msslib.features import *
from msslib.prepare import *

def prepare_page_accessors(window_size:int or tuple, page_size:tuple, page_paths:[str]):
    """ Returns a tuple of functions for accessing the original image,
        the label image, and the bordered greyscale image. 
        """
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
    return access_img, access_label, access_f_img

def random_observations_within_blocks(page_size: tuple, block_size: tuple or int, no_of_samples: int, group_label, group_obs, label_func, obs_func):
    """  """
    labels = result_matrix(page_size, block_size)
    # Make a single observation to work out how big the results will be - 
    # avoiding calculating elsewhere and passing the variable about.  
    _s = obs_func((0,0))
    data = result_matrix(page_size, block_size, result_shape=_s.shape)
    blocks = win_iter(page_size, block_size)
    result_locations = point_iter(labels)
    for b, i in zip(blocks, result_locations):
        labels_and_obs = take_n_samples(no_of_samples, random_point_in_window(b), applier(label_func, obs_func))
        labels[i] = group_label([l[0] for l in labels_and_obs])
        data[i] = group_obs([l[1] for l in labels_and_obs])
    return labels, data
