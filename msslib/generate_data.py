import functools as f
from scipy import misc
from .data import *
from .observe import *
from .utils import *

def prepare_page_accessors(window_size:int or tuple, img_path: str, label_path: str):
    # Initialise a function to get a window of this size from an image
    windower = f.partial(win_centred_on, window=window_size)
    # Initialise a function to shift a point to accomodate a border from this window size. 
    shifter = f.partial(point_shift, window=window_size)
    # Open the two images 
    img = misc.imread(img_path)
    label = misc.imread(label_path)
    # Create a bordered greyscale image for fft windows. 
    f_img = prepare_fft_image(img, window_size)
    access_img = img_accessor(img, identity)
    access_label = img_accessor(label, identity)
    access_f_img = img_accessor(f_img, compose(windower, shifter))
    return access_img, access_label, access_f_img

def random_observations_within_page(img_path: str, label_path: str, no_of_samples=10000, window_size=41):
    """ This is a bit of a hacky yin just now.  
        """
    random_coord = random_xy_coord_gen(0,1200,0,900)
    img, label, f_img = prepare_page_accessors(window_size, img_path, label_path) 
    make_observations = compose(prepare_features, real_fft, std_dev_contrast_stretch, f_img)
    samples = take_n_samples(no_of_samples, random_coord, applier(label, make_observations))
    return np.asarray([s[1] for s in samples]), np.asarray([s[0] for s in samples])

def perform_rowwise_labelling(img_path: str, label_path: str, classifier, window_size=41):
    img, label, f_img = prepare_page_accessors(window_size, img_path, label_path) 
    make_observations = compose(prepare_features, real_fft, std_dev_contrast_stretch, f_img)
    new_label = np.zeros((1200,900))
    for i in range(new_label.shape[0]):
        obs = np.asarray([make_observations((i, j)) for j in range(new_label.shape[1])])
        # This is a genero function incase I want to do more than just clf.predict()...
        preds = classifier(obs)
        new_label[i] = preds
    return new_label

def perform_pair_rowwise_labelling(img_path: str, label_path: str, classifier1, classifier2, window_size=41):
    """ Even more of a hack than the above to allow me to do this with two classifiers without having to 
        read the same data twice. """
    img, label, f_img = prepare_page_accessors(window_size, img_path, label_path) 
    make_observations = compose(prepare_features, real_fft, std_dev_contrast_stretch, f_img)
    new_label1 = np.zeros((1200,900))
    new_label2 = np.zeros((1200,900))
    for i in range(new_label1.shape[0]):
        obs = np.asarray([make_observations((i, j)) for j in range(new_label1.shape[1])])
        # This is a genero function incase I want to do more than just clf.predict()...
        new_label1[i] = classifier1(obs)
        new_label2[i] = classifier2(obs)
    return new_label1, new_label2


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
