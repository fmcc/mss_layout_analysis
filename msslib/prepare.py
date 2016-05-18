import numpy as np
from scipy import misc
from msslib.page_prima.page import PrimaPage
from msslib.utils import *

def prepare_fft_image(img: np.ndarray, win_size: int):
    """ Rescales an image and adds a mid-grey border.
        """
    return pad_img(to_greyscale(img), win_size-1)

def page_img_opener(page: PrimaPage):
    """ Generates function to open images and crop to a page boundary.
        """
    return lambda path: misc.imread(path)[page.border.as_slice()]

def page_path_img_opener(page_path: str):
    """ As the function it wraps, but opens the file.  """
    return page_img_opener(PrimaPage(page_path))

def img_scaler(scale):
    """ Generates a function that will scale one or more images by the provided
        scaling function. 
        """
    return lambda *args: [scale_img(a, scale) for a in args]

def img_scaler(scale):
    """ Generates a function that will scale one or more images by the provided
        scaling function. 
        """
    return lambda *args: [scale_img(a, scale) for a in args]

def img_resizer(size):
    """ Generates a function that will resize one or more images to the provided
        size. 
        """
    return lambda *args: [misc.imresize(a, size) for a in args]

def open_image_label(page_path, img_path, label_path): 
    """ Opens the page image and label and crops both to the maximum bounding
        box of the page border.
        """
    page = PrimaPage(page_path)
    p_img_open = page_img_opener(page)
    img = p_img_open(img_path)
    l_img = p_img_open(label_path)
    return img, l_img 

