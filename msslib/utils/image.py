"""
Functions for image manipulation. 
"""
from scipy import misc
from PIL import Image
import numpy as np
from .modify import *

def scale_img(i, s, interpolation=0):
    """ Resizes an image on the basis of a tuple of scaling factors.
        """
    i = misc.toimage(i)
    s = scale_tuple(i.size, s)
    return misc.fromimage(i.resize(s, interpolation))

def resize_img(i, s, interpolation=0):
    """ Resizes an image to the provided height and width, which are flipped to maintain
        the same ordering as is used elsewhere in numpy. 
        misc.imresize would do the same, it just defaults to bilinear interpolation
        rather than nearest, which is required for scaling labels. Doing it this way
        for consistency with scale_img. 
        """
    i = misc.toimage(i)
    return misc.fromimage(i.resize(s[::-1], interpolation))

def to_greyscale(i):
    """ Convert an image to greyscale using PIL. 
        """
    i = misc.toimage(i)
    return misc.fromimage(i.convert('L'))

def h_w(i):
    """ Height and width from image, regardless of format.  
        """
    return i.shape[:2]

def pad_img(img, w, v=128):
    """ Pads an image with a border of size w containing value v.
        """
    if len(img.shape) == 3:
        t = ((w,w),(w,w),(0,0))
    else: 
        t = ((w,w),(w,w))
    return np.lib.pad(img, t, mode='constant', constant_values=v)

def overlay_imgs(img: np.ndarray, overlay: np.ndarray, opacity=0.5):
    """ Takes two images and overlays the latter over the former at the 
        given opacity. 
        """
    img = misc.toimage(img).convert("RGBA")
    overlay = misc.toimage(overlay).convert("RGBA")
    new_img = Image.blend(img, overlay, opacity)
    return misc.fromimage(new_img)
