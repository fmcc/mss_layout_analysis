from scipy import misc
import numpy as np

def resize_img(i, s, interpolation=0):
    """ Resizes an image to the provided height and width, which are flipped to maintain
        the same ordering as is used elsewhere in numpy. 
        misc.imresize would do the same, it just defaults to bilinear interpolation
        rather than nearest, which is required for scaling labels. Doing it this way
        for consistency with scale_img. """
    i = misc.toimage(i)
    return misc.fromimage(i.resize(s[::-1], interpolation))

def to_greyscale(i):
    i = misc.toimage(i)
    return misc.fromimage(i.convert('L'))

def h_w(i):
    """ Height and width from image, regardless of format """
    return i.shape[:2]

def pad_img(img, w, v=128):
    if len(img.shape) == 3:
        t = ((w,w),(w,w),(0,0))
    else: 
        t = ((w,w),(w,w))
    return np.lib.pad(img, t, mode='constant', constant_values=v)

