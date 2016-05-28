import numpy as np
from ..utils import *

def prepare_features(arr: np.ndarray):
    """ Prepare an FFT matrix for use as a set of features. """
    return arr.flatten()

def real_fft(arr: np.ndarray):
    """ Performs a 2D FFT on an array and returns the real components. 
        This gives us just the frequency components of the transform, rather
        than the phase. 
        """
    return np.abs(np.fft.fft2(arr))

def gaussian_matrix(size: tuple, width: tuple, centre=None):
    """
        """
    x = np.arange(0, size[0], 1, float)
    y = np.arange(0, size[1], 1, float)[:,np.newaxis]
    if not centre:
        c_x = size[0] // 2
        c_y = size[1] // 2
    else:
        c_x, c_y = centre
    return np.exp(-4*np.log(2) * (((x-c_x)**2 / width[0]**2) + ((y-c_y)**2/width[1]**2)))

def gaussian_weight(arr: np.ndarray):
    """ Weights an array on the basis of a Gaussian distribution from 1 at
        the centre, to ~0 at the edges of the array. 
        """
    gaussian = gaussian_matrix(arr.shape, (arr.shape[0]/2, arr.shape[1]/2))
    return arr*gaussian

def gaussian_weighter(size:tuple or int):
    """ Provides a function for weighting with a gaussian matrix of a 
        defined size. 
        """
    size = as_pair(size)
    gaussian = gaussian_matrix(size, (size[0]/2, size[1]/2))
    return lambda arr: arr*gaussian

def std_dev_contrast_stretch(arr: np.ndarray, n=2):
    """ Performs a contrast stretch from +/-2σ around the mean to 
        -1 to 1. 
        """
    sigma = arr.std()*n
    m = arr.mean()
    return np.interp(arr,[m-sigma,m+sigma],[-1,1])

