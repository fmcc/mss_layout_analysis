import random
import numpy as np
import itertools as it 
import collections as c
from msslib.utils import *

def random_xy_coord_gen(x1:int, x2:int, y1:int, y2:int):
    """ Create a generator for a random x,y coordinate within 
        an x and y range. 
        """
    return lambda: (random.randrange(x1,x2), random.randrange(y1,y2))

def win_centred_on(point:tuple, window: tuple or int):
    """ Generate a window slice centred on a given x,y coordinate. 
        """ 
    win_x, win_y = format_window(window)
    s_x = point[0] - win_x // 2
    s_y = point[1] - win_y // 2
    return np.s_[s_x:s_x+win_x,s_y:s_y+win_y]

def win_iter(img_shape: tuple, win_shape: tuple or int, step=False):
    """ Iterator of 2d slices over a numpy ndarray.
        Both img_shape and win_shape are in (height, width) format, 
        and iteration will be in row order. """
    win_y, win_x = format_window(win_shape)
    if not step:
        step_y, step_x = win_y, win_x
    else: 
        step_y, step_x = format_window(step)
    img_y, img_x = img_shape
    for i in range(0, img_y-win_y+1, step_y):
        for j in range(0, img_x-win_x+1, step_x):
            yield np.s_[i:i+win_y:1, j:j+win_x:1]

def point_iter(arr:np.ndarray):
    """ Iterator of indices in a numpy ndarray.
        Intended to be zipped with win_iter. 
        """
    y, x = arr.shape[0], arr.shape[1]
    for i in range(0, y):
        for j in range(0, x):
            yield np.s_[i,j]

def result_matrix(img_shape: tuple, win_shape: tuple or int, step=False, result_shape=(1,)):
    win_y, win_x = format_window(win_shape)
    if not step:
        step_y, step_x = win_y, win_x
    else: 
        step_y, step_x = format_window(step)
    img_y, img_x = img_shape
    y = int((img_y-win_y)/step_y) + 1
    x = int((img_x-win_x)/step_x) + 1
    return np.empty((y,x)+result_shape)

def mean_vector(vectors:[np.ndarray]):
    """ Mean vector from a list of vectors.  
        """
    return np.asarray(vectors).mean(axis=0)

def most_common(l:[]):
    """ Find the most common item in a list.  
        """
    return c.Counter(l).most_common(1)[0][0] 

def point_shift(point:tuple, window: tuple or int):
    """ Moves a point by height-1 and width-1 of a given window to 
        accommodate the margin added to a image. 
        """
    win_x, win_y = format_window(window)
    return (point[0] + win_x-1, point[1] + win_y-1)

def random_point_in_window(win: slice):
    """ Create a random point generator for a point within a 2D window.   
        """
    return random_xy_coord_gen(win[0].start, win[0].stop, win[1].start, win[1].stop)

def img_accessor(img:np.ndarray, modifier):
    """ Create a getter function which binds the image with a function
        modifying the selection provided. 
        """
    return lambda selection: img[modifier(selection)]

def take_n_samples(n:int, get_data, process_data):
    """ Given a function which provides a piece of data, and a function which
        processes that data and returns some data structure, get and process 
        n times, storing the results in a list which is returned. 
        """
    samples = []
    for _ in it.repeat(None, n):
        samples.append(process_data(get_data()))
    return samples

def sample_all_in_area(area_slice:slice, process_data):
    """ Iterates over indices in a area and return values.
        This is a poor way of doing this just now, but will do for now.
        """
    width = area_slice[0]
    height = area_slice[1]
    samples = []
    for i in range(width.start, width.stop): 
        for j in range(height.start, height.stop): 
            samples.append(process_data((i,j)))
    return samples

