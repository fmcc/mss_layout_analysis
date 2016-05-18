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
    """ Returns an iterator of slices over a shape """
    win_x, win_y = format_window(win_shape)
    if not step:
        step_x, step_y = win_x, win_y
    else: 
        step_x, step_y = format_window(step)
    img_x, img_y = img_shape
    for i in range(0, img_x-win_x+1, step_x):
        for j in range(0, img_y-win_y+1, step_y):
            yield np.s_[i:i+win_x:1, j:j+win_y:1]

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

