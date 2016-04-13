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

def nine_coords_around(point:tuple, window: tuple or int):
    """ Creates a list of nine coordinates regularly spaced around 
        a point that all fall within a window. 
        """
    win_x, win_y = format_window(window)
    s_x = point[0] - win_x // 2
    s_y = point[1] - win_y // 2
    xs = [s_x, point[0], s_x+win_x-1]
    ys = [s_y, point[1], s_y+win_y-1]
    return [(x,y) for x in xs for y in ys]

def win_centred_on(point:tuple, window: tuple or int):
    """ Generate a window slice centred on a given x,y coordinate. 
        """ 
    win_x, win_y = format_window(window)
    s_x = point[0] - win_x // 2
    s_y = point[1] - win_y // 2
    return np.s_[s_x:s_x+win_x,s_y:s_y+win_y]

def point_shift(point:tuple, window: tuple or int):
    """ Moves a point by height-1 and width-1 of a given window to 
        accommodate the margin added to a image. 
        """
    win_x, win_y = format_window(window)
    return (point[0] + win_x-1, point[1] + win_y-1)

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
