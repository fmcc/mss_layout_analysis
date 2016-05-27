import scipy as sp
import numpy as np
import collections as c
import operator as op
import functools as f
import itertools as it 
import os

def format_window(window):
    """ Deals with a window being an int or tuple of two ints,
        consistently returning a tuple of ints to allow for
        square or rectangular windows. """
    if isinstance(window, tuple):
        win_x, win_y = window
    else:
        win_x = win_y = window
    return win_x, win_y

def as_pair(p: int or float or tuple) -> (int or float):
    """ Deals with a window being an int or tuple of two ints,
        consistently returning a tuple of ints to allow for
        square or rectangular windows. """
    if isinstance(p, tuple):
        a, b = p
    else:
        a = b = p
    return a, b
