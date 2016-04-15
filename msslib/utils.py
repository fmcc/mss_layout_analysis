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

###
### Functional utils
def identity(x):
    return x

def compose(*functions):
    """ A compose function which I've taken directly from:
        https://mathieularose.com/function-composition-in-python/
        """
    def compose2(g, h):
        return lambda x: g(h(x))
    return f.reduce(compose2, functions, lambda x: x)

def applier(*functions):
    """ Create a function which returns a tuple of results of 
        applying the bound functions to a provided variable. 
        """
    return lambda x: tuple(func(x) for func in functions)

def to_dict_list(t:[tuple]):
    """ Transforms a list of (key, value) tuples into a dictionary of
        lists. 
        """
    d = c.defaultdict(list)
    for k, v in t:
        d[k].append(v)
    return d

def default(f,a,b):
    return b if f(a) else a

isNone = f.partial(op.eq, None)

###
### Path utils

def listpaths(d):
    """ Create a list of all paths for files in a directory """ 
    return [os.path.join(d, f) for f in os.listdir(d)]

def img_path(path, page, ext='.png'):
    return os.path.join(path, page.filename + ext)

###
### Scaling utils

def modify_tuple(t,i,a):
    l = list(t)
    op.setitem(l,i,a)
    return tuple(l)

def slice_tuple(s):
    """ Convert a slice to a tuple with appropriate default values for None."""
    return (default(isNone, s.start, 1), s.stop, default(isNone, s.step, 1))

def tuple_slice(s):
    return slice(*s)

def scale_tuple(a,b):
    """ Scales a tuple by a scaling factor """
    return map(lambda x,y: round(op.mul(x,y)), a, b)

def scale_tuples(a,b):
    """ Scales a number of tuples by a scaling factor """
    return map(lambda x,y: tuple(scale_tuple(x, it.repeat(y))), a, b)

def scale_slice(sl, s):
    """ Scales a slice by a scaling factor. 
        Ensures that it returns a valid step value (>1).
        The scaling factor is reversed  """
    valid_step = lambda x: tuple_slice(
            default(lambda a: op.lt(a[-1], 1), x, modify_tuple(x,-1,1))
            )
    return tuple(valid_step(s_t) for s_t in scale_tuples(map(slice_tuple, sl), reversed(s)))

def scale_img(i, s, interpolation=0):
    """ Resizes an image on the basis of a tuple of scaling factors."""
    i = sp.misc.toimage(i)
    s = scale_tuple(i.size, s)
    return sp.misc.fromimage(i.resize(s, interpolation))

def to_greyscale(i):
    i = sp.misc.toimage(i)
    return sp.misc.fromimage(i.convert('L'))

def h_w(i):
    """ height and width from image, regardless of format """
    return i.shape[:2]

def pad_img(img, w, v=128):
    if len(img.shape) == 3:
        t = ((w,w),(w,w),(0,0))
    else: 
        t = ((w,w),(w,w))
    return np.lib.pad(img, t, mode='constant', constant_values=v)

def win_iter(img_shape, win_shape, step):
    """ Returns an iterator of slices over a shape """
    h,w  = img_shape
    w_h, w_w = win_shape
    for i in range(0, h-w_h+1 ,step):
        for j in range(0, w-w_w+1 ,step):
            # (slice(),slice())
            yield np.s_[i:i+w_h:1, j:j+w_w:1]
