import scipy as sp
import numpy as np
import operator as op
import functools as f
import itertools as it 

def default(f,a,b):
    return b if f(a) else a

isNone = f.partial(op.eq, None)

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

def pad_img(img, w, v=127):
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
