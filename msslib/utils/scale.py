import operator as op

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

