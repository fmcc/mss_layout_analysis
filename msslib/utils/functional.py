"""
Functions used for programming in a functional style in Python.
"""
import collections as c
import functools as f
import itertools as it
import operator as op

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

def pairwise(iterable):
    """ From the itertools recipes.
        """
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)

def default(f,a,b):
    return b if f(a) else a

isNone = f.partial(op.eq, None)
