from collections import namedtuple
import numpy as np
import scipy.misc

Scale = namedtuple('Scale', ['x','y'])

def _scale(a: float, b: float) -> int:
    return int(np.floor(a*b))

def _default_scale(a: int, s: float, d: int):
    v = a if a else d 
    r = _scale(v,s)
    return r if r >= d else d 

class ScaledImage():
    def __init__(self, img: np.ndarray, scale=Scale(1,1), shift=Scale(0,0)):
        self.img = img
        self.scale = scale
        self.shift = shift

    def _scale_index(self, a, sc=None, sh=None):
        if not sc:
            sc = self.scale.y
        if not sh:
            sh = self.shift.y

        if isinstance(a, tuple):
            return (self._scale_index(a[0], self.scale.y, self.shift.y),
                    self._scale_index(a[1], self.scale.x, self.shift.x))

        if isinstance(a, slice):
            return slice(*(_default_scale(a.start + sh, sc, 0),
                _default_scale(a.stop + sh, sc, -1), 
                _default_scale(a.step, sc, 1)))

        if isinstance(a, int):
            return _scale(a + sh, sc)

    def get(self, a):
        k = self._scale_index(a)
        return self.img[k]

def scale_img(i: np.ndarray, s: Scale, interpolation=0):
    """ Resizes an image on the basis of a tuple of scaling factors."""
    i = scipy.misc.toimage(i)
    new_dim = (_scale(i.size[0], s.x), _scale(i.size[1], s.y))
    return scipy.misc.fromimage(i.resize(new_dim, interpolation))

