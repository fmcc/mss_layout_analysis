"""
Functions for matrix manipulation. 
"""
import numpy as np
from .modify import *

def scale_matrix(a: np.ndarray, s:int or tuple) -> np.ndarray:
    """ Uses Kronecker product to scale a matrix by a scaling factor, or 
        factors that consist of positive integers. 
        """
    return np.kron(a, np.ones(as_pair(s)))

def flatten_vector_matrix(a: np.ndarray) -> np.ndarray:
    """ Flattens a matrix so that the size of the final dimension is always preserved. 
        """
    return np.reshape(a, (-1, a.shape[-1]))

class Spiraliser():
    """ Flatten a matrix in a outward spiral from the centre
        and restore to the original shape. """

    def _gen_spiral(self, size: int):
        x = y = size // 2
        x_dir, y_dir = -1, 0
        out_arr = np.empty((size, size))
        x_pos_lim = x + 1
        x_neg_lim = x - 1
        y_pos_lim = y + 1
        y_neg_lim = y - 1
        for i in range(out_arr.size):
            out_arr[x,y] = i
            if x == x_pos_lim:
                x_pos_lim += 1
                x_dir, y_dir = -y_dir, x_dir
            if x == x_neg_lim:
                x_neg_lim -= 1
                x_dir, y_dir = -y_dir, x_dir        
            if y == y_pos_lim:
                y_pos_lim += 1
                x_dir, y_dir = -y_dir, x_dir
            if y == y_neg_lim:
                y_neg_lim -= 1
                x_dir, y_dir = -y_dir, x_dir                          
            x,y = x+x_dir, y+y_dir
        return out_arr.astype(int)

    def __init__(self, height: int, width: int):
        max_dim = max(height, width)
        x_shift = (max_dim-width)//2
        y_shift = (max_dim-height)//2
        spiral = self._gen_spiral(max_dim)
        self.flatten_ = np.zeros((height, width), dtype=int)
        self.reshape_ = np.zeros((self.flatten_.size,2), dtype=int)
        mask = np.zeros_like(spiral)
        mask[0+y_shift:height+y_shift,0+x_shift:width+x_shift] = 1
        counter = 0
        for i in range(spiral.size):
            c = np.where(spiral==i)
            if mask[c]:
                out_c = c[0]-y_shift, c[1]-x_shift
                self.flatten_[out_c] = counter
                self.reshape_[counter] = out_c
                counter +=1
    
    def flatten(self, arr: np.ndarray):
        """ Flattens an array the spiraliser way."""
        if len(arr.shape) == 2:
            out_arr = np.empty(arr.size)
        else:
            h,w = arr.shape[:2]
            out_arr = np.empty((h*w,)+arr.shape[2:])
        for i in range(out_arr.shape[0]):
            out_arr[i] = arr[np.where(self.flatten_==i)]
        return out_arr
        
    
    def reshape(self, arr: np.ndarray):
        """ Reshapes a matrix with some index tricks. """
        if len(arr.shape) == 1:
            out_arr = np.empty(self.flatten_.shape[:2])
        else:
            out_arr = np.empty(self.flatten_.shape[:2] + arr.shape[1:])
        for i in range(arr.shape[0]):
            out_arr[tuple(self.reshape_[i])] = arr[i]
        return out_arr
