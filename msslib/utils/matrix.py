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
