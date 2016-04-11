import numpy as np
import collections as c

def label_dict(arr: np.ndarray):
    """ Creates a dictionary to collect labelled observations, 
        ensuring that all labels in a label image are used as keys.
        """
    labels = c.defaultdict(list)
    for i in np.unique(arr).tolist():
        labels[i]
    return labels 

