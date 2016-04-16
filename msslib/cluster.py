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

def calculate_centroid(vectors:[np.ndarray]):
    """ Find the centroid of a list of vectors. 
        """
    return np.asarray(vectors).mean(axis=0)

def labelled_centroids(observations_dict:dict) -> (()):
    centroids = []
    for label, observations in observations_dict.items():
        centroids.append((label, calculate_centroid(observations)))
    return centroids

