import numpy as np
import collections as c
import itertools as it
from scipy.cluster import vq
from ..utils import *

def calculate_centroid(vectors:[np.ndarray]):
    """ Find the centroid of a list of vectors. 
        Also in observe.sample as mean_vector, I'm just keeping this 
        file relatively independent for now. 
        """
    return np.asarray(vectors).mean(axis=0)

def labelled_centroids(observations_dict:dict) -> (()):
    centroids = []
    for label, observations in observations_dict.items():
        centroids.append((label, calculate_centroid(observations)))
    return centroids

class CentroidVQ():
    """ 
        """
    def __init__(self):
        self.centroids_ = False

    def centroids(self):
        return np.asarray([a[1] for a in self.centroids_])

    def labels(self):
        return  [a[0] for a in self.centroids_]

    def fit(self, X, y):
        obs = to_dict_list(zip(y.tolist(), X.tolist()))
        new_centroids = labelled_centroids(obs)
        if not self.centroids_:
            self.centroids_ = new_centroids
        else: 
            # This is a little lazy, but I'm going with it for now. 
            self.centroids_ = labelled_centroids(to_dict_list(it.chain([*self.centroids_, *new_centroids])))

    def predict(self, X):
        cent = self.centroids()
        lab = self.labels()
        codes, dist = vq.vq(X, cent)
        return np.asarray([lab[i] for i in codes])

