import numpy as np
from sklearn.metrics import confusion_matrix

def normalised_confusion_matrix(label:np.ndarray, image:np.ndarray):
    """ Create a normalised confusion matrix for a label and image in the same format. 
        """
    cm = confusion_matrix(label.flatten(), image.flatten())
    cm_normalised = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm_normalised

def normalise_confusion_matrix(cm: np.ndarray):
    return cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

def percent_correct(label:np.ndarray, image:np.ndarray):
    all_correct = label == image
    return all_correct.sum()/(all_correct.size/100)
