import numpy as np
from matplotlib import colors, cm
import seaborn as sns 

def to_rgb(rgba: np.ndarray):
    """Converts a RGBA colour in the [0,1] range to RGB in the [0,255] range"""
    return np.asarray(rgba[:3])*255

def label_colour_image(label_img: np.ndarray, colour_name='Set2'):
    """ Takes a greyscale labelled image and converts it to a RGB colour image.  
        """
    h,w = label_img.shape
    labels = [l for l in np.sort(np.unique(label_img))]
    colours = sns.color_palette(colour_name, len(labels))
    colourised = np.zeros((h,w,3))
    for colour, label in zip(colours, labels):
        colourised[np.where(label_img==label)] = colour
    return colourised 
