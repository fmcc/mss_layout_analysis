"""
Functions for creating and colouring label images. 
"""
from PIL import Image, ImageDraw
from matplotlib import colors, cm
import seaborn as sns 
import functools as f
import numpy as np

mss_labels = ['image_background', 'page', 'marginalia', 'note', 'main_text']

def colour_polygon(_draw, polygon, colour):
    _draw.polygon(polygon, outline=colour, fill=colour)
    
def label_mss_image(page):
    # Create an image that's all 0, labelling as the image_background
    img = Image.new('L', page.dimensions, 0)
    draw = ImageDraw.Draw(img)
    polygon_drawer = f.partial(colour_polygon, draw)
    # Draw the page itself - the support on which the writing appears
    polygon_drawer(page.border.get_coords(), 1)
    for region in page.text_regions:
        # I assume that the block of text that includes the centre of
        # the page is the main_text of the page - there are a few exceptions
        # to this, so manual correction may be necessary. 
        if region.polygon.contains(page.border.centroid):
            polygon_drawer(region.get_coords(),4)
        # Label any text regions that are over this size as being marginalia.
        # This value was arrived at by experimentation.  
        elif region.polygon.area > 500000.0:
            polygon_drawer(region.get_coords(),2)
        # Label any text regions that are under this size as being small notes. 
        else:
            polygon_drawer(region.get_coords(),3)
    return np.array(img)

def to_rgb(rgba: np.ndarray):
    """Converts a RGBA colour in the [0,1] range to RGB in the [0,255] range"""
    return np.asarray(rgba[:3])*255

def label_colour_image(label_img: np.ndarray, colour_name='Set2'):
    """ Takes a greyscale labelled image and converts it to a RGB colour image.  
        """
    h,w = label_img.shape
    colours = sns.color_palette(colour_name, len(mss_labels))
    colourised = np.zeros((h,w,3))
    for label, colour in enumerate(colours):
        colourised[np.where(label_img==label)] = to_rgb(colour)
    return colourised 
