"""
Functions for creating label images. 
"""
from PIL import Image, ImageDraw
import functools as f
import numpy as np

mss_labels = ['image_background', 'page', 'marginalia', 'note', 'main_text']

def colour_polygon(d,p,l):
    d.polygon(p, outline=col, fill=col)
    
def label_mss_image(page):
    # Create an image that's all 0, labelling as the image_background
    img = Image.new('L', page.dimensions, 0)
    draw = ImageDraw.Draw(img)
    polygon_drawer = f.partial(colout_polygon, draw)
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


