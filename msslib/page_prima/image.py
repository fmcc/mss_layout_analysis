from PIL import Image, ImageDraw
import functools as f
import numpy as np

c = [0,10,20,30,40,50]

def col_poly(d,p,l):
    col = c[l]
    d.polygon(p, outline=col, fill=col)
    
def label_image(page):
    img = Image.new('L', page.dimensions, c[0])
    draw = ImageDraw.Draw(img)
    d_p = f.partial(col_poly, draw)
    d_p(page.border.get_coords(), 1)
    for region in page.text_regions:
        if region.polygon.contains(page.border.centroid):
            d_p(region.get_coords(),4)
        elif region.polygon.area > 500000.0:
            d_p(region.get_coords(),2)
        else:
            d_p(region.get_coords(),3)
            colour = (0,0,255)
    return np.array(img)


