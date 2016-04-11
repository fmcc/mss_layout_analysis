from page import *
import os
import numpy as np
from scipy import misc
from PIL import Image, ImageDraw

from mask import *
from histogram import *

def adj_page_img(adj, page):
    img = Image.new('RGB', page.dimensions, (0,0,0))
    ImageDraw.Draw(img).polygon(page.border.adjusted_coords(adj), outline=(255,255,255), fill=(255,255,255))
    for region in page.text_regions:
        if region.polygon.area > 500000.0:
            colour = (255,0,0)
        else:
            colour = (0,0,255)
        ImageDraw.Draw(img).polygon(region.adjusted_coords(adj), outline=colour, fill=colour)
    
    return np.array(img)

def centroid_difference(a, b):
    return (a[0] - b[0], a[1] - b[1])

#r_d = '/home/finlay/HMT_data/data/VenetusA/region_xml/'
#paths = [os.path.join(r_d,p) for p in os.listdir(r_d)]
#paths = list(filter(lambda p: "VN" in p, paths))

#pages = [PrimaPage(etree.parse(p)) for p in paths]

#page = PrimaPage(etree.parse('./VA012RN-0013.xml'))
img = misc.imread('./VA012RN-0013.jpeg')
border_mask = create_mask(page.dimensions, page.border.coords)
colour_histogram(img, border_mask, 256)

for i in page.text_regions:
    region_mask = create_mask(page.dimensions, i.coords)
    border_mask = np.logical_xor(border_mask, region_mask)
    colour_histogram(img, region_mask, 256)
print(mask)
#misc.imsave('aye.jpeg', border_mask)
