import click
import os

from scipy import misc
import numpy as np
import random
from page_prima.page import PrimaPage
from collections import defaultdict
from img_lib.fourier import *
from img_lib.scaled_img import ScaledImage, Scale
from img_lib.utils import *

###########
def r_xy(x1,x2,y1,y2):
    return (random.randrange(x1,x2), random.randrange(y1,y2))

def fft_img(img, window):
    """ Formats an image for FFT """
    w = window - 1
    return pad_img(to_greyscale(img), w)

def dist(a,b):
    return np.linalg.norm(a-b)

def closest_key(l_dict, m):
    return min([(k, dist(v,m)) for k, v in l_dict.items()], key=lambda x: x[1])

def update_lookup_dict(img, l_img, avg_dict, window, win):
    f_img = fft_img(img, window)
    height, width, _ = img.shape
    for i in range(1000):
        r = r_xy(0,height,0,width)
        l = l_img[r]
        r_f = (r[0] + win, r[1] + win)
        #a = avg_fft(f_img, r_f, window)
        a = avg_fft(f_img, r_f, window, window/2)
        try:
            curr = avg_dict[l]
            avg_dict[l] = (a + curr)/2
        except KeyError:
            avg_dict[l] = a
    return avg_dict

def lookup_from_page(page_path, img_path, label_path, window, avg_dict): 
    win = window - 1 
    
    page = PrimaPage(page_path)
    img = misc.imread(img_path)[page.border.as_slice()]
    l_img = misc.imread(label_path)[page.border.as_slice()]

    img = scale_img(img, (0.2,0.2))
    l_img = scale_img(l_img, (0.2,0.2))

    return update_lookup_dict(img, l_img, avg_dict, window, win) 

def process_page(page_path, img_path, window, avg_dict): 

    page = PrimaPage(page_path)
    img = misc.imread(img_path)[page.border.as_slice()]
    img = scale_img(img, (0.2,0.2))
    f_img = fft_img(img, window)
    new_label = np.zeros_like(l_img)

    height, width, _ = img.shape
    for h in range(0,height):
        for w in range(0,width):
            k, _ = closest_key(avg_dict, avg_fft(f_img, (h+win,w+win), window, window/2))
            new_label[h,w] = k

    return new_label

######
def listpaths(d):
    """ Create a list of all paths for files in a directory """ 
    print(d)
    return [os.path.join(d, f) for f in os.listdir(d)]

def img_path(path, page, ext='.png'):
    return os.path.join(path, page.filename + ext)


@click.command()
@click.argument('page_dir', type=click.Path(exists=True, resolve_path=True))
@click.argument('img_dir', type=click.Path(exists=True, resolve_path=True))
@click.argument('label_dir', type=click.Path(exists=True, resolve_path=True))
@click.argument('output_dir', type=click.Path(exists=True, resolve_path=True))
def avg_area(page_dir, img_dir, label_dir, output_dir):
    paths = random.shuffle(filter(lambda x: 'RN' in x, listpaths(page_dir)))
    first_ten_r = paths[:10] 
    next_ten_r = paths[10:20]

    avg_dict = {}
    window = 40
    for p in first_ten_r:
        page = PrimaPage(p)
        i_path = img_path(img_dir, page, ".jpg")
        click.echo("Updating avgs from %s" % i_path)
        label_path = img_path(label_dir, page)
        avg_dict = lookup_from_page(p, i_path, label_path, window, avg_dict)

    for p in next_ten_r:
        page = PrimaPage(p)
        out_path = img_path(output_dir, page)
        i_path = img_path(img_dir, page, ".jpg")
        click.echo("Creating %s" % out_path)
        out_img = process_page(p, i_path, window, avg_dict)
        misc.imsave(out_path, labelled_img)


if __name__ =='__main__':
    avg_area()
