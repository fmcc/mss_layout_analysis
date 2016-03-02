import click 
import os

from page_prima.page import PrimaPage

from img_lib.utils import *
from scipy import misc
from PIL import Image

def listpaths(d):
    """ Create a list of all paths for files in a directory """ 
    print(d)
    return [os.path.join(d, f) for f in os.listdir(d)]

def ext_path(path, name, ext='.png'):
    return os.path.join(path, name + ext)

def cropped_imgs(img_dir, page_dir, label_dir, name):
    page = PrimaPage(ext_path(page_dir, name, ".xml"))
    img = misc.imread(ext_path(img_dir, name, ".jpg"))[page.border.as_slice()]
    l_img = misc.imread(ext_path(label_dir, name, ".png"))[page.border.as_slice()]
    return misc.toimage(img), misc.toimage(l_img)

@click.command()
@click.argument('in_dir', type=click.Path(exists=True, resolve_path=True))
@click.argument('page_dir', type=click.Path(exists=True, resolve_path=True))
@click.argument('img_dir', type=click.Path(exists=True, resolve_path=True))
@click.argument('label_dir', type=click.Path(exists=True, resolve_path=True))
@click.argument('out_dir', type=click.Path(exists=True, resolve_path=True))
def merge_three(in_dir, page_dir, img_dir, label_dir, out_dir):
    for p in listpaths(in_dir):
        b = os.path.splitext(os.path.basename(p))[0] 
        out_path = ext_path(out_dir, b, "_combined.jpg")
        click.echo(out_path)
        img, l_img = cropped_imgs(img_dir, page_dir, label_dir, b)
        
        img = img.resize(scale_tuple(img.size, (0.2,0.2)), 0)
        l_img = l_img.resize(scale_tuple(l_img.size, (0.2,0.2)), 0)
        n_img = Image.open(p)
        w,h = img.size
        new_im = Image.new('RGB', (w*3, h))
        new_im.paste(l_img, (0,0))
        new_im.paste(img, (w,0))
        new_im.paste(n_img, (w*2,0))
        ext_path(out_dir, b, "_combined.jpg")
        new_im.save(out_path)

if __name__ =='__main__':
    merge_three()
