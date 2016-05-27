import click
import os
from msslib.utils import *
from msslib.data import *
from scipy import misc

@click.command()
@click.option('--height', default=1200)
@click.option('--width', default=900)
@click.argument('page_dir', type=click.Path(exists=True, resolve_path=True))
@click.argument('image_dir', type=click.Path(exists=True, resolve_path=True))
@click.argument('label_dir', type=click.Path(exists=True, resolve_path=True))
@click.argument('output_dir', type=click.Path(exists=True, resolve_path=True))
def generate_resized_images(height, width, page_dir, image_dir, label_dir, output_dir):
    """ Crops images to the extents of the page defined in the PrimaPage
        file, then resizing the image down to the defined size. 
        """
    get_page_path = compose(f.partial(format_path, page_dir, 'xml'), only_basename)
    get_img_path = compose(f.partial(format_path, image_dir, 'jpg'), only_basename)
    get_label_path = compose(f.partial(format_path, label_dir, 'png'), only_basename)
    
    label_out_dir = os.path.join(output_dir, 'labels') 
    os.makedirs(label_out_dir)
    image_out_dir = os.path.join(output_dir, 'images') 
    os.makedirs(image_out_dir)

    create_label_out_path = compose(f.partial(format_path, label_out_dir, 'png'), only_basename)
    create_image_out_path = compose(f.partial(format_path, image_out_dir, 'jpg'), only_basename)

    path_formatter = applier(get_page_path, get_img_path, get_label_path)
    # I assume that the label_dir will have the most useful set of data to get the other paths from. 
    paths = listpaths(label_dir)
    # A resizing function. 
    resizer = img_resizer((height, width))
    
    for p in paths:
        img, label = open_image_label(*path_formatter(p))
        img, label = resizer(img, label)
        misc.imsave(create_label_out_path(p), label)
        misc.imsave(create_image_out_path(p), img)

if __name__=='__main__':
    generate_resized_images()
