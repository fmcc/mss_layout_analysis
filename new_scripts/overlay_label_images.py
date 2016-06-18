import click
from msslib.utils import *
from msslib.data import *
from scipy import misc

@click.command()
@click.option('--label_type', default='npy')
@click.argument('image_dir', type=click.Path(exists=True, resolve_path=True))
@click.argument('label_dir', type=click.Path(exists=True, resolve_path=True))
@click.argument('output_dir', type=click.Path(exists=True, resolve_path=True))
def overlay_label_images(label_type, image_dir, label_dir, output_dir):
    """ Converts a label image to a colour image and overlays it over
        the original images of the same size, saving it to the output directory.  
        """
    get_img_path = compose(f.partial(format_path, image_dir, 'jpg'), only_basename)
    get_label_path = compose(f.partial(format_path, label_dir, label_type), only_basename)

    create_overlay_out_path = compose(f.partial(format_path, output_dir, 'jpg'), only_basename)
    # I assume that the label_dir will have the most useful set of data to get the other paths from. 
    paths = listpaths(label_dir)
    
    for p in paths:
        if label_type == 'png':
            label = misc.imread(get_label_path(p))
        else:
            label = np.load(get_label_path(p))
        image = misc.imread(get_img_path(p))
        colour_label = label_colour_image(label) 
        overlay_img = overlay_imgs(image, colour_label) 
        misc.imsave(create_overlay_out_path(p), overlay_img)

if __name__=='__main__':
    overlay_label_images()
