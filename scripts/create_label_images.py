import click
import os
from msslib.page_prima.page import PrimaPage
from msslib.page_prima.image import label_image
from scipy import misc

def listpaths(d):
    """ Create a list of all paths for files in a directory """ 
    print(d)
    return [os.path.join(d, f) for f in os.listdir(d)]

def img_path(path, page, ext='.png'):
    return os.path.join(path, page.filename + ext)


@click.command()
@click.argument('page_dir', type=click.Path(exists=True, resolve_path=True))
@click.argument('label_dir', type=click.Path(exists=True, resolve_path=True))
def create_label_images(page_dir, label_dir):
    for p in listpaths(page_dir):
        page = PrimaPage(p)
        out_path = img_path(label_dir, page)
        click.echo("Creating %s" % out_path)
        labelled_img = label_image(page)
        misc.imsave(out_path, labelled_img)


if __name__ =='__main__':
    create_label_images()
