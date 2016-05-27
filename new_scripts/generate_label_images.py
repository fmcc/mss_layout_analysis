import click
from msslib.utils import *
from msslib.data import *
from scipy import misc

@click.command()
@click.argument('page_dir', type=click.Path(exists=True, resolve_path=True))
@click.argument('label_dir', type=click.Path(exists=True, resolve_path=True))
def generate_label_images(page_dir, label_dir):
    paths = listpaths(page_dir)
    for p in paths:
        page = PrimaPage(p)
        img = label_mss_image(page)
        out_path = format_path(label_dir, 'png', only_basename(p))
        misc.imsave(out_path, img)

if __name__=='__main__':
    generate_label_images()
