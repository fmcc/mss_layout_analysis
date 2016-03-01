import click
import os

from scipy import misc
import numpy as np

###########
def r_xy(x1,x2,y1,y2):
    return (random.randrange(x1,x2), random.randrange(y1,y2))

def fft_img(img, window):
    """ Formats an image for FFT """
    w = window - 1
    return pad_img(to_greyscale(img), w)

def local_fft(img, point, window):
    half = window / 2
    x, y = point
    x = x - half
    y = y - half
    return np.abs(np.fft.fft2(img[x:x+window+1, y:y+window+1]))

def dist(a,b):
    return np.linalg.norm(a-b)

def closest_key(l_dict, m):
    return min([(k, dist(v,m)) for k, v in l_dict.items()], key=lambda x: x[1])

def process_page(page_path, img_path, label_path): 
    page = PrimaPage(page_path)
    img = misc.imread(img_path)[page.border.as_slice()]
    l_img = misc.imread(label_path)[page.border.as_slice()]

    img = scale_img(img, (0.2,0.2))
    l_img = scale_img(l_img, (0.2,0.2))

    window = 40
    win = window - 1 
    avg_dict = {}
    f_img = fft_img(img, window)
    height, width, _ = img.shape
    # create a lookup table
    for i in range(1000):
        r = r_xy(0,height,0,width)
        l = l_img[r]
        r_f = (r[0] + win, r[1] + win)
        #a = avg_fft(f_img, r_f, window)
        a = local_fft(f_img, r_f, window)
        try:
            curr = avg_dict[l]
            avg_dict[l] = (a + curr)/2
        except KeyError:
            avg_dict[l] = a

    new_label = np.zeros_like(l_img)

    for h in range(0,height):
        for w in range(0,width):
        #h, w = r_xy(0,height,0,width)
        #k, _ = closest_key(avg_dict, avg_fft(f_img, (h+win,w+win), window))
            k, _ = closest_key(avg_dict, local_fft(f_img, (h+win,w+win), window))
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
def all_nearest_by_fourier(page_dir, img_dir, label_dir, output_dir):
    for p in listpaths(page_dir):
        page = PrimaPage(p)
        out_path = img_path(output_dir, page)
        img_path = img_path(img_dir, page, ".jpg")
        label_path = img_path(label_dir, page)
        click.echo("Creating %s" % out_path)
        out_img = process_page(p, img_path, label_path)
        misc.imsave(out_path, labelled_img)


if __name__ =='__main__':
    all_nearest_by_fourier()
