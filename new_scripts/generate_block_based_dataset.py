import click
import os

from msslib.generate_data import *
from msslib.features import *
from msslib.sample import *

def init_sampler():
    block_size = (20,20)
    no_of_samples = 10
    group_label = most_common
    group_obs = mean_vector
    return f.partial(random_observations_within_blocks, page_size, block_size, no_of_samples, group_label, group_obs) 

def label_and_data(page_paths:[str]):
    window_size = 41
    page_size = (1200,900)
    img, label, f_img = prepare_page_accessors(window_size, page_size, page_paths)
    get_data = compose(prepare_features, real_fft, std_dev_contrast_stretch, f_img)
    return label, get_data 

@click.command()
@click.option('--block_size', default=20)
@click.argument('img_dir', type=click.Path(exists=True, resolve_path=True))
@click.argument('label_dir', type=click.Path(exists=True, resolve_path=True))
@click.argument('output_dir', type=click.Path(exists=True, resolve_path=True))
def generate_block_based_dataset(block_size, img_dir, label_dir, output_dir):
    get_img_path = compose(f.partial(format_path, img_dir, 'jpg'), only_basename)
    get_label_path = compose(f.partial(format_path, label_dir, 'png'), only_basename)

    label_out_dir = os.path.join(output_dir, 'labels') 
    os.makedirs(label_out_dir)
    data_out_dir = os.path.join(output_dir, 'data') 
    os.makedirs(data_out_dir)

    create_label_out_path = compose(f.partial(format_path, label_out_dir, 'npy'), only_basename)
    create_data_out_path = compose(f.partial(format_path, data_out_dir, 'npy'), only_basename)

    path_formatter = applier(get_img_path, get_label_path)
    sampler = init_sampler()
    paths = listpaths(label_dir)

    for p in paths:
        label, data = sampler(*label_and_data(path_formatter(p)))
        data_out = create_data_out_path(p)
        label_out = create_label_out_path(p)
        print("Saving %s" % label_out)
        np.save(label_out, label)
        print("Saving %s" % data_out)
        np.save(data_out, data)

    return 


if __name__=='__main__':
    generate_block_based_dataset()
