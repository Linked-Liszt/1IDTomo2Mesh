print('Initializing GPU libs...')

import numpy as np
from PIL import Image
import json
import os
import argparse
from tqdm import tqdm
import cupy as cp
import math

os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from tomo2mesh.projects.steel_am.coarse2fine import coarse_map, process_subset
from tomo2mesh.misc.voxel_processing import TimerGPU
from tomo2mesh.structures.voids import Voids
from tomo2mesh.porosity.params_3dunet import *
from tomo2mesh.unet3d.surface_segmenter import SurfaceSegmenter
from tomo2mesh.fbp.recon import recon_slice, recon_binned, recon_all

import tensorflow as tf

tf.config.experimental.set_visible_devices([], 'GPU')
"""
TODO: Reenable once env issues fixed
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
"""

def parse_args():
    parser = argparse.ArgumentParser(
        description='Script to course-map voids from a series of raw tomography images'
    )
    parser.add_argument(
        'config_fp', 
        help='path to the config .json describing scan data'
    )
    parser.add_argument(
        'output_fp', 
        help='path to place output file'
    )

    return parser.parse_args()


def crop_projs(projs, scaling_factor, wd):
    CROP_IDXS = [0, 2]
    crop_edges = []
    for crop_idx in CROP_IDXS:
        stage_1_crop = projs.shape[crop_idx] - (projs.shape[crop_idx] % scaling_factor)
        offset_x = (math.ceil(stage_1_crop / scaling_factor)) % (wd // scaling_factor)
        crop_edges.append(stage_1_crop - (offset_x * scaling_factor))

    original_shape = projs.shape
    projs = projs[:crop_edges[0], :,:crop_edges[1]]
    print(f'Auto-Cropped from {original_shape} to {projs.shape}')
    return projs


def find_voids_coarse(config, output_prefix):
    if not os.path.exists(output_prefix):
        os.mkdir(output_prefix)

    prefix = config['img_dir']
    start_file = config['img_range'][0]
    end_file = config['img_range'][1]
    omega = np.asarray(config['omega'])

    omega = np.asarray(config['omega'])
    omega = omega / 180 * np.pi # NOTE: Why 1pi and not 2pi

    scaling_factor = config['mesh_settings']['sf']
    b=scaling_factor
    b_K = scaling_factor
    wd = scaling_factor

    projs = []
    for im_idx in tqdm(range(start_file, end_file + 1), desc='Loading imgs'):
        im = Image.open(os.path.join(prefix, config['img_prefix'] + f'_{im_idx:06d}.tif'))
        projs.append(np.array(im))
    projs = np.stack(projs)
    projs = projs.swapaxes(0,1)
    projs = crop_projs(projs, scaling_factor, wd)


    if 'center' not in config:
        print(f'Assuming image is centered...')
        center = projs.shape[-1]/2.0 # assuming that object is perfectly centered
    else:
        center = config['center']

    if projs.shape[1] != len(omega):
        raise ValueError("Number of projections and omegas are not equal!")

    

    cp._default_memory_pool.free_all_blocks(); cp.fft.config.get_plan_cache().clear()
    voids_b = coarse_map(projs, omega, center, b, b_K, 2)

    voids_b.export_void_mesh_mproc("sizes", edge_thresh=0).write_ply(
        os.path.join(output_prefix, 
        f"{config['img_prefix']}_{config['img_range'][0]}_{config['img_range'][1]}_coarse.ply"))

    # Insert Filtering Here
    voids_b.select_by_size(config['mesh_settings']['vs'], config['mesh_settings']['ps'], sel_type="geq")

    p_voids, r_fac = voids_b.export_grid(wd)
    cp._default_memory_pool.free_all_blocks(); cp.fft.config.get_plan_cache().clear()        

    model_tag = 'M_a07'
    model_names = {'segmenter': "segmenter_Unet_M_a07"}

    model_params = get_model_params(model_tag)
    segmenter = SurfaceSegmenter(model_initialization = 'load-model', \
                            model_names = model_names, \
                            model_path = config['mesh_settings']['model_path'])

    print('Finished grid export...')
    # process subset reconstruction
    x_voids, p_voids = process_subset(projs, omega, center, segmenter, p_voids, voids_b["rec_min_max"])
    
    # import voids data from subset reconstruction
    voids = Voids().import_from_grid(voids_b, x_voids, p_voids)


    voids.export_void_mesh_mproc("sizes", edge_thresh=0).write_ply(
        os.path.join(output_prefix, 
        f"{config['img_prefix']}_{config['img_range'][0]}_{config['img_range'][1]}.ply"))

if __name__ == '__main__':
    args = parse_args()
    with open(args.config_fp, 'r') as config_f:
        config = json.load(config_f)

    find_voids_coarse(config, args.output_fp)
