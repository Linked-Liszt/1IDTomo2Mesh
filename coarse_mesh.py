import numpy as np
from PIL import Image
import json
import os
import copy
import argparse

from tomo2mesh.projects.steel_am.coarse2fine import coarse_map, process_subset
from tomo2mesh.misc.voxel_processing import TimerGPU
from tomo2mesh.structures.voids import Voids
from tomo2mesh.porosity.params_3dunet import *
from tomo2mesh.unet3d.surface_segmenter import SurfaceSegmenter
from tomo2mesh.fbp.recon import recon_slice, recon_binned, recon_all


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script to course-map voids from a series of raw tomography images'
    )
    parser.add_argument(
        'config_fp', 
        help='path to the config .json describing scan data'
    )

    return parser.parse_args()


def find_voids_coarse(config):
    prefix = config['img_dir']
    start_file = config['img_range'][0]
    end_file = config['img_range'][1]
    center = config['center']
    omega = np.asarray(config['omega'])

    omega = np.asarray(config['omega'])
    omega = omega / 180 * np.pi # NOTE: Why 1pi and not 2pi

    projs = []
    for im_idx in range(start_file, end_file + 1):
        im = Image.open(os.path.join(prefix, config['img_prefix'] + f'_{im_idx}.tif'))
        projs.append(np.array(im))
    projs = np.stack(projs)

    b=4
    b_K = 4

    voids_b = coarse_map(projs, omega, center, b, b_K, 2)

    # Insert Filtering Here

    voids_b.export_void_mesh_mproc("sizes", edge_thresh=0).write_ply(
        f"working_dir/{config['img_prefix']}_{config['img_range'][0]}_{config['img_range'][1]}.ply")


if __name__ == '__main__':
    args = parse_args()
    with open(args.config_fp, 'r') as config_f:
        config = json.load(config_f)

    find_voids_coarse(config)