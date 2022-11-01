import dataclasses
from typing import List
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import cupy
import tomo2mesh.fbp.subset as subset

@dataclasses.dataclass
class ScanData():
    img_range: List[int]
    omega: np.ndarray
    img_dir: str
    img_prefix: str

"""
==========================================
Processing Functions
==========================================
"""

def extract_scan_data(metadata_fp, override_path=None, override_pfx=None):
    """
    Reads the specified metadata file and produces 
    a list of ScanData for each scan in the file

    Params:
        metadata_fp: path to the metadata file
        
        override_path: Override the metadata image path
            if the images were moved after metadata file was created
        
        override_prefix: Override the image prefixes if the
            image format was changed after the metadata file was created
        
    Returns:
        List of ScanData objects contained in the metadata file
    """
    scans = []
    NUM_FIELD_BEGIN = 10
    NUM_FIELD_END = 20

    with open(metadata_fp) as dat_f:
        read_counter = 0
        cur_scan = {}
        img_range = [0, 0]
        dat_lines = dat_f.readlines()

        is_override_path = override_path is not None
        is_override_pfx = override_pfx is not None
        for i, line in enumerate(dat_lines):
            if line.startswith('End'):
                read_counter = 2
            elif read_counter == 2:
                read_counter -= 1
                img_range[0] = int(line.split(' ')[4].strip()) + NUM_FIELD_BEGIN
            elif read_counter == 1:
                read_counter -= 1
                img_range[1] = int(line.split(' ')[4].strip()) - NUM_FIELD_END

                omega, path, img_pfx = _find_img_data(dat_lines, 
                                                     img_range, 
                                                     i,
                                                     override_path,
                                                     override_pfx)
                if is_override_path: path = override_path
                if is_override_pfx: img_pfx = override_pfx

                scans.append(ScanData(img_range, omega, path, img_pfx))
    return scans


def load_images(scan_data: ScanData):
    """
    Loads images from disk into memory from a scan_data 
    object extracted from the metadata file.

    Params:
        scan_data: ScanData object pointing to a series of scans
    
    Returns:
        projs: stack of projections in shape (x, scan_num, y)

    """
    prefix = scan_data.img_dir
    start_file = scan_data.img_range[0]
    end_file = scan_data.img_range[1]
    omega = scan_data.omega

    omega = omega / 180 * np.pi # NOTE: Why 1pi and not 2pi


    projs = []
    for im_idx in tqdm(range(start_file, end_file + 1), desc='Loading Imgs'):
        im = Image.open(os.path.join(prefix, scan_data.img_prefix + f'_{im_idx:06d}.tif'))
        projs.append(np.array(im))
    projs = np.stack(projs)
    projs = projs.swapaxes(0,1)

    if projs.shape[1] != len(omega):
        raise ValueError("Number of projections and omegas are not equal!")
    
    return projs


def reconstruct(projs, omega, center, pixel_ds, scan_ds, gpu_batch_size):
    """
    Performs reconstruction from projections and angles
    on GPU.

    Params:
        projs: stack of projections. Shape: (x,depth,y)

        omega: stack of angles for projection images Shape (depth)

        center: int, center pixel value of the sample

        pixel_ds: pixel downsampling factor. Subsamples skipping this many pixels 
            in both dims
        
        scan_ds: scan downsampling factor. Subsamples skipping this many frames

        gpu_batch_size: number of frames to process simultaneously. Reduce this to
            save memory, but increase processing time
    
    Returns: 
        reconstruction: reconstructed images. Shape: (stack,x,y)
    """
    raw_data = projs[::pixel_ds,::scan_ds,::pixel_ds], omega[::scan_ds,...], center/pixel_ds
    recon = subset.recon_all(*raw_data, gpu_batch_size)
    return recon


"""
==========================================
Plotting & Printing Functions
==========================================
"""

def plot_recon_compare(original, processed, recon_slice):
    fig, ax = plt.subplots(1, 2, figsize=(30,10))
    pos0 = ax[0].imshow(original[recon_slice])
    fig.colorbar(pos0, ax=ax[0])
    ax[0].set_title('Orignal')
    pos1 = ax[1].imshow(processed[recon_slice])
    fig.colorbar(pos1, ax=ax[1])
    ax[1].set_title('Post Processsing')
    return fig


def plot_recon(original, recon_slice):
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    pos0 = ax.imshow(original[recon_slice])
    fig.colorbar(pos0, ax=ax)
    return fig


def print_avail_scans(scans: List[ScanData]):
    print('\n'.join([f"[{i}]: {scan.img_range}" for i, scan in enumerate(scans)]))


"""
==========================================
Internal processing functions
==========================================
"""

def _find_img_data(dat_lines, 
                  scan_range,
                  entry_line, 
                  override_path=False, 
                  override_pfx=False):
    METADATA_LINE_LEN = 51 
    SCAN_ID_IDX = 7
    OMEGA_IDX = 31
    PATH_LEN = 6
    IMG_PFX_LEN = 14

    start_idx = scan_range[0]
    end_idx = scan_range[1]

    omegas = np.zeros(end_idx - start_idx + 1)
    found_omegas = np.zeros(end_idx - start_idx + 1)

    path = None
    img_pfx = None

    for i in range(entry_line, 0, -1):
        line = dat_lines[i].split(' ')
        if len(line) == METADATA_LINE_LEN:
            scan_num = int(line[SCAN_ID_IDX])

            if scan_num < start_idx:
                break

            if scan_num <= end_idx:
                store_idx = scan_num - start_idx
                omegas[store_idx] = float(line[OMEGA_IDX])
                found_omegas[store_idx] = True

        elif dat_lines[i].startswith('Path:'):
            path = dat_lines[i][PATH_LEN:].strip()
        
        elif dat_lines[i].startswith('Image prefix:'):
            img_pfx = dat_lines[i][IMG_PFX_LEN:].strip()


    if not np.all(found_omegas):
        raise ValueError(f"Unable to find all omegas for given scan range {start_idx}-{end_idx}")
    
    if path is None and not override_path:
        raise ValueError(f"Unable to find image path. Please check metadata file structure.")
    
    if img_pfx is None and not override_pfx:
        raise ValueError(f"Unable to find image prefixes. Please check metadata file structure.")
    
    pfx_split = os.path.split(img_pfx)
    path = os.path.join(path, pfx_split[0])
    img_pfx = pfx_split[1]

    return omegas, path, img_pfx


