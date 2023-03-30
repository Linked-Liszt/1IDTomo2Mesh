import dataclasses
from typing import List
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import asyncio
import datetime
import tomo2mesh.fbp.subset as subset

@dataclasses.dataclass
class ScanMetaData():
    img_range: List[int]
    omega: np.ndarray
    img_dir: str
    img_prefix: str
    found_omega: bool

@dataclasses.dataclass
class ScanData():
    projs: np.ndarray
    omega: np.ndarray
    dark_fields: np.ndarray
    white_fields: np.ndarray
    center: float = None

"""
==========================================
Processing Functions
==========================================
"""

def extract_scan_data(metadata_fp, override_path=None, override_pfx=None):
    """
    Reads the specified metadata file and produces 
    a list of ScanMetaData for each scan in the file

    Params:
        metadata_fp: path to the metadata file
        
        override_path: Override the metadata image path
            if the images were moved after metadata file was created
        
        override_prefix: Override the image prefixes if the
            image format was changed after the metadata file was created
        
    Returns:
        List of ScanMetaData objects contained in the metadata file
    """
    scans = []

    with open(metadata_fp) as dat_f:
        read_counter = 0
        cur_scan = {}
        dat_lines = dat_f.readlines()

        is_override_path = override_path is not None
        is_override_pfx = override_pfx is not None
        for i, line in enumerate(dat_lines):
            if line.startswith('End'):
                read_counter = 2
            elif read_counter == 2:
                img_range = [0, 0]
                read_counter -= 1
                img_range[0] = int(line.split(' ')[4].strip()) 
            elif read_counter == 1:
                read_counter -= 1
                img_range[1] = int(line.split(' ')[4].strip()) 

                omega, path, img_pfx, found_omega = _find_img_data(dat_lines, 
                                                     img_range, 
                                                     i,
                                                     override_path,
                                                     override_pfx)
                if is_override_path: path = override_path
                if is_override_pfx: img_pfx = override_pfx

                scans.append(ScanMetaData(img_range, omega, path, img_pfx, found_omega))
    return scans


def load_images(scan_data: ScanMetaData, num_dark_white: int, override_prog = None) -> ScanData:
    """
    Loads images from disk into memory from a scan_data 
    object extracted from the metadata file.

    Params:
        scan_data: ScanMetaData object pointing to a series of scans

        num_dark_white: number of dark and white projections to extract, assumes 1x dark at the 
            beginning and 2x white at the end

        use_async: use asyncio loading. Set false for serial loading
    
    TODO: Loading from top to bottom
    
    Returns:
        projs: stack of projections in shape (x, scan_num, y)

    """
    prefix = scan_data.img_dir
    start_file = scan_data.img_range[0]
    end_file = scan_data.img_range[1]
    omega = scan_data.omega

    omega = omega / 180 * np.pi # NOTE: Why 1pi and not 2pi

    if override_prog is not None:
        prog = override_prog.tqdm
    else:
        prog = tqdm

    projs = []
    for im_idx in prog(range(start_file, end_file + 1), desc='Loading Imgs'):
        im = Image.open(os.path.join(prefix, scan_data.img_prefix + f'_{im_idx:06d}.tif'))
        projs.append(np.array(im))
    projs = np.stack(projs)

    # Assume Wite, Projs, White, Dark
    if num_dark_white != 0:
        dark = projs[-num_dark_white:]
        white = np.concatenate((projs[:num_dark_white], projs[-num_dark_white * 2 : -num_dark_white]), axis=0)
        projs = projs[num_dark_white:-num_dark_white * 2]
    else:
        dark = np.expand_dims(np.ones(projs.shape[1:]), axis=0)
        white = np.expand_dims(np.ones(projs.shape[1:]), axis=0)


    projs = projs.swapaxes(0,1)
    dark = dark.swapaxes(0,1)
    white = white.swapaxes(0,1)


    #if projs.shape[1] != len(omega):
    #    raise ValueError("Number of projections and omegas are not equal!")
    
    return ScanData(projs, omega, dark, white)


async def async_load_img(im_fp):
    im = Image.open(os.path.join(im_fp))
    return np.array(im)

def crop_projs(scan_data, x_start, x_end, y_start, y_end):
    scan_data.projs = scan_data.projs[:, y_start:y_end, x_start:x_end]
    scan_data.dark_fields = scan_data.dark_fields[:, y_start:y_end, x_start:x_end]
    scan_data.white_fields = scan_data.white_fields[:, y_start:y_end, x_start:x_end]


def reconstruct(scan_data: ScanData, gpu_batch_size):
    """
    Performs reconstruction from projections and angles
    on GPU.

    Params:
        scan_data Scan data object containing projections and omegas. 
    
    Returns: 
        reconstruction: reconstructed images. Shape: (stack,x,y)
    """
    raw_data = scan_data.projs, scan_data.omega, scan_data.center
    print(raw_data[0].shape)
    recon = subset.recon_all(*raw_data, gpu_batch_size)
    return recon

def downsample_scan(scan_data: ScanData, pixel_ds, scan_ds):
    """
    Downsamples a scan based on pixel and scan downsampling factors. 

    Params:
        scan_data: scan data to downsample NOTE: Destructive action

        pixel_ds: pixel downsampling factor. Subsamples skipping this many pixels 
            in both dims
        
        scan_ds: scan downsampling factor. Subsamples skipping this many frames
    
    return:
        scan_data: downsampled scan
    """
    scan_data.projs = scan_data.projs[::pixel_ds,::scan_ds,::pixel_ds]
    scan_data.omega = scan_data.omega[::scan_ds,...]
    scan_data.dark_fields = scan_data.dark_fields[::pixel_ds,::scan_ds,::pixel_ds]
    scan_data.white_fields = scan_data.white_fields[::pixel_ds,::scan_ds,::pixel_ds]
    if scan_data.center is not None:
        scan_data.center = scan_data.center / pixel_ds

    return scan_data


def norm_whitefield(scan_data: ScanData) -> ScanData:
    """
    Performs normalization by dividing the projections by
    the average of all white fields. 

    NOTE: REPLACES projs with normalized projs. This is a destructive action.

    TODO: Accelerate with GPU
    
    TODO: Average first 10, average last 10, 10, Interp & average for projections,
        each frame of proj gets divided by interp
        
    TODO: DF correction: (proj-df / (wf-df)
    """
    scan_data.projs = scan_data.projs / np.mean(scan_data.white_fields, axis=0) 
    return scan_data


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


def plot_recon(original, recon_slice, color_range=None): # TODO: Add range to plots
    # TODO: Look for imagej alternative. Potentially custom GUI
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    pos0 = ax.imshow(original[recon_slice], cmap='gray')
    if color_range is not None:
        pos0.set_clim(color_range[0], color_range[1])
    fig.colorbar(pos0, ax=ax)
    return fig

def plot_proj(original, proj_slice, color_range=None):
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    pos0 = ax.imshow(original[:, proj_slice, :])
    if color_range is not None:
        pos0.set_clim(color_range[0], color_range[1])
    fig.colorbar(pos0, ax=ax)
    return fig

def print_avail_scans(scans: List[ScanMetaData]):
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
    DARK_WHITE_COUNT = 30
    WHITE_COUNT_BEGIN = 10
    METADATA_LINE_LEN = 51 
    SCAN_ID_IDX = 7
    OMEGA_IDX = 31
    PATH_LEN = 6
    IMG_PFX_LEN = 14

    start_idx = scan_range[0]
    end_idx = scan_range[1]

    omegas = np.zeros(end_idx - start_idx + 1 - DARK_WHITE_COUNT)
    found_omegas = np.zeros(end_idx - start_idx + 1 - DARK_WHITE_COUNT)

    path = None
    img_pfx = None

    for i in range(entry_line, 0, -1):
        line = dat_lines[i].split(' ')
        if len(line) == METADATA_LINE_LEN:
            scan_num = int(line[SCAN_ID_IDX])

            if scan_num < start_idx:
                break

            if scan_num <= end_idx:
                store_idx = scan_num - start_idx - WHITE_COUNT_BEGIN
                omegas[store_idx] = float(line[OMEGA_IDX])
                found_omegas[store_idx] = True

        elif dat_lines[i].startswith('Path:'):
            path = dat_lines[i][PATH_LEN:].strip()
        
        elif dat_lines[i].startswith('Image prefix:'):
            img_pfx = dat_lines[i][IMG_PFX_LEN:].strip()

    if path is None and not override_path:
        raise ValueError(f"Unable to find image path. Please check metadata file structure.")
    
    if img_pfx is None and not override_pfx:
        raise ValueError(f"Unable to find image prefixes. Please check metadata file structure.")
    
    pfx_split = os.path.split(img_pfx)
    path = os.path.join(path, pfx_split[0])
    img_pfx = pfx_split[1]

    return omegas, path, img_pfx, np.all(found_omegas)


async def async_time_func(func, args=[], kwargs={}):
    start_time = datetime.datetime.now()
    result = await func(*args, **kwargs)
    print(f'Elapsed Time: {(datetime.datetime.now() - start_time).total_seconds()}')
    return result
