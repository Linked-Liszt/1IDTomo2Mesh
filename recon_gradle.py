CUDA_DEVICE = "1"
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE

import gradio as gr

import numpy as np
from PIL import Image
import json
import copy
import argparse
from tqdm import tqdm
import notebook_mesh_utils as nu
import matplotlib.pyplot as plt
import tomopy
import cv2


class ReconUI:
    def __init__(self):
        self.init_shared_components()
        self.main_interface()

    # Workaround for combobox error
    def init_shared_components(self):
        self.scan_drop = gr.Dropdown(label='Selected Scan', type='index')
        self.loaded_scans = None

    def get_avaliable_scans(self, metadata_fp, is_override, override_path):
        if is_override:
            override_path = override_path
        else:
            override_path = None
        scans = nu.extract_scan_data(metadata_fp, override_path)
        new_choices = [f"[{i}]: {scan.img_range}" for i, scan in enumerate(scans)]
        self.scan_drop.choices = new_choices
        return gr.Dropdown.update(choices=new_choices, value=new_choices[0]), scans


    def hide_override(self, ckbx_state):
        return gr.Textbox.update(visible=ckbx_state)


    def load_scan(self, scans, scan_idx, progress=gr.Progress(track_tqdm=True)):
        self.loaded_scans = nu.load_images(scans[scan_idx], 10)
        return "Loaded"
        

    def main_interface(self):
        metadata_fp = '/home/beams/S1IDUSER/new_data/alshibli_nov22/F50_sp5_tomo/F50_sp5_tomo_TomoFastScan.dat'
        override_path = '/home/beams/S1IDUSER/mnt/s1c/alshibli_nov22/tomo/F50_sp5_tomo'

        with gr.Blocks() as scan_if:
            # Layout
            with gr.Tab("Loading Data"):
                metadata_fp_fld = gr.Textbox(label='Metadata File Path', value=metadata_fp)
                override_ckbx = gr.Checkbox(label='Override File Path', value=True)
                override_path_fld = gr.Textbox(label='Image File Path', value=override_path)
                find_scans_btn = gr.Button('Find Scans')

                self.scan_drop.render()

                load_scans_btn = gr.Button('Load Scans')

                # Functionality 
                avail_scans = gr.State()
                load_txt = gr.Textbox(label='Loading Area', interactive=False)

                override_ckbx.change(fn=self.hide_override, 
                                    inputs=override_ckbx, 
                                    outputs=override_path_fld)

                find_scans_btn.click(fn=self.get_avaliable_scans, 
                                    inputs=[metadata_fp_fld, override_ckbx, override_path_fld], 
                                    outputs=[self.scan_drop, avail_scans])

                load_scans_btn.click(fn=self.load_scan, 
                                    inputs=[avail_scans, self.scan_drop], 
                                    outputs=load_txt,
                                    )
                
            #with gr.Tab('Reconstruction'):


        scan_if.queue(concurrency_count=3).launch()


if __name__ == '__main__':
    ReconUI()



    """


scan = scans[1]
scan




print(scan_data.projs.shape)
print(scan_data.dark_fields.shape)
print(scan_data.white_fields.shape)
print(scan_data.omega.shape)

scan_data.projs = scan_data.projs.swapaxes(0,1)
scan_data.dark_fields = scan_data.dark_fields.swapaxes(0,1)
scan_data.white_fields = scan_data.white_fields.swapaxes(0,1)

nu.crop_projs(scan_data, [200, 1720])

proj = tomopy.normalize(scan_data.projs, scan_data.white_fields, scan_data.dark_fields)
#proj = tomopy.minus_log(proj)
print(proj.shape)

rot_center = tomopy.find_center(proj, scan_data.omega, init=scan_data.projs.shape[2]//2, ind=300, tol=0.5)

print(rot_center)

print(scan_data.projs.shape[2]/2)
#print(rot_center)
print(proj.shape)
print(scan_data.omega.shape)
#rot_center = scan_data.projs.shape[2]//2

adjusted = proj * 4 # increase contrast

options = {'proj_type': 'linear', 'method': 'FBP_CUDA'}
recon = tomopy.recon(adjusted[:-1],
                     scan_data.omega[:-1],
                     center=adjusted.shape[2]//2,
                     algorithm=tomopy.astra,
                     options=options,
                     ncore=20)

#recon_clipped = np.clip(recon, 0, 0.002)
recon_clipped = recon
recon_no_ring = tomopy.misc.corr.circ_mask(recon_clipped, 0, ratio=0.95, val=np.mean(recon_clipped))

# %%
_ = nu.plot_recon(recon_clipped, 500) 

# %%
print(recon_clipped[0][0][0])

# %%
norm[0][0]

# %%
print(np.max(recon_clipped[0]))
print(np.max(recon_clipped[0] * (254 / np.max(recon_clipped[0]))))

# %%
for layer_idx in range(recon_clipped.shape[0]):
    norm = recon_clipped[layer_idx]
    norm = norm + (0 - norm.min())
    norm = norm * (254 / np.max(norm))
    im = Image.fromarray(norm)
    im = im.convert("L")
    im.save(f'recon/{layer_idx:05d}.tiff')


"""