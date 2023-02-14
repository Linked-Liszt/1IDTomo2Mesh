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
        self.cur_projs = None
        self.cur_recon = None
        self.crop = [0, 0, 0, 0]

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
        self.loaded_scans.projs = self.loaded_scans.projs.swapaxes(0,1)
        self.loaded_scans.dark_fields = self.loaded_scans.dark_fields.swapaxes(0,1)
        self.loaded_scans.white_fields = self.loaded_scans.white_fields.swapaxes(0,1)
        return "Loaded"
        

    def reload_reset_scans(self):
        if self.loaded_scans is None:
            raise ValueError("Scans not Loaded!")
        else:
            self.cur_projs = copy.deepcopy(self.loaded_scans)
        sl_update = gr.Slider.update(minimum=0, maximum=self.cur_projs.projs.shape[0]-1, value=0, interactive=True)
        xs_update = gr.Number.update(value=0)
        xe_update = gr.Number.update(value=self.cur_projs.projs.shape[2])
        ys_update = gr.Number.update(value=0)
        ye_update = gr.Number.update(value=self.cur_projs.projs.shape[1])
        self.crop = [0, self.cur_projs.projs.shape[1], 0, self.cur_projs.projs.shape[2]]
        im_update = self._render_im(0)
        return sl_update, im_update, xs_update, xe_update, ys_update, ye_update
    
    def update_proj_slide(self, slide_value):
        return self._render_im(slide_value)


    def crop_img(self, slide_value, start_x, end_x, start_y, end_y):
        self.crop = [start_y, end_y, start_x, end_x]
        return self._render_im(slide_value)

    def _render_im(self, slide_value):
        norm_im = self.cur_projs.projs[slide_value]
        if norm_im.dtype is not np.dtype(np.uint16):
            norm_im /= np.max(np.abs(norm_im))
        return gr.Image.update(value=norm_im[self.crop[0]:self.crop[1], 
                                             self.crop[2]:self.crop[3]])

    def _render_recon(self, slide_value):
        norm_im = self.cur_recon[slide_value]
        norm_im /= np.max(np.abs(norm_im))
        return gr.Image.update(value=norm_im)


    def update_recon_slide(self, slide_value):
        return self._render_recon(slide_value)


    def reconstruct(self, slide_value, norm):
        projs = self.cur_projs.projs[:, self.crop[0]:self.crop[1], 
                                        self.crop[2]:self.crop[3]]
        if norm  != 'None':
            wf = self.cur_projs.white_fields[:, self.crop[0]:self.crop[1], 
                                             self.crop[2]:self.crop[3]]
            df = self.cur_projs.dark_fields[:, self.crop[0]:self.crop[1], 
                                            self.crop[2]:self.crop[3]]

            if norm == 'TomoPy':
                projs = tomopy.normalize(projs, wf, df)
            
            elif norm == 'Basic':
                projs = projs / wf.mean(axis=0)
                            
        projs = projs * 4
        options = {'proj_type': 'linear', 'method': 'FBP_CUDA'}
        center = (self.cur_projs.projs.shape[2] // 2) - self.crop[2]
        self.cur_recon = tomopy.recon(projs[:-1],
                            self.cur_projs.omega[:-1],
                            center=center,
                            algorithm=tomopy.astra,
                            options=options,
                            ncore=20)

        sl_update = gr.Slider.update(minimum=0, maximum=self.cur_recon.shape[0]-1, value=0, interactive=True)
        return self._render_recon(slide_value), sl_update


    def main_interface(self):
        metadata_fp = '/home/beams/S1IDUSER/new_data/alshibli_nov22/F50_sp5_tomo/F50_sp5_tomo_TomoFastScan.dat'
        override_path = '/home/beams/S1IDUSER/mnt/s1c/alshibli_nov22/tomo/F50_sp5_tomo'

        with gr.Blocks() as scan_if:
            # Loading Tab Layout
            with gr.Tab("Loading Data"):
                metadata_fp_fld = gr.Textbox(label='Metadata File Path', value=metadata_fp)
                override_ckbx = gr.Checkbox(label='Override File Path', value=True)
                override_path_fld = gr.Textbox(label='Image File Path', value=override_path)
                find_scans_btn = gr.Button('Find Scans')

                self.scan_drop.render()

                load_scans_btn = gr.Button('Load Scans')
                load_txt = gr.Textbox(label='Loading Progress', interactive=False)

                
            with gr.Tab('Projection'):
                proj_img = gr.Image(label='Projection',
                                    image_mode="L",
                                interactive=False)
                proj_slide = gr.Slider(label='Projection', interactive=True)
                reset_proj_btn = gr.Button('RESET and Reload Projs')
            
                with gr.Row():
                    proj_xs_num = gr.Number(label='X-Start', precision=0)
                    proj_xe_num = gr.Number(label='X-End', precision=0)
                    proj_ys_num = gr.Number(label='Y-Start', precision=0)
                    proj_ye_num = gr.Number(label='Y-End', precision=0)

                proj_crop_btn = gr.Button('Crop Projections')
        
            with gr.Tab('Reconstruction'):
                recon_img = gr.Image(label='Reconstruction',
                                    image_mode="L",
                                    interactive=False)
                recon_slide = gr.Slider(label='Reconstruction', interactive=True)
                norm_rdo = gr.Radio(label='Normalization', choices=['TomoPy', 'Basic', 'None'], value='TomoPy')
                recon_btn = gr.Button('Reconstruct')
                recon_btn = gr.Button('Reconstruct')
                

            # Functionality Loading Tab
            avail_scans = gr.State()

            override_ckbx.change(fn=self.hide_override, 
                                inputs=override_ckbx, 
                                outputs=override_path_fld)

            find_scans_btn.click(fn=self.get_avaliable_scans, 
                                inputs=[metadata_fp_fld, override_ckbx, override_path_fld], 
                                outputs=[self.scan_drop, avail_scans])

            load_scans_btn.click(fn=self.load_scan, 
                                inputs=[avail_scans, self.scan_drop], 
                                outputs=load_txt
                                )

            # Functionality Recon Tab
            reset_proj_btn.click(fn=self.reload_reset_scans,
                                    outputs=[proj_slide, proj_img,
                                            proj_xs_num, proj_xe_num,
                                            proj_ys_num, proj_ye_num]
            )

            proj_slide.change(fn=self.update_proj_slide,
                                inputs=proj_slide,
                                outputs=proj_img)

            proj_crop_btn.click(fn=self.crop_img,
                                inputs=[proj_slide, 
                                        proj_xs_num, proj_xe_num,
                                        proj_ys_num, proj_ye_num],
                                outputs=proj_img
            )

            # Recon Functionality

            recon_slide.change(fn=self.update_recon_slide,
                                inputs=recon_slide,
                                outputs=recon_img)

            recon_btn.click(fn=self.reconstruct,
                                inputs=[recon_slide, norm_rdo],
                                outputs=[recon_img, recon_slide])


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