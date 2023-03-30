CUDA_DEVICE = "1"
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE

import gradio as gr

import matplotlib.pyplot as plt
import matplotlib
# Fix for linux rendering off the main thread
matplotlib.use('agg')

import numpy as np
from PIL import Image
import json
import copy
import argparse
import io
from tqdm import tqdm
import notebook_mesh_utils as nu
import tomopy
import pprint
import cv2


DEFAULTS = {
    'crop_x_start': 200,
    'crop_x_end': 1720,
    'rot': 0,
    'denoise_template': 15,
    'denoise_search': 7,
    'clip_low': -0.005,
    'clip_high': 0.005
}

META_PATH_BASE = '/home/beams/S1IDUSER/new_data/'
IM_PATH_BASE = '/home/beams/S1IDUSER/mnt/s1c/'

class ReconUI:
    def __init__(self):
        self.init_shared_components()
        self.main_interface()

    def init_shared_components(self):
        # Workaround for combobox error
        self.scan_drop = gr.Dropdown(label='Selected Scan', type='index')
        self.loaded_scans = None
        self.cur_projs = None
        self.cur_recon = None
        self.crop = [0, 0, 0, 0]
        self.rot = DEFAULTS['rot']

        self.recon_layer_start = 0
        self.recon_center_offset = 0

        self.recon_is_clip = True
        self.recon_clip = [DEFAULTS['clip_low'], DEFAULTS['clip_high']] # Replace with const
        self.recon_circ_crop = False
        self.recon_circ_ratio = 1.0
        self.recon_is_denoise = False
        self.recon_denoise_params = [DEFAULTS['denoise_template'], DEFAULTS['denoise_search']]

    def get_avaliable_scans(self, is_manual, exp_name, scan_name, metadata_fp, override_path):
        if not is_manual:
            metadata_fp = os.path.join(META_PATH_BASE, exp_name, scan_name, f'{scan_name}_TomoFastScan.dat')
            override_path = os.path.join(IM_PATH_BASE, exp_name, 'tomo', scan_name)
            

        scans = nu.extract_scan_data(metadata_fp, override_path)
        new_choices = []
        for i, scan in enumerate(scans):
            if scan.found_omega:
                new_choices.append(f"[{i}]: {scan.img_range}")
            else:
                new_choices.append(f"[{i}]: {scan.img_range} WARN: INTERP OMEGA")

        self.scan_drop.choices = new_choices
        return gr.Dropdown.update(choices=new_choices, value=new_choices[0]), scans


    def hide_override(self, ckbx_state):
        return gr.Textbox.update(visible=ckbx_state), gr.Textbox.update(visible=ckbx_state)

    def hide_omega(self, ckbx_state):
        return gr.Number.update(visible=ckbx_state), gr.Number.update(visible=ckbx_state)

    def load_scan(self, scans, scan_idx, is_omgea, omega_low, omega_high, progress=gr.Progress(track_tqdm=True)):
        scan = scans[scan_idx]
        if is_omgea:
            print('interperating omega')
            num_ims = len(scan.omega)
            scan.omega = np.linspace(omega_low, omega_high, num=num_ims)
            


        self.loaded_scans = nu.load_images(scan, 10)
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
        xs_update = gr.Number.update(value=DEFAULTS['crop_x_start'])
        xe_update = gr.Number.update(value=DEFAULTS['crop_x_end'])
        ys_update = gr.Number.update(value=0)
        ye_update = gr.Number.update(value=self.cur_projs.projs.shape[1])
        self.crop = [0, self.cur_projs.projs.shape[1], DEFAULTS['crop_x_start'], DEFAULTS['crop_x_end']]
        im_update = self._render_proj(0)
        return sl_update, im_update, xs_update, xe_update, ys_update, ye_update
    

    def update_proj_slide(self, slide_value):
        return self._render_proj(slide_value)


    def crop_img(self, slide_value, start_x, end_x, start_y, end_y, rot):
        self.crop = [start_y, end_y, start_x, end_x]
        self.rot = -1 * rot
        return self._render_proj(slide_value)

    def _render_proj(self, slide_value):
        norm_im = self.cur_projs.projs[slide_value]
        if norm_im.dtype is not np.dtype(np.uint16):
            norm_im /= np.max(np.abs(norm_im))
        
        w = norm_im.shape[1]
        h = norm_im.shape[0]
        center = (w / 2, h / 2)

        # Perform the rotation
        M = cv2.getRotationMatrix2D(center, self.rot, scale=1)
        norm_im = cv2.warpAffine(norm_im, M, (w, h))

        norm_im = norm_im[self.crop[0]:self.crop[1], self.crop[2]:self.crop[3]]

        fig, ax = plt.subplots(figsize=(20, 13))
        ax.imshow(norm_im, cmap='gray')
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        norm_im = Image.open(buf)
        plt.close(fig)

        return gr.Image.update(value=norm_im)

    def _norm_recon(self, norm_im):
        norm_im = np.copy(norm_im)
        if self.recon_is_clip:
            norm_im = np.clip(norm_im, self.recon_clip[0], self.recon_clip[1])
        
        if self.recon_circ_crop:
            norm_im = np.expand_dims(norm_im, 0)
            norm_im = tomopy.misc.corr.circ_mask(norm_im, 0, ratio=self.recon_circ_ratio, val=np.mean(norm_im), ncore=5)[0]
        
        if self.recon_is_denoise:
            min_norm = np.min(norm_im)
            norm_im = norm_im - min_norm
            max_norm = np.max(norm_im)
            norm_im  = norm_im * (254.0 / max_norm)
            norm_im = np.uint8(norm_im)
            norm_im = cv2.fastNlMeansDenoising(norm_im ,None, self.recon_denoise_params[0], self.recon_denoise_params[1])
            norm_im = np.float_(norm_im)
            norm_im = norm_im / (254.0 / max_norm)
            norm_im = norm_im + min_norm

        return norm_im

    def _render_recon(self, slide_value, return_im=False):
        norm_im = self.cur_recon[slide_value]
        norm_im = self._norm_recon(norm_im)

        fig, ax = plt.subplots(figsize=(15, 1.5))
        ax.hist(norm_im.flatten(), bins=100)
        ax.set_title("Pixel Distribution")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        hist_im = Image.open(buf)
        plt.close(fig)


        width, height = hist_im.size
        new_height = height + 100
        pad_im = Image.new(hist_im.mode, (width, new_height), (255, 255, 255))
        pad_im.paste(hist_im, (0, 50))

        norm_im = norm_im / np.max(np.abs(norm_im))

        if not return_im:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(norm_im, cmap='gray')
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf)
            buf.seek(0)
            norm_im = Image.open(buf)
            plt.close(fig)
        # Decrease rendering size
        #pad = norm_im.shape[1] // 2
        # Padding can be used to reduce image size
        #norm_im = np.pad(norm_im, ((0, 0), (pad, pad)), constant_values=np.max(norm_im))

        pprint.pp(self.export_params())

        if return_im:
            return norm_im, hist_im
        else:
            return gr.Image.update(value=norm_im), gr.Image.update(value=hist_im)


    def update_recon_slide(self, slide_value):
        return self._render_recon(slide_value)


    def _reconstruct(self, recon_algorithm, norm, center_offset, recon_slice=None, return_im=False):
        if self.cur_projs is None:
            self.cur_projs = copy.deepcopy(self.loaded_scans)
            self.crop = [0, self.cur_projs.projs.shape[1], 0, self.cur_projs.projs.shape[2]]

        projs = self.cur_projs.projs[:, self.crop[0]:self.crop[1], 
                                        self.crop[2]:self.crop[3]]
        
        wf = self.cur_projs.white_fields[:, self.crop[0]:self.crop[1], 
                                            self.crop[2]:self.crop[3]]
        df = self.cur_projs.dark_fields[:, self.crop[0]:self.crop[1], 
                                        self.crop[2]:self.crop[3]]
        
        # Prevent rotation from staying

        print('Rotating')
        if self.rot != 0:
            projs = np.copy(projs)
            wf = np.copy(wf)
            df = np.copy(df)

            w = projs.shape[2]
            h = projs.shape[1]
            center = (w / 2, h / 2)

            M = cv2.getRotationMatrix2D(center, self.rot, scale=1)
            for i in range(len(projs)):
                projs[i] = cv2.warpAffine(projs[i], M, (w, h))
            for i in range(len(wf)):
                wf[i] = cv2.warpAffine(wf[i], M, (w, h))
            for i in range(len(df)):
                df[i] = cv2.warpAffine(df[i], M, (w, h))
                
        print('Finished Rotating')

        self.recon_layer_start = 0
        if recon_slice is not None:
            self.recon_layer_start = recon_slice[0]
            projs = projs[:, recon_slice[0]:recon_slice[0] + recon_slice[1], :]
            wf = wf[:, recon_slice[0]:recon_slice[0] + recon_slice[1], :]
            df = df[:, recon_slice[0]:recon_slice[0] + recon_slice[1], :]


        if norm  != 'None':
            if norm == 'TomoPy':
                projs = tomopy.normalize(projs, wf, df)
            
            elif norm == 'Standard':
                projs = projs / wf.mean(axis=0)
                            
        projs = projs * 4
        options = {'proj_type': 'linear', 'method': 'FBP_CUDA'}
        center = (self.cur_projs.projs.shape[2] // 2) - self.crop[2] + center_offset
        self.recon_center_offset = center_offset
        self.cur_recon = tomopy.recon(projs[:-1],
                            self.cur_projs.omega[:-1],
                            center=center,
                            algorithm=tomopy.astra,
                            options=options,
                            ncore=20)
        
        sl_update = gr.Slider.update(minimum=0, maximum=self.cur_recon.shape[0]-1, value=0, interactive=True)
        im_update, hist_update = self._render_recon(0, return_im)
        return im_update, hist_update, sl_update


    def reconstruct_all(self, slide_value, norm, center_offset):
        return self._reconstruct(slide_value, norm, center_offset)

    def reconstruct_slice(self, slide_value, norm, center_offset, slice_start, slice_num):
        return self._reconstruct(slide_value, norm, center_offset, recon_slice=[slice_start, slice_num])
        
    def apply_filters(self, slide_value, 
                      is_clip, clip_lower, clip_upper,
                      is_circ, circ_ratio,
                      is_denoise, template_window, search_window):
        self.recon_is_clip = is_clip
        self.recon_clip = [clip_lower, clip_upper]

        self.recon_circ_crop = is_circ
        self.recon_circ_ratio = circ_ratio

        self.recon_is_denoise = is_denoise
        self.recon_denoise_params = [template_window, search_window]

        return self._render_recon(slide_value=slide_value)
        
    

    def update_center_render(self, is_visible):
        return gr.Gallery.update(visible=is_visible)
    

    def render_center(self, recon_slice, norm):
        CENTERS = list(range(-10, 11))
        center_ims = []
        for center in CENTERS:
            im, _, _ = self._reconstruct(0, norm, center, recon_slice=[recon_slice, 1], return_im=True)
            center_ims.append((im, str(center)))

        return gr.Gallery.update(value=center_ims)

    def export_recon(self, export_dir, progress=gr.Progress(track_tqdm=True)):
        if not os.path.exists(os.path.join(export_dir, 'recon')):
            os.makedirs(os.path.join(export_dir, 'recon'))
        
        with open(os.path.join(export_dir, 'recon_params.json'), 'w') as params_f:
            json.dump(self.export_params(), params_f, indent=4)
        
        start_id = self.recon_layer_start
        for i in tqdm(range(self.cur_recon.shape[0])):
            norm = self._norm_recon(self.cur_recon[i])
            norm -= norm.min()
            norm = norm * (254 / np.max(norm))
            im = Image.fromarray(norm)
            im = im.convert("L")
            im_path = os.path.join(export_dir, 'recon', f'{start_id:05d}.tiff')
            im.save(im_path)
            start_id += 1
        
        return gr.Text.update(value="Done")
    
    def export_params(self):
        return {
            'proj_crop': self.crop,
            'center_offset': self.recon_center_offset,
            'is_clipped': self.recon_is_clip,
            'clip': self.recon_clip,
            'is_circ_crop': self.recon_circ_crop,
            'circ_ratio': self.recon_circ_ratio,
            'is_denoise': self.recon_is_denoise,
            'denoise_params': self.recon_denoise_params
        }


    def main_interface(self):
        metadata_fp = '/home/beams/S1IDUSER/new_data/alshibli_nov22/F50_sp5_tomo/F50_sp5_tomo_TomoFastScan.dat'
        override_path = '/home/beams/S1IDUSER/mnt/s1c/alshibli_nov22/tomo/F50_sp5_tomo'
        exp_name = 'alshibli_nov22'
        scan_name = 'F50_sp5_tomo'

        with gr.Blocks(title='1ID Tomo Reconstruction') as scan_if:
            # Loading Tab Layout
            with gr.Tab("Loading Data"):
                experiment_fld = gr.Textbox(label='Experiment Folder', value=exp_name)
                scan_fld = gr.Textbox(label='Scan Name', value=scan_name)
                override_ckbx = gr.Checkbox(label='Manual File Paths', value=False)
                metadata_fp_fld = gr.Textbox(label='Metadata File Path', value=metadata_fp, visible=False)
                override_path_fld = gr.Textbox(label='Image File Path', value=override_path, visible=False)
                omega_override_ckbx = gr.Checkbox(label='Override Angles', value=False)
                with gr.Row():
                    omega_start = gr.Number(label='Start Angle', value=-180.0, visible=False, interactive=True)
                    omega_end = gr.Number(label='End Angle', value=180.0, visible=False, interactive=True)

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

                rot_num = gr.Number(label='Rotation (deg)')

                proj_crop_btn = gr.Button('Crop & Rotate Projections')
        
            with gr.Tab('Reconstruction'):
                with gr.Row():
                    with gr.Column(scale=1.5):
                        recon_img = gr.Image(label='Reconstruction',
                                            image_mode="L",
                                            interactive=False).style(height='3')
                        recon_dist = gr.Image(label='Pixel Intensity Distribution')
                        recon_slide = gr.Slider(label='Reconstruction', interactive=True)


                    with gr.Column():
                        # Reset slider on recon
                        norm_rdo = gr.Radio(label='Normalization', choices=['TomoPy', 'Standard', 'None'], value='TomoPy')
                        center_num = gr.Number(label='Center Offset', precision=0, value=0)

                        with gr.Row():
                            recon_slice_btn = gr.Button("Reconstruct Slice")
                            recon_slice_start = gr.Number(label="Slice Start", precision=0, value=500)
                            recon_slice_num = gr.Number(label="Num Slices", precision=0, value=5)
                        recon_btn = gr.Button('Reconstruct All')

                        with gr.Row():
                            clip_ckbx = gr.Checkbox(label='Enable Clipping', value=True)
                            clip_low = gr.Number(label='Lower Bound', value=DEFAULTS['clip_low'])
                            clip_high = gr.Number(label='High Bound', value=DEFAULTS['clip_high'])
                        
                        with gr.Row():
                            circ_ckbx = gr.Checkbox(label='Enable Circle Crop')
                            circ_ratio = gr.Number(label='Crop Ratio', value=1.0)

                        with gr.Row():
                            denoise_ckbx = gr.Checkbox(label='Enable Denoise', value=False)
                            template_wdw_num = gr.Number(label='Template Window', precision=0, value=DEFAULTS['denoise_search'])
                            search_wdw_num = gr.Number(label='Search Window', precision=0, value=DEFAULTS['denoise_template'])
                        
                        apply_filters_btn = gr.Button('Apply Filters')

                        
                        with gr.Row():
                            # TODO: 16 bit int out
                            export_btn = gr.Button('Export Scans')
                            export_fld = gr.Textbox(label='Export Location', value='recon/test')
                        export_txt = gr.Textbox(label='Export Progress', interactive=False)

                with gr.Row():
                    center_render_btn = gr.Button("Generate Centering Renders")
                    center_render_ckbx = gr.Checkbox(label="Show Centering Render")
                    center_slice = gr.Number(label="Centering Slice", precision=0, value=500)

                center_render_imgrid = gr.Gallery(visible=False)

            # Functionality Loading Tab
            avail_scans = gr.State()

            override_ckbx.change(fn=self.hide_override, 
                                inputs=override_ckbx, 
                                outputs=[metadata_fp_fld, override_path_fld])

            omega_override_ckbx.change(fn=self.hide_omega, 
                                inputs=omega_override_ckbx, 
                                outputs=[omega_start, omega_end])


            find_scans_btn.click(fn=self.get_avaliable_scans, 
                                inputs=[override_ckbx, experiment_fld, scan_fld, metadata_fp_fld, override_path_fld], 
                                outputs=[self.scan_drop, avail_scans])

            load_scans_btn.click(fn=self.load_scan, 
                                inputs=[avail_scans, self.scan_drop, omega_override_ckbx, omega_start, omega_end], 
                                outputs=load_txt
                                )

            # Functionality Proj Tab
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
                                        proj_ys_num, proj_ye_num,
                                        rot_num],
                                outputs=proj_img
            )

            # Functionality Proj Tab

            recon_slide.change(fn=self.update_recon_slide,
                                inputs=recon_slide,
                                outputs=[recon_img, recon_dist])

            recon_btn.click(fn=self.reconstruct_all,
                                inputs=[recon_slide, norm_rdo, center_num],
                                outputs=[recon_img, recon_dist, recon_slide])

            recon_slice_btn.click(fn=self.reconstruct_slice,
                                inputs=[recon_slide, norm_rdo, center_num, recon_slice_start, recon_slice_num],
                                outputs=[recon_img, recon_dist, recon_slide])

            # Center Rendering

            center_render_ckbx.change(fn=self.update_center_render,
                                      inputs=center_render_ckbx,
                                      outputs=center_render_imgrid)
            
            center_render_btn.click(fn=self.render_center,
                                    inputs=[center_slice, norm_rdo],
                                    outputs=center_render_imgrid
            )

            # Modifications
             
            apply_filters_btn.click(fn=self.apply_filters, inputs=[recon_slide,
                                            clip_ckbx, clip_low, clip_high,
                                            circ_ckbx, circ_ratio,
                                            denoise_ckbx, template_wdw_num, search_wdw_num],
                                            outputs=[recon_img, recon_dist])

            # Export

            export_btn.click(fn=self.export_recon,
                             inputs=[export_fld],
                             outputs=export_txt
            )


        scan_if.queue(concurrency_count=3).launch()


if __name__ == '__main__':
    ReconUI()


