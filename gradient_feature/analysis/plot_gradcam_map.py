"""
Grad-CAM Map Vizualizer
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
import numpy as np
from PIL import Image
import os
import argparse
from tqdm import tqdm
from functools import partial
import torch

matplotlib.use('Agg')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

# ============================================================================
# Core Visualization Function
# ============================================================================

def plot_gradcam(
    gradcam_npz_path: str, output_viz_dir: str, save_npz: bool = True,
    overlay_viz: bool = True, alpha: float = 0.6, colormap: str = 'jet'
):
    
    try:
        data = np.load(gradcam_npz_path)
    except:
        return
        
    if 'gradcam_feature' not in data or data['gradcam_feature'] is None:
        return
        
    gradcam_1d = data['gradcam_feature']
    grid_h, grid_w = data['grid_h'], data['grid_w']
    jpg_path = str(data['jpg_path'])
    filename_base = gradcam_npz_path.split('/')[-1].replace('_gradcam', '')
    
    expected_len = int(grid_h * grid_w)
    actual_len = len(gradcam_1d)
    
    if expected_len != actual_len:
        if expected_len == actual_len * 4:
            grid_h = grid_h // 2
            grid_w = grid_w // 2
        else:
            print(f"Warning: Unexpected scale. 1D length ({actual_len}) vs grid ({grid_h}x{grid_w}) for {filename_base}.")
            ratio = np.sqrt(expected_len / actual_len)
            grid_h = int(np.round(grid_h / ratio))
            grid_w = int(np.round(grid_w / ratio))
            
            if int(grid_h * grid_w) != actual_len:
                grid_h = int(np.sqrt(actual_len))
                grid_w = actual_len // grid_h

    try:
        spatial_2d = gradcam_1d.reshape((int(grid_h), int(grid_w)))
    except:
        return

    vmin = np.min(spatial_2d)
    vmax = np.max(spatial_2d)
    
    if vmax > vmin:
        normalized_spatial = (spatial_2d - vmin) / (vmax - vmin + 1e-8)
    else:
        normalized_spatial = np.zeros_like(spatial_2d)
    
    os.makedirs(output_viz_dir, exist_ok=True)
    
    if save_npz:
        save_npz_path = os.path.join(output_viz_dir, f"{filename_base}_gradcam_2d_viz.npz")
        np.savez_compressed(save_npz_path, gradcam_2d=normalized_spatial, grid_h=grid_h, grid_w=grid_w, jpg_path=jpg_path, filename=filename_base)
    
    fig = None
    if overlay_viz and os.path.exists(jpg_path):
        
        try:
            img_pil = Image.open(jpg_path).convert('RGB')
        except:
            return
            
        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        
        ax.imshow(img_pil)

        im_overlay = ax.imshow(
            normalized_spatial,
            cmap=colormap,
            alpha=alpha,
            norm=Normalize(vmin=0, vmax=1),
            interpolation='bilinear',
            extent=[0, img_pil.width, img_pil.height, 0]
        )
        
        ax.set_axis_off()
        
        cax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
        cb = plt.colorbar(im_overlay, cax=cax, ticks=[0, 0.5, 1])
        cb.set_label('Grad-CAM Map (Normalized)', size=14)
        cax.set_yticklabels(['0', '0.5', '1'])
        
        ax.set_title(f"Grad-CAM Heatmap Overlay\nTarget Token: '{data['target_token']}'", size=16, pad=20)
        
        viz_save_path = os.path.join(output_viz_dir, f"{filename_base}_gradcam_viz.jpg")
        fig.savefig(viz_save_path, bbox_inches='tight', pad_inches=0.1)

        plt.close(fig)
        del img_pil


# ============================================================================
# Main Loop (Sequential Pipeline)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Visualize Grad-CAM Features from Qwen3-VL")
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--iters', type=str, default=None)
    
    args = parser.parse_args()

    extract_dir_base = os.path.dirname(os.path.abspath(__file__))
    model_short = args.model_name.split('/')[-1]
    
    gradcam_feature_dir = os.path.join(extract_dir_base, "..", "extract_result", args.dataset_name, "image_gradcam", model_short)
    
    if not os.path.exists(gradcam_feature_dir):
        print(f"Error: Grad-CAM features not found: {gradcam_feature_dir}")
        return

    plot_viz_dir_base = os.path.dirname(os.path.abspath(__file__))
    output_viz_dir = os.path.join(plot_viz_dir_base, "..", "plots", args.dataset_name, "image_gradcam", model_short)
    
    os.makedirs(output_viz_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Running Grad-CAM Map Visualizer Sequential Pipeline")
    print(f"Input Features: {gradcam_feature_dir}")
    print(f"Output Viz: {output_viz_dir}")
    print(f"{'='*60}")
    
    iters_filter = set(args.iters.split(',')) if args.iters else None

    npz_files = [f for f in os.listdir(gradcam_feature_dir) if f.endswith('_gradcam.npz')]
    
    success_count = 0
    
    for f in tqdm(npz_files, desc="Visualizing Grad-CAMs"):
        
        if iters_filter:
            filename_part = f.split('_')[1] # dimensions, dilemma, dilemma_instance
            found = False
            for iter_t in iters_filter:
                if iter_t in f: found = True; break
            if not found: continue

        gradcam_npz_path = os.path.join(gradcam_feature_dir, f)
        
        plot_gradcam(
            gradcam_npz_path=gradcam_npz_path, output_viz_dir=output_viz_dir, save_npz=True,
            overlay_viz=True, alpha=0.6, colormap='jet'
        )
        success_count += 1
        
        if success_count % 100 == 0:
            torch.cuda.empty_cache()

    print(f"\n{'='*60}\nVisualization Finished.\nDirectory: {output_viz_dir}\nTotal Visualized: {success_count}\n{'='*60}")

if __name__ == '__main__':
    main()