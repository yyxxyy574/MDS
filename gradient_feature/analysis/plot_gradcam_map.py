"""
Grad-CAM Map Vizualizer - v1.0 (Direct Overlay Pipeline)

核心修复: 针对 Grad-CAM 特征的归一化和处理进行全面更新。
1. 将 1D 提取特征 reshape 回 2D Spatial Grid (grid_h x grid_w)。
2. 【核心修复】: 移除了 Attention Sinks (边缘抑制) 逻辑。Grad-CAM 本身就是目标导向的，不需要这种修复。
3. 【核心修复】: 改进 2D 数据预处理。不再使用百分位数截断，而是使用标准的 min-max 归一化在 2D 空间上缩放特征。
4. 【核心修复】: 更新标题和文字为 Grad-CAM 热力图。
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

# 使用 Agg 后端，不显示弹窗
matplotlib.use('Agg')

# 设置英文字体
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
        # 🌟 修复: 自动检测 Qwen-VL 的 2x2 Patch Merging (即 4:1 压缩)
        if expected_len == actual_len * 4:
            grid_h = grid_h // 2
            grid_w = grid_w // 2
        else:
            # 如果不是标准的 4 倍关系，按照面积比例自动推算新的长宽
            print(f"Warning: Unexpected scale. 1D length ({actual_len}) vs grid ({grid_h}x{grid_w}) for {filename_base}.")
            ratio = np.sqrt(expected_len / actual_len)
            grid_h = int(np.round(grid_h / ratio))
            grid_w = int(np.round(grid_w / ratio))
            
            # 极限 Fallback (如果推算的长宽乘积依然对不上)
            if int(grid_h * grid_w) != actual_len:
                grid_h = int(np.sqrt(actual_len))
                grid_w = actual_len // grid_h

    # ==========================================
    # 🌟 核心步骤 1: 重塑成 2D 网格
    # shape: [grid_h, grid_w]
    # ==========================================
    try:
        spatial_2d = gradcam_1d.reshape((int(grid_h), int(grid_w)))
    except:
        return

    # ==========================================
    # 🌟 核心步骤 2: 归一化 (Min-Max Scaling on 2D)
    # Grad-CAM 范围是 arbitrary 0+, 不再经过 Softmax 0~1 分布。
    # 需要将其缩放到 0~1，作为 Colormap 输入。
    # 彻底移除了 Attention Sinks 边缘修复。
    # ==========================================
    vmin = np.min(spatial_2d)
    vmax = np.max(spatial_2d)
    
    if vmax > vmin:
        normalized_spatial = (spatial_2d - vmin) / (vmax - vmin + 1e-8)
    else:
        # 如果整个特征图是全黑的
        normalized_spatial = np.zeros_like(spatial_2d)

    # ==========================================
    # 🌟 核心步骤 3: 可视化生成
    # ==========================================
    
    os.makedirs(output_viz_dir, exist_ok=True)
    
    if save_npz:
        save_npz_path = os.path.join(output_viz_dir, f"{filename_base}_gradcam_2d_viz.npz")
        # 如果存在且内容没变，可以跳过 (可选逻辑)
        # np.savez_compressed(save_npz_path, gradcam_2d=normalized_spatial, grid_h=grid_h, grid_w=grid_w, jpg_path=jpg_path, filename=filename_base)
    
    # 生成可视化结果
    fig = None
    if overlay_viz and os.path.exists(jpg_path):
        
        try:
            img_pil = Image.open(jpg_path).convert('RGB')
        except:
            return
            
        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        
        # 1. 显示底色图像
        ax.imshow(img_pil)
        
        # 2. 生成 Colormap 热力图并通过上采样覆盖
        # extent=[left, right, bottom, top]
        im_overlay = ax.imshow(
            normalized_spatial,
            cmap=colormap,
            alpha=alpha,
            norm=Normalize(vmin=0, vmax=1),  # 固定使用 0~1 的标准分布
            interpolation='bilinear',       # 双线性插值上采样
            extent=[0, img_pil.width, img_pil.height, 0] # 坐标对齐
        )
        
        ax.set_axis_off() # 隐藏坐标轴
        
        # 增加 Colorbar，更新标题
        cax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
        cb = plt.colorbar(im_overlay, cax=cax, ticks=[0, 0.5, 1])
        cb.set_label('Grad-CAM Map (Normalized)', size=14)
        cax.set_yticklabels(['0', '0.5', '1'])
        
        ax.set_title(f"Grad-CAM Heatmap Overlay\nTarget Token: '{data['target_token']}'", size=16, pad=20)
        
        # 保存图像
        viz_save_path = os.path.join(output_viz_dir, f"{filename_base}_gradcam_viz.jpg")
        fig.savefig(viz_save_path, bbox_inches='tight', pad_inches=0.1)
        
        # 清理内存
        plt.close(fig)
        del img_pil


# ============================================================================
# Main Loop (Sequential Pipeline)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Visualize Grad-CAM Features from Qwen3-VL")
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--model-name', type=str, required=True)
    # Grad-CAM 提取默认使用 image 模式
    parser.add_argument('--iters', type=str, default=None)
    
    args = parser.parse_args()
    
    # 保持原有的提取路径架构
    extract_dir_base = os.path.dirname(os.path.abspath(__file__))
    model_short = args.model_name.split('/')[-1]
    
    gradcam_feature_dir = os.path.join(extract_dir_base, "..", "extract_result", args.dataset_name, "image_gradcam", model_short)
    
    if not os.path.exists(gradcam_feature_dir):
        print(f"Error: Grad-CAM features not found: {gradcam_feature_dir}")
        return

    # 输出可视化路径架构
    plot_viz_dir_base = os.path.dirname(os.path.abspath(__file__))
    output_viz_dir = os.path.join(plot_viz_dir_base, "..", "plots", args.dataset_name, "image_gradcam", model_short)
    
    os.makedirs(output_viz_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Running Grad-CAM Map Visualizer Sequential Pipeline")
    print(f"Input Features: {gradcam_feature_dir}")
    print(f"Output Viz: {output_viz_dir}")
    print(f"{'='*60}")
    
    # 解析 iters 用于过滤
    iters_filter = set(args.iters.split(',')) if args.iters else None

    # 获取所有 npz 文件
    # 必须匹配模式 "_gradcam.npz"
    npz_files = [f for f in os.listdir(gradcam_feature_dir) if f.endswith('_gradcam.npz')]
    
    success_count = 0
    
    # 顺序运行，防止 matplotlib 显存问题
    for f in tqdm(npz_files, desc="Visualizing Grad-CAMs"):
        
        # 应用过滤逻辑 (假设 filename_base 模式仍然适用)
        if iters_filter:
            filename_part = f.split('_')[1] # dimensions, dilemma, dilemma_instance
            # 这是一个简单的过滤逻辑，你可能需要重写它以适配真实数据
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
        
        # 防止显存爆炸
        if success_count % 100 == 0:
            torch.cuda.empty_cache()

    print(f"\n{'='*60}\nVisualization Finished.\nDirectory: {output_viz_dir}\nTotal Visualized: {success_count}\n{'='*60}")

if __name__ == '__main__':
    main()