"""
Grad-CAM Feature Extraction for Qwen3-VL - v1.0 (SLURM Distributed)

核心原理: 此脚本执行反向传播 (Backward) 以计算目标 Token 的 Logit 对视觉层特征图的梯度。
1. Hook 目标视觉层 (`model.model.visual`) 的输出。
2. 构造 Prompt ("Is this ethical? Answer yes or no.")。
3. 获取 "Yes" Token 的 Logit。
4. 执行 backward()。
5. 计算 Grad-CAM 特征 = ReLU(Mean(Gradients) * Activations)。

显存提示: Grad-CAM 比 Attention 极其耗显存。强制使用 4-bit 量化运行。
"""

import argparse
import torch
import os
import numpy as np
from PIL import Image
import sys
import yaml
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm
from functools import partial

current_file_path = os.path.abspath(__file__)
extract_dir = os.path.dirname(current_file_path)

# SLURM 兼容: 支持环境变量覆盖
_extract_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.environ.get('PROJECT_ROOT',
    os.path.dirname(os.path.dirname(_extract_dir))
)
_data_root = os.environ.get('DATA_ROOT', os.path.join(_project_root, 'data'))

project_root = _project_root
if project_root not in sys.path:
    sys.path.append(project_root)

# 确保这些模块可供项目加载
try:
    from config.constants import ROOT
    from Attention_Map.baseline.utils import prepare_data, parse_value, parse_description
except ImportError:
    print("Warning: Project utils not found, ensure paths are correct or self-contained data loading logic.")
    # 如果 utils 没找到，你需要在这里提供 self-contained 的数据加载逻辑或占位符

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

# ============================================================================
# Hook 钩子管理 (用于捕获特征和梯度)
# ============================================================================

class HookManager:
    def __init__(self):
        self.activations = None
        self.gradients = None
        self.forward_hook = None
        self.backward_hook = None
    
    def forward_hook_fn(self, module, input, output):
        # 🌟 修复：如果输出是 tuple，提取第一个元素（实际的特征张量）
        target_output = output[0] if isinstance(output, tuple) else output
        
        self.activations = target_output.detach()
        if not target_output.requires_grad:
            target_output.requires_grad_(True)
            
        # 激活值需要 retain_grad，以便在 backward 后获取梯度
        target_output.retain_grad()
    
    def backward_hook_fn(self, module, grad_input, grad_output):
        # 🌟 修复：反向传播的梯度也可能是 tuple
        target_grad = grad_output[0] if isinstance(grad_output, tuple) else grad_output
        
        self.gradients = target_grad.detach()

    def register_hooks(self, layer):
        # 注册前向钩子捕获激活值
        self.forward_hook = layer.register_forward_hook(self.forward_hook_fn)
        # 注册反向钩子捕获梯度
        self.backward_hook = layer.register_full_backward_hook(self.backward_hook_fn)

    def remove_hooks(self):
        if self.forward_hook: self.forward_hook.remove()
        if self.backward_hook: self.backward_hook.remove()
        self.activations = None
        self.gradients = None

# ============================================================================
# 核心 Grad-CAM 提取函数 (Image Only)
# ============================================================================

def extract_gradcam_core(
    model,
    processor,
    image,
    content: str = "Answer the question with only yes or no.", # Targeted Question
    target_layer_name: str = 'model.visual',     # Correct for Qwen3-VL ViT
    target_token_text: str = None                     # Target output
) -> Dict[str, Any]:
    
    device = model.device
    # print([name for name, _ in model.named_modules() if 'visual' in name or 'vision' in name])
    
    # 获取目标层
    target_layer = dict(model.named_modules())[target_layer_name]
    hook_manager = HookManager()
    hook_manager.register_hooks(target_layer)
    
    # 准备 messages
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": content}
        ]}
    ]
    
    # 预处理
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
    
    # 获取图像网格
    grid_h, grid_w = 0, 0
    if 'image_grid_thw' in inputs:
        thw = inputs['image_grid_thw'].cpu().numpy()
        if len(thw) > 0:
            _, grid_h, grid_w = thw[0]
        
    # ==========================================
    # 🌟 核心修复: 执行 Forward + Backward 获取梯度
    # 这正是 Grad-CAM 与 Attention 的本质区别
    # ==========================================

    if target_token_text is None:
        # 让模型自己先跑一次，看它到底想输出什么词 (Yes 还是 No 还是其他)
        with torch.no_grad():
            gen_outputs = model.generate(**inputs, max_new_tokens=1)
            # 拿到生成的最新一个 Token 的 ID
            token_id = gen_outputs[0, -1].item()
            # 解码成文字，方便画图的时候显示在标题上
            target_token_text = processor.tokenizer.decode([token_id]).strip()
    else:
        # 如果你依然想强行写死某个词进行测试
        token_id = processor.tokenizer.convert_tokens_to_ids(target_token_text)
        if token_id is None:
            print(f"Error: Token '{target_token_text}' not found in vocabulary.")
            return None
    
    # 确保梯度为零
    model.zero_grad()
    
    # 1. 前向传播 (Forward)
    outputs = model(**inputs)
    logits = outputs.logits  # Shape: [batch, seq_len, vocab_size]
    
    # 获取最后一个 Token 的 Logit (针对 Prompt 结尾的决策时刻)
    target_logit = logits[:, -1, token_id]
    
    # 2. 反向传播 (Backward)
    # 计算 target_logit 对视觉层特征图的梯度
    target_logit.backward()
    
    # ==========================================
    # 🌟 核心计算: Grad-CAM 特征生成
    # 1D 视觉 token sequence (未 reshape 成 2D 之前)
    # ==========================================
    
    # 获取 Hook 捕获的数据
    activations = hook_manager.activations  # 可能是 [N, D] 或是 [B, N, D]
    gradients = hook_manager.gradients      # 同上
    
    if activations is None or gradients is None:
        print(f"Error: Hooks failed to capture data for layer '{target_layer_name}'.")
        return None
        
    # 计算权重 (Mean Gradient per channel 跨越空间维度)
    # Qwen-VL 采用了动态分辨率，视觉输出去掉了 Batch 维度，变成了 2D 张量 [num_tokens, hidden_dim]
    if activations.dim() == 2:
        # 形状是 [num_visual_tokens, hidden_dim]
        # 在 token 维度 (dim=0) 上求平均，得到每个通道的权重
        weights = torch.mean(gradients, dim=0)  # [hidden_dim]
        
        # 计算 Grad-CAM 特征图: sum(weights * activations)
        # weights 自动广播: [hidden_dim] * [num_visual_tokens, hidden_dim]
        gradcam_feature_1d = torch.sum(weights * activations, dim=-1)  # [num_visual_tokens]
        
        # ReLU 激活
        gradcam_feature_1d = torch.clamp(gradcam_feature_1d, min=0)
        gradcam_np = gradcam_feature_1d.float().cpu().numpy()
        
    elif activations.dim() == 3:
        # 如果模型版本更新切回了 3D 形状 [batch, num_visual_tokens, hidden_dim]
        weights = torch.mean(gradients, dim=1)  # [batch, hidden_dim]
        gradcam_feature_1d = torch.sum(weights.unsqueeze(1) * activations, dim=-1) # [batch, num_visual_tokens]
        
        gradcam_feature_1d = torch.clamp(gradcam_feature_1d, min=0)
        gradcam_np = gradcam_feature_1d[0].float().cpu().numpy()
    else:
        print(f"Error: Unexpected activation dimensions: {activations.dim()}")
        return None
    
    # 释放资源
    hook_manager.remove_hooks()
    del outputs, logits, activations, gradients, target_logit
    
    result = {
        'gradcam_feature': gradcam_np,
        'grid_h': int(grid_h),
        'grid_w': int(grid_w),
        'content': content,
        'target_layer_name': target_layer_name,
        'target_token_text': target_token_text,
    }
    
    return result


# ============================================================================
# Distributed Extraction Loop (SLURM Distributed aware)
# ============================================================================

def run_extraction_loop_gradcam(
    output_dir, dataset_dir, extract_func, iters=None, dataset_name=None,
    model_short=None, multi_head_config=None, sample_ratio=None, sample_seed=None,
    sample_index=None
):
    print(f"\n{'='*60}")
    print(f"Running Grad-CAM Extraction Loop | Distributed")
    if sample_ratio is not None:
        print(f"Sampling ratio: {sample_ratio} (seed={sample_seed})")
    print(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 分布式设置
    global_rank = 0
    local_rank = 0
    world_size = 1
    
    if 'SLURM_PROCID' in os.environ:
        global_rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        # 设置显卡索引
        device = torch.device(f"cuda:{local_rank}")
        print(f"[Rank {global_rank}/{world_size}] Using GPU: {device}")
    
    allowed_basenames: set = None
    if sample_index:
        import json
        if os.path.exists(sample_index):
            with open(sample_index, 'r', encoding='utf-8') as f:
                idx_data = json.load(f)
            allowed_basenames = {entry['filename_base'] for entry in idx_data.get('tasks', {}).values()}
        else:
            print(f"[sample-index] WARNING: index file not found: {sample_index}")

    effective_sample_ratio = None if (sample_index and allowed_basenames) else sample_ratio
    # 分布式加载数据切片
    data = prepare_data(dataset_dir, sample_ratio=effective_sample_ratio, seed=sample_seed)

    success_count, skip_count, missing_data_count = 0, 0, 0
    total_dilemmas = len(data)
    
    for idx, (dilemma, data_list) in enumerate(data.items(), start= global_rank + 1):
        outer_tag = f"{idx}/{total_dilemmas}"
        
        # 针对每个 Dilemma 的 tqdm，附带分布式状态
        pbar_desc = f"[Rank {global_rank}] {dilemma} ({outer_tag})"
        for sample in tqdm(data_list, desc=pbar_desc, disable=(local_rank != 0)):
            if iters:
                values = parse_value(sample['value'])
                if values[-1] not in iters: continue

            base_name = f"{sample['dimension']}_{sample['dilemma_instance']}_{sample['feature']}_{sample['filename']}"
            if allowed_basenames is not None and base_name not in allowed_basenames:
                continue

            # 保存路径强制使用 npz
            save_path = os.path.join(output_dir, f"{base_name}_gradcam.npz")
            
            if os.path.exists(save_path):
                # 检查文件是否损坏
                try:
                    existing = np.load(save_path)
                    if 'gradcam_feature' in existing and existing['gradcam_feature'] is not None:
                        skip_count += 1
                        continue
                except:
                    pass
            
            # Grad-CAM 核心只处理图像模式
            image_input = sample['jpg_path']
            if not os.path.exists(image_input):
                missing_data_count += 1
                continue
                
            try:
                # 执行核心提取
                result = extract_gradcam_core(model, processor, image_input)
                
                if result is None: continue
                
                # 构造保存字典
                save_dict = {
                    'gradcam_feature': result.get('gradcam_feature'),
                    'grid_h': result.get('grid_h', 0),
                    'grid_w': result.get('grid_w', 0),
                    'content': result.get('content', ''),
                    'target_layer': result.get('target_layer_name', ''),
                    'target_token': result.get('target_token_text', ''),
                    'jpg_path': str(sample.get('jpg_path', '')),
                    'filename': str(sample.get('filename', '')),
                }
                
                np.savez_compressed(save_path, **save_dict)
                success_count += 1
                
                # 显存清理
                if success_count % 100 == 0:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                # 分布式下的错误不应中断其他任务
                pbar_desc = f"[Rank {global_rank}] ERROR processing {sample.get('filename', 'unknown')}"
                if local_rank == 0:
                    print(f"{pbar_desc}: {e}")
                else:
                    print(pbar_desc)
                    
    # 释放 GPU
    torch.cuda.empty_cache()
    pbar_desc = f"[Rank {global_rank}] Extraction Finished."
    
    if local_rank == 0:
        print(f"\n{'='*60}\nRank 0 Reporting: Directory: {output_dir}\nRank 0 Extracted: {success_count}\nRank 0 Skipped: {skip_count}\n{'='*60}")
    else:
        print(f"{pbar_desc}")


# ============================================================================
# 主入口 (针对 SLURM 分布式优化)
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract Grad-CAM Features from Qwen3-VL (SLURM Distributed)")
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--model-name', type=str, required=True)
    # Grad-CAM 模式固定为 image
    parser.add_argument('--iters', type=str, default=None)
    parser.add_argument('--quantize', action='store_true', help="Force 4bit quantization manually")
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--sample-ratio', type=float, default=None)
    parser.add_argument('--sample-seed', type=int, default=None)
    parser.add_argument('--sample-index', type=str, default=None)

    args = parser.parse_args()
    
    dataset_dir = os.path.join(_data_root, args.dataset_name, "samples")
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset not found: {dataset_dir}")
        sys.exit(1)

    # ==========================================
    # 设备与模型加载配置 (自动量化逻辑)
    # ==========================================
    model_load_kwargs = {}
    
    if args.device != "auto" and args.device.isdigit():
        model_load_kwargs["device_map"] = {"": int(args.device)}
    else:
        model_load_kwargs["device_map"] = "auto"
        
    model_name_lower = args.model_name.lower()
    
    # 逻辑：如果是 32B 模型，或者手动传入了 --quantize，则开启 4-bit 量化
    if '32b' in model_name_lower or args.quantize:
        print(f"[{args.model_name}] Detected 32B model (or --quantize set). Enabling 4-bit quantization for Grad-CAM...")
        model_load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_use_double_quant=True,
        )
        model_load_kwargs["torch_dtype"] = torch.bfloat16
    else:
        # 8B 模型：不开量化，但推荐使用 bfloat16 以防止显存溢出并加速 backward
        print(f"[{args.model_name}] Detected 8B model. Loading in standard precision (bfloat16)...")
        model_load_kwargs["torch_dtype"] = torch.bfloat16 

    # 加载模型和 Processor
    try:
        model_path = os.path.expanduser(args.model_name)
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **model_load_kwargs)
        processor = AutoProcessor.from_pretrained(model_path)
        model.eval()
        
    except Exception as e:
        print(f"Error loading model {args.model_name}: {e}")
        sys.exit(1)

    # 构造输出路径
    model_short = args.model_name.split('/')[-1]
    save_root = os.path.join(extract_dir, "extract_result", args.dataset_name, "image_gradcam", model_short)
    
    # 解析 iters
    iters = set(args.iters.split(',')) if args.iters else None

    # 执行分布式提取循环
    run_extraction_loop_gradcam(
        output_dir=save_root, dataset_dir=dataset_dir, extract_func=extract_gradcam_core,
        iters=iters, dataset_name=args.dataset_name, model_short=model_short,
        sample_ratio=args.sample_ratio, sample_seed=args.sample_seed, sample_index=args.sample_index
    )