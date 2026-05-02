"""
Grad-CAM Feature Extraction for Qwen3-VL - v1.0 (SLURM Distributed)

Core Logic:
- This script performs backward propagation to compute gradients of target token logits with respect to visual layer feature maps.
- Uses hooks to capture activations and gradients.
- Computes Grad-CAM features = ReLU(Mean(Gradients) * Activations).
- Requires 4-bit quantization for memory efficiency.
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

# Support environment variable overrides for SLURM
_extract_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.environ.get('PROJECT_ROOT',
    os.path.dirname(os.path.dirname(_extract_dir))
)
_data_root = os.environ.get('DATA_ROOT', os.path.join(_project_root, 'data'))

project_root = _project_root
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from config.constants import ROOT
    from Attention_Map.baseline.utils import prepare_data, parse_value, parse_description
except ImportError:
    print("Warning: Project utils not found, ensure paths are correct or self-contained data loading logic.")

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

# ============================================================================
# Hook Management (for capturing features and gradients)
# ============================================================================

class HookManager:
    def __init__(self):
        self.activations = None
        self.gradients = None
        self.forward_hook = None
        self.backward_hook = None
    
    def forward_hook_fn(self, module, input, output):
        target_output = output[0] if isinstance(output, tuple) else output
        self.activations = target_output.detach()
        if not target_output.requires_grad:
            target_output.requires_grad_(True)
        target_output.retain_grad()
    
    def backward_hook_fn(self, module, grad_input, grad_output):
        target_grad = grad_output[0] if isinstance(grad_output, tuple) else grad_output
        self.gradients = target_grad.detach()

    def register_hooks(self, layer):
        self.forward_hook = layer.register_forward_hook(self.forward_hook_fn)
        self.backward_hook = layer.register_full_backward_hook(self.backward_hook_fn)

    def remove_hooks(self):
        if self.forward_hook: self.forward_hook.remove()
        if self.backward_hook: self.backward_hook.remove()
        self.activations = None
        self.gradients = None

# ============================================================================
# Core Grad-CAM Extraction Function (Image Only)
# ============================================================================

def extract_gradcam_core(
    model,
    processor,
    image,
    content: str = "Answer the question with only yes or no.",
    target_layer_name: str = 'model.visual',
    target_token_text: str = None
) -> Dict[str, Any]:
    
    device = model.device
    target_layer = dict(model.named_modules())[target_layer_name]
    hook_manager = HookManager()
    hook_manager.register_hooks(target_layer)
    
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": content}
        ]}
    ]
    
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
    
    grid_h, grid_w = 0, 0
    if 'image_grid_thw' in inputs:
        thw = inputs['image_grid_thw'].cpu().numpy()
        if len(thw) > 0:
            _, grid_h, grid_w = thw[0]
        
    if target_token_text is None:
        with torch.no_grad():
            gen_outputs = model.generate(**inputs, max_new_tokens=1)
            token_id = gen_outputs[0, -1].item()
            target_token_text = processor.tokenizer.decode([token_id]).strip()
    else:
        token_id = processor.tokenizer.convert_tokens_to_ids(target_token_text)
        if token_id is None:
            print(f"Error: Token '{target_token_text}' not found in vocabulary.")
            return None
    
    model.zero_grad()
    outputs = model(**inputs)
    logits = outputs.logits
    target_logit = logits[:, -1, token_id]
    target_logit.backward()
    
    activations = hook_manager.activations
    gradients = hook_manager.gradients
    
    if activations is None or gradients is None:
        print(f"Error: Hooks failed to capture data for layer '{target_layer_name}'.")
        return None
        
    if activations.dim() == 2:
        weights = torch.mean(gradients, dim=0)
        gradcam_feature_1d = torch.sum(weights * activations, dim=-1)
        gradcam_feature_1d = torch.clamp(gradcam_feature_1d, min=0)
        gradcam_np = gradcam_feature_1d.float().cpu().numpy()
    elif activations.dim() == 3:
        weights = torch.mean(gradients, dim=1)
        gradcam_feature_1d = torch.sum(weights.unsqueeze(1) * activations, dim=-1)
        gradcam_feature_1d = torch.clamp(gradcam_feature_1d, min=0)
        gradcam_np = gradcam_feature_1d[0].float().cpu().numpy()
    else:
        print(f"Error: Unexpected activation dimensions: {activations.dim()}")
        return None
    
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
# Distributed Extraction Loop (SLURM Distributed)
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
    
    global_rank = 0
    local_rank = 0
    world_size = 1
    
    if 'SLURM_PROCID' in os.environ:
        global_rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        world_size = int(os.environ['SLURM_NTASKS'])
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
    data = prepare_data(dataset_dir, sample_ratio=effective_sample_ratio, seed=sample_seed)

    success_count, skip_count, missing_data_count = 0, 0, 0
    total_dilemmas = len(data)
    
    for idx, (dilemma, data_list) in enumerate(data.items(), start=global_rank + 1):
        outer_tag = f"{idx}/{total_dilemmas}"
        pbar_desc = f"[Rank {global_rank}] {dilemma} ({outer_tag})"
        for sample in tqdm(data_list, desc=pbar_desc, disable=(local_rank != 0)):
            if iters:
                values = parse_value(sample['value'])
                if values[-1] not in iters: continue

            base_name = f"{sample['dimension']}_{sample['dilemma_instance']}_{sample['feature']}_{sample['filename']}"
            if allowed_basenames is not None and base_name not in allowed_basenames:
                continue

            save_path = os.path.join(output_dir, f"{base_name}_gradcam.npz")
            if os.path.exists(save_path):
                try:
                    existing = np.load(save_path)
                    if 'gradcam_feature' in existing and existing['gradcam_feature'] is not None:
                        skip_count += 1
                        continue
                except:
                    pass
            
            image_input = sample['jpg_path']
            if not os.path.exists(image_input):
                missing_data_count += 1
                continue
                
            try:
                result = extract_gradcam_core(model, processor, image_input)
                if result is None: continue
                
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
                
                if success_count % 100 == 0:
                    torch.cuda.empty_cache()
            except Exception as e:
                error_msg = f"[Rank {global_rank}] ERROR processing {sample.get('filename', 'unknown')}: {e}"
                if local_rank == 0:
                    print(error_msg)
                    
    torch.cuda.empty_cache()
    if local_rank == 0:
        print(f"\n{'='*60}\nExtraction finished. Directory: {output_dir}\nExtracted: {success_count}, Skipped: {skip_count}\n{'='*60}")


# ============================================================================
# Main Entry Point (SLURM Distributed)
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract Grad-CAM Features from Qwen3-VL (SLURM Distributed)")
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--model-name', type=str, required=True)
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

    model_load_kwargs = {}
    if args.device != "auto" and args.device.isdigit():
        model_load_kwargs["device_map"] = {"": int(args.device)}
    else:
        model_load_kwargs["device_map"] = "auto"
    
    model_name_lower = args.model_name.lower()
    if '32b' in model_name_lower or args.quantize:
        print(f"[{args.model_name}] Enabling 4-bit quantization for Grad-CAM...")
        model_load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_use_double_quant=True,
        )
        model_load_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_load_kwargs["torch_dtype"] = torch.bfloat16 

    try:
        model_path = os.path.expanduser(args.model_name)
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **model_load_kwargs)
        processor = AutoProcessor.from_pretrained(model_path)
        model.eval()
    except Exception as e:
        print(f"Error loading model {args.model_name}: {e}")
        sys.exit(1)

    model_short = args.model_name.split('/')[-1]
    save_root = os.path.join(extract_dir, "extract_result", args.dataset_name, "image_gradcam", model_short)
    iters = set(args.iters.split(',')) if args.iters else None
    
    run_extraction_loop_gradcam(
        output_dir=save_root, dataset_dir=dataset_dir, extract_func=extract_gradcam_core,
        iters=iters, dataset_name=args.dataset_name, model_short=model_short,
        sample_ratio=args.sample_ratio, sample_seed=args.sample_seed, sample_index=args.sample_index
    )