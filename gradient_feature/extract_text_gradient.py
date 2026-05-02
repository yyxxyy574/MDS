"""
Text Gradient Extraction - v1.0 (Input x Gradient)

核心逻辑:
1. 专门用于处理纯文本 (text) 和文本描述 (caption) 模式。
2. 动态生成目标 Token（看模型自己想输出什么）。
3. 提取 `inputs_embeds` (词嵌入) 并执行反向传播。
4. 计算 Saliency Score = L2_Norm(Embedding * Gradient)。
"""

import argparse
import torch
import os
import numpy as np
import sys
import yaml
from typing import Dict, Any
from tqdm import tqdm

current_file_path = os.path.abspath(__file__)
extract_dir = os.path.dirname(current_file_path)
_project_root = os.environ.get('PROJECT_ROOT', os.path.dirname(os.path.dirname(extract_dir)))
_data_root = os.environ.get('DATA_ROOT', os.path.join(_project_root, 'data'))

if _project_root not in sys.path:
    sys.path.append(_project_root)

from baseline.utils import prepare_data, parse_value, parse_description
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

def extract_text_gradient_core(
    model, processor, content: str, target_token_text: str = None
) -> Dict[str, Any]:
    
    device = model.device
    
    # 构建纯文本消息
    messages = [{"role": "user", "content": [{"type": "text", "text": content}]}]
    
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
    input_ids = inputs['input_ids']
    
    # 动态获取 Target Token (如果未指定)
    if target_token_text is None:
        with torch.no_grad():
            gen_outputs = model.generate(**inputs, max_new_tokens=1)
            token_id = gen_outputs[0, -1].item()
            target_token_text = processor.tokenizer.decode([token_id]).strip()
    else:
        token_id = processor.tokenizer.convert_tokens_to_ids(target_token_text)
        if token_id is None:
            return None

    # --- 获取词嵌入 (Embeddings) ---
    # Qwen 的词嵌入层可以直接通过 get_input_embeddings() 获取
    embedding_layer = model.get_input_embeddings()
    inputs_embeds = embedding_layer(input_ids)
    inputs_embeds.retain_grad() # 挂载钩子，保留梯度
    
    # 移除 input_ids，改用 inputs_embeds 喂给模型
    forward_inputs = {k: v for k, v in inputs.items() if k != 'input_ids'}
    forward_inputs['inputs_embeds'] = inputs_embeds

    model.zero_grad()
    
    # --- 前向传播与反向传播 ---
    outputs = model(**forward_inputs)
    logits = outputs.logits
    
    # 获取生成词的 logit 并反向传播
    target_logit = logits[0, -1, token_id]
    target_logit.backward()
    
    # --- 计算 Input x Gradient ---
    grad = inputs_embeds.grad  # [1, seq_len, hidden_dim]
    embeds = inputs_embeds.detach() # [1, seq_len, hidden_dim]
    
    # 核心公式: 输入特征乘梯度
    input_x_grad = (embeds * grad).squeeze(0) # [seq_len, hidden_dim]
    
    # 对 hidden_dim 维度求 L2 范数，得到每个 Token 的标量显著性得分
    saliency_scores = torch.norm(input_x_grad, dim=-1).float().cpu().numpy() # [seq_len] # [seq_len]
    
    # 提取文本 Token
    token_texts = processor.tokenizer.convert_ids_to_tokens(input_ids[0].cpu())
    
    torch.cuda.empty_cache()
    
    return {
        'gradient_weights': saliency_scores,
        'token_texts': token_texts,
        'prompt': content,
        'target_token': target_token_text
    }

def run_extraction_loop_text(
    output_dir, dataset_dir, mode, model, processor, iters=None, 
    sample_index=None
):
    print(f"\n{'='*60}\nRunning Text Gradient Extraction | Mode: {mode}\n{'='*60}")
    os.makedirs(output_dir, exist_ok=True)
    
    allowed_basenames = None
    if sample_index and os.path.exists(sample_index):
        import json
        with open(sample_index, 'r', encoding='utf-8') as f:
            idx_data = json.load(f)
            allowed_basenames = {entry['filename_base'] for entry in idx_data.get('tasks', {}).values()}

    data = prepare_data(dataset_dir)
    success_count = 0
    model_short = output_dir.rstrip('/').split('/')[-1]
    visual_key, ocr_key = f"generated_visual_{model_short}", f"generated_orc_{model_short}"
    
    for dilemma, data_list in data.items():
        for sample in tqdm(data_list, desc=f"Extracting {dilemma}"):
            if iters:
                values = parse_value(sample['value'])
                if values[-1] not in iters: continue

            base_name = f"{sample['dimension']}_{sample['dilemma_instance']}_{sample['feature']}_{sample['filename']}"
            if allowed_basenames and base_name not in allowed_basenames:
                continue

            save_path = os.path.join(output_dir, f"{base_name}_textgrad.npz")
            if os.path.exists(save_path): continue
            
            # --- 模式解析 ---
            prompt = ""
            if mode == 'text':
                with open(sample['yaml_path'], 'r', encoding='utf-8') as f:
                    desc = yaml.safe_load(f).get('description', '')
                if not desc: continue
                prompt = f"{parse_description(desc)} Answer the question with only yes or no."
            elif mode == 'caption':
                with open(sample['yaml_path'], 'r', encoding='utf-8') as f:
                    meta = yaml.safe_load(f)
                v_text, o_text = meta.get(visual_key, ""), meta.get(ocr_key, "")
                if not v_text and not o_text: continue
                prompt = f"{v_text}\n{o_text}\n Answer the question with only yes or no."
            
            try:
                result = extract_text_gradient_core(model, processor, prompt)
                if not result: continue
                
                np.savez_compressed(
                    save_path, 
                    gradient_weights=result['gradient_weights'],
                    token_texts=np.array(result['token_texts'], dtype=object),
                    prompt=str(result['prompt']),
                    target_token=str(result['target_token']),
                    filename=str(sample.get('filename', '')),
                    dilemma=str(dilemma)
                )
                success_count += 1
            except Exception as e:
                print(f"Error on {base_name}: {e}")

    print(f"\nFinished. Extracted: {success_count}. Dir: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True, choices=['text', 'caption'])
    parser.add_argument('--quantize', action='store_true')
    parser.add_argument('--sample-index', type=str, default=None)
    args = parser.parse_args()
    
    dataset_dir = os.path.join(_data_root, args.dataset_name, "samples")
    
    kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
    if '32b' in args.model_name.lower() or args.quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )
        
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_name, **kwargs)
    processor = AutoProcessor.from_pretrained(args.model_name)
    model.eval()
    
    model_short = args.model_name.split('/')[-1]
    save_root = os.path.join(extract_dir, "extract_result", args.dataset_name, f"{args.mode}_grad", model_short)
    
    run_extraction_loop_text(save_root, dataset_dir, args.mode, model, processor, sample_index=args.sample_index)