"""
Text Gradient Plotter - v1.0
功能:
1. 生成高分 Token 柱状图 (继承 Attention Map 的清洗逻辑)。
2. 生成 HTML 文本高亮文件 (NLP 领域标准展示方法)。
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from tqdm import tqdm

# 借用你之前的清洗逻辑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from plot_attention_map import clean_tokens_and_weights, is_neutral_token
except ImportError:
    print("Warning: Ensure plot_attention_map.py is in the same directory for token cleaning logic.")

def generate_html_highlight(token_texts, weights, target_token, save_path):
    """生成漂亮的 HTML 文本高亮"""
    # 归一化权重到 0-1
    w_min, w_max = np.min(weights), np.max(weights)
    if w_max > w_min:
        norm_weights = (weights - w_min) / (w_max - w_min)
    else:
        norm_weights = np.zeros_like(weights)

    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.8; padding: 20px; }}
            .container {{ max-width: 800px; margin: auto; background: #f9f9f9; padding: 20px; border-radius: 8px; }}
            .word {{ padding: 2px 4px; border-radius: 4px; display: inline-block; }}
            h3 {{ color: #333; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h3>Target Decision: <span style="color:red;">"{target_token}"</span></h3>
            <p style="font-size: 18px;">
    """
    
    for token, weight in zip(token_texts, norm_weights):
        clean = token.replace('Ġ', '').replace('Ċ', ' ').replace('<|im_start|>', '').strip()
        if not clean: continue
        # 根据权重计算透明度 (红色系)
        alpha = float(weight) * 0.85 # 限制最高不完全不透明
        color = f"rgba(255, 0, 0, {alpha})"
        font_weight = "bold" if weight > 0.5 else "normal"
        html_content += f'<span class="word" style="background-color: {color}; font-weight: {font_weight};">{clean}</span> '
        
    html_content += "</p></div></body></html>"
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

def plot_text_gradient(npz_path, output_dir):
    data = np.load(npz_path, allow_pickle=True)
    weights = data['gradient_weights']
    token_texts = data['token_texts']
    target_token = str(data.get('target_token', 'Unknown'))
    filename_base = npz_path.split('/')[-1].replace('_textgrad', '')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 生成柱状图
    cleaned_tokens, cleaned_weights = clean_tokens_and_weights(
        token_texts, weights, filter_neutral=True, deduplicate=True
    )
    
    if len(cleaned_tokens) > 0:
        top_k = 15
        n_display = min(len(cleaned_weights), top_k)
        top_indices = np.argsort(cleaned_weights)[-n_display:][::-1]
        
        display_tokens = [cleaned_tokens[i] for i in top_indices]
        display_weights = cleaned_weights[top_indices]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(display_tokens))
        w_max = float(np.max(display_weights)) + 1e-8
        colors = plt.cm.Reds(display_weights / w_max)
        
        ax.barh(y_pos, display_weights, color=colors, edgecolor='white')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(display_tokens, fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        ax.set_title(f"Input x Gradient Saliency\nTarget: '{target_token}'", fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{filename_base}_bar.png"), dpi=150)
        plt.close()

    # 2. 生成 HTML 高亮
    html_path = os.path.join(output_dir, f"{filename_base}_highlight.html")
    generate_html_highlight(token_texts, weights, target_token, html_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True, choices=['text', 'caption'])
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_short = args.model_name.split('/')[-1]
    
    input_dir = os.path.join(base_dir, "extract_result", args.dataset_name, f"{args.mode}_grad", model_short)
    output_dir = os.path.join(base_dir, "plots", args.dataset_name, f"{args.mode}_grad", model_short)
    
    if not os.path.exists(input_dir):
        print(f"Error: Not found {input_dir}")
        sys.exit(1)
        
    npz_files = [f for f in os.listdir(input_dir) if f.endswith('.npz')]
    for f in tqdm(npz_files, desc=f"Plotting {args.mode} Gradients"):
        plot_text_gradient(os.path.join(input_dir, f), output_dir)