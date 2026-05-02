"""
Text Gradient Plotter
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from tqdm import tqdm
import re
import unicodedata

# ============================================================================
# Token Filter
# ============================================================================

NEUTRAL_WORDS = {
    'i', 'me', 'my', 'mine', 'myself', 'you', 'your', 'yours', 'yourself',
    'he', 'she', 'it', 'they', 'them', 'their', 'theirs', 'we', 'us', 'our', 'ours', 'ourselves',
    'this', 'that', 'these', 'those', 'who', 'whom', 'whose', 'which', 'what',
    'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under', 'over', 'underneath', 'beside', 'behind',
    'and', 'or', 'but', 'nor', 'so', 'yet', 'both', 'either', 'neither',
    'if', 'unless', 'until', 'while', 'when', 'where', 'because', 'since', 'although', 'though', 'whereas', 'whether',
    'be', 'is', 'am', 'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'doing', 'done', 'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must', 'need', 'needs',
    'there', 'here', 'where', 'now', 'then', 'how', 'all', 'some', 'any', 'no', 'not', 'none', 'every', 'each',
    'more', 'most', 'less', 'least', 'much', 'many', 'few', 'several', 'other', 'another', 'such', 'same', 'different',
    'very', 'really', 'quite', 'rather', 'too', 'also', 'even', 'just', 'only', 'still', 'already', 'always', 'never', 'often', 'sometimes',
    'about', 'like', 'than', 'scene', 'image', 'picture', 'photo', 'depicts', 'showing',
    'text', 'caption', 'description', 'figure', 'visual', 'background',
    'answer', 'question', 'ask', 'asked', 'asking', 'whether', 'yes', 'no', 'maybe', 'perhaps', 'explain', 'reason',
    'area', 'space', 'place', 'part', 'side', 'region', 'near', 'next', 'adjacent', 'close', 'far',
    'thing', 'stuff', 'matter', 'object', 'item', 'situation', 'condition', 'state', 'status',
    'way', 'means', 'method', 'manner', 'time', 'moment', 'period', 'point', 'case',
    'choice', 'choose', 'decision', 'decide', 'act', 'action', 'acting',
    'result', 'consequence', 'outcome', 'effect', 'risk', 'chance', 'probability',
    'life', 'death', 'die', 'live', 'save', 'kill', 'harm', 'help', 'hurt',
    'one', 'two', 'three', 'four', 'five', 'first', 'second', 'third',
    'let', 'say', 'said', 'get', 'got', 'see', 'saw', 'look', 'looking',
    'know', 'knew', 'think', 'thought', 'want', 'wanted',
}

CHAT_TEMPLATE_WORDS = {
    'assistant', 'user', 'system', 'human', 'bot', 'teacher', 'student', 'dilemma',
}

_BPE_SPACE_CHARS = (
    "\u0120",
    "\u010a",
    "\u2581",
)


def _strip_bpe_markers(s: str) -> str:
    for ch in _BPE_SPACE_CHARS:
        s = s.replace(ch, " " if ch == "\u010a" else "")
    return s

def _normalize_for_neutral_check(token: str) -> str:
    cleaned = token.lower().strip()
    for prefix in ('Ġ', 'Ċ', '##', '▁'):
        cleaned = cleaned.replace(prefix, '')
    cleaned = re.sub(r'^[^\w]+|[^\w]+$', '', cleaned)
    return cleaned

def is_neutral_token(token: str) -> bool:
    cleaned = _normalize_for_neutral_check(token)
    if not cleaned:
        return True
    return cleaned in NEUTRAL_WORDS or cleaned in CHAT_TEMPLATE_WORDS

def is_displayable_token(s: str) -> bool:
    if not s or not s.strip():
        return False
    s = s.strip()

    if "\ufffd" in s:
        return False
    _MOJIBAKE_SUBSTRINGS = (
        "Ã", "Â", "â€™", "â€œ", "â€", "Å", "ðŁ", "â", "Ê", "â€˜", "â€™", "â€œ", "â€\x9d",
    )
    if any(bad in s for bad in _MOJIBAKE_SUBSTRINGS):
        return False

    for c in s:
        o = ord(c)
        if o < 0x20 or o > 0x7E:
            return False
        if not c.isprintable():
            return False

    if not any(c.isalnum() for c in s):
        return False
    if len(s) == 1 and not s.isalnum():
        return False
    return True

def clean_token(token: str) -> str | None:
    """Clean individual token"""
    if token in ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<s>", "</s>"]:
        return None

    cleaned = str(token).replace("Ċ", " ").replace("Ġ", "")
    cleaned = _strip_bpe_markers(cleaned)
    cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", cleaned)
    cleaned = unicodedata.normalize("NFKC", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"^[\s.,!?;:'\"()\[\]{}]+|[\s.,!?;:'\"()\[\]{}]+$", "", cleaned).strip()

    if not cleaned:
        return None
    if cleaned in ",.!?;:'\"()-[]{}":
        return None
    if not is_displayable_token(cleaned):
        return None

    return cleaned

def clean_tokens_and_weights(tokens: list, weights: np.ndarray, filter_neutral: bool = False, deduplicate: bool = False):
    cleaned_tokens = []
    cleaned_weights = []

    for t, w in zip(tokens, weights):
        ct = clean_token(t)
        if ct is not None:
            if filter_neutral and is_neutral_token(ct):
                continue
            cleaned_tokens.append(ct)
            cleaned_weights.append(float(w))

    if deduplicate:
        accum = {}
        for tok, w in zip(cleaned_tokens, cleaned_weights):
            accum[tok] = accum.get(tok, 0.0) + w
        cleaned_tokens = list(accum.keys())
        cleaned_weights = list(accum.values())

    return cleaned_tokens, np.array(cleaned_weights)

def generate_html_highlight(token_texts, weights, target_token, save_path):
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
        alpha = float(weight) * 0.85
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