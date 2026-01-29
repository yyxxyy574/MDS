import pandas as pd
import re
import os

from config.constants import DILEMMA

DILEMMA_ORDER = [d for dimension in DILEMMA for d in DILEMMA[dimension]]

# Mapping for dimension analysis
DILEMMA_DIMENSION_MAP = {
    'Care vs Care': ['trolley', 'footbridge', 'vaccine_policy', 'environmental_policy', 'lifeboat', 'prevent_spread', 'crying_baby', 'shark_attack', 'transplant', 'terrorist'],
    'Care vs Fairness': ['bonus_allocation'],
    'Care vs Loyalty': ['self-harming'],
    'Care vs Authority': ['guarded_speedboat', 'save_dying'],
    'Care vs Purity': ['party'],
    'Fairness vs Loyalty': ['resume', 'report_cheating'],
    'Fairness vs Authority': ['hiring'],
    'Fairness vs Purity': ['inpurity'],
    'Loyalty vs Authority': ['feed', 'report_stealing'],
    'Loyalty vs Purity': ['ceremony'],
    'Authority vs Purity': ['dirty']
}

DILEMMA_DIMENSION_ORDER = list(DILEMMA_DIMENSION_MAP.keys())

def get_model_name_pretty(model_str):
    """Prettify model names for plots."""
    if 'deepseek' in model_str: return 'DeepSeek-VL-7B'
    if 'llava-v1_5' in model_str: return 'LLaVA-v1.5-13B'
    if 'llava-v1_6' in model_str: return 'LLaVA-v1.6-34B'
    if 'Qwen3-VL-8B' in model_str: return 'Qwen3-VL-8B'
    if 'Qwen3-VL-32B' in model_str: return 'Qwen3-VL-32B'
    if 'gemini-2.5-flash' in model_str: return 'Gemini-2.5-flash'
    if 'llama-3.2-90b' in model_str: return 'LLaMA-3.2-90B'
    if 'gpt-4o-mini' in model_str: return 'GPT-4o-mini'
    return model_str

def parse_model_info(model_str):
    name = get_model_name_pretty(model_str)
    modality = model_str.split('_')[-1].capitalize()
    return name, modality

MODALITY_LIST = ['Text', 'Caption', 'Image']
# MODALITY_PALETTE = {'Text': '#4c72b0', 'Caption': '#e6b800', 'Image': '#c44e52'}
MODALITY_PALETTE = {'Text': '#70A6CF', 'Caption': '#FACD6E', 'Image': '#E78489'}
MODEL_LIST = [
    # "deepseek-vl-7b-chat_text",
    # "deepseek-vl-7b-chat_caption",
    # "deepseek-vl-7b-chat_image",

    # "llava-v1_5-13b_text",
    # "llava-v1_5-13b_caption",
    # "llava-v1_5-13b_image",

    "llava-v1_6-34b_text",
    "llava-v1_6-34b_caption",
    "llava-v1_6-34b_image",

    "Qwen3-VL-8B-Instruct_text",
    "Qwen3-VL-8B-Instruct_caption",
    "Qwen3-VL-8B-Instruct_image",

    "Qwen3-VL-32B-Instruct_text",
    "Qwen3-VL-32B-Instruct_caption",
    "Qwen3-VL-32B-Instruct_image",

    'llama-3.2-90b-vision-instruct_text',
    'llama-3.2-90b-vision-instruct_caption',
    'llama-3.2-90b-vision-instruct_image',

    'gpt-4o-mini-2024-07-18_text',
    'gpt-4o-mini-2024-07-18_caption',
    'gpt-4o-mini-2024-07-18_image',

    "gemini-2.5-flash-preview-09-2025_text",
    "gemini-2.5-flash-preview-09-2025_caption",
    "gemini-2.5-flash-preview-09-2025_image",
]
# MODEL_TYPE_LIST = ['DeepSeek-VL-7B', 'LLaVA-v1.5-13B', 'LLaVA-v1.6-34B', 'Qwen3-VL-8B', 'Qwen3-VL-32B', 'Gemini-2.5-flash']
# MODEL_TYPE_LIST = ['LLaVA-v1.6-34B', 'Qwen3-VL-8B', 'Qwen3-VL-32B', 'Gemini-2.5-flash']
MODEL_TYPE_LIST = ['LLaVA-v1.6-34B', 'Qwen3-VL-8B', 'Qwen3-VL-32B', 'LLaMA-3.2-90B', 'GPT-4o-mini', 'Gemini-2.5-flash']
MODEL_NAME_LIST = [f"{model_name} - {modality}" for model_name, modality in (parse_model_info(model_str) for model_str in MODEL_LIST)]

COLOR_PALETTE = {
    'DeepSeek-VL-7B - Text': '#457b9d',      'DeepSeek-VL-7B - Image': '#a8dadc',    'DeepSeek-VL-7B - Caption': '#e9c46a',
    'LLaVA-v1.5-13B - Text': '#558b2f',      'LLaVA-v1.5-13B - Image': '#c5e1a5',    'LLaVA-v1.5-13B - Caption': '#fff176',
    'LLaVA-v1.6-34B - Text': '#33691e',      'LLaVA-v1.6-34B - Image': '#aed581',    'LLaVA-v1.6-34B - Caption': '#dce775',
    'Qwen3-VL-8B - Text': '#c62828',         'Qwen3-VL-8B - Image': '#ef9a9a',       'Qwen3-VL-8B - Caption': '#ffcc80',
    'Qwen3-VL-32B - Text': '#ad1457',        'Qwen3-VL-32B - Image': '#f48fb1',      'Qwen3-VL-32B - Caption': '#ffab91',
    'Gemini-2.5-flash - Text': '#6200ea',    'Gemini-2.5-flash - Image': '#b388ff',  'Gemini-2.5-flash - Caption': '#ea80fc',
}

def get_stars(p):
    """Convert p-value to significance stars."""
    if pd.isna(p): return ""
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""

def get_mft(dilemmama):
    for key in DILEMMA_DIMENSION_MAP:
        for t in DILEMMA_DIMENSION_MAP[key]:
            if dilemmama.startswith(t): return key
    return 'Other'

def preprocess_data(df):
    if df.empty: return df
    # Filter significant & non-intercept
    sig_df = df[(df['P-value'] < 0.05) & (df['Effect_Type'] != 'Intercept')].copy()
    if sig_df.empty: return sig_df
    
    return sig_df

def get_feature_type(feature_name):
    """Classifies feature name into types for coloring."""
    f_lower = feature_name.lower()
    if 'quantity' in f_lower:
        # Distinguish 1vs1 (baseline bias) from 1vsX (utilitarian reasoning)
        if '1vs1' in f_lower and '1vs10' not in f_lower:
            return 'action_bias'
        return 'quantity'
    elif 'gender' in f_lower:
        return 'gender'
    elif 'color' in f_lower: 
        return 'color'
    elif 'profession' in f_lower: 
        return 'profession'
    else: 
        return None
    
FEATURE_TYPE_COLORS = {
    'action_bias': '#000000',  # Black - Baseline preference (1vs1)
    'quantity': '#808080',     # Gray - Rational utilitarian (1vs5, etc.)
    'gender': '#E74C3C',       # Red
    'color': '#F39C12',        # Turquoise/Orange
    'profession': '#45B7D1',   # Blue
}

FEATURE_TYPE_LABEL_COLORS = {
    'action_bias': "#000000",
    'quantity': "#504F4F",
    'gender': "#AE291A",
    'color': "#9F6B18",
    'profession': "#0D5A8E",
}

def parse_feature_components(feature_name):
    """
    Parses a feature string (potentially an interaction) into its components.
    
    Returns:
        dict: {
            'type': 'Main' | '2-way' | '3-way',
            'components': [list of sub-features],
            'base_features': [list of base feature names e.g. person1_gender]
        }
    """
    # Check for interactions (delimited by '&' in the new analyze_shap logic)
    # The feature name might look like "person1_gender=Male&person1_profession=Doctor"
    
    if '&' not in feature_name:
        return {
            'type': 'Main',
            'components': [feature_name],
            'order': 1
        }
    
    componenets = feature_name.split(' & ')
    order = len(componenets)
    
    label = '2-way' if order == 2 else '3-way' if order == 3 else f'{order}-way'
    
    return {
        'type': label,
        'components': componenets,
        'order': order
    }
