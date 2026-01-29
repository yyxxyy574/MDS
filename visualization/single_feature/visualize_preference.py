import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml
import glob

from config.constants import ROOT
from visualization.utils import MODALITY_LIST, MODALITY_PALETTE, MODEL_LIST, MODEL_TYPE_LIST, parse_model_info

# Display Order on Y-axis
ORDER_MAP = {
    'global': 0,
    'species': 1,
    'gender': 2,
    'age': 3,
    'fitness': 4,
    'profession': 5,
    'wealth': 6,
    'education': 7,
    'color': 8
}

def load_model_data(model_str, modality, analyze_dir):
    fname = f"preferences_{model_str}.yaml"
    fpath = os.path.join(analyze_dir, fname)
    if os.path.exists(fpath):
        with open(fpath, 'r') as f:
            data = yaml.safe_load(f)
        if data:
            data['modality'] = modality
    return data

def plot_single_model(model_type, data_list, output_dir):
    rows = []
    for data in data_list:
        modality = data['modality']
        
        # 1. Global
        gp = data.get('global_preference', {})
        if gp:
            rows.append({
                'Modality': modality,
                'Category': 'Global',
                'Left_Label': gp.get('left_label', 'Inaction').capitalize(),
                'Right_Label': gp.get('right_label', 'Action').capitalize(),
                'Delta': gp.get('delta', 0.0),
                'Order': 0
            })
            
        # 2. Features
        fp = data.get('feature_preference', {})
        for cat, stats in fp.items():
            if cat in ORDER_MAP:
                rows.append({
                    'Modality': modality,
                    'Category': cat.capitalize(),
                    'Left_Label': stats['left_label'],
                    'Right_Label': stats['right_label'],
                    'Delta': stats['delta'],
                    'Order': ORDER_MAP.get(cat, 99)
                })

    if not rows: return
    df = pd.DataFrame(rows).sort_values(by=['Order', 'Category'])

    # Setup Plot
    fig, ax = plt.subplots(figsize=(11, 7)) # Slightly taller for more dimensions

    markers = {'Text': 'o', 'Image': 's', 'Caption': '^'}
    
    # Draw Scatter/Strip Plot
    sns.scatterplot(
        data=df, 
        y='Category', 
        x='Delta', 
        hue='Modality', 
        style='Modality',     
        markers=markers,   
        palette=MODALITY_PALETTE, 
        s=200,                 
        alpha=0.9, 
        edgecolor='white', 
        linewidth=1, 
        ax=ax
    )
    
    # Add Central Line
    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    
    # Labels
    ax.set_title(f"Preference Analysis: {model_type}", fontsize=16, pad=20)
    ax.set_xlabel("Change in probability of being spared", fontsize=12, labelpad=10)
    ax.set_ylabel("")
    
    # Axis Limits
    max_val = max(abs(df['Delta'].min()), abs(df['Delta'].max()), 0.15) * 1.3
    ax.set_xlim(-max_val, max_val)
    
    # Side Labels
    categories = df['Category'].unique()
    for i, cat in enumerate(categories):
        subset = df[df['Category'] == cat]
        if subset.empty: continue
        sample = subset.iloc[0]
        
        left_text = sample['Left_Label']
        right_text = sample['Right_Label']
        
        ax.text(-max_val, i, left_text, ha='left', va='center', fontsize=11, fontweight='bold', color='#555555')
        ax.text(max_val, i, right_text, ha='right', va='center', fontsize=11, fontweight='bold', color='#555555')

    # Legend
    ax.legend(title='Modality', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    out_file = os.path.join(output_dir, f"preference_{model_type}.pdf")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {out_file}")

if __name__ == "__main__":
    analyze_dir = f"{ROOT}/../results/single_feature/analyze_results"
    output_dir = f"{ROOT}/../visualization/single_feature/preference"
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir, exist_ok=True)
    
    data = {}
    for model_str in MODEL_LIST:
        model_name, modality = parse_model_info(model_str)
        if model_name not in data:
            data[model_name] = []
        data[model_name].append(load_model_data(model_str, modality, analyze_dir))
    
    sns.set_theme(style="whitegrid")
    for model_name in MODEL_TYPE_LIST:
        if data[model_name]: 
            plot_single_model(model_name, data[model_name], output_dir)