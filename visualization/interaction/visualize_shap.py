import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from config.constants import ROOT
from visualization.utils import (
    MODALITY_LIST, MODEL_LIST, MODEL_TYPE_LIST, 
    MODALITY_PALETTE, FEATURE_TYPE_COLORS, 
    get_feature_type, parse_model_info, parse_feature_components
)

# ================= CONFIGURATION =================
# Path settings
RESULT_DIR = os.path.join(ROOT, "..", "results", "interaction", "analyze_results")
OUTPUT_DIR = os.path.join(ROOT, "..", "visualization", "interaction", "shap")

# Scheme A Settings
FEATURE_ORDER = ['Action Bias', 'Quantity', 'Character']
FEATURE_COLORS_DICT = {
    'Action Bias': '#CACED1', 
    'Quantity':    '#F9B505',
    'Character':   '#C6E377'
}

# Scheme B Settings
INTERACTION_CATS = [
    'Quant(1vs1) × Char', 
    'Intra-Char',  # Same person (e.g. Race x Gender)
    'Inter-Char'     # Different people (e.g. P1 x P2)
]
INTERACTION_PALETTE = {
    'Quant(1vs1) × Char': '#d62728',      # Red: Conditional Trigger
    'Intra-Char': '#9467bd', # Purple: Visual Stereotype
    'Inter-Char': '#1f77b4'    # Blue: Social Relation/Comparison
}

# ================= DATA LOADING FUNCTIONS =================

def load_general_data(model_str, classifier_name="RandomForest"):
    """Loads flattened SHAP data for Scheme A (Composition)."""
    result_path = os.path.join(RESULT_DIR, f"{model_str.rpartition('_')[0]}/{model_str.rpartition('_')[-1]}")
    data_path = os.path.join(result_path, f"test_data_{model_str}.joblib")
    
    if not os.path.exists(data_path): return None
    try:
        data = joblib.load(data_path)
        base_feature_names = data.get('feature_names', [])
    except: return None

    shap_path = os.path.join(result_path, f"shap_interactions_{model_str}_{classifier_name}.npy")
    if not os.path.exists(shap_path): return None
    
    shap_interaction_values = np.load(shap_path)
    if len(shap_interaction_values.shape) == 4:
        shap_interaction_values = shap_interaction_values[:, 1, :, :]
    
    flat_feature_names = []
    flat_mean_abs_shap = []
    n_features = len(base_feature_names)
    
    # Global Mean for Normalization
    global_mean_matrix = np.abs(shap_interaction_values).mean(axis=0)
    total_importance = np.sum(global_mean_matrix)
    
    # A. Main Effects
    for i in range(n_features):
        feat_name = base_feature_names[i]
        main_vals = shap_interaction_values[:, i, i]
        flat_feature_names.append(feat_name)
        flat_mean_abs_shap.append(np.mean(np.abs(main_vals)))
        
    # B. Interaction Effects
    for i in range(n_features):
        for j in range(i + 1, n_features):
            feat_i = base_feature_names[i]
            feat_j = base_feature_names[j]
            if '_' in feat_i and '_' in feat_j:
                parent_i = feat_i.rpartition('_')[0]
                parent_j = feat_j.rpartition('_')[0]
                if parent_i == parent_j: continue

            inter_name = f"{feat_i} & {feat_j}"
            inter_vals = shap_interaction_values[:, i, j]
            mean_imp = np.mean(np.abs(inter_vals)) * 2
            
            if mean_imp < 1e-6: continue
                
            flat_feature_names.append(inter_name)
            flat_mean_abs_shap.append(mean_imp)

    flat_mean_abs_shap = np.array(flat_mean_abs_shap)
    if total_importance > 0:
        norm_shap = flat_mean_abs_shap / total_importance
    else:
        norm_shap = flat_mean_abs_shap
        
    return {'feature_names': flat_feature_names, 'norm_shap': norm_shap}

def load_interaction_matrix(model_str, classifier_name="RandomForest"):
    """
    Loads Interaction Matrix for Scheme B with SIGNED values.
    Returns:
        norm_matrix: Absolute values normalized (for height)
        signed_matrix: Signed values normalized (for direction check)
    """
    result_path = os.path.join(RESULT_DIR, f"{model_str.rpartition('_')[0]}/{model_str.rpartition('_')[-1]}")
    data_path = os.path.join(result_path, f"test_data_{model_str}.joblib")
    if not os.path.exists(data_path): return None
    try:
        data = joblib.load(data_path)
        feature_names = data.get('feature_names', [])
    except: return None

    shap_path = os.path.join(result_path, f"shap_interactions_{model_str}_{classifier_name}.npy")
    if not os.path.exists(shap_path): return None
    
    # Shape: (Samples, M, M)
    shap_interactions = np.load(shap_path)
    if len(shap_interactions.shape) == 4:
        shap_interactions = shap_interactions[:, 1, :, :]
    
    # 1. Absolute Mean (Magnitude)
    mean_abs_matrix = np.abs(shap_interactions).mean(axis=0)
    total = np.sum(mean_abs_matrix)
    
    # 2. Signed Mean (Direction)
    mean_signed_matrix = shap_interactions.mean(axis=0)
    
    if total > 0:
        norm_matrix = mean_abs_matrix / total
        # Normalize signed matrix by same total to keep scale consistent
        norm_signed_matrix = mean_signed_matrix / total
    else:
        norm_matrix = mean_abs_matrix
        norm_signed_matrix = mean_signed_matrix
        
    return {
        'feature_names': feature_names, 
        'norm_matrix': norm_matrix,
        'signed_matrix': norm_signed_matrix
    }

# ================= PLOTTING SCHEME A =================

def plot_scheme_a_bias_fingerprint(data_map, save_dir):
    """
    Scheme A: Bias Composition Fingerprint (Normalized to 100%)
    Stacked Bar Chart: Rows=Modality, X=Model, Stack=FeatureType
    """
    # 1. Aggregate Data
    records = []
    
    for model_type in MODEL_TYPE_LIST:
        for modality in MODALITY_LIST:
            key = (model_type, modality)
            if key not in data_map: continue
            
            d = data_map[key]
            features = d['feature_names']
            importances = d['norm_shap']
            
            type_sums = {'Quantity': 0, 'Action Bias': 0, 'Character': 0}
            
            for feat, imp in zip(features, importances):
                info = parse_feature_components(feat)
                components = info['components']
                if len(components) == 0: continue
                
                share_imp = imp / len(components)
                for comp in components:
                    ftype = get_feature_type(comp)
                    if ftype == 'quantity':
                        type_sums['Quantity'] += share_imp
                    elif ftype == 'action_bias':
                        type_sums['Action Bias'] += share_imp
                    elif ftype in ['gender', 'color', 'profession']:
                        type_sums['Character'] += share_imp
            total_visible = sum(type_sums.values())
            if total_visible > 0:
                for k in type_sums:
                    type_sums[k] /= total_visible
            # ------------------------------------------------------
                        
            for ftype, total_imp in type_sums.items():
                records.append({
                    'Model': model_type,
                    'Modality': modality.capitalize(),
                    'Feature Type': ftype,
                    'Importance': total_imp
                })

    df = pd.DataFrame(records)
    if df.empty: return

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.05) # Tighter vertical spacing
    
    modalities_ordered = [m.capitalize() for m in MODALITY_LIST]
    
    for i, mode in enumerate(modalities_ordered):
        ax = axes[i]
        df_mode = df[df['Modality'] == mode]
        
        # Pivot for stacking
        df_pivot = df_mode.pivot(index='Model', columns='Feature Type', values='Importance')
        # Reindex to ensure order and missing columns
        df_pivot = df_pivot.reindex(index=MODEL_TYPE_LIST, columns=FEATURE_ORDER).fillna(0)
        
        # Plot Stacked Bar
        df_pivot.plot(kind='bar', stacked=True, ax=ax, width=0.9, 
                      color=[FEATURE_COLORS_DICT[c] for c in FEATURE_ORDER], 
                      edgecolor='black', linewidth=0.5)
        
        ax.set_ylabel('Composition', fontsize=18)
        ax.tick_params(axis='y', labelsize=16)
        # Add Percentage Formatting to Y axis
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        
        # Modality Label inside the plot area or to the right for compactness? 
        # Title is cleaner.
        ax.set_title(f"{mode} Mode", loc='left', fontsize=20, fontweight='bold', pad=3)
        ax.legend().remove()
        
        # Add labels inside bars (only if > 5%)
        for c in ax.containers:
            # Create label: e.g. "20%"
            labels = [f'{v:.0%}' if v > 0.05 else '' for v in c.datavalues]
            ax.bar_label(c, labels=labels, label_type='center', fontsize=16, color='white', weight='bold')

    # X-axis formatting
    axes[-1].set_xticklabels(axes[-1].get_xticklabels(), rotation=10, fontsize=16, fontweight='bold')
    axes[-1].set_xlabel("")

    # Global Legend
    handles = [Line2D([0], [0], color=FEATURE_COLORS_DICT[f], lw=10, label=f) for f in FEATURE_ORDER]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=5, frameon=False, fontsize=18)
    
    # plt.suptitle("Bias Composition Fingerprint (Normalized)", fontsize=16, y=1.05)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "Summary_SchemeA_Composition.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    save_path = os.path.join(save_dir, "Summary_SchemeA_Composition.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved Scheme A (Normalized) to: {save_path}")
    plt.close()

# ================= PLOTTING SCHEME B =================

def get_person_id(feature_name):
    """
    Extracts person ID from feature name (e.g., 'person1_gender' -> 'person1').
    Assumes format: 'personX_featureName=Value'
    """
    # Split by first underscore to get potential 'personX'
    parts = feature_name.split('_')
    if len(parts) > 0 and 'person' in parts[0]:
        return parts[0]
    return 'unknown'

def classify_interaction_effect(f1, f2, abs_int, s_int, char_abs, char_sgn):
    t1 = get_feature_type(f1)
    t2 = get_feature_type(f2)
    char_types = {'gender', 'color', 'profession'}
    
    category = None
    effect_type = None
    
    is_char_char = False
    
    # 1. Determine Category
    if t1 in char_types and t2 in char_types:
        # Exclude sibling features (same person, same attribute group - e.g. color=Red & color=Blue)
        # Sibling check is usually done in main loop, but here we check person ID identity
        if f1.split('=')[0] == f2.split('=')[0]: return None, None
        
        is_char_char = True
        
        # --- NEW: Check if same person or different person ---
        p1 = get_person_id(f1)
        p2 = get_person_id(f2)
        
        if p1 != 'unknown' and p2 != 'unknown':
            if p1 == p2:
                category = 'Intra-Char'
            else:
                category = 'Inter-Char'
        else:
            # Fallback if parsing fails
            category = 'Inter-Char'
        
    elif (t1 == 'quantity' and t2 in char_types) or (t2 == 'quantity' and t1 in char_types):
        if '1vs1' in f1 or '1vs1' in f2:
            category = 'Quant(1vs1) × Char'
        else:
            return None, None
    
    if not category: return None, None
    
    # 2. Determine Effect Type (Direction)
    if is_char_char:
        # All Char x Char are assumed to be Amplification of complexity/bias
        effect_type = 'Amplification'
    else:
        # Quant x Char logic
        if char_abs < (0.2 * abs_int):
            effect_type = 'Amplification'
        elif (s_int * char_sgn) >= 0:
            effect_type = 'Amplification'
        else:
            # effect_type = 'Correction'
            effect_type = None
            
    return category, effect_type

def plot_scheme_b_bidirectional(df, save_dir):
    """Plot 1: Bidirectional Bar Chart (Absolute Intensity)"""
    categories = INTERACTION_CATS
    models = MODEL_TYPE_LIST
    modalities = MODALITY_LIST
    bar_width = 0.25
    
    # Adjusted figsize for 3 rows
    fig, axes = plt.subplots(len(categories), 1, figsize=(10, 2 * len(categories)), sharex=True)
    if len(categories) == 1: axes = [axes]
    
    for row_idx, cat in enumerate(categories):
        ax = axes[row_idx]
        subset = df[df['Interaction Type'] == cat]
        x = np.arange(len(models))
        
        for i, mode in enumerate(modalities):
            offset = (i - 1) * bar_width 
            vals_amp = []
            vals_corr = []
            
            for model in models:
                m_data = subset[(subset['Model'] == model) & (subset['Modality'] == mode)]
                amp = m_data[m_data['Effect Type'] == 'Amplification']['Intensity'].sum()
                corr = m_data[m_data['Effect Type'] == 'Correction']['Intensity'].sum()
                vals_amp.append(amp)
                vals_corr.append(-corr) 
            
            ax.bar(x + offset, vals_amp, width=bar_width, label=mode if row_idx==0 else "",
                   color=MODALITY_PALETTE[mode], edgecolor='black', linewidth=0.5, alpha=0.9)
            ax.bar(x + offset, vals_corr, width=bar_width, 
                   color=MODALITY_PALETTE[mode], edgecolor='black', linewidth=0.5, alpha=0.5, hatch='///')

        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_ylabel("Intensity", fontsize=18)
        ax.tick_params(axis='y', labelsize=16)
        ax.set_title(cat, loc='left', fontsize=18, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=1.0)

    axes[-1].set_xticks(range(len(models)))
    axes[-1].set_xticklabels(models, fontsize=16, rotation=10, fontweight='bold')
    
    handles = [Line2D([0], [0], color=MODALITY_PALETTE[m], lw=10, label=m) for m in modalities]
    # handles.append(Patch(facecolor='white', edgecolor='black', hatch='///', label='Correction'))
    
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=4, frameon=False, fontsize=16, labelspacing=0.3, columnspacing=0.8, handlelength=1.2)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "Summary_SchemeB_Interaction_Bidirectional.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    save_path = os.path.join(save_dir, "Summary_SchemeB_Interaction_Bidirectional.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved Scheme B (Bidirectional) to: {save_path}")
    plt.close()

def process_and_plot_scheme_b(data_map, save_dir):
    records = []
    
    for model_type in MODEL_TYPE_LIST:
        for modality in MODALITY_LIST:
            key = (model_type, modality)
            if key not in data_map: continue
            
            d = data_map[key]
            features = d['feature_names']
            abs_mat = d['norm_matrix'] 
            sgn_mat = d['signed_matrix']
            
            for i in range(len(features)):
                abs_m1 = abs_mat[i, i]
                s_m1 = sgn_mat[i, i]
                
                for j in range(i+1, len(features)):
                    abs_m2 = abs_mat[j, j]
                    s_m2 = sgn_mat[j, j]
                    
                    f1, f2 = features[i], features[j]
                    
                    if '=' in f1 and '=' in f2:
                        p1, p2 = f1.split('=')[0], f2.split('=')[0]
                        if p1 == p2: continue
                    
                    abs_int = abs_mat[i, j] * 2
                    s_int = sgn_mat[i, j] * 2
                    
                    if abs_int < 1e-5: continue 
                    
                    char_abs, char_sgn = 0, 0
                    t1 = get_feature_type(f1)
                    t2 = get_feature_type(f2)
                    
                    if t1 == 'quantity' and t2 in ['gender', 'color', 'profession']:
                        char_abs, char_sgn = abs_m2, s_m2
                    elif t2 == 'quantity' and t1 in ['gender', 'color', 'profession']:
                        char_abs, char_sgn = abs_m1, s_m1
                    
                    cat, eff_type = classify_interaction_effect(
                        f1, f2, abs_int, s_int, char_abs, char_sgn
                    )
                    
                    if cat and eff_type:
                        records.append({
                            'Model': model_type,
                            'Modality': modality.capitalize(),
                            'Interaction Type': cat,
                            'Effect Type': eff_type,
                            'Intensity': abs_int
                        })
                
    df = pd.DataFrame(records)
    if df.empty: return
    
    plot_scheme_b_bidirectional(df, save_dir)

# ================= MAIN =================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Starting SHAP Summary Visualization...")
    
    # Load Data
    data_general = {}
    data_interaction = {}
    
    print("Loading data...")
    for model_str in MODEL_LIST:
        model_name, modality = parse_model_info(model_str)
        
        # Load General (Scheme A)
        res_gen = load_general_data(model_str)
        if res_gen:
            data_general[(model_name, modality)] = res_gen
            
        # Load Interaction (Scheme B)
        res_int = load_interaction_matrix(model_str)
        if res_int:
            data_interaction[(model_name, modality)] = res_int
            
    print(f"Loaded {len(data_general)} general records and {len(data_interaction)} interaction records.")

    # Plot Scheme A
    if data_general:
        plot_scheme_a_bias_fingerprint(data_general, OUTPUT_DIR)
        
    # Plot Scheme B
    if data_interaction:
        process_and_plot_scheme_b(data_interaction, OUTPUT_DIR)
        
    print("Done.")

if __name__ == "__main__":
    main()