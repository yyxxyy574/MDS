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
RESULT_DIR = os.path.join(ROOT, "..", "results", "interaction", "analyze_results_implicit")
OUTPUT_DIR = os.path.join(ROOT, "..", "visualization", "interaction", "shap_group")

# Scheme A Settings
FEATURE_ORDER = ['Action Bias', 'Quantity', 'Character']
FEATURE_COLORS_DICT = {
    'Action Bias': '#CACED1', 
    'Quantity':    '#F9B505',
    'Character':   '#C6E377'
}

# Scheme B Settings
INTERACTION_CATS = [
    'Quant 1vs1 × Char', 
    'Intra-Char',  # Same person (e.g. Race x Gender)
    'Inter-Char'     # Different people (e.g. P1 x P2)
]
INTERACTION_PALETTE = {
    'Quant 1vs1 × Char': '#d62728',      # Red: Conditional Trigger
    'Intra-Char': '#9467bd', # Purple: Visual Stereotype
    'Inter-Char': '#1f77b4'    # Blue: Social Relation/Comparison
}

# ================= DATA LOADING FUNCTIONS =================

def load_general_data(model_str, classifier_name="RandomForest"):
    """Loads flattened SHAP data and Error Margins for Scheme A (Composition)."""
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

    csv_path = os.path.join(result_path, f"interaction_analysis_{classifier_name}.csv")
    error_dict = {}
    if os.path.exists(csv_path):
        df_csv = pd.read_csv(csv_path)
        if 'Error_Margin' in df_csv.columns:
            error_dict = dict(zip(df_csv['Feature'], df_csv['Error_Margin']))
    
    flat_feature_names = []
    flat_mean_abs_shap = []
    flat_error_margins = []
    n_features = len(base_feature_names)
    
    global_mean_matrix = np.abs(shap_interaction_values).mean(axis=0)
    total_importance = np.sum(global_mean_matrix)
    
    # A. Main Effects
    for i in range(n_features):
        feat_name = base_feature_names[i]
        main_vals = shap_interaction_values[:, i, i]
        flat_feature_names.append(feat_name)
        flat_mean_abs_shap.append(np.mean(np.abs(main_vals)))
        flat_error_margins.append(error_dict.get(feat_name, 0.0))
        
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
            flat_error_margins.append(error_dict.get(inter_name, 0.0))

    flat_mean_abs_shap = np.array(flat_mean_abs_shap)
    flat_error_margins = np.array(flat_error_margins)
    
    if total_importance > 0:
        norm_shap = flat_mean_abs_shap / total_importance
        norm_error = flat_error_margins / total_importance 
    else:
        norm_shap = flat_mean_abs_shap
        norm_error = flat_error_margins
        
    return {'feature_names': flat_feature_names, 'norm_shap': norm_shap, 'norm_error': norm_error}

def load_interaction_matrix(model_str, classifier_name="RandomForest"):
    """
    Loads Interaction Matrix for Scheme B with SIGNED values and Errors.
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
    
    shap_interactions = np.load(shap_path)
    if len(shap_interactions.shape) == 4:
        shap_interactions = shap_interactions[:, 1, :, :]
        
    csv_path = os.path.join(result_path, f"interaction_analysis_{classifier_name}.csv")
    error_dict = {}
    if os.path.exists(csv_path):
        df_csv = pd.read_csv(csv_path)
        if 'Error_Margin' in df_csv.columns:
            error_dict = dict(zip(df_csv['Feature'], df_csv['Error_Margin']))
    
    mean_abs_matrix = np.abs(shap_interactions).mean(axis=0)
    total = np.sum(mean_abs_matrix)
    mean_signed_matrix = shap_interactions.mean(axis=0)
    
    err_matrix = np.zeros_like(mean_abs_matrix)
    for i in range(len(feature_names)):
        err_matrix[i, i] = error_dict.get(feature_names[i], 0)
        for j in range(i+1, len(feature_names)):
            inter_name = f"{feature_names[i]} & {feature_names[j]}"
            err_matrix[i, j] = error_dict.get(inter_name, 0) / 2.0 
            err_matrix[j, i] = err_matrix[i, j]
    
    if total > 0:
        norm_matrix = mean_abs_matrix / total
        norm_signed_matrix = mean_signed_matrix / total
        norm_err_matrix = err_matrix / total
    else:
        norm_matrix = mean_abs_matrix
        norm_signed_matrix = mean_signed_matrix
        norm_err_matrix = err_matrix
        
    return {
        'feature_names': feature_names, 
        'norm_matrix': norm_matrix,
        'signed_matrix': norm_signed_matrix,
        'err_matrix': norm_err_matrix
    }

# ================= MD EXPORT FUNCTIONS =================

def export_scheme_a_markdown(df, save_dir):
    """Exports Scheme A Error Bars to a Markdown Table."""
    md_lines = [
        "### Statistical Variance of Effect Composition (Scheme A)",
        "Values denote `Mean Contribution (%) ± 95% CI (%)`.",
        "",
        "| Model | Modality | Action Bias | Quantity | Character |",
        "|---|---|---|---|---|"
    ]
    
    for model in MODEL_TYPE_LIST:
        for mod in [m.capitalize() for m in MODALITY_LIST]:
            sub = df[(df['Model'] == model) & (df['Modality'] == mod)]
            if sub.empty: continue
            
            row_dict = {}
            for _, r in sub.iterrows():
                ftype = r['Feature Type']
                imp = r['Importance'] * 100
                err = r['Error'] * 100
                row_dict[ftype] = f"{imp:.2f}% ± {err:.2f}%"
                
            md_lines.append(f"| {model} | {mod} | {row_dict.get('Action Bias', '-')} | {row_dict.get('Quantity', '-')} | {row_dict.get('Character', '-')} |")
            
    with open(os.path.join(save_dir, "Summary_SchemeA_Error_Stats.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

def export_scheme_b_markdown(df, save_dir):
    """Exports Scheme B Error Bars to a Markdown Table."""
    md_lines = [
        "### Statistical Variance of Interaction Intensities (Scheme B)",
        "Values denote `Absolute Intensity ± 95% CI`.",
        "",
        "| Category | Model | Modality | Amplification (Mean ± CI) |",
        "|---|---|---|---|"
    ]
    
    for cat in INTERACTION_CATS:
        sub_cat = df[df['Interaction Type'] == cat]
        for model in MODEL_TYPE_LIST:
            for mod in [m.capitalize() for m in MODALITY_LIST]:
                sub = sub_cat[(sub_cat['Model'] == model) & (sub_cat['Modality'] == mod)]
                if sub.empty: continue
                
                amp_imp = sub[sub['Effect Type'] == 'Amplification']['Intensity'].sum()
                amp_err = np.sqrt((sub[sub['Effect Type'] == 'Amplification']['Error']**2).sum())
                
                # corr_imp = sub[sub['Effect Type'] == 'Correction']['Intensity'].sum()
                # corr_err = np.sqrt((sub[sub['Effect Type'] == 'Correction']['Error']**2).sum())
                
                amp_str = f"{amp_imp:.4f} ± {amp_err:.4f}" if amp_imp > 0 else "-"
                # corr_str = f"{corr_imp:.4f} ± {corr_err:.4f}" if corr_imp > 0 else "-"
                
                md_lines.append(f"| {cat} | {model} | {mod} | {amp_str} |")
                
    with open(os.path.join(save_dir, "Summary_SchemeB_Error_Stats.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))


# ================= PLOTTING SCHEME A =================

def plot_scheme_a_bias_fingerprint(data_map, save_dir):
    records = []
    
    for model_type in MODEL_TYPE_LIST:
        for modality in MODALITY_LIST:
            key = (model_type, modality)
            if key not in data_map: continue
            
            d = data_map[key]
            features = d['feature_names']
            importances = d['norm_shap']
            errors = d['norm_error'] 
            
            type_sums = {'Quantity': 0, 'Action Bias': 0, 'Character': 0}
            type_errors = {'Quantity': 0, 'Action Bias': 0, 'Character': 0}
            
            for feat, imp, err in zip(features, importances, errors):
                info = parse_feature_components(feat)
                components = info['components']
                if len(components) == 0: continue
                
                share_imp = imp / len(components)
                share_err = err / len(components)
                
                for comp in components:
                    ftype = get_feature_type(comp)
                    if ftype == 'quantity':
                        type_sums['Quantity'] += share_imp
                        type_errors['Quantity'] += share_err**2 
                    elif ftype == 'action_bias':
                        type_sums['Action Bias'] += share_imp
                        type_errors['Action Bias'] += share_err**2
                    elif ftype in ['gender', 'color', 'profession']:
                        type_sums['Character'] += share_imp
                        type_errors['Character'] += share_err**2
                        
            total_visible = sum(type_sums.values())
            if total_visible > 0:
                for k in type_sums:
                    type_sums[k] /= total_visible
                    type_errors[k] = np.sqrt(type_errors[k]) / total_visible 
                        
            for ftype, total_imp in type_sums.items():
                records.append({
                    'Model': model_type,
                    'Modality': modality.capitalize(),
                    'Feature Type': ftype,
                    'Importance': total_imp,
                    'Error': type_errors[ftype]
                })

    df = pd.DataFrame(records)
    if df.empty: return
    
    export_scheme_a_markdown(df, save_dir)

    modalities_ordered = [m.capitalize() for m in MODALITY_LIST]
    num_models = len(MODEL_TYPE_LIST)
    
    fig, axes = plt.subplots(1, num_models, figsize=(3 * num_models, 3), sharey=True)
    if num_models == 1: axes = [axes]

    for i, model_name in enumerate(MODEL_TYPE_LIST):
        ax = axes[i]
        df_model = df[df['Model'] == model_name]
        
        df_pivot = df_model.pivot(index='Modality', columns='Feature Type', values='Importance')
        df_pivot = df_pivot.reindex(index=modalities_ordered, columns=FEATURE_ORDER).fillna(0)
        
        df_error = df_model.pivot(index='Modality', columns='Feature Type', values='Error')
        df_error = df_error.reindex(index=modalities_ordered, columns=FEATURE_ORDER).fillna(0)
        
        df_pivot.plot(
            kind='bar',
            stacked=True,
            ax=ax,
            width=0.85,
            color=[FEATURE_COLORS_DICT.get(c, '#CCCCCC') for c in FEATURE_ORDER],
            edgecolor='black',
            linewidth=0.5,
            yerr=df_error, 
            capsize=0,
            error_kw={'elinewidth': 15, 'ecolor': 'black', 'alpha': 0.8}
        )
        
        ax.set_title(f"{model_name}", fontsize=20, fontweight='bold', pad=6)
        
        if i == 0:
            ax.set_ylabel('Composition', fontsize=20)
            ax.tick_params(axis='y', rotation=30, labelsize=18)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis='y', left=False, labelleft=False)
        
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, fontsize=18)
        ax.set_xlabel("")
        ax.legend().remove()
        
        for c in ax.containers:
            if hasattr(c, 'datavalues'):
                labels = [f'{v:.0%}' if v > 0.05 else '' for v in c.datavalues]
                ax.bar_label(c, labels=labels, label_type='center',
                            fontsize=16, color='white', weight='bold')

    handles = [Line2D([0], [0], color=FEATURE_COLORS_DICT[f], lw=10, label=f) for f in FEATURE_ORDER]
    fig.legend(
        handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.12),
        ncol=len(FEATURE_ORDER), frameon=False, fontsize=20
    )

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.0)

    save_path_pdf = os.path.join(save_dir, "Summary_SchemeA_Composition.pdf")
    plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight')
    save_path_png = os.path.join(save_dir, "Summary_SchemeA_Composition.png")
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    print(f"Saved Scheme A to: {save_path_png}")
    plt.close()

# ================= PLOTTING SCHEME B =================

def get_person_id(feature_name):
    parts = feature_name.split('_')
    if len(parts) > 0 and 'person' in parts[0]:
        return parts[0]
    return 'unknown'

def classify_interaction_effect(f1, f2, abs_int, s_int, char_abs, char_sgn):
    t1 = get_feature_type(f1)
    t2 = get_feature_type(f2)
    char_types = {'gender', 'color', 'profession'}
    
    category, effect_type = None, None
    is_char_char = False
    
    if t1 in char_types and t2 in char_types:
        if f1.split('=')[0] == f2.split('=')[0]: return None, None
        is_char_char = True
        p1 = get_person_id(f1)
        p2 = get_person_id(f2)
        if p1 != 'unknown' and p2 != 'unknown':
            category = 'Intra-Char' if p1 == p2 else 'Inter-Char'
        else:
            category = 'Inter-Char'
    elif (t1 == 'quantity' and t2 in char_types) or (t2 == 'quantity' and t1 in char_types):
        if '1vs1' in f1 or '1vs1' in f2:
            category = 'Quant 1vs1 × Char'
        else:
            return None, None
    
    if not category: return None, None
    
    if is_char_char:
        effect_type = 'Amplification'
    else:
        if char_abs < (0.2 * abs_int):
            effect_type = 'Amplification'
        elif (s_int * char_sgn) >= 0:
            effect_type = 'Amplification'
        else:
            effect_type = None 
            
    return category, effect_type

def plot_scheme_b_bidirectional(df, save_dir):
    export_scheme_b_markdown(df, save_dir)
    
    categories = INTERACTION_CATS
    models = MODEL_TYPE_LIST
    modalities = MODALITY_LIST
    bar_width = 0.25
    
    fig, axes = plt.subplots(1, len(categories), figsize=(18, 3)) 
    if len(categories) == 1: axes = [axes]
    
    for row_idx, cat in enumerate(categories):
        ax = axes[row_idx]
        subset = df[df['Interaction Type'] == cat]
        x = np.arange(len(models))
        
        for i, mode in enumerate(modalities):
            offset = (i - 1) * bar_width 
            vals_amp, vals_corr = [], []
            err_amp, err_corr = [], []
            
            for model in models:
                m_data = subset[(subset['Model'] == model) & (subset['Modality'] == mode)]
                
                amp = m_data[m_data['Effect Type'] == 'Amplification']['Intensity'].sum()
                corr = m_data[m_data['Effect Type'] == 'Correction']['Intensity'].sum()
                
                a_err = np.sqrt((m_data[m_data['Effect Type'] == 'Amplification']['Error']**2).sum())
                c_err = np.sqrt((m_data[m_data['Effect Type'] == 'Correction']['Error']**2).sum())
                
                vals_amp.append(amp)
                vals_corr.append(-corr) 
                err_amp.append(a_err)
                err_corr.append(c_err)
            
            ax.bar(x + offset, vals_amp, width=bar_width, label=mode if row_idx==0 else "",
                   color=MODALITY_PALETTE[mode], edgecolor='black', linewidth=0.5, alpha=0.9,
                   yerr=err_amp, capsize=0, error_kw={'elinewidth': 5, 'ecolor': 'black'})
                   
            ax.bar(x + offset, vals_corr, width=bar_width, 
                   color=MODALITY_PALETTE[mode], edgecolor='black', linewidth=0.5, alpha=0.5, hatch='///',
                   yerr=err_corr, capsize=0, error_kw={'elinewidth': 5, 'ecolor': 'black'})

        ax.axhline(0, color='black', linewidth=0.8)
        
        if row_idx == 0:
            ax.set_ylabel("Intensity", fontsize=22, fontweight='bold')
            
        ax.tick_params(axis='y', rotation=30, labelsize=20)
        ax.set_title(cat, loc='left', fontsize=22, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=1.0)
        
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=18, rotation=25, ha='right')

    handles = [Line2D([0], [0], color=MODALITY_PALETTE[m], lw=10, label=m) for m in modalities]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.13), 
               ncol=3, frameon=False, fontsize=20, labelspacing=0.3, columnspacing=1.5)
               
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15)
    
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
            err_mat = d['err_matrix']
            
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
                    err_int = err_mat[i, j] * 2 
                    
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
                            'Intensity': abs_int,
                            'Error': err_int
                        })
                
    df = pd.DataFrame(records)
    if df.empty: return
    
    plot_scheme_b_bidirectional(df, save_dir)

# ================= MAIN =================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Starting SHAP Summary Visualization...")
    
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

    if data_general:
        plot_scheme_a_bias_fingerprint(data_general, OUTPUT_DIR)
        
    if data_interaction:
        process_and_plot_scheme_b(data_interaction, OUTPUT_DIR)
        
    print("Done. Check generated Markdown tables in the output directory.")

if __name__ == "__main__":
    main()
