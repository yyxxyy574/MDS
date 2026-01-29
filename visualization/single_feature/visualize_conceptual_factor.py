import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import yaml
import os

from config.constants import ROOT
from visualization.utils import DILEMMA_ORDER, DILEMMA_DIMENSION_ORDER, DILEMMA_DIMENSION_MAP, parse_model_info, get_stars, MODEL_LIST, MODEL_NAME_LIST, MODEL_TYPE_LIST, MODALITY_LIST, MODALITY_PALETTE

CONCEPTUAL_FACTORS_MAP = {
    # Main Effects
    'C(personal_force, Treatment(reference=0))[T.1]': 'Personal Force', 
    'C(intention_of_harm, Treatment(reference=0))[T.1]': 'Intention of Harm',
    'C(self_benefit, Treatment(reference=0))[T.1]': 'Self Benefit',
    
    # 2-way Interactions
    'C(personal_force, Treatment(reference=0))[T.1]:C(intention_of_harm, Treatment(reference=0))[T.1]': 'PF x IH',
    'C(personal_force, Treatment(reference=0))[T.1]:C(self_benefit, Treatment(reference=0))[T.1]': 'PF x SB',
    'C(intention_of_harm, Treatment(reference=0))[T.1]:C(self_benefit, Treatment(reference=0))[T.1]': 'IH x SB',

    # 3-way Interaction
    'C(personal_force, Treatment(reference=0))[T.1]:C(intention_of_harm, Treatment(reference=0))[T.1]:C(self_benefit, Treatment(reference=0))[T.1]': 'PF x IH x SB'
}

FACTOR_TYPE_PALETTE = {
    'Main Effect': {'Text': '#5c8aae', 'Caption': '#8ab6d6', 'Image': '#aebcd1'},        
    '2-Way Interaction': {'Text': '#7f9e74', 'Caption': '#a3c299', 'Image': '#c6dab6'}, 
    '3-Way Interaction': {'Text': '#dcb65b', 'Caption': '#e5cf85', 'Image': '#eee2b3'}  
}

EFFECT_COLOR_MAP = {
    'Personal Force': '#d62728',    # Red
    'Intention of Harm': '#d62728', # Red
    'Self Benefit': '#1f77b4'       # Blue
}

FACTOR_TYPE_MAP = {
    'Personal Force': 'Main Effect',
    'Intention of Harm': 'Main Effect',
    'Self Benefit': 'Main Effect',
    'PF x IH': '2-Way Interaction',
    'PF x SB': '2-Way Interaction',
    'IH x SB': '2-Way Interaction',
    'PF x IH x SB': '3-Way Interaction'
}

def prepare_results(model_str, results_file, data_list, intercept_data):
    """
    Robustly processes regression results.
    - Captures Main Effects and Interactions.
    - Crucially, captures the INTERCEPT (log_odds) for scenario reconstruction.
    """
    print(f"Processing results for {model_str} from {results_file}...")
    
    try:
        with open(results_file, 'r') as f:
            results = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {results_file}")
        return
    
    model_type, modality = parse_model_info(model_str)
    group_name = f"{model_type} - {modality}"

    for dilemma in results:
        
        dilemma_results = results[dilemma]
        
        # 1. Capture Intercept (Baseline Log Odds)
        if 'Intercept' in dilemma_results:
            intercept_stats = dilemma_results['Intercept']
            # Store intercept for scenario reconstruction later
            intercept_data.append({
                'model_str': group_name,
                'model_type': model_type,
                'modality': modality,
                'dilemma': dilemma,
                'intercept_log_odds': intercept_stats.get('log_odds', 0),
                'status': intercept_stats.get('Status', 'OK')
            })
            
            # Check for constant target (edge case)
            if intercept_stats.get('Status') == 'Constant_Target':
                # Fill dummy data for heatmap to show failure
                for factor_label in CONCEPTUAL_FACTORS_MAP.values():
                    data_list.append({
                        'model_str': group_name,
                        'model_type': model_type,
                        'modality': modality,
                        'dilemma': dilemma,
                        'factor': factor_label,
                        'factor_type': FACTOR_TYPE_MAP[factor_label],
                        'odds_ratio': np.nan, 
                        'log_odds': np.nan,
                        'p_value': 1.0,
                        'status_note': 'Constant_Target'
                    })
                continue 

        # 2. Capture Factors
        for factor_key, factor_label in CONCEPTUAL_FACTORS_MAP.items():
            if factor_key in dilemma_results:
                stats = dilemma_results[factor_key]
                or_val = stats.get('Odds_ratio', np.nan)
                log_val = stats.get('log_odds', np.nan) # Needed for reconstruction
                p_val = stats.get('p_value', np.nan)
                status = stats.get('Status', 'Normal')

                # Handle Separation (Infinite OR)
                if status in ['Perfect_Separation', 'Quasi_Complete_Separation', 'Manual_Sum_Coding_Exact']:
                     # For visualization, clip log_odds if they are inf
                     if log_val == np.inf: log_val = 10
                     if log_val == -np.inf: log_val = -10

                data_list.append({
                    'model_str': group_name,
                    'model_type': model_type,
                    'modality': modality,
                    'dilemma': dilemma,
                    'factor': factor_label,
                    'factor_type': FACTOR_TYPE_MAP[factor_label],
                    'odds_ratio': or_val,
                    'log_odds': log_val,
                    'p_value': p_val,
                    'status_note': status
                })
            else:
                data_list.append({
                    'model_str': group_name,
                    'model_type': model_type,
                    'modality': modality,
                    'dilemma': dilemma,
                    'factor': factor_label,
                    'factor_type': FACTOR_TYPE_MAP[factor_label],
                    'odds_ratio': np.nan,
                    'log_odds': 0, # Assume 0 effect if missing
                    'p_value': np.nan,
                    'status_note': 'Missing'
                })

def create_dilemma_to_type_mapping():
    """Create mapping from dilemma to conflict type using DILEMMA_DIMENSION_MAP."""
    dilemma_to_type = {}
    for ctype, dilemmas in DILEMMA_DIMENSION_MAP.items():
        for s in dilemmas:
            dilemma_to_type[s] = ctype
    return dilemma_to_type

def prepare_dimension_data(df_main):
    """Prepare data for dimension-based analysis."""
    dilemma_to_type = create_dilemma_to_type_mapping()
    
    df_dim = df_main.copy()
    # Map dilemma to its dimension/conflict type
    df_dim['conflict_type'] = df_dim['dilemma'].map(dilemma_to_type)
    df_dim['model_modality'] = df_dim['model_str']
    
    # Drop rows where conflict_type is NaN (dilemma not in map)
    df_dim = df_dim.dropna(subset=['conflict_type', 'model_modality'])
    return df_dim

def plot_heatmaps(df_p, save_dir):
    """Generate heatmap visualizations for ALL factors."""
    print("Generating heatmaps...")
    
    sns.set(style="white", font_scale=1.0)
    unique_factors = list(CONCEPTUAL_FACTORS_MAP.values())

    for factor_label in unique_factors:
        # [Note] Heatmaps generally visualize specific dilemmas, so 'total' might be excluded naturally if not in DILEMMA_ORDER
        # or it can be kept if added to order. Here we proceed with DILEMMA_ORDER.
        subset = df_p[df_p['factor'] == factor_label]
        if subset.empty: continue

        pivot_or = subset.pivot(index='dilemma', columns='model_str', values='odds_ratio')
        pivot_p = subset.pivot(index='dilemma', columns='model_str', values='p_value')

        pivot_or = pivot_or.reindex(index=DILEMMA_ORDER, columns=MODEL_NAME_LIST)
        pivot_p = pivot_p.reindex(index=DILEMMA_ORDER, columns=MODEL_NAME_LIST)

        pivot_or = pivot_or.dropna(how='all')
        pivot_p = pivot_p.dropna(how='all')

        processed_or = pivot_or.replace(0.0, 0.01).replace(np.inf, 100).clip(lower=0.01, upper=100)
        log_or = np.log10(processed_or)
        annot_matrix = pivot_p.map(get_stars)

        fig, ax = plt.subplots(figsize=(16, 14)) # Slightly wider for more columns
        ax.set_facecolor('#EAEAEA') 

        sns.heatmap(log_or, cmap='vlag', center=0, 
                    annot=annot_matrix, fmt='', 
                    mask=pivot_or.isna(),
                    cbar_kws={'label': 'Log10 Odds Ratio'},
                    linewidths=.5, linecolor='white',
                    square=False, ax=ax)
        
        num_mods = len(MODALITY_LIST)
        for i, model_type in enumerate(MODEL_TYPE_LIST):
            start_x = i * num_mods
            center_x = start_x + num_mods / 2
            end_x = start_x + num_mods
            
            ax.text(center_x, -0.2, model_type, 
                    ha='center', va='bottom', 
                    fontsize=12, fontweight='bold', color='#333333')
            
            if i < len(MODEL_TYPE_LIST) - 1:
                ax.axvline(x=end_x, color='white', linewidth=8)

        x_labels = MODALITY_LIST * len(MODEL_TYPE_LIST)
        
        ax.set_xticks([i + 0.5 for i in range(len(MODEL_NAME_LIST))])
        ax.set_xticklabels(x_labels, rotation=0, fontsize=10)
        ax.set_xlabel("")

        readable_title = factor_label.replace("_", " ").title()
        plt.title(f"Sensitivity to {readable_title}", fontsize=18, fontweight='bold', pad=40)
        plt.tight_layout()

        safe_name = factor_label.lower().replace(" ", "_").replace(":", "_")
        plt.savefig(f'{save_dir}/heatmap_{safe_name}.pdf', bbox_inches='tight', dpi=300)
        plt.close()

def plot_model_complexity_profile(df_dim, save_dir):
    """Generates stacked bar chart for model complexity."""
    print("Generating model complexity profile...")
    
    df_calc = df_dim.copy()
    df_calc['odds_ratio_clean'] = df_calc['odds_ratio'].replace(np.inf, 100).replace(0, 0.01).clip(0.01, 100)
    df_calc['effect_strength'] = np.abs(np.log10(df_calc['odds_ratio_clean']))
    
    # Filter only significant effects
    df_sig = df_calc[df_calc['p_value'] < 0.05].copy()
    
    if df_sig.empty:
        print("No significant effects found for complexity profile.")
        return

    agg_df = df_sig.groupby(['model_type', 'modality', 'factor_type'])['effect_strength'].mean().reset_index()
    
    # Pivot for stacking
    pivot_df = agg_df.pivot_table(index=['model_type', 'modality'], columns='factor_type', values='effect_strength').fillna(0)

    factor_types = ['Main Effect', '2-Way Interaction', '3-Way Interaction']
    
    # Get unique base models in order
    unique_base_models = []
    for m in MODEL_NAME_LIST:
        base = m.split(' - ')[0]
        if base not in unique_base_models:
            unique_base_models.append(base)
            
    # Prepare Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bar_width = 0.25
    indices = np.arange(len(unique_base_models))
    offsets = [-bar_width, 0, bar_width]
    
    # Plotting loop
    for i, model_str in enumerate(unique_base_models):
        for modality, offset in zip(MODALITY_LIST, offsets):
            if (model_str, modality) in pivot_df.index:
                row = pivot_df.loc[(model_str, modality)]
                bottom = 0
                for f_type in factor_types:
                    if f_type in row:
                        val = row[f_type]
                        color = FACTOR_TYPE_PALETTE[f_type].get(modality, '#cccccc')
                        hatch = '///' if modality == 'Image' else ('.' if modality == 'Caption' else '')
                        ax.bar(indices[i] + offset, val, bar_width, bottom=bottom, 
                               color=color, edgecolor='white', linewidth=0.5, hatch=hatch)
                        bottom += val
    
    simple_handles = []
    for f_type in factor_types:
        # Create a proxy artist for the legend group
        simple_handles.append(mpatches.Patch(facecolor=FACTOR_TYPE_PALETTE[f_type]['Text'], label=f"{f_type} (Text)"))
        simple_handles.append(mpatches.Patch(facecolor=FACTOR_TYPE_PALETTE[f_type]['Caption'], hatch='.', label=f"{f_type} (Caption)"))
        simple_handles.append(mpatches.Patch(facecolor=FACTOR_TYPE_PALETTE[f_type]['Image'], hatch='///', label=f"{f_type} (Image)"))

    ax.set_xticks(indices)
    ax.set_xticklabels(unique_base_models, ha='center')
    plt.title('Model Complexity Profile', fontsize=16)
    plt.ylabel('Mean Absolute Effect Strength', fontsize=12)
    plt.legend(handles=simple_handles, bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/model_complexity_profile.pdf', bbox_inches='tight', dpi=300)
    plt.close()

def plot_dimension_analysis_grouped(df_dim, save_dir):
    """Generate dimension-based analysis for Main Effects and Interactions."""
    print("Generating grouped dimension analysis plots...")
    
    if df_dim.empty:
        print("No data available for dimension analysis.")
        return

    df_dim['odds_ratio_clean'] = df_dim['odds_ratio'].replace(np.inf, 1000).replace(0, 0.001)
    
    for group_name, factors in {'Main_Effects': [k for k, v in FACTOR_TYPE_MAP.items() if v == 'Main Effect'], 
                                'Interaction_Effects': [k for k, v in FACTOR_TYPE_MAP.items() if 'Interaction' in v]}.items():
        
        subset_df = df_dim[df_dim['factor'].isin(factors)]
        agg_df = subset_df.groupby(['conflict_type', 'factor', 'model_type', 'modality'])['odds_ratio_clean'].median().reset_index()
        
        for factor in factors:
            subset = agg_df[agg_df['factor'] == factor].copy()
            
            if subset.empty:
                # print(f"Skipping {factor} (no data)")
                continue
                
            subset['Log OR'] = np.log10(subset['odds_ratio_clean'])
            
            # Catplot
            g = sns.catplot(
                data=subset, kind="bar",
                x="Log OR", y="conflict_type", hue="modality", col="model_type",
                palette=MODALITY_PALETTE, height=6, aspect=0.8, sharex=True, 
                order=DILEMMA_DIMENSION_ORDER, 
                col_order=MODEL_TYPE_LIST,
                hue_order=MODALITY_LIST
            )
            
            for ax in g.axes.flat:
                ax.axvline(0, color='black', linestyle='--', linewidth=1)
                ax.set_xlabel('Log10 Odds Ratio (Median)')
            
            title_prefix = "Impact of" if group_name == 'Main_Effects' else "Interaction Strength of"
            g.fig.suptitle(f'{title_prefix} {factor} across Moral Dimensions', y=1.02, fontsize=16, fontweight='bold')
            
            safe_name = factor.lower().replace(" ", "_").replace(":", "_")
            save_path = f'{save_dir}/dimension_analysis_{group_name}_{safe_name}.pdf'
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"Saved {group_name} analysis to: {save_path}")

def plot_main_effect_balance(df_p, save_dir, target_dilemma='total'):
    """
    Visual 1: The 'Tug-of-War' Chart.
    [Modified] Uses 'total' (Pooled) results for robust global comparison.
    """
    print("Generating Main Effect Balance (Tug-of-War) plot...")
    
    # Filter for Main Effects only
    main_effects = ['Personal Force', 'Intention of Harm', 'Self Benefit']
    
    df_agg = df_p[
        (df_p['factor'].isin(main_effects)) & 
        (df_p['dilemma'] == target_dilemma)
    ].copy()
    
    if df_agg.empty:
        print("No 'total' data found for Tug-of-War plot.")
        return

    # Alternative: Standard Barplot with Hue
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.set_theme(style="whitegrid")
    sns.barplot(data=df_agg, x='model_str', y='log_odds', hue='factor', 
                palette=EFFECT_COLOR_MAP, errorbar=None,
                order=MODEL_NAME_LIST,
                ax=ax, edgecolor='white', linewidth=0.5,
                width=0.75)
    
    plt.axhline(0, color='black', linewidth=1.5)

    hatch_styles = ['', '///', '...']
    for patch in ax.patches:
        current_x = patch.get_x() + patch.get_width() / 2
        
        if np.isnan(current_x): 
            continue
        
        col_idx = int(round(current_x))
        modality_idx = col_idx % len(MODALITY_LIST)
        
        patch.set_hatch(hatch_styles[modality_idx])
        patch.set_edgecolor('black')
        patch.set_linewidth(0.5)

    group_size = len(MODALITY_LIST)
    new_xticklabels = []
    xticks_positions = []
    for i, model_base in enumerate(MODEL_TYPE_LIST):
        start_idx = i * group_size
        end_idx = start_idx + group_size
        center_idx = start_idx + (group_size - 1) / 2.0
        
        ax.text(center_idx, -0.07, model_base, 
                ha='center', va='top', 
                transform=ax.get_xaxis_transform(),
                fontsize=11, fontweight='bold', color='#333333')
        
        if i < len(MODEL_TYPE_LIST) - 1:
            sep_x = end_idx - 0.5
            ax.axvline(x=sep_x, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)

        for j in range(group_size):
            xticks_positions.append(start_idx + j)
            new_xticklabels.append(MODALITY_LIST[j])

    ax.set_xticks(xticks_positions)
    ax.set_xticklabels(new_xticklabels, rotation=0, fontsize=10)
    ax.set_xlabel("")
    ax.set_xlim(-0.5, len(MODEL_NAME_LIST) - 0.5)

    scope_title = "Global (Total)" if target_dilemma == 'total' else "Care vs. Care"
    plt.ylabel(f"Log Odds ({scope_title} Effect Strength)", fontsize=13)
    plt.title(f"The Moral Tug-of-War: Promoting vs. Inhibiting Factors ({scope_title})",
              fontsize=18, fontweight='bold', y=1.1)
    
    color_handles = [mpatches.Patch(facecolor=c, label=l, edgecolor='black') 
                     for l, c in EFFECT_COLOR_MAP.items()]
    hatch_handles = [
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='', label='Text'),
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='///', label='Caption'),
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='...', label='Image')
    ]
    empty_handle = mpatches.Patch(color='none', label='')
    final_handles = color_handles + [empty_handle] + hatch_handles
    leg = ax.legend(handles=final_handles, 
              bbox_to_anchor=(0.5, 1.01), loc='lower center', 
              ncol=7,
              frameon=False, fontsize=11, handletextpad=0.5, columnspacing=1)
    
    plt.tight_layout(rect=[0, 0, 1, 0.9]) 
    plt.subplots_adjust(bottom=0.12)
    plt.savefig(f'{save_dir}/tug_of_war_main_effects_{target_dilemma}.pdf', bbox_inches='tight', dpi=300)
    plt.close()

def plot_three_way_interaction_scenarios(df_p, df_intercepts, save_dir):
    """
    Visual 2 (Upgraded): 3-Way Interaction Scenario Reconstruction.
    [Modified] Uses 'total' (Pooled) intercepts and coefficients.
    """
    print("Generating 3-Way Interaction Scenario Reconstruction plot...")
    
    # 1. Prepare Coefficients
    # We need ALL coefficients: Intercept, PF, IH, SB, PFxIH, PFxSB, IHxSB, PFxIHxSB
    
    # [Modified] Filter for 'total' only
    df_int_total = df_intercepts[df_intercepts['dilemma'] == 'total']
    agg_intercept = df_int_total.set_index('model_str')['intercept_log_odds']
    
    required_factors = [
        'Personal Force', 'Intention of Harm', 'Self Benefit',
        'PF x IH', 'PF x SB', 'IH x SB', 'PF x IH x SB'
    ]
    
    # [Modified] Filter for 'total' only
    df_sub = df_p[
        (df_p['factor'].isin(required_factors)) & 
        (df_p['dilemma'] == 'total')
    ].copy()
    
    # Pivot to get coefficients. Using mean() here is safe as there is only 1 row per (model_str, factor) for 'total'
    agg_factors = df_sub.groupby(['model_str', 'factor'])['log_odds'].mean().unstack().fillna(0)
    
    reconstruction_data = []
    
    def sigmoid(x): 
        return 1 / (1 + np.exp(-x))
    
    for model_str in agg_intercept.index:
        b0 = agg_intercept[model_str]
        
        # Extract Coeffs (default to 0 if missing)
        coeffs = {k: agg_factors.loc[model_str, k] if k in agg_factors.columns else 0 for k in required_factors}
        
        # --- Context 1: Altruistic (SB=0) ---
        # Only terms NOT involving SB
        # 1. Baseline (Switch)
        l_base = b0
        # 2. Force Only
        l_pf = b0 + coeffs['Personal Force']
        # 3. Intention Only
        l_ih = b0 + coeffs['Intention of Harm']
        # 4. Combined (Footbridge)
        l_comb = b0 + coeffs['Personal Force'] + coeffs['Intention of Harm'] + coeffs['PF x IH']
        
        # --- Context 2: Selfish (SB=1) ---
        # Add SB main effect AND all interactions involving SB
        sb_base = coeffs['Self Benefit'] # The base shift caused by SB
        
        # 1. Benefit Only (Switch but for Self)
        l_sb = b0 + sb_base
        
        # 2. Benefit + Force
        # Intercept + PF + SB + (PFxSB)
        l_sb_pf = b0 + coeffs['Personal Force'] + sb_base + coeffs['PF x SB']
        
        # 3. Benefit + Intention
        # Intercept + IH + SB + (IHxSB)
        l_sb_ih = b0 + coeffs['Intention of Harm'] + sb_base + coeffs['IH x SB']
        
        # 4. Benefit + Combined (Self-Footbridge)
        # ALL TERMS ACTIVE
        l_sb_comb = (b0 + coeffs['Personal Force'] + coeffs['Intention of Harm'] + sb_base + 
                     coeffs['PF x IH'] + coeffs['PF x SB'] + coeffs['IH x SB'] + 
                     coeffs['PF x IH x SB']) # <--- This captures the 3-way effect!
        
        meta = df_int_total[df_int_total['model_str'] == model_str].iloc[0]

        # Append Data
        scenarios = [
            ('Altruistic (SB=0)', '1. Baseline', sigmoid(l_base)),
            ('Altruistic (SB=0)', '2. +Force', sigmoid(l_pf)),
            ('Altruistic (SB=0)', '3. +Intention', sigmoid(l_ih)),
            ('Altruistic (SB=0)', '4. Combined', sigmoid(l_comb)),
            
            ('Selfish (SB=1)', '1. Baseline', sigmoid(l_sb)),
            ('Selfish (SB=1)', '2. +Force', sigmoid(l_sb_pf)),
            ('Selfish (SB=1)', '3. +Intention', sigmoid(l_sb_ih)),
            ('Selfish (SB=1)', '4. Combined', sigmoid(l_sb_comb)),
        ]
        
        for context, scene, prob in scenarios:
            reconstruction_data.append({
                'model_str': model_str, 
                'model_type': meta['model_type'],
                'modality': meta['modality'],
                'Context': context, 
                'Scenario': scene, 
                'Prob': prob
            })

    df_recon = pd.DataFrame(reconstruction_data)
    
    if df_recon.empty: return

    # Plotting
    g = sns.catplot(
        data=df_recon, kind="bar",
        x="Scenario", y="Prob", hue="Context", row="model_type", col="modality",
        palette={"Altruistic (SB=0)": "#95a5a6", "Selfish (SB=1)": "#e74c3c"}, # Grey vs Red
        height=3.5, aspect=1.3,
        sharey=True, margin_titles=True,
        row_order=MODEL_TYPE_LIST,
        col_order=MODALITY_LIST
    )
    
    g.set_axis_labels("", "Probability of Approval")
    g.set_xticklabels(rotation=45, ha='right')
    
    # Add a reference line for 0.5 probability
    for ax in g.axes.flat:
        ax.axhline(0.5, ls='--', color='k', alpha=0.3)
        
    g.fig.suptitle("3-Way Interaction (Pooled): How Self-Interest Reshapes Moral Judgments", y=1.02, fontsize=16)
    
    save_path = f'{save_dir}/interaction_3way_reconstruction.pdf'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved 3-way interaction plot to: {save_path}")

def plot_modality_gap_decomposition(df_p, save_dir):
    """
    Visual: Decomposes the difference in coefficients into 'Cognitive Cost' and 'Visual Alignment Cost'.
    [Modified] Uses 'total' results for accurate gap calculation.
    """
    print("Generating Modality Gap Decomposition plot...")
    
    # Filter for Main Effects only to keep it clean
    target_factors = ['Personal Force', 'Intention of Harm', 'Self Benefit']
    
    # [Modified] Filter for 'total' only
    df_sub = df_p[
        (df_p['factor'].isin(target_factors)) & 
        (df_p['dilemma'] == 'total')
    ].copy()
    
    # Pivot to get columns: Text, Caption, Image
    df_pivot = df_sub.pivot_table(index=['model_type', 'factor'], columns='modality', values='log_odds').reset_index()
    
    # Ensure all columns exist
    for mod in ['Text', 'Caption', 'Image']:
        if mod not in df_pivot.columns:
            df_pivot[mod] = np.nan
            
    # Calculate Gaps
    # We take Absolute Difference to measure magnitude of change (Stability), not direction
    df_pivot['Cognitive_Gap'] = (df_pivot['Caption'] - df_pivot['Text']).abs()
    df_pivot['Visual_Gap'] = (df_pivot['Image'] - df_pivot['Caption']).abs()
    
    # Remove rows with NaNs
    df_plot = df_pivot.dropna(subset=['Cognitive_Gap', 'Visual_Gap'])
    
    if df_plot.empty:
        return

    # Plotting
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot
    sns.scatterplot(data=df_plot, x='Cognitive_Gap', y='Visual_Gap', 
                    hue='model_type', style='factor', 
                    s=150, alpha=0.8, ax=ax, palette='deep')
    
    # Add diagonal line (Visual Gap = Cognitive Gap)
    max_val = max(df_plot['Cognitive_Gap'].max(), df_plot['Visual_Gap'].max())
    ax.plot([0, max_val], [0, max_val], ls="--", c=".3", alpha=0.5)
    
    # Annotations for Quadrants
    ax.text(max_val*0.9, max_val*0.1, "High Cognitive Instability\n(Text vs Caption issue)", 
            ha='right', va='bottom', fontsize=10, color='gray')
    ax.text(max_val*0.1, max_val*0.9, "High Visual Misalignment\n(Caption vs Image issue)", 
            ha='left', va='top', fontsize=10, color='gray')
    ax.text(0, 0, "Consistent/Robust", ha='left', va='bottom', fontsize=10, color='green', fontweight='bold')

    ax.set_xlabel("Cognitive Gap (|Text - Caption|)", fontsize=12)
    ax.set_ylabel("Visual Gap (|Caption - Image|)", fontsize=12)
    ax.set_title("Source of Inconsistency (Pooled): Cognitive Load vs. Visual Modality", fontsize=15, fontweight='bold')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/modality_gap_decomposition.pdf', bbox_inches='tight', dpi=300)
    plt.close()


def plot_factor_trajectory_slope(df_p, save_dir, target_dilemma='total'):
    """
    Visual: Slope Chart tracking the coefficient trajectory: Text -> Caption -> Image.
    [Modified] Uses 'total' results for global trend tracking.
    """
    print("Generating Factor Trajectory (Slope) plot...")
    
    target_factors = ['Personal Force', 'Intention of Harm', 'Self Benefit']
    modality_order = ['Text', 'Caption', 'Image']
    
    df_sub = df_p[
        (df_p['factor'].isin(target_factors)) & 
        (df_p['dilemma'] == target_dilemma)
    ].copy()
    
    # Setup plot
    fig, axes = plt.subplots(1, len(target_factors), figsize=(16, 6), sharey=True)
    
    for i, factor in enumerate(target_factors):
        ax = axes[i]
        subset = df_sub[df_sub['factor'] == factor]
        
        # We need to plot a line for each model
        for model in subset['model_type'].unique():
            model_data = subset[subset['model_type'] == model]
            
            # Reindex to ensure order
            model_data = model_data.set_index('modality').reindex(modality_order).reset_index()
            
            # Plot
            ax.plot(model_data['modality'], model_data['log_odds'], 
                    marker='o', linewidth=2, label=model)
            
            # Add text annotation at the end
            last_val = model_data.iloc[-1]['log_odds']
            if not np.isnan(last_val):
                 ax.text(2.1, last_val, model, va='center', fontsize=9)

        ax.set_title(f"Trajectory of {factor}", fontsize=14, fontweight='bold')
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        scope_label = "Pooled" if target_dilemma == 'total' else "Care-Care"
        ax.set_ylabel(f"Log Odds ({scope_label} Strength)")
        ax.grid(axis='x', linestyle=':')
        
    # Only legend on first plot to save space, or handle external legend
    axes[0].legend(fontsize=8, loc='best')
    
    scope_title = "Global (Total)" if target_dilemma == 'total' else "Care vs. Care"
    plt.suptitle(f"The Modality Shift ({scope_title}): How Context & Vision Alter Moral Weights", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/factor_trajectory_slope_{target_dilemma}.pdf', bbox_inches='tight', dpi=300)
    plt.close()

def plot_main_effect_distribution(df_p, save_dir):
    """
    Visual: Boxplot showing the distribution of Main Effects across ALL individual dilemmas.
    Complementary to 'Tug-of-War' which shows the pooled average.
    """
    print("Generating Main Effect Distribution (Boxplot)...")
    
    # Filter for Main Effects and Exclude Aggregates
    target_factors = ['Personal Force', 'Intention of Harm', 'Self Benefit']
    aggregates = ['total', 'care_vs_care']
    
    df_dist = df_p[
        (df_p['factor'].isin(target_factors)) & 
        (~df_p['dilemma'].isin(aggregates))
    ].copy()
    
    if df_dist.empty:
        print("No individual dilemma data found for distribution plot.")
        return

    # Create FacetGrid: Rows = Factors, Cols = None (or Models?)
    # Better: Col = Model, X = Factor, Y = Log Odds, Hue = Modality
    
    g = sns.catplot(
        data=df_dist, kind="box",
        x="factor", y="log_odds", hue="modality", col="model_type",
        hue_order=MODALITY_LIST, col_order=MODEL_TYPE_LIST,
        palette=MODALITY_PALETTE, # Use consistent modality colors
        height=5, aspect=1.2,
        sharey=True,
        fliersize=2, linewidth=1
    )
    
    # Add a stripplot on top for detail (optional, but good for "distribution")
    # Mapping stripplot to the same grid is tricky with catplot. 
    # Let's stick to boxplot for clarity, or hybrid violin. 
    # Boxplot is standard for "distribution".
    
    for ax in g.axes.flat:
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.grid(axis='y', linestyle=':', alpha=0.5)

    g.set_axis_labels("", "Log Odds (Effect Strength)")
    g.set_titles("{col_name}")
    g._legend.set_title("Modality")
    
    g.fig.suptitle("Distribution of Moral Factors across Dilemmas (Boxplot)", y=1.05, fontsize=16, fontweight='bold')
    
    save_path = f'{save_dir}/main_effect_distribution_boxplot.pdf'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved distribution boxplot to: {save_path}")

def main():
    """Main function to generate all visualizations."""
    result_path = f"{ROOT}/../results/single_feature/analyze_results"
    data_list = []
    intercept_list = [] # Separate list to store intercepts
    
    for model_str in MODEL_LIST:
        file_path = f"{result_path}/conceptual_factor_{model_str}.yaml"
        prepare_results(model_str, file_path, data_list, intercept_list)
    
    if not data_list:
        print("No data loaded. Exiting.")
        return

    df_p = pd.DataFrame(data_list)
    df_intercepts = pd.DataFrame(intercept_list)
    
    viz_dir = f'{ROOT}/../visualization/single_feature/conceptual_factor'
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. Heatmaps
    plot_heatmaps(df_p, viz_dir)
    
    # 2. Data Prep for Dimensions
    df_dim = prepare_dimension_data(df_p)
    
    # 3. Dimension Analysis (Fixed)
    plot_dimension_analysis_grouped(df_dim, viz_dir)
    
    # 4. Complexity Profile
    plot_model_complexity_profile(df_dim, viz_dir)

    if not df_intercepts.empty:
        plot_three_way_interaction_scenarios(df_p, df_intercepts, viz_dir)

    for scope in ['total', 'care_vs_care']:
        plot_main_effect_balance(df_p, viz_dir, target_dilemma=scope)
        plot_factor_trajectory_slope(df_p, viz_dir, target_dilemma=scope)

    plot_main_effect_distribution(df_p, viz_dir)

    print("All visualizations completed!")

if __name__ == '__main__':
    main()