import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import colorsys
import matplotlib.colors as mc
import seaborn as sns
import yaml
import os
import re

from config.constants import ROOT
from visualization.utils import DILEMMA_DIMENSION_MAP, DILEMMA_DIMENSION_ORDER, DILEMMA_ORDER, MODEL_LIST, MODEL_NAME_LIST, MODEL_TYPE_LIST, MODALITY_LIST, parse_model_info

LEVEL_PALETTE = {
    'Agent Intrinsic (L1)': '#F7E7A6',       # Butter Yellow
    'Target Valuation (L2)': '#A5D6A7',      # Soft Mint Green
    'Target-Target Interaction (L3)': '#90CAF9', # Powder Blue
    'Agent-Target Interaction (L4)': '#EF9A9A'   # Soft Rose Pink
}

# Helper for MFT Sorting
def get_dilemma_order_by_mft():
    ordered_dilemmas = []
    for mft_key in DILEMMA_DIMENSION_ORDER:
        if mft_key in DILEMMA_DIMENSION_MAP:
            dims = sorted(DILEMMA_DIMENSION_MAP[mft_key])
            ordered_dilemmas.extend(dims)
    known = set(ordered_dilemmas)
    remaining = [d for d in DILEMMA_ORDER if d not in known]
    return ordered_dilemmas + remaining

SORTED_DILEMMAS = get_dilemma_order_by_mft()

def prepare_results(model_str, results_file, data_list):
    print(f"Processing results for {model_str} from {results_file}...")
    try:
        with open(results_file, 'r') as f:
            results = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {results_file}")
        return

    model_type, modality = parse_model_info(model_str)
    group_name = f"{model_type} - {modality}"

    for scenario, conditions in results.items():
        if not isinstance(conditions, dict): continue
        parts = scenario.rsplit('_', 1)
        dilemma, attribute = (parts[0], parts[1]) if len(parts) == 2 else (scenario, 'Unknown')

        mft_dim = 'Other'
        for key in DILEMMA_DIMENSION_MAP:
            for t in DILEMMA_DIMENSION_MAP[key]:
                if dilemma.startswith(t):
                    mft_dim = key
                    break

        for condition, stats in conditions.items():
            if not isinstance(stats, dict): continue
            if 'Odds_ratio' in stats and 'p_value' in stats:
                or_val = stats['Odds_ratio']
                p_val = stats['p_value']
                status = stats.get('Status', 'Normal')

                if status in ['Perfect_Separation', 'Quasi_Complete_Separation', 'Constant_Target']:
                    if pd.isna(p_val) or p_val > 0.05: p_val = 1e-15 
                    if or_val == 0: or_val = 1e-6
                    if np.isinf(or_val): or_val = 1e6
                
                is_interaction = ":" in condition
                has_agent = "C(agent" in condition 
                
                if "Intercept" in condition: level = "Intercept"
                elif is_interaction:
                    level = "Agent-Target Interaction (L4)" if has_agent else "Target-Target Interaction (L3)"
                elif has_agent: level = "Agent Intrinsic (L1)"
                else: level = "Target Valuation (L2)"

                data_list.append({
                    'model_str': group_name, 'model_type': model_type, 'modality': modality,
                    'dilemma': dilemma, 'attribute': attribute,
                    'mft_dimension': mft_dim, 'condition': condition, 'level': level,
                    'odds_ratio': or_val, 'p_value': p_val, 'status_note': status
                })

def plot_bias_heatmap(df, save_dir):
    print("Generating global bias heatmap...")
    sig_data = df[(df['p_value'] < 0.05) & (df['condition'] != 'Intercept')].copy()
    
    pivot = None
    max_val = 1.0

    if not sig_data.empty:
        sig_data['log_or'] = np.log10(sig_data['odds_ratio'].clip(1e-6, 1e6))
        sig_data['abs_log_or'] = sig_data['log_or'].abs()
        
        idx = sig_data.groupby(['model_str', 'dilemma'])['abs_log_or'].idxmax()
        top_bias = sig_data.loc[idx]
        
        pivot = top_bias.pivot(index='dilemma', columns='model_str', values='log_or')
        max_val = top_bias['abs_log_or'].max()
    else:
        pivot = pd.DataFrame()
        max_val = 1.0

    pivot = pivot.reindex(index=SORTED_DILEMMAS, columns=MODEL_NAME_LIST)

    fig, ax = plt.subplots(figsize=(18, 14))
    ax.set_facecolor('#EAEAEA')

    sns.heatmap(pivot, cmap='RdBu_r', center=0, vmin=-max_val, vmax=max_val, 
                annot=True, fmt='.1f', 
                mask=pivot.isna(),
                cbar_kws={'label': 'Log10 OR (Max Bias)', 'shrink': 0.8},
                linewidths=1, linecolor='white',
                square=False, ax=ax)
    
    num_mods = len(MODALITY_LIST)
    for i, model_type in enumerate(MODEL_TYPE_LIST):
        start_x = i * num_mods
        center_x = start_x + num_mods / 2.0
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
    
    plt.title('Global Landscape: Max Bias Intensity by Dilemma (Sorted by MFT)', fontsize=16, pad=40)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '1_global_bias_heatmap_mft.pdf'), dpi=300)
    plt.close()

def plot_attribute_faceted_heatmap(df, save_dir):
    print("Generating attribute faceted heatmap...")
    sig_data = df[(df['p_value'] < 0.05) & (df['condition'] != 'Intercept')].copy()
    if sig_data.empty: return
    sig_data['log_or'] = np.log10(sig_data['odds_ratio'].clip(1e-6, 1e6))
    sig_data['abs_log_or'] = sig_data['log_or'].abs()
    
    idx = sig_data.groupby(['model_type', 'modality', 'dilemma', 'attribute'])['abs_log_or'].idxmax()
    agg = sig_data.loc[idx].drop_duplicates(subset=['model_type', 'modality', 'dilemma', 'attribute'])
    
    all_attributes = ['species', 'color', 'profession', 'gender', 'age', 'wealth', 'fitness', 'education']
    
    fig, axes = plt.subplots(len(MODEL_TYPE_LIST), len(MODALITY_LIST), figsize=(20, 6 * len(MODEL_TYPE_LIST)), sharex=True, sharey=False)
    global_max = agg['abs_log_or'].max()
    if len(MODEL_TYPE_LIST) == 1: 
        axes = np.array([axes])

    for i, model_type in enumerate(MODEL_TYPE_LIST):
        for j, mod in enumerate(MODALITY_LIST):
            ax = axes[i, j] if len(MODEL_TYPE_LIST) > 1 else axes[j]
            subset = agg[(agg['model_type'] == model_type) & (agg['modality'] == mod)].copy()
            if not subset.empty:
                subset['dilemma'] = pd.Categorical(subset['dilemma'], categories=SORTED_DILEMMAS, ordered=True)
                subset['attribute'] = pd.Categorical(subset['attribute'], categories=all_attributes, ordered=True)
                pivot = subset.pivot(index='dilemma', columns='attribute', values='log_or')
                pivot = pivot.reindex(index=SORTED_DILEMMAS, columns=all_attributes)
                sns.heatmap(pivot, ax=ax, cmap='RdBu_r', center=0, 
                            vmin=-global_max, vmax=global_max, 
                            annot=True, fmt='.1f', annot_kws={"size": 8},
                            cbar=False, mask=pivot.isna(),
                            linewidths=0.5, linecolor='white')
            else: 
                ax.text(0.5, 0.5, 'No Significant Bias', 
                        ha='center', va='center', color='grey', 
                        transform=ax.transAxes, fontsize=12)
                ax.set_facecolor('#F0F0F0')
                ax.tick_params(left=False, bottom=False)

            ax.set_title(f"{model_type} - {mod}", fontsize=14, fontweight='bold', pad=2)
            if j == 0: 
                ax.set_ylabel('Dilemma (MFT Sorted)')
            else:
                ax.set_ylabel("")
            if i == len(MODEL_TYPE_LIST) - 1: 
                ax.set_xlabel('Attribute')
                ax.set_xticks(np.arange(len(all_attributes)) + 0.5)
                ax.set_xticklabels(all_attributes, rotation=45, ha='right')
            else:
                ax.set_xlabel("")
                ax.set_xticks([])

    plt.tight_layout(rect=[0, 0, 0.92, 0.97], h_pad=2.0)

    cbar_ax = fig.add_axes([0.93, 0.3, 0.015, 0.4])
    norm = plt.Normalize(-global_max, global_max)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label='Log10 Odds Ratio (Bias Strength)')
    
    plt.suptitle("Attribute-Level Bias Intensity", y=0.98, fontsize=18, fontweight='bold')
    plt.savefig(os.path.join(save_dir, '2_attribute_faceted_heatmap.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_mechanism_composition(df, save_dir):
    """
    Layer 3: Mechanism Composition Analysis.
    """
    print("Generating mechanism composition analysis...")
    sig_data = df[(df['p_value'] < 0.05) & (df['condition'] != 'Intercept')].copy()
    if sig_data.empty: return
    sig_data['magnitude'] = np.log10(sig_data['odds_ratio'].clip(1e-6, 1e6)).abs()
    
    # 1. Sum of magnitudes per (Model, Modality, MFT, Level) - Numerator Component
    agg_sum = sig_data.groupby(['model_type', 'modality', 'mft_dimension', 'level'], observed=True)['magnitude'].sum().reset_index(name='sum_mag')
    
    # 2. Count of significant items per (Model, Modality, MFT) - Denominator
    agg_cnt = sig_data.groupby(['model_type', 'modality', 'mft_dimension'], observed=True)['magnitude'].count().reset_index(name='sig_count')
    
    # 3. Merge to compute Mean Contribution per Level
    merged = pd.merge(agg_sum, agg_cnt, on=['model_type', 'modality', 'mft_dimension'])
    merged['mean_contribution'] = merged['sum_mag'] / merged['sig_count']
    
    mfts = DILEMMA_DIMENSION_ORDER
    levels = [l for l in LEVEL_PALETTE.keys() if l != 'Intercept']

    width = 0.25
    offsets = {'Text': -width,  'Caption': 0, 'Image': width}
    hatch_map = {'Text': '', 'Caption': '///', 'Image': '...'}

    for model_type in MODEL_TYPE_LIST:
        model_data = merged[merged['model_type'] == model_type].copy()
        
        fig, ax = plt.subplots(figsize=(20, 10), constrained_layout=True)
        x = np.arange(len(mfts))
        bottoms = {m: np.zeros(len(mfts)) for m in offsets.keys()}
        
        for level in levels:
            color = LEVEL_PALETTE[level]
            for mode in MODALITY_LIST:
                y_vals = []
                for mft in mfts:
                    row = model_data[(model_data['mft_dimension'] == mft) & 
                                     (model_data['modality'] == mode) & 
                                     (model_data['level'] == level)]
                    y_vals.append(row['mean_contribution'].values[0] if not row.empty else 0)
                
                ax.bar(x + offsets[mode], y_vals, width, bottom=bottoms[mode],
                       color=color, edgecolor='white', linewidth=0.5, hatch=hatch_map[mode])
                bottoms[mode] += np.array(y_vals)

        ax.set_xticks(x)
        ax.set_xticklabels(mfts, fontsize=11, rotation=45, ha='right')
        ax.set_xlabel("Conflict of MFT", fontsize=12)
        ax.set_ylabel("Mean Bias Intensity (Avg |Log OR| per Significant)", fontsize=12)

        color_handles = [mpatches.Patch(color=c, label=l) for l, c in LEVEL_PALETTE.items() if l != 'Intercept']
        mode_handles = [mpatches.Patch(facecolor='#cccccc', edgecolor='black', hatch=h, label=m) for m, h in hatch_map.items()]
        ax.legend(handles=color_handles + mode_handles, title="Level & Modality",
                  loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
        
        fig.suptitle(f"Model: {model_type} - Mean Bias Intensity & Composition", 
                    fontsize=16, y=1.02) 
        plt.savefig(os.path.join(save_dir, f'3_mechanism_composition_{model_type}.pdf'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    print("Generating Care-Care summary across all models...")
    
    care_care_data = merged[merged['mft_dimension'] == 'Care vs Care'].copy()
    
    if not care_care_data.empty:
        fig, ax = plt.subplots(figsize=(16, 8), constrained_layout=True)
        
        models = MODEL_TYPE_LIST
        x = np.arange(len(models))
        
        bottoms = {m: np.zeros(len(models)) for m in offsets.keys()}
        
        for level in levels:
            if level not in LEVEL_PALETTE: continue
            color = LEVEL_PALETTE[level]
            
            for mode in MODALITY_LIST:
                y_vals = []
                for model in models:
                    row = care_care_data[(care_care_data['model_type'] == model) & 
                                         (care_care_data['modality'] == mode) & 
                                         (care_care_data['level'] == level)]
                    
                    val = row['mean_contribution'].values[0] if not row.empty else 0
                    y_vals.append(val)
                
                ax.bar(x + offsets[mode], y_vals, width, bottom=bottoms[mode],
                       color=color, edgecolor='white', linewidth=0.5, hatch=hatch_map[mode])
                
                bottoms[mode] += np.array(y_vals)

        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=12, ha='right')
        
        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel("Mean Bias Intensity (Avg |Log OR| per Significant)", fontsize=12)
        
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        color_handles = [mpatches.Patch(color=c, label=l) for l, c in LEVEL_PALETTE.items() if l != 'Intercept']
        mode_handles = [mpatches.Patch(facecolor='#cccccc', edgecolor='black', hatch=h, label=m) for m, h in hatch_map.items()]
        ax.legend(handles=color_handles + mode_handles, title="Level & Modality",
                  loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
        
        fig.suptitle("Care-Care Conflict: Mean Bias Intensity & Composition Across Models", 
                    fontsize=16, y=1.02)
        
        plt.savefig(os.path.join(save_dir, '3_mechanism_composition_care_care_summary.pdf'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def plot_mft_analysis(df, save_dir):
    print("Generating MFT gap analysis...")
    df_no_intercept = df[(df['p_value'] < 0.05) & (df['condition'] != 'Intercept')].copy()
    if df_no_intercept.empty: return

    df_no_intercept['log_or'] = np.log10(df_no_intercept['odds_ratio'].clip(1e-6, 1e6))
    pivot = df_no_intercept.pivot_table(
        index=['model_type', 'dilemma', 'attribute', 'condition', 'mft_dimension'], 
        columns='modality', 
        values='log_or'
    ).dropna(subset=['Text', 'Caption', 'Image'], how='any')
    for m in MODALITY_LIST:
        if m not in pivot.columns: 
            pivot[m] = np.nan

    pivot['Info_Gap'] = (pivot['Text'] - pivot['Caption']).abs()
    pivot['Modality_Gap'] = (pivot['Caption'] - pivot['Image']).abs()
    
    melted = pivot[['Info_Gap', 'Modality_Gap']].reset_index().melt(
        id_vars=['mft_dimension', 'model_type', 'attribute'], 
        value_vars=['Info_Gap', 'Modality_Gap'], 
        var_name='Comparison', value_name='Gap'
    )
    idx = melted.groupby(['mft_dimension', 'Comparison'])['Gap'].idxmax()
    agg = melted.loc[idx].copy()
    
    agg['mft_dimension'] = pd.Categorical(agg['mft_dimension'], categories=DILEMMA_DIMENSION_ORDER, ordered=True)
    agg = agg.sort_values(['mft_dimension', 'Comparison'])
    
    fig, ax = plt.subplots(figsize=(16, 9))
    sns.set_theme(style="whitegrid")

    hue_order = ['Info_Gap', 'Modality_Gap']
    palette = {'Info_Gap': '#8da0cb', 'Modality_Gap': '#fc8d62'}
    
    sns.barplot(data=agg, x='mft_dimension', y='Gap', hue='Comparison', 
                palette=palette, hue_order=hue_order, 
                order=DILEMMA_DIMENSION_ORDER, ax=ax, edgecolor='white')
    
    n_hues = len(hue_order)
    bar_width = 0.8 / n_hues

    x_map = {cat: i for i, cat in enumerate(DILEMMA_DIMENSION_ORDER)}

    for _, row in agg.iterrows():
        mft = row['mft_dimension']
        comp = row['Comparison']
        gap_val = row['Gap']
        model = row['model_type']
        attr = row['attribute']
        
        if mft not in x_map or pd.isna(gap_val): 
            continue
        
        # Calculate X coordinate: Center index + offset
        x_idx = x_map[mft]
        hue_idx = hue_order.index(comp)
        # Offset logic: Center is 0. If 2 bars, indices are 0 and 1. 
        # 0 -> -0.2, 1 -> +0.2 (approx)
        offset = (hue_idx - (n_hues - 1) / 2) * bar_width
        x_pos = x_idx + offset
        
        # Format text
        label_text = f"{gap_val:.2f}\n{model}\n({attr})"
        
        ax.text(x_pos, gap_val + 0.005, label_text, 
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333333')
    
    plt.title('Moral Foundation: Maximum Gap Magnitude & Contributing Factors', fontsize=18, pad=20)
    plt.xlabel('Conflict of MFT', fontsize=14)
    plt.ylabel('Maximum Absolute Log Odds Difference', fontsize=14)
    ax.set_xticks(np.arange(len(DILEMMA_DIMENSION_ORDER)))
    ax.set_xticklabels(DILEMMA_DIMENSION_ORDER, rotation=45, ha='right', fontsize=11)

    ax.legend(title='Comparison Type', loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
    
    plt.margins(y=0.15)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '4_mft_gap_chart.pdf'), dpi=300)
    plt.close()

def plot_top_influential_factors(df, save_dir):
    print("Generating top influential factors analysis by Feature...")
    
    sig_data = df[(df['p_value'] < 0.05) & (df['condition'] != 'Intercept')].copy()
    sig_data['log_or'] = np.log10(sig_data['odds_ratio'].clip(1e-6, 1e6))
    sig_data['magnitude'] = sig_data['log_or'].abs()
    
    def clean_label(row):
        dilemma = row['dilemma']
        s = row['condition']
        matches = re.findall(r'C\((.*?)(?:,.*?)?\)\[[ST]\.(.*?)\]', s)
        content = " & ".join([f"{m[1].strip()}" for m in matches]) if matches else s
        return f"[{row['model_type']}] {dilemma}\n{content}"

    sig_data['pretty_label'] = sig_data.apply(clean_label, axis=1)
    attributes = sig_data['attribute'].unique()
    
    for attr in attributes:
        attr_df = sig_data[sig_data['attribute'] == attr].copy()
        pivot = attr_df.pivot_table(
            index=['mft_dimension', 'pretty_label', 'level'], 
            columns='modality', 
            values='log_or',
            aggfunc='mean'
        ).reset_index()

        for mode in MODALITY_LIST:
            if mode not in pivot.columns: 
                pivot[mode] = np.nan
        
        pivot['max_mag'] = pivot[MODALITY_LIST].abs().max(axis=1)
        top_k = 25 
        top_items = (
            pivot.sort_values('max_mag', ascending=False)
            .head(top_k)
            .sort_values('max_mag', ascending=True) 
        )
        
        if top_items.empty: 
            continue

        def get_range(row):
            values = row[MODALITY_LIST].dropna()
            if len(values) >= 2:
                return values.min(), values.max()
            else:
                return np.nan, np.nan

        ranges = top_items.apply(get_range, axis=1, result_type='expand')
        top_items['min_val'] = ranges[0]
        top_items['max_val'] = ranges[1]
        
        fig_height = max(8, len(top_items) * 0.5)
        
        fig_height = max(6, len(top_items) * 0.4)
        plt.figure(figsize=(12, fig_height))
        y_pos = range(len(top_items))

        valid_mask = top_items[['min_val', 'max_val']].notna().all(axis=1)
        if valid_mask.any():
            plt.hlines(y=np.array(y_pos)[valid_mask], 
                      xmin=top_items.loc[valid_mask, 'min_val'], 
                      xmax=top_items.loc[valid_mask, 'max_val'], 
                      color='#bbbbbb', alpha=0.6, linewidth=1.5, zorder=1)
        
        colors = top_items['level'].map(LEVEL_PALETTE).fillna('#999999')
        
        plt.hlines(y=y_pos, xmin=top_items['min_val'], xmax=top_items['max_val'], 
                   color='#888888', alpha=0.8, linewidth=2.0, zorder=1)
        
        colors = top_items['level'].map(LEVEL_PALETTE).fillna('#999999')
        
        if 'Text' in top_items.columns:
            mask = top_items['Text'].notna()
            plt.scatter(top_items.loc[mask, 'Text'], np.array(y_pos)[mask], 
                       c=colors[mask], marker='o', s=100, zorder=2, label='Text Mode')
        if 'Caption' in top_items.columns:
            mask = top_items['Caption'].notna()
            plt.scatter(top_items.loc[mask, 'Caption'], np.array(y_pos)[mask], 
                       c=colors[mask], marker='s', s=100, zorder=2, label='Caption Mode')
        if 'Image' in top_items.columns:
            mask = top_items['Image'].notna()
            plt.scatter(top_items.loc[mask, 'Image'], np.array(y_pos)[mask], 
                       c=colors[mask], marker='D', s=100, zorder=2, label='Image Mode')
        
        plt.yticks(y_pos, top_items['pretty_label'], fontsize=10)
        plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
        plt.xlabel("Log10 Odds Ratio (Bias Direction)")
        
        level_handles = [Line2D([0], [0], marker='s', color='w', markerfacecolor=c, markersize=10, label=l) 
                         for l, c in LEVEL_PALETTE.items()]
        modality_handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=10, label='Text (Circle)'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='grey', markersize=10, label='Caption (Square)'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='grey', markersize=10, label='Image (Diamond)'),
        ]
        
        plt.legend(handles=level_handles + modality_handles, loc='upper right', title="Mechanism & Modality")
        
        plt.title(f"Top Influential Factors: {attr.capitalize()} (Across Models)", fontsize=16, pad=20)
        plt.tight_layout(rect=[0, 0, 1, 0.95]) 
        
        plt.savefig(os.path.join(save_dir, f'5_top_factors_{attr}.pdf'), dpi=300)
        plt.close()

def main():
    result_path = f"{ROOT}/../results/single_feature/analyze_results"
    data_list = []
    
    for model_str in MODEL_LIST:
        file_path = f"{result_path}/character_factor_{model_str}.yaml"
        prepare_results(model_str, file_path, data_list)
    
    if not data_list:
        print("No data loaded.")
        return

    df = pd.DataFrame(data_list)
    vis_dir = f'{ROOT}/../visualization/single_feature/character_factor'
    if not os.path.exists(vis_dir): os.makedirs(vis_dir)
    
    plot_bias_heatmap(df, vis_dir)
    plot_attribute_faceted_heatmap(df, vis_dir)
    plot_mechanism_composition(df, vis_dir)
    plot_mft_analysis(df, vis_dir)
    plot_top_influential_factors(df, vis_dir)
    
    print(f"Visualization complete. Results saved to {vis_dir}")

if __name__ == '__main__':
    main()