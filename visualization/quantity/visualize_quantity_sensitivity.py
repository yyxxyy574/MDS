import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as mtick
import seaborn as sns
import os
import yaml

from config.constants import ROOT
from visualization.utils import parse_model_info, MODALITY_LIST, MODALITY_PALETTE, MODEL_LIST

def get_model_order(available_models):
    """
    Returns a list of model names sorted according to their appearance in MODEL_LIST.
    """
    ordered = []
    seen = set()
    
    for model_str in MODEL_LIST:
        model_name, _ = parse_model_info(model_str)
        if model_name in available_models and model_name not in seen:
            ordered.append(model_name)
            seen.add(model_name)
    
    for model in available_models:
        if model not in seen:
            ordered.append(model)
            
    return ordered

def set_style():
    """Sets publication-quality plotting style."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.titlesize'] = 18
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 14

def load_all_stats():
    """
    Iterates through MODEL_LIST, loads corresponding YAML analysis files,
    and aggregates them into DataFrames.
    """
    all_dilemma_stats = []
    all_global_stats = []
    all_slopes = []
    all_refusals = []
    
    analyze_dir = os.path.join(ROOT, "..", "results", "quantity", "analyze_results")
    processed_files = set()

    for model_str in MODEL_LIST:
        model_name, modality = parse_model_info(model_str)
        filename = f"quantity_sensitivity_{model_str}.yaml"
        file_path = os.path.join(analyze_dir, filename)
        
        if file_path in processed_files:
            continue
            
        if not os.path.exists(file_path):
            continue
            
        try:
            print(f"Loading analysis results from {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            processed_files.add(file_path)
            
            def append_data(source_list, target_list):
                if source_list:
                    df = pd.DataFrame(source_list)
                    df['Model'] = model_name
                    df['Modality'] = modality
                    target_list.append(df)

            append_data(data.get('dilemma_stats'), all_dilemma_stats)
            append_data(data.get('global_stats'), all_global_stats)
            append_data(data.get('slopes'), all_slopes)

            data_info = data.get('data_info', {})
            refusal_rate = data_info.get('refusal_rate', 0.0) * 100
            
            all_refusals.append({
                'Model': model_name,
                'Modality': modality,
                'Refusal Rate': refusal_rate
            })
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    df_dilemma = pd.concat(all_dilemma_stats, ignore_index=True) if all_dilemma_stats else pd.DataFrame()
    df_global = pd.concat(all_global_stats, ignore_index=True) if all_global_stats else pd.DataFrame()
    df_slope = pd.concat(all_slopes, ignore_index=True) if all_slopes else pd.DataFrame()
    df_refusal = pd.DataFrame(all_refusals) if all_refusals else pd.DataFrame()
    
    return df_dilemma, df_global, df_slope, df_refusal

def plot_refusal_rate_comparison(df, output_dir):
    """
    [Figure 0] Safety Refusal Rate Comparison.
    Bar chart showing refusal rates across models and modalities.
    """
    if df.empty: return

    plt.figure(figsize=(15, 6))
    
    unique_models = set(df['Model'].unique())
    model_order = get_model_order(unique_models)
    
    # Draw Barplot
    ax = sns.barplot(
        data=df, 
        x='Model', 
        y='Refusal Rate', 
        hue='Modality', 
        palette=MODALITY_PALETTE,
        order=model_order,
        hue_order=MODALITY_LIST,
        edgecolor='black'
    )
    
    # Format Y-axis as percentage
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    
    # Add values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f%%', padding=3, fontsize=10)
    
    plt.title('Safety Filter Trigger Rate (Refusal Rate)', fontsize=18, fontweight='bold')
    plt.ylabel('Refusal Rate (%)', fontsize=14)
    plt.xlabel('', fontsize=14)
    
    plt.xticks(rotation=0, ha='right', fontsize=12)
    plt.legend(title='Modality', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "0_refusal_rate_comparison.pdf")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path}")

def plot_global_curve_by_model(df, output_dir):
    """
    [Figure 1] Global Curve - Faceted by Model.
    """
    if df.empty: return

    sns.set_context("talk", font_scale=1.2) 
    
    df = df.sort_values('Net_Benefit')
    unique_models = set(df['Model'].unique())
    model_order = get_model_order(unique_models)
    
    n_models = len(model_order)
    n_cols = n_models // 2 if n_models > 2 else 3

    hue_kws = {
        'linewidth': [10, 8, 8],           
        'linestyle': ['-', '-', '--'],    
        'alpha': [0.5, 1.0, 1.0]
    }

    g = sns.FacetGrid(
        df, 
        col="Model", 
        col_wrap=n_cols, 
        col_order=model_order,
        hue="Modality", 
        hue_order=MODALITY_LIST,
        hue_kws=hue_kws,
        sharey=True, 
        height=3.2,
        aspect=1.2,
        palette=MODALITY_PALETTE
    )
    
    g.map(sns.lineplot, "Net_Benefit", "Action", marker="o", markersize=15)
    
    g.map(plt.axvline, x=0, linestyle=':', color='gray', alpha=0.5, linewidth=4)
    
    ratio_map = {
        -9: "1:10", -4: "1:5", -1: "1:2", 0: "1:1",
         1: "2:1", 4: "5:1", 9: "10:1"
    }
    
    data_values = sorted(df['Net_Benefit'].unique())
    ticks_to_use = [v for v in data_values if v in ratio_map]
    labels_to_use = [ratio_map[v] for v in ticks_to_use]
    
    g.set(xticks=ticks_to_use)
    
    g.set_xticklabels(labels_to_use, rotation=90, ha='center', fontsize=16)
    
    g.set_titles("{col_name}", size=24, fontweight='bold')
    
    g.set(xlim=(min(ticks_to_use)-0.8, max(ticks_to_use)+0.8))
    
    g.set_axis_labels("Ratio (Saved : Sacrificed)", "Action Probability", fontsize=18)

    g.set(ylim=(-0.02, 1.05))

    # 调整子图间距
    g.fig.subplots_adjust(
        wspace=0.005,  # 水平间距，减小值以减少间距
        hspace=0.2   # 垂直间距，减小值以减少间距
    )

    # 增大y轴标签和刻度大小
    for ax in g.axes:
        ax.tick_params(axis='y', labelsize=18)  # 增大y轴刻度标签大小
        ax.set_ylabel(ax.get_ylabel(), fontsize=20)  # 增大y轴标题大小
        
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.yaxis.get_offset_text().set_visible(False)
        ax.ticklabel_format(style='plain', axis='y', useOffset=False)

        ax.tick_params(right=False, labelright=False)
    
    target_ax_idx = n_cols - 1
    if target_ax_idx < len(g.axes):
        target_ax = g.axes[target_ax_idx]
        
        legend_elements = [
            Line2D([0], [0], color=MODALITY_PALETTE['Text'], lw=8, linestyle='-', alpha=0.5, label='Text'),
            Line2D([0], [0], color=MODALITY_PALETTE['Caption'], lw=6, linestyle='-', label='Caption'),
            Line2D([0], [0], color=MODALITY_PALETTE['Image'], lw=6, linestyle='--', label='Image'),
        ]
        
        target_ax.legend(
            handles=legend_elements, 
            title="Mode", 
            loc='upper right', 
            fontsize=18, 
            title_fontsize=20, 
            frameon=True,
            handlelength=3
        )
    
    save_path = os.path.join(output_dir, "1_global_curve_by_model.pdf")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    save_path = os.path.join(output_dir, "1_global_curve_by_model.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # 恢复默认设置 (可选，避免影响后续画图)
    sns.set_context("notebook")
    print(f"Saved {save_path}")

def plot_dilemma_breakdown_per_model(df, output_dir):
    """
    [Figure 2] Dilemma Breakdown - One File per Model.
    """
    if df.empty: return
    
    unique_models = set(df['Model'].unique())
    models_sorted = get_model_order(unique_models)
    
    # Define style strategy (Consistent with Figure 1)
    palette = MODALITY_PALETTE
    hue_kws = {
        'linewidth': [5, 3, 3],           
        'linestyle': ['-', '-', '--'],    
        'alpha': [0.6, 0.8, 1.0]          
    }
    
    for model in models_sorted:
        df_model = df[df['Model'] == model].sort_values('Net_Benefit')
        
        if df_model.empty: continue
        
        g = sns.FacetGrid(
            df_model, 
            col="Dilemma", 
            col_wrap=5, 
            hue="Modality",
            hue_order=MODALITY_LIST,
            hue_kws=hue_kws,
            sharey=True,
            height=3, 
            aspect=1.1,
            palette=palette
        )
        
        g.map(sns.lineplot, "Net_Benefit", "Action", marker="o", markersize=6)
        
        # [MODIFIED] Add vertical line at 0
        g.map(plt.axvline, x=0, linestyle=':', color='gray', alpha=0.5, linewidth=1)
        
        g.set_titles("{col_name}", size=11, fontweight='bold')
        g.set_axis_labels("Net Benefit", "Action Rate")
        g.set(ylim=(-0.05, 1.05))

        # Manual Legend matching the new styles
        legend_elements = [
            Line2D([0], [0], color='#1f77b4', lw=4, linestyle='-', alpha=0.8, label='Text'),
            Line2D([0], [0], color='#ff7f0e', lw=3, linestyle='--', label='Caption'),
            Line2D([0], [0], color='#d62728', lw=3, linestyle='-', label='Image'),
        ]
        g.add_legend(handles=legend_elements, title="Modality", fontsize=11)
        
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle(f'{model}: Scenario Breakdown', fontsize=16, fontweight='bold')
        
        clean_name = model.replace('/', '_')
        save_path = os.path.join(output_dir, f"2_breakdown_{clean_name}.pdf")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved Breakdown for {model}")

def plot_sensitivity_slope_comparison(df, output_dir):
    """
    [Figure 3] Sensitivity Slope Comparison (Model vs Model).
    [NOTE] Slopes are now calculated against Net Benefit.
    """
    if df.empty: return

    plt.figure(figsize=(10, 7))
    
    unique_models = set(df['Model'].unique())
    model_order = get_model_order(unique_models)
    
    sns.boxplot(
        data=df,
        x='Model',
        y='Slope',
        hue='Modality',
        palette=MODALITY_PALETTE,
        order=model_order,
        hue_order=MODALITY_LIST,
        width=0.9,
        fliersize=0, 
        boxprops=dict(alpha=.8)
    )
    plt.ylim(bottom=-0.04)
    # plt.axhline(0, color='gray', linestyle='--', linewidth=3)
    
    # plt.title('Quantity Sensitivity Slope (Action vs Net Benefit)', fontsize=18, fontweight='bold')
    plt.ylabel('Marginal Sensitivity (Slope k)', fontsize=18)
    plt.xlabel('', fontsize=20)
    
    plt.xticks(rotation=15, ha='right', fontsize=16)
    plt.yticks(fontsize=14)
    plt.legend(title='Mode', loc='upper left', fontsize=15, title_fontsize=16)
    
    # plt.text(0.01, 0.02, "y=0: Insensitive", 
    #          transform=plt.gca().transAxes, fontsize=16, color='gray')

    plt.tight_layout()

    save_path = os.path.join(output_dir, "3_sensitivity_slope_comparison.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    save_path = os.path.join(output_dir, "3_sensitivity_slope_comparison.pdf")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved {save_path}")

def main():
    viz_dir = f'{ROOT}/../visualization/quantity/quantity_sensitivity'
    os.makedirs(viz_dir, exist_ok=True)
    
    print("Starting Quantity Visualization (Per Model) with Net Benefit...")
    set_style()
    
    # 1. Load Data
    df_dilemma, df_global, df_slope, df_refusal = load_all_stats()
    
    if df_global.empty:
        print("No data found. Please run analysis scripts first.")
        return

    # 2. Generate Plots
    # plot_refusal_rate_comparison(df_refusal, viz_dir)
    plot_global_curve_by_model(df_global, viz_dir)
    # plot_dilemma_breakdown_per_model(df_dilemma, viz_dir)
    plot_sensitivity_slope_comparison(df_slope, viz_dir)
    
    print(f"\nVisualization Complete. Files saved to: {viz_dir}")

if __name__ == "__main__":
    main()