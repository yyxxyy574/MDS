import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml

from config.constants import ROOT
from visualization.utils import parse_model_info, MODALITY_LIST, MODEL_LIST

FACTOR_ORDER = ['personal_force', 'intention_of_harm', 'self_benefit']

def set_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

def load_interaction_stats():
    analyze_dir = os.path.join(ROOT, "..", "results", "quantity", "analyze_results")
    all_curves = []
    all_slopes = []
    
    print(f"Loading interaction results from {analyze_dir}...")
    
    processed = set()
    for model_str in MODEL_LIST:
        model_name, modality = parse_model_info(model_str)
        filename = f"interaction_stats_{model_str}.yaml"
        file_path = os.path.join(analyze_dir, filename)
        
        if file_path in processed or not os.path.exists(file_path):
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            processed.add(file_path)
            
            for fac, content in data.get('interactions', {}).items():
                # Curves
                for r in content['curve']:
                    r['Model'] = model_name
                    r['Modality'] = modality
                    all_curves.append(r)
                # Slopes
                for s in content['slopes']:
                    s['Model'] = model_name
                    s['Modality'] = modality
                    all_slopes.append(s)
                    
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    return pd.DataFrame(all_curves), pd.DataFrame(all_slopes)

def plot_interaction_curves(df, output_dir):
    """
    Plot Action Rate vs Net_Benefit, split by Factor Value (0 vs 1).
    Shows how factors affect the Utility Curve (S-Curve).
    """
    if df.empty: return
    
    models = df['Model'].unique()
    
    for model in models:
        df_model = df[df['Model'] == model].sort_values('Net_Benefit')
        
        if df_model.empty: continue
        
        g = sns.FacetGrid(
            df_model, 
            col="Factor_Name", 
            col_order=FACTOR_ORDER,
            height=4, 
            aspect=1.2
        )
        
        g.map_dataframe(
            sns.lineplot, 
            x="Net_Benefit",
            y="Action", 
            hue="Factor_Value", 
            style="Modality", 
            hue_order=[0, 1],
            style_order=MODALITY_LIST,
            markers=True,
            palette={0: '#2ca02c', 1: '#d62728'}, 
            dashes={'Text': (None, None), 'Image': (2, 2), 'Caption': (3, 1)},
            linewidth=2.5,
            markersize=8
        )
        
        g.map(plt.axvline, x=0, linestyle='--', color='gray', alpha=0.5)

        g.set_titles("{col_name}")
        g.set_axis_labels("Net Benefit (Saved - Sacrificed)", "Action Rate")
        g.set(ylim=(-0.05, 1.05))
        
        # Ensure x-ticks show the full range
        all_ticks = sorted(df_model['Net_Benefit'].unique())
        g.set(xticks=all_ticks)
        
        g.add_legend()
        
        plt.subplots_adjust(top=0.85)
        g.fig.suptitle(f'{model}: Interaction Analysis (Net Benefit)', fontsize=15, fontweight='bold')
        
        clean_name = model.replace('/', '_')
        save_path = os.path.join(output_dir, f"1_interaction_curve_{clean_name}.pdf")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved interaction curve for {model}")

def plot_slope_impact(df, output_dir):
    """
    Bar chart comparing Slopes (Utility Sensitivity) for Factor=0 vs Factor=1.
    """
    if df.empty: return
    
    factors = ['personal_force', 'intention_of_harm', 'self_benefit']
    df = df[df['Factor_Name'].isin(factors)]
    
    models = df['Model'].unique()
    
    for model in models:
        df_model = df[df['Model'] == model]
        if df_model.empty: continue

        g = sns.catplot(
            data=df_model,
            x="Modality", 
            y="Slope", 
            hue="Factor_Value", 
            col="Factor_Name",
            col_order=FACTOR_ORDER,
            order=MODALITY_LIST,
            kind="bar",
            height=5, 
            aspect=0.8,
            palette={0: '#2ca02c', 1: '#d62728'},
            alpha=0.8,
            legend=True
        )
        
        for ax in g.axes.flat:
            ax.axhline(0, color='gray', linestyle='--', linewidth=1)
            
            # Hatching for different modalities to make them distinct in B&W too
            factor_name = ax.get_title()
            for i, bar in enumerate(ax.patches):
                # Simple logic to guess if it's a different modality bar based on x-position or order
                # (Seaborn makes this tricky, but hue is distinct enough usually)
                pass

        g.set_axis_labels("", "Utility Sensitivity (Slope)")
        g.set_titles("{col_name}")
        
        sns.move_legend(g, "upper right", bbox_to_anchor=(1, 1), title="Factor Presence\n(0=No, 1=Yes)")
        
        plt.subplots_adjust(top=0.85)
        g.fig.suptitle(f"{model}: Factor Impact on Utility Sensitivity", fontsize=16, fontweight='bold')
        
        clean_name = model.replace('/', '_')
        save_path = os.path.join(output_dir, f"2_slope_impact_{clean_name}.pdf")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved slope impact chart for {model}")

def main():
    viz_dir = f'{ROOT}/../visualization/quantity/quantity_conceptual'
    os.makedirs(viz_dir, exist_ok=True)
    
    set_style()
    df_curves, df_slopes = load_interaction_stats()
    
    if df_curves.empty:
        print("No interaction data found. Run analyze_quantity_conceptual.py first.")
        return
        
    print("Generating Interaction Curves (Net Benefit)...")
    plot_interaction_curves(df_curves, viz_dir)
    
    print("Generating Slope Impact Chart...")
    plot_slope_impact(df_slopes, viz_dir)
    
    print(f"Done. Files saved to {viz_dir}")

if __name__ == "__main__":
    main()