import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
import yaml
import os
from math import pi
from matplotlib.lines import Line2D

from config.constants import ROOT
from visualization.utils import parse_model_info, MODEL_LIST, MODEL_TYPE_LIST, MODALITY_LIST, MODALITY_PALETTE

def prepare_results(model_str, data_list):
    """Load and process YAML results."""
    model_type, modality = parse_model_info(model_str)
    
    path = f"{ROOT}/../results/single_feature/analyze_results/general_stats_{model_str}.yaml"
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            data['model_type'] = model_type
            data['modality'] = modality
            data['model_str'] = f"{model_type} - {modality}"
            data_list.append(data)
    else:
        print(f"Warning: Results not found for {model_str}")

def plot_refusal_rate(data_list, save_dir):
    """
    [Figure 0] Safety Refusal Rate Comparison.
    Visualizes the global refusal rate for each model across different modalities.
    """
    rows = []
    for entry in data_list:
        data_info = entry.get('data_info', {})
        refusal_rate = data_info.get('global_refusal_rate', 0.0)
        
        rows.append({
            'Model': entry['model_type'],
            'Modality': entry['modality'],
            'Refusal_Rate': refusal_rate * 100  # Convert to percentage
        })
    
    if not rows: return
    df = pd.DataFrame(rows)
    
    # Ensure Modality Order matches MODALITY_LIST (Text, Image, Caption)
    hue_order = [m for m in MODALITY_LIST if m in df['Modality'].unique()]
    
    # Dynamic Figure Size
    # Base width 8, plus 1.2 inch for each model to ensure spacing
    n_models = len(MODEL_TYPE_LIST)
    fig_width = max(15, n_models * 2.5)
    plt.figure(figsize=(fig_width, 7))
    
    # Draw Barplot
    ax = sns.barplot(
        data=df, 
        x='Model', 
        y='Refusal_Rate', 
        hue='Modality', 
        palette=MODALITY_PALETTE,
        order=MODEL_TYPE_LIST,     # Apply Model Order
        hue_order=hue_order,   # Apply Modality Order
        edgecolor='black'
    )
    
    # Format Y-axis as percentage
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    
    # Add values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=9)
        
    plt.title('Safety Filter Trigger Rate (Refusal Rate)', fontsize=16, fontweight='bold')
    plt.ylabel('Refusal Rate (%)', fontsize=14)
    plt.xlabel('Model', fontsize=14)
    
    # Rotate X-axis labels to prevent overlap
    plt.xticks(rotation=0, ha='right', fontsize=12)
    
    plt.legend(title='Modality', bbox_to_anchor=(1.01, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/general_refusal_rate.pdf", dpi=300)
    plt.close()
    print(f"Saved Refusal Rate Plot.")

def plot_radar_compass(data_list, save_dir):
    """
    [Figure 1a] Radar Chart for Dimension Preferences with Tangential Labels.
    """
    rows = []
    for entry in data_list:
        scores = entry.get('dimension_scores', {})
        for dim, stats in scores.items():
            rows.append({
                'Model': entry['model_type'],
                'Modality': entry['modality'],
                'Dimension': dim,
                'Score': stats['win_rate']
            })
    
    if not rows: return
    df = pd.DataFrame(rows)

    categories = ['Care', 'Fairness', 'Loyalty', 'Authority', 'Purity']
    N = len(categories)
    # 基础弧度
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles_val = angles + angles[:1]

    models = df['Model'].unique()
    n_cols = max(1, len(models) // 2)
    n_rows = 2 if len(models) > 1 else 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5.5*n_rows), 
                             subplot_kw=dict(polar=True))
    fig.subplots_adjust(
        wspace=0.002,  # 水平间距，减小值以减少间距
        hspace=0.2   # 垂直间距，减小值以减少间距
    )
    axes = np.atleast_1d(axes).flatten()
    
    for ax, model_type in zip(axes, models):
        # 1. 设置坐标系：12点钟方向开始，顺时针
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)

        # 2. 关键：清除默认标签，手动添加
        ax.set_xticks(angles)
        ax.set_xticklabels([]) # 清空默认文字

        for i, angle in enumerate(angles):
            # 计算旋转角度（角度制）
            # 顺时针坐标系下，切线旋转角 = -当前角度
            angle_deg = np.rad2deg(angle)
            rotation = -angle_deg 
            
            # 防倒置逻辑：如果文字在左半球或底部，翻转180度提高可读性
            if 90 < angle_deg % 360 < 270:
                rotation += 180
            
            # 使用 ax.text 手动放置标签
            # r=1.1 表示放在半径 1.0 之外一点点
            ax.text(angle, 1.15, categories[i], 
                    rotation=rotation,
                    rotation_mode='anchor', # 确保旋转轴心正确
                    ha='center', va='center',
                    size=30, fontweight='bold', color='#333333')

        # 3. 辅助线与范围
        ax.set_rlabel_position(0)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.4, 0.8])
        ax.set_yticklabels(["0.4", "0.8"], color="grey", size=14)
        ax.set_title(model_type, weight='black', size=32, pad=45)
        
        # 4. 绘图数据
        for modality in MODALITY_LIST:
            subset = df[(df['Model'] == model_type) & (df['Modality'] == modality)]
            if subset.empty: continue
            
            values = []
            for cat in categories:
                val = subset[subset['Dimension'] == cat]['Score'].values
                values.append(val[0] if len(val) > 0 else 0)
            values += values[:1]
            
            color = MODALITY_PALETTE.get(modality, '#333333')
            ax.plot(angles_val, values, linewidth=4, label=modality, color=color, zorder=3)
            ax.fill(angles_val, values, color=color, alpha=0.15)
        
    for i in range(len(models), len(axes)):
        fig.delaxes(axes[i])
    
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        legend = fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=30, 
                   frameon=False, bbox_to_anchor=(0.5, -0.05))
        

        for line in legend.get_lines():
            line.set_linewidth(12)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/general_radar_compass.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/general_radar_compass.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Radar Compass with tangential labels.")

def plot_severity_point(data_list, save_dir):
    """
    [Figure 2] Severity Analysis (High vs Low) - Point Plot.
    """
    rows = []
    for entry in data_list:
        dilemmas = entry.get('dilemma_stats', {})
        for name, stats in dilemmas.items():
            if stats['severity'] in ['High', 'Low']:
                rows.append({
                    'Model': entry['model_type'],
                    'Modality': entry['modality'],
                    'Severity': stats['severity'],
                    'Action_Rate': stats['action_rate']
                })
                
    if not rows: return
    df = pd.DataFrame(rows)

    markers_map = {'Text': 'o', 'Image': 's', 'Caption': '^'}
    linestyles_map = {'Text': '-', 'Image': '--', 'Caption': '-.'}

    hue_order = [m for m in MODALITY_LIST if m in df['Modality'].unique()]
    
    plt.figure(figsize=(12, 6))
    
    g = sns.catplot(
        data=df, kind="point",
        x="Severity", y="Action_Rate", hue="Modality", col="Model",
        col_wrap=4, palette=MODALITY_PALETTE,
        order=['Low', 'High'], dodge=True, 
        markers=[markers_map[m] for m in hue_order],
        linestyles=[linestyles_map[m] for m in hue_order],
        height=3.5, aspect=0.9
    )
    
    g.set_axis_labels("Severity", "Action Probability (Yes)")
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle("Dual Process Effect: Severity x Modality Interaction", fontsize=16, fontweight='bold')
    
    plt.savefig(f"{save_dir}/general_severity_interaction.pdf", dpi=300)
    plt.close()
    print(f"Saved Severity Analysis.")

def plot_robustness_scatter(data_list, save_dir):
    """
    [Figure 3] Robustness Landscape.
    """
    rows = []
    for entry in data_list:
        # Aggregate dilemma stats to get one point per model
        dilemmas = entry.get('dilemma_stats', {})
        iter_vals = [d['iter_robustness'] for d in dilemmas.values()]
        ctx_vals = [d['context_sensitivity'] for d in dilemmas.values()]
        
        rows.append({
            'Model': entry['model_type'],
            'Modality': entry['modality'],
            'Iter_Robustness': np.nanmean(iter_vals),
            'Context_Sensitivity': np.nanmean(ctx_vals)
        })
        
    df = pd.DataFrame(rows)
    
    plt.figure(figsize=(11, 9))
    markers_map = {'Text': 'o', 'Image': 's', 'Caption': '^'}
    
    # Draw arrows
    models = df['Model'].unique()
    palette = sns.color_palette("bright", n_colors=len(models))
    model_color_map = dict(zip(models, palette))
    
    # 画散点图
    sns.scatterplot(
        data=df, x='Iter_Robustness', y='Context_Sensitivity',
        hue='Model', style='Modality', 
        markers=markers_map, s=300, palette=model_color_map, # s=300 稍微把点也调大了
        zorder=10
    )

    # 画箭头 (逻辑保持不变)
    arrow_props = dict(color='gray', alpha=0.4, head_width=0.005, length_includes_head=True, zorder=1)
    for m in models:
        t = df[(df['Model']==m) & (df['Modality']=='Text')]
        i = df[(df['Model']==m) & (df['Modality']=='Image')]
        c = df[(df['Model']==m) & (df['Modality']=='Caption')]

        if t.empty or i.empty: continue
        
        xt, yt = t.iloc[0]['Iter_Robustness'], t.iloc[0]['Context_Sensitivity']
        xi, yi = i.iloc[0]['Iter_Robustness'], i.iloc[0]['Context_Sensitivity']

        line_color = model_color_map[m]
        # 注意：如果箭头看起来太细，可以增加 head_width 或 width
        arrow_props = dict(color=line_color, alpha=0.6, head_width=0.003, length_includes_head=True, zorder=1)

        if not c.empty:
            # Text -> Caption -> Image
            xc, yc = c.iloc[0]['Iter_Robustness'], c.iloc[0]['Context_Sensitivity']
            plt.arrow(xt, yt, xc-xt, yc-yt, linestyle=':', **arrow_props)
            plt.arrow(xc, yc, xi-xc, yi-yc, linestyle='-', **arrow_props)
        else:
            # Text -> Image
            plt.arrow(xt, yt, xi-xt, yi-yt, linestyle='-', **arrow_props)        

    # --- 修改部分开始 ---

    # 1. 设置坐标轴标签字体大小
    plt.xlabel('Iterative Robustness', fontsize=22, fontweight='bold')
    plt.ylabel('Context Sensitivity', fontsize=22, fontweight='bold')

    # 2. 设置刻度字体大小
    plt.tick_params(axis='both', which='major', labelsize=18)

    plt.grid(True, linestyle='--', alpha=0.75)

    # 3. 设置 Legend 位置到左下角，并增大字体
    # loc='lower left' 表示放在图内左下角
    # framealpha=0.9 给图例加个半透明背景，防止遮挡住背后的网格线太乱
    plt.legend(
        loc='lower left', 
        fontsize=16,           # 标签字体
        title_fontsize=18,     # 标题字体
        framealpha=0.9,        # 背景不透明度
        borderpad=1            # 边框内边距
    )
    
    # --- 修改部分结束 ---

    plt.tight_layout()
    plt.savefig(f"{save_dir}/general_robustness.pdf", dpi=300)
    plt.savefig(f"{save_dir}/general_robustness.png", dpi=300)
    plt.close()
    print(f"Saved Robustness.")
def main():
    data_list = []
    
    # Load all models
    for model_str in MODEL_LIST:
        prepare_results(model_str, data_list)
        
    if not data_list:
        print("No data found.")
        return

    viz_dir = f'{ROOT}/../visualization/single_feature/general_stats'
    os.makedirs(viz_dir, exist_ok=True)
    
    # plot_refusal_rate(data_list, viz_dir)
    plot_radar_compass(data_list, viz_dir)
    # plot_pairwise_bar(data_list, viz_dir)
    # plot_severity_point(data_list, viz_dir)
    plot_robustness_scatter(data_list, viz_dir)

if __name__ == '__main__':
    main()