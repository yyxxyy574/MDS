import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from config.constants import ROOT
from visualization.utils import MODALITY_LIST, MODALITY_PALETTE, MODEL_LIST, MODEL_TYPE_LIST, MODEL_NAME_LIST, parse_model_info
from visualization.single_feature.visualize_conceptual_factor import prepare_results

MODALITY_MARKERS = {'Text': 'o', 'Caption': 'd', 'Image': '>'}

MODEL_CATEGORY_MAP = {
    'LLaVA-v1.6-34B': 'Open-Weight',
    'Qwen3-VL-8B': 'Open-Weight',
    'Qwen3-VL-32B': 'Open-Weight',
    'LLaMA-3.2-90B': 'Open-Weight',
    'GPT-4o-mini': 'Proprietary',
    'Gemini-2.5-flash': 'Proprietary'
}

LINE_COLORS = {
    'Context': '#b7be8c',  # Green (Text -> Caption)
    'Modality': '#d2a6a3'  # Purple (Caption -> Image)
}

def get_base_model_name(model_str):
    return parse_model_info(model_str)[0]

def plot_dumbbell_scheme_a(df_p, save_dir, target_factor='Self Benefit'):
    """
    Scheme A: Dumbbell Plot showing the shift Text -> Caption -> Image.
    Sorted by MODEL_TYPE_LIST with Open-Weight vs Proprietary separation.
    Distinct lines for Context Shift vs Modality Shift.
    """
    print(f"Generating Scheme A (Dumbbell) for {target_factor}...")
    
    # 1. 数据准备
    df_sub = df_p[
        (df_p['factor'] == target_factor) & 
        (df_p['dilemma'] == 'total') # 使用 pooled 结果
    ].copy()
    
    if df_sub.empty:
        print(f"No data for {target_factor}")
        return

    # 使用全局定义的 MODEL_TYPE_LIST 顺序 (假设前4个是Open，后2个是Proprietary)
    available_models = set(df_sub['model_type'].unique())
    sorted_models = [m for m in MODEL_TYPE_LIST if m in available_models]
    
    # 设置画布
    fig, ax = plt.subplots(figsize=(8, 6)) # 适合半栏的比例
    
    # 2. 绘制线条和点
    for i, model in enumerate(sorted_models):
        model_data = df_sub[df_sub['model_type'] == model]
        
        # 获取各模态的值
        val_text = model_data[model_data['modality'] == 'Text']['log_odds'].values
        val_cap = model_data[model_data['modality'] == 'Caption']['log_odds'].values
        val_img = model_data[model_data['modality'] == 'Image']['log_odds'].values
        
        val_text = val_text[0] if len(val_text) > 0 else np.nan
        val_cap = val_cap[0] if len(val_cap) > 0 else np.nan
        val_img = val_img[0] if len(val_img) > 0 else np.nan
        
        # # --- 绘制连线 (Segments) ---
        
        # # Segment 1: Text -> Caption (Context Shift)
        # if not np.isnan(val_text) and not np.isnan(val_cap):
        #     ax.plot([val_text, val_cap], [i, i], 
        #             color=LINE_COLORS['Context'], alpha=0.6, zorder=1, lw=2)
            
        # # Segment 2: Caption -> Image (Modality Shift)
        # if not np.isnan(val_cap) and not np.isnan(val_img):
        #     ax.plot([val_cap, val_img], [i, i], 
        #             color=LINE_COLORS['Modality'], alpha=0.6, zorder=1, lw=2)

        # --- 绘制点 (Markers) ---
        
        # Text Point
        ax.scatter(val_text, i, color=MODALITY_PALETTE['Text'], marker=MODALITY_MARKERS['Text'], 
                   s=100, zorder=3, label='Text' if i == 0 else "")
        
        # Caption Point
        ax.scatter(val_cap, i, color=MODALITY_PALETTE['Caption'], marker=MODALITY_MARKERS['Caption'], 
                   s=80, zorder=3, label='Caption' if i == 0 else "")
        
        # Image Point
        ax.scatter(val_img, i, color=MODALITY_PALETTE['Image'], marker=MODALITY_MARKERS['Image'], 
                   s=120, zorder=3, label='Image' if i == 0 else "")

    # 3. 添加分隔线和标注 (Open/Proprietary)
    if len(sorted_models) >= 5: 
        split_index = 3.5
        ax.axhline(y=split_index, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # Open-Weight 标注
        ax.text(1.02, 1.5, 'Open-Weight', transform=ax.get_yaxis_transform(), 
                rotation=270, va='center', ha='left', fontsize=12, fontweight='bold', color='#555555')
        
        # Proprietary 标注
        ax.text(1.02, 4.5, 'Proprietary', transform=ax.get_yaxis_transform(), 
                rotation=270, va='center', ha='left', fontsize=12, fontweight='bold', color='#555555')

    # 4. 美化
    ax.set_yticks(range(len(sorted_models)))
    ax.set_yticklabels(sorted_models, fontsize=11, fontweight='bold')
    ax.set_xlabel('Log Odds (Regression Coefficient)', fontsize=12)
    
    # 翻转 Y 轴，让列表第一个元素(Open-Weight)在顶部
    ax.invert_yaxis()
    
    # 标题
    readable_title = target_factor.replace("_", " ").title()
    ax.set_title(f"Visual Shift in {readable_title}", fontsize=14, fontweight='bold', pad=15)
    
    # 垂直零线
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    
    # 5. 更新图例 (包含点和线)
    handles = [
        # Dots
        mlines.Line2D([], [], color=MODALITY_PALETTE['Text'], marker=MODALITY_MARKERS['Text'], linestyle='None', markersize=8, label='Text'),
        mlines.Line2D([], [], color=MODALITY_PALETTE['Caption'], marker=MODALITY_MARKERS['Caption'], linestyle='None', markersize=8, label='Caption'),
        mlines.Line2D([], [], color=MODALITY_PALETTE['Image'], marker=MODALITY_MARKERS['Image'], linestyle='None', markersize=8, label='Image'),
        # # Gaps (Lines) - 使用空数据创建图例项
        # mlines.Line2D([], [], color=LINE_COLORS['Context'], linewidth=2, label='Context Shift (Text→Cap)'),
        # mlines.Line2D([], [], color=LINE_COLORS['Modality'], linewidth=2, label='Modality Shift (Cap→Img)')
    ]
    
    # 调整图例列数，使其美观 (3列可能放不下5个，改用2列或者3列)
    ax.legend(
        handles=handles,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.22),
        ncol=5,
        frameon=False,
        fontsize=10,
        handletextpad=0.6,
        columnspacing=1.2
    )
    
    # 保存
    plt.tight_layout()
    safe_name = target_factor.lower().replace(" ", "_")
    plt.savefig(f'{save_dir}/scheme_A_dumbbell_{safe_name}.pdf', bbox_inches='tight', dpi=300)
    plt.close()

def plot_combined_dumbbell(df_p, save_dir):
    """
    Combined Dumbbell Plot.
    Features: Shared Y-axis, Slanted Labels, Inside Legend, No Titles.
    """
    factors = ['Intention of Harm', 'Self Benefit']
    print(f"Generating Combined Dumbbell Plot for {factors}...")
    
    # 1. 数据过滤与准备
    df_sub = df_p[
        (df_p['factor'].isin(factors)) & 
        (df_p['dilemma'] == 'total')
    ].copy()
    
    if df_sub.empty:
        print("No data found for target factors.")
        return

    # 排序模型 (Open -> Proprietary)
    available_models = set(df_sub['model_type'].unique())
    sorted_models = [m for m in MODEL_TYPE_LIST if m in available_models]
    
    # 2. 设置画布: 1行2列, 共用Y轴
    # figsize=(10, 5) 既保证了宽度适中(适合半栏+斜体字)，又保证了高度足够容纳大字体
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.5), sharey=True)
    plt.subplots_adjust(wspace=0.002) # 极窄的子图间距
    
    # 3. 循环绘制每个因子
    for ax_idx, (ax, factor) in enumerate(zip(axes, factors)):
        factor_data = df_sub[df_sub['factor'] == factor]
        
        # 绘制背景网格
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

        for i, model in enumerate(sorted_models):
            model_data = factor_data[factor_data['model_type'] == model]
            
            # 提取数据 (安全获取)
            def get_val(mod):
                vals = model_data[model_data['modality'] == mod]['log_odds'].values
                return vals[0] if len(vals) > 0 else np.nan

            val_text = get_val('Text')
            val_cap = get_val('Caption')
            val_img = get_val('Image')
            
            # # --- 绘制连线 (加粗 lw=3) ---
            # if not np.isnan(val_text) and not np.isnan(val_cap):
            #     ax.plot([val_text, val_cap], [i, i], 
            #             color=LINE_COLORS['Context'], alpha=0.7, zorder=2, lw=3)
            
            # if not np.isnan(val_cap) and not np.isnan(val_img):
            #     ax.plot([val_cap, val_img], [i, i], 
            #             color=LINE_COLORS['Modality'], alpha=0.7, zorder=2, lw=3)

            # --- 绘制点 (加大 s=180) ---
            ax.scatter(val_text, i, color=MODALITY_PALETTE['Text'], marker=MODALITY_MARKERS['Text'], 
                       s=250, zorder=3, edgecolors='white', linewidth=1.2, label='Text' if i==0 else "")
            ax.scatter(val_cap, i, color=MODALITY_PALETTE['Caption'], marker=MODALITY_MARKERS['Caption'], 
                       s=250, zorder=3, edgecolors='white', linewidth=1.2, label='Caption' if i==0 else "")
            ax.scatter(val_img, i, color=MODALITY_PALETTE['Image'], marker=MODALITY_MARKERS['Image'], 
                       s=250, zorder=3, edgecolors='white', linewidth=1.2, label='Image' if i==0 else "")

        # --- 添加分隔线 (Open vs Proprietary) ---
        if len(sorted_models) >= 5:
            split_index = 3.5
            ax.axhline(y=split_index, color='gray', linestyle='--', linewidth=3, alpha=0.6)

        # --- 内部标题 (替代 ax.set_title) ---
        # 根据因子名称放置在图表上方内部，节省外部空间
        panel_label = f"({chr(ord('a') + ax_idx)})"
        readable_title = f"{panel_label} {factor.replace('_', ' ').title()}"
        ax.text(0.5, 1.09, readable_title, transform=ax.transAxes, 
                ha='center', va='top', fontsize=24, fontweight='bold', 
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=2))

        # --- 坐标轴美化 ---
        ax.set_yticks(range(len(sorted_models)))
        if ax_idx == 0:
            ax.invert_yaxis() # 只需翻转一次
        
        # 垂直零线
        ax.axvline(0, color='black', linestyle='-', linewidth=4, alpha=0.4)
        
        # X轴标签
        ax.tick_params(axis='x', labelsize=19)
        ax.set_xlabel('Log Odds', fontsize=21, fontweight='bold')

    # 4. 全局修饰
    
    # --- Y轴标签斜着排 (Slanted Labels) ---
    # ha='right' 保证文字尾部对齐坐标轴
    axes[0].set_yticklabels(sorted_models, fontsize=18, fontweight='bold', rotation=20, ha='right')
    axes[0].tick_params(axis='y', pad=0) # 减少标签与轴的距离
    
    # 第二个图隐藏 Y 轴刻度
    axes[1].tick_params(left=False, labelleft=False)

    # --- 标注 Open-Weight / Proprietary ---
    # 放在最右侧图表的右边
    axes[1].text(1.02, 1.5, 'Open-Weight', transform=axes[1].get_yaxis_transform(), 
            rotation=270, va='center', ha='left', fontsize=20, fontweight='bold', color='#555555')
    axes[1].text(1.02, 4.5, 'Proprietary', transform=axes[1].get_yaxis_transform(), 
            rotation=270, va='center', ha='left', fontsize=20, fontweight='bold', color='#555555')

    # --- 统一图例 (Inside Axis) ---
    # 创建自定义 Handles
    legend_handles = [
        mlines.Line2D([], [], color=MODALITY_PALETTE['Text'], marker=MODALITY_MARKERS['Text'], linestyle='None', markersize=18, label='Text'),
        mlines.Line2D([], [], color=MODALITY_PALETTE['Caption'], marker=MODALITY_MARKERS['Caption'], linestyle='None', markersize=18, label='Caption'),
        mlines.Line2D([], [], color=MODALITY_PALETTE['Image'], marker=MODALITY_MARKERS['Image'], linestyle='None', markersize=18, label='Image'),
        # mlines.Line2D([], [], color=LINE_COLORS['Context'], linewidth=3, label='Context Shift'),
        # mlines.Line2D([], [], color=LINE_COLORS['Modality'], linewidth=3, label='Modality Shift')
    ]
    
    # 将图例放在第一个子图的右下角 (或者根据数据分布选择较空的位置)
    # loc='lower right' 通常是个好位置
    axes[0].legend(handles=legend_handles, loc='lower left', 
                   fontsize=18, frameon=True, framealpha=0.9, 
                   edgecolor='gray', ncol=1, bbox_to_anchor=(0.02, 0.02))

    # 保存
    plt.tight_layout()
    # 再次微调，给斜体字留出左边距
    plt.subplots_adjust(left=0.1, top=0.75) 
    
    save_path = f'{save_dir}/combined_dumbbell_slanted.pdf'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    save_path = f'{save_dir}/combined_dumbbell_slanted.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved combined plot to {save_path}")

def plot_slope_scheme_b(df_p, save_dir, target_factor='Intention of Harm'):
    """
    Scheme B: Slope Chart split by Model Type (Open vs Proprietary).
    """
    print(f"Generating Scheme B (Grouped Slope) for {target_factor}...")

    # 1. 数据准备
    df_sub = df_p[
        (df_p['factor'] == target_factor) & 
        (df_p['dilemma'] == 'total')
    ].copy()

    if df_sub.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    
    categories = ['Open-Weight', 'Proprietary']
    df_sub['category'] = df_sub['model_type'].map(MODEL_CATEGORY_MAP)
    
    # 统一 Y 轴范围以便对比
    y_min = df_sub['log_odds'].min() - 0.5
    y_max = df_sub['log_odds'].max() + 0.5

    for ax, cat in zip(axes, categories):
        cat_data = df_sub[df_sub['category'] == cat]
        
        # 如果该类别没数据，跳过
        if cat_data.empty:
            ax.set_title(cat)
            continue

        # 对每个模型画线
        unique_models = cat_data['model_type'].unique()
        
        for model in unique_models:
            model_df = cat_data[cat_data['model_type'] == model]
            
            # 确保顺序 Text -> Caption -> Image
            model_df = model_df.set_index('modality').reindex(MODALITY_LIST).reset_index()
            
            # 准备数据
            x_vals = range(len(MODALITY_LIST))
            y_vals = model_df['log_odds'].values
            
            # 绘制折线
            # 可以给不同模型不同颜色，或者统一颜色
            line = ax.plot(x_vals, y_vals, marker='o', linewidth=2, label=model)
            color = line[0].get_color()
            
            # 在终点 (Image) 旁边标注模型名字
            if not np.isnan(y_vals[-1]):
                ax.text(2.1, y_vals[-1], model, va='center', fontsize=9, color=color, fontweight='bold')

        # 设置轴
        ax.set_xticks(range(len(MODALITY_LIST)))
        ax.set_xticklabels(MODALITY_LIST)
        ax.set_title(f"{cat} Models", fontsize=12, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.axhline(0, color='black', linewidth=1)
        ax.set_xlim(-0.2, 2.8) # 留出右侧写字的空间
        ax.set_ylim(y_min, y_max)

    # 3. 全局设置
    axes[0].set_ylabel('Log Odds (Effect Strength)', fontsize=12)
    
    readable_title = target_factor.replace("_", " ").title()
    plt.suptitle(f"Trajectory of {readable_title} by Model Type", fontsize=14, y=1.02)
    
    plt.tight_layout()
    safe_name = target_factor.lower().replace(" ", "_")
    plt.savefig(f'{save_dir}/scheme_B_slope_{safe_name}.pdf', bbox_inches='tight', dpi=300)
    plt.close()

def plot_interaction_log_odds(df_p, save_dir, target_dilemma='total'):
    """
    Visualization: Interaction Effects Bar Chart (Log Odds).
    Plots the coefficients of all 2-way and 3-way interactions.
    """
    print(f"Generating Interaction Effects Log Odds plot ({target_dilemma})...")
    
    # 1. Define Interaction Terms
    interaction_factors = [
        'PF x IH',      # Personal Force * Intention of Harm (Instrumental Harm)
        'PF x SB',      # Personal Force * Self Benefit (Self-interested Violence)
        'IH x SB',      # Intention of Harm * Self Benefit (Malicious Intent)
        'PF x IH x SB'  # The "Moral Collapse" combination
    ]
    
    # 2. Filter Data
    df_agg = df_p[
        (df_p['factor'].isin(interaction_factors)) & 
        (df_p['dilemma'] == target_dilemma)
    ].copy()
    
    if df_agg.empty:
        print("No data found for interaction plots.")
        return

    # 3. Define Colors for Interactions
    # Distinct colors to separate the types of complex reasoning
    interaction_colors = {
        'PF x IH': '#d62728',    # Red (Harm-related)
        'PF x SB': '#ff7f0e',    # Orange (Self-interest mixed)
        'IH x SB': '#9467bd',    # Purple (Intentional malice)
        'PF x IH x SB': '#2ca02c' # Green (Complex 3-way)
    }

    # 4. Setup Plot
    fig, ax = plt.subplots(figsize=(18, 8))
    sns.set_theme(style="whitegrid")
    
    # Create grouped bar chart
    sns.barplot(data=df_agg, x='model_str', y='log_odds', hue='factor', 
                palette=interaction_colors, errorbar=None,
                order=MODEL_NAME_LIST,
                ax=ax, edgecolor='white', linewidth=0.5,
                width=0.8)
    
    # Add Zero Line
    ax.axhline(0, color='black', linewidth=1.5)

    # 5. Apply Modality Textures (Hatches)
    # Pattern: Text=None, Caption=///, Image=...
    hatch_styles = ['', '///', '...']
    
    # Iterate through patches to apply hatches based on their x-position (Modality)
    # Note: Seaborn groups bars by Hue within each X category.
    # The bars are drawn in sequence. We need to map them correctly.
    
    # Helper to determine modality from x-coordinate is tricky with multiple hues.
    # Better approach: Iterate patches and check the data they represent? 
    # Hard in matplotlib. We rely on the periodic structure of grouped bars.
    
    num_hues = len(interaction_factors)
    # Bars are ordered: Model1-Text(Hue1, Hue2...), Model1-Cap(...), Model1-Img(...)
    # Actually, sns.barplot draws all bars for Hue1, then all for Hue2, etc.
    # So we need to calculate modality based on x-tick index.
    
    for patch in ax.patches:
        # Get the center x position of the bar
        current_x = patch.get_x() + patch.get_width() / 2
        
        if np.isnan(current_x): continue
        
        # Determine which Model-Modality group this x belongs to
        # X ticks are integers 0, 1, 2... corresponding to MODEL_NAME_LIST
        # MODEL_NAME_LIST structure: [M1-Text, M1-Cap, M1-Img, M2-Text...]
        
        tick_idx = int(round(current_x))
        if tick_idx < 0 or tick_idx >= len(MODEL_NAME_LIST): continue
        
        modality_idx = tick_idx % 3 # Assuming 3 modalities: Text, Caption, Image
        
        patch.set_hatch(hatch_styles[modality_idx])
        patch.set_edgecolor('black')
        patch.set_linewidth(0.5)

    # 6. Formatting X-Axis (Group by Model)
    group_size = len(MODALITY_LIST)
    new_xticklabels = []
    xticks_positions = []
    
    for i, model_base in enumerate(MODEL_TYPE_LIST):
        start_idx = i * group_size
        center_idx = start_idx + (group_size - 1) / 2.0
        end_idx = start_idx + group_size
        
        # Model Name Label
        ax.text(center_idx, ax.get_ylim()[0] - (abs(ax.get_ylim()[0])*0.15), model_base, 
                ha='center', va='top', 
                fontsize=12, fontweight='bold', color='#333333')
        
        # Vertical Separators
        if i < len(MODEL_TYPE_LIST) - 1:
            sep_x = end_idx - 0.5
            ax.axvline(x=sep_x, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)

        # Modality Labels
        for j in range(group_size):
            xticks_positions.append(start_idx + j)
            new_xticklabels.append(MODALITY_LIST[j])

    ax.set_xticks(xticks_positions)
    ax.set_xticklabels(new_xticklabels, rotation=0, fontsize=10)
    ax.set_xlabel("")
    ax.set_xlim(-0.5, len(MODEL_NAME_LIST) - 0.5)

    # 7. Labels and Legend
    scope_title = "Global (Total)" if target_dilemma == 'total' else "Care vs. Care"
    plt.ylabel(f"Log Odds ({scope_title})", fontsize=13)
    plt.title(f"Interaction Effects Profile: How Factors Combine ({scope_title})",
              fontsize=18, fontweight='bold', y=1.05)
    
    # Custom Legend
    color_handles = [mpatches.Patch(facecolor=c, label=l, edgecolor='black') 
                     for l, c in interaction_colors.items()]
    hatch_handles = [
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='', label='Text'),
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='///', label='Caption'),
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='...', label='Image')
    ]
    
    # Combine legends
    ax.legend(handles=color_handles + [mpatches.Patch(visible=False)] + hatch_handles, 
              bbox_to_anchor=(0.5, 1.12), loc='upper center', 
              ncol=4, frameon=False, fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15) # Make room for legend and x-labels
    
    filename = f'interaction_effects_log_odds_{target_dilemma}.pdf'
    plt.savefig(f'{save_dir}/{filename}', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved Interaction Log Odds plot to: {filename}")

def plot_interaction_per_factor_subplots(df_p, save_dir, target_dilemma='total'):
    """
    Visualization: Detailed Subplots for Each Interaction Factor.
    Generates 4 separate figures (one per factor).
    Each figure contains subplots for each Model Type.
    """
    print(f"Generating Interaction Detail Subplots ({target_dilemma})...")
    
    target_factors = ['Personal Force', 'Intention of Harm', 'Self Benefit', 'PF x IH', 'PF x SB', 'IH x SB', 'PF x IH x SB']
    
    # Define consistent palette for Modalities
    # You can reuse MODALITY_PALETTE if defined, or define explicitly
    modality_colors = {
        'Text': '#5c8aae',    # Blue-ish
        'Caption': '#7f9e74', # Green-ish
        'Image': '#d62728'    # Red-ish
    }
    
    # Get list of unique models to determine subplot grid
    unique_models = [m for m in MODEL_TYPE_LIST if m in df_p['model_type'].unique()]
    n_models = len(unique_models)
    n_cols = 3 
    n_rows = (n_models + n_cols - 1) // n_cols # Ceiling division

    for factor in target_factors:
        # Filter data for this specific factor
        df_factor = df_p[
            (df_p['factor'] == factor) & 
            (df_p['dilemma'] == target_dilemma)
        ].copy()
        
        if df_factor.empty:
            continue

        # Setup Figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=True)
        axes = axes.flatten() # Flatten 2D array to 1D for easy iteration
        
        # Determine global min/max for Y-axis limits to keep scales consistent
        y_max = df_factor['log_odds'].max()
        y_min = df_factor['log_odds'].min()
        # Add some padding
        padding = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
        y_limit = max(abs(y_max + padding), abs(y_min - padding))
        
        # Iterate over models
        for i, model_type in enumerate(unique_models):
            ax = axes[i]
            
            # Get data for this model
            subset = df_factor[df_factor['model_type'] == model_type]
            
            # Reindex to ensure Modality order: Text -> Caption -> Image
            subset = subset.set_index('modality').reindex(MODALITY_LIST).reset_index()
            
            # Plot Bar Chart
            sns.barplot(
                data=subset, x='modality', y='log_odds',
                palette=modality_colors,
                ax=ax, edgecolor='black', linewidth=1
            )
            
            # Aesthetics
            ax.set_title(model_type, fontsize=14, fontweight='bold')
            ax.axhline(0, color='black', linestyle='-', linewidth=1)
            ax.set_xlabel("")
            ax.set_ylabel("Log Odds" if i % n_cols == 0 else "") # Only label Y on left-most plots
            ax.grid(axis='y', linestyle=':', alpha=0.5)
            
            # Add value annotations on bars
            for p in ax.patches:
                height = p.get_height()
                if not np.isnan(height):
                    ax.annotate(f'{height:.2f}', 
                                (p.get_x() + p.get_width() / 2., height), 
                                ha='center', va='bottom' if height > 0 else 'top', 
                                fontsize=9, color='black', xytext=(0, 2 if height > 0 else -2),
                                textcoords='offset points')

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
            
        # Global Title
        readable_factor = factor.replace("PF", "Personal Force").replace("IH", "Intention").replace("SB", "Self Benefit")
        scope_title = "Global (Pooled)" if target_dilemma == 'total' else "Care vs. Care"
        plt.suptitle(f"Interaction Analysis: {readable_factor} ({scope_title})", fontsize=18, y=1.02)
        
        # Legend (Create custom handles)
        handles = [mpatches.Patch(color=color, label=mod) for mod, color in modality_colors.items()]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=12)

        plt.tight_layout()
        
        # Save
        safe_name = factor.lower().replace(" ", "_").replace(":", "_")
        filename = f'interaction_detail_{safe_name}_{target_dilemma}.pdf'
        plt.savefig(f'{save_dir}/{filename}', bbox_inches='tight', dpi=300)
        filename = f'interaction_detail_{safe_name}_{target_dilemma}.png'
        plt.savefig(f'{save_dir}/{filename}', bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Saved detail plot: {filename}")

# ================= 集成到 Main =================

def main():
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
    
    viz_dir = f'{ROOT}/../visualization/single_feature/conceptual_factor_general'
    os.makedirs(viz_dir, exist_ok=True)
    
    # 执行方案 A (分别针对两个变量)
    # plot_dumbbell_scheme_a(df_p, viz_dir, target_factor='Self Benefit')
    # plot_dumbbell_scheme_a(df_p, viz_dir, target_factor='Intention of Harm')
    # plot_dumbbell_scheme_a(df_p, viz_dir, target_factor='Personal Force')
    plot_combined_dumbbell(df_p, viz_dir)
    
    # 执行方案 B (分别针对两个变量)
    # plot_slope_scheme_b(df_p, viz_dir, target_factor='Self Benefit')
    # plot_slope_scheme_b(df_p, viz_dir, target_factor='Intention of Harm')

    # for scope in ['total', 'care_vs_care']:
    #     # plot_interaction_log_odds(df_p, viz_dir, target_dilemma=scope)
    #     plot_interaction_per_factor_subplots(df_p, viz_dir, target_dilemma=scope)

if __name__ == '__main__':
    main()