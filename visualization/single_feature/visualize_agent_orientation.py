import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yaml
import os
import glob

from config.constants import ROOT
from visualization.utils import MODEL_LIST, MODALITY_LIST, parse_model_info

# ==========================================
# 配置与常量
# ==========================================
MORAL_DIMENSIONS = ['Care', 'Fairness', 'Loyalty', 'Authority', 'Purity']
DIMENSION_PALETTE = {
    'Care': '#4E79A7',       # 蓝色
    'Fairness': '#F28E2B',   # 橙色
    'Loyalty': '#E15759',    # 红色
    'Authority': '#76B7B2',  # 青色
    'Purity': '#59A14F'      # 绿色
}

# 自定义排序顺序
CATEGORY_ORDER = [
    'Species', 'Profession', 'Age', 'Wealth', 
    'Gender', 'Education', 'Fitness', 'Color', 'Other'
]

# 指定 Agent 在 Category 内部的特定顺序 (可选)
AGENT_INTERNAL_ORDER = {
    'Species': ['human', 'non-human'],
    'Profession': ['criminal', 'low', 'high'],
    'Wealth': ['poor', 'normal', 'rich'],
    'Fitness': ['unhealthy', 'normal'],
    'Education': ['low-educated', 'well-educated'],
    'Age': ['infant', 'child', 'teenager', 'middle-age', 'elderly']
}

def load_orientation_data(model_str, analyze_dir):
    combined_data = []
    model_type, mode = parse_model_info(model_str)
    
    fname = f"orientation_{model_str}.yaml"
    fpath = os.path.join(analyze_dir, fname)

    if os.path.exists(fpath):
        with open(fpath, 'r') as f:
            yaml_data = yaml.safe_load(f)
            
        if yaml_data:
            for category, agents_dict in yaml_data.items():
                for agent, scores in agents_dict.items():
                    record = {
                        'model_type': model_type,
                        'model_raw': model_str,
                        'mode': mode,
                        'category': category,
                        'agent': agent
                    }
                    record.update(scores)
                    combined_data.append(record)
    
    return combined_data

def get_sort_key(row):
    # 1. Category Rank
    try:
        cat_rank = CATEGORY_ORDER.index(row['category'])
    except ValueError:
        cat_rank = 999
    
    # 2. Agent Rank (Internal)
    agent_rank = 999
    if row['category'] in AGENT_INTERNAL_ORDER:
        order_list = AGENT_INTERNAL_ORDER[row['category']]
        # 模糊匹配：比如 "human" in "human"
        agent_clean = str(row['agent']).lower()
        for idx, target in enumerate(order_list):
            if target in agent_clean:
                agent_rank = idx
                break
    
    # 3. 如果没有预定义顺序，按字母序
    return (cat_rank, agent_rank, row['agent'])

def plot_grouped_stacked_orientation(df, save_dir):
    if df.empty:
        print("DataFrame is empty, skip plotting.")
        return

    # 添加排序键
    df['sort_tuple'] = df.apply(get_sort_key, axis=1)
    df = df.sort_values(by=['sort_tuple'])
    
    # 获取唯一的 Agent 顺序 (骨架)，包含 Category 和 Agent
    # 必须保留 category，因为 'normal' 在 Wealth 和 Fitness 中都存在
    unique_agents_df = df[['category', 'agent', 'sort_tuple']].drop_duplicates().sort_values('sort_tuple')
    
    # 获取用于 X 轴标签的 agent 名称列表
    agents_ordered_labels = unique_agents_df['agent'].values
    
    models = df['model_type'].unique()
    modes = ['Text', 'Caption', 'Image'] # 确保与 load_orientation_data 中的 mode.capitalize() 一致
    
    fig, axes = plt.subplots(len(models), 1, figsize=(20, 6 * len(models)), sharex=True)
    if len(models) == 1: axes = [axes]
    
    bar_width = 0.25
    offsets = {'Text': -bar_width, 'Caption': 0, 'Image': bar_width}
    hatches = {'Text': '', 'Caption': '///', 'Image': '..'}
    
    for ax, model in zip(axes, models):
        model_data = df[df['model_type'] == model]
        
        x_indices = np.arange(len(unique_agents_df))
        
        for mode in modes:
            subset = model_data[model_data['mode'] == mode]
            
            # 【核心修复】：使用 category 和 agent 双重键构建骨架
            # 这样 'Wealth-normal' 和 'Fitness-normal' 就是不同的行
            skeleton = unique_agents_df[['category', 'agent']].copy()
            
            # 使用双键合并 (left join)
            subset_aligned = skeleton.merge(subset, on=['category', 'agent'], how='left').fillna(0)
            
            # 准备堆叠数据
            bottoms = np.zeros(len(x_indices))
            
            for dim in MORAL_DIMENSIONS:
                values = subset_aligned[dim].values
                
                # 安全检查：如果长度依然不匹配，说明数据源可能有其他问题（如重复行）
                if len(values) != len(x_indices):
                    # 尝试再次去重作为最后的防线
                    subset_aligned = subset_aligned.drop_duplicates(subset=['category', 'agent'])
                    values = subset_aligned[dim].values
                    if len(values) != len(x_indices):
                        print(f"Skipping {model}-{mode}-{dim} due to shape mismatch: {len(values)} vs {len(x_indices)}")
                        continue

                ax.bar(
                    x_indices + offsets[mode],
                    values,
                    width=bar_width,
                    bottom=bottoms,
                    color=DIMENSION_PALETTE[dim],
                    edgecolor='white',
                    linewidth=0.5,
                    hatch=hatches[mode]
                )
                bottoms += values
                
        ax.set_title(f"Model: {model}", fontsize=16, fontweight='bold')
        ax.set_ylabel("Moral Dimension Proportion (Sum=1)", fontsize=12)
        ax.set_xticks(x_indices)
        ax.set_xticklabels(agents_ordered_labels, rotation=45, ha='right', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_xlim(-0.5, len(x_indices) - 0.5)
        
        # 绘制分类分隔线
        cat_series = unique_agents_df['category'].values
        for i in range(1, len(cat_series)):
            if cat_series[i] != cat_series[i-1]:
                ax.axvline(i - 0.5, color='gray', linestyle=':', alpha=0.5)
                # 可选：在上方添加类别标签
                ax.text(i - 0.5, 1.02, cat_series[i], ha='center', fontsize=9, fontweight='bold', color='gray')

    # 全局图例
    dim_handles = [mpatches.Patch(color=DIMENSION_PALETTE[d], label=d) for d in MORAL_DIMENSIONS]
    mode_handles = [
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='', label='Text'),
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='///', label='Caption'),
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='..', label='Image')
    ]
    
    fig.legend(
        handles=dim_handles + mode_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.95),
        ncol=8,
        fontsize=12,
        title="Moral Dimensions (Colors) & Modalities (Patterns)"
    )
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    output_path = os.path.join(save_dir, "agent_orientation_stacked.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")

def main():
    analyze_dir = f"{ROOT}/../results/single_feature/analyze_results"
    vis_dir = f'{ROOT}/../visualization/single_feature/agent_orientation'
    if not os.path.exists(vis_dir): os.makedirs(vis_dir)

    all_data = []
    
    # 遍历 MODEL_LIST
    for model_str in MODEL_LIST:
        model_data = load_orientation_data(model_str, analyze_dir)
        if model_data:
            all_data.extend(model_data)
    
    df = pd.DataFrame(all_data)
    
    # 可选：全局去重，防止 parse_model_info 导致名字重复时的数据重复
    if not df.empty:
        df = df.drop_duplicates(subset=['model_type', 'mode', 'category', 'agent'])

    plot_grouped_stacked_orientation(df, vis_dir)

if __name__ == '__main__':
    main()