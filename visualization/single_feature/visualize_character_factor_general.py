import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import yaml
import os
import re

from config.constants import ROOT
from visualization.utils import DILEMMA_DIMENSION_MAP, MODEL_LIST, parse_model_info, MODALITY_PALETTE

# ==========================================
# 1. Constants & Mapping Logic
# ==========================================

# Moral Dimension Mapping
DILEMMA_DIMENSION_LOGIC = {
    'bonus_allocation': {1: 'Care', 0: 'Fairness'},
    'self_harming': {1: 'Care', 0: 'Loyalty'},
    'guarded_speedboat': {1: 'Care', 0: 'Authority'},
    'save_dying': {1: 'Care', 0: 'Authority'},
    'party': {1: 'Care', 0: 'Purity'},
    'resume': {1: 'Loyalty', 0: 'Fairness'},
    'report_cheating': {1: 'Fairness', 0: 'Loyalty'},
    'hiring': {1: 'Authority', 0: 'Fairness'},
    'inpurity': {1: 'Purity', 0: 'Fairness'},
    'feed': {1: 'Loyalty', 0: 'Authority'},
    'report_stealing': {1: 'Authority', 0: 'Loyalty'},
    'ceremony': {1: 'Loyalty', 0: 'Purity'},
    'dirty': {1: 'Authority', 0: 'Purity'},
}

MORAL_DIMENSIONS = ['Care', 'Fairness', 'Loyalty', 'Authority', 'Purity']

# Target Role Sign Mapping (+1: Beneficiary, -1: Victim)
POSITIVE_ROLES = [
    'rail_horizontal', 'road_left', 'immunity', 'hazard', 'remaining_member', 
    'dying', 'diver', 'organ_receiver', 'crowd', 'urgent_colleague', 
    'tourist', 'child', 'children', 'elder', 'friend', 'neighbor'
]
NEGATIVE_ROLES = [
    'rail_vertical', 'road_right', 'disease', 'policy', 'thrown_member', 
    'hiv_patient', 'injured_diver', 'organ_source', 'terrorist', 
    'common_colleague', 'candidate', 'cheating_classmate', 'baby'
]
ROLE_OVERRIDES = {
    'save_dying': {'neighbor': -1}, 
    'crying_baby': {'neighbor': 1},
    'report_cheating': {'friend': -1}, 
    'report_stealing': {'friend': -1}, 
}

def get_target_sign(dilemma, feature_name):
    if dilemma in ROLE_OVERRIDES and feature_name in ROLE_OVERRIDES[dilemma]:
        return ROLE_OVERRIDES[dilemma][feature_name]
    if feature_name in POSITIVE_ROLES: return 1
    if feature_name in NEGATIVE_ROLES: return -1
    return 0

# ==========================================
# 2. Data Preparation
# ==========================================

def prepare_results(model_str, results_file, data_list):
    try:
        with open(results_file, 'r') as f:
            results = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: {results_file} not found")
        return

    model_type, modality = parse_model_info(model_str)
    
    for scenario, conditions in results.items():
        if not isinstance(conditions, dict): continue
        parts = scenario.rsplit('_', 1)
        dilemma, attribute = (parts[0], parts[1]) if len(parts) == 2 else (scenario, 'Unknown')
        
        for condition, stats in conditions.items():
            if not isinstance(stats, dict): continue
            if 'Odds_ratio' in stats and 'p_value' in stats:
                or_val = stats['Odds_ratio']
                if pd.isna(or_val) or or_val == 0: continue
                
                log_or = np.log10(or_val)
                log_or = np.clip(log_or, -5, 5)

                is_interaction = ":" in condition
                has_agent = "C(agent" in condition 
                
                level = "Unknown"
                if "Intercept" in condition: level = "Intercept"
                elif is_interaction:
                    level = "Agent-Target (L4)" if has_agent else "Target-Target (L3)"
                elif has_agent: level = "Agent Intrinsic (L1)"
                else: level = "Target Valuation (L2)"

                # Extract features: Returns list of tuples [(Variable, Level), ...]
                # e.g., C(wealth)[T.rich] -> [('wealth', 'rich')]
                features = re.findall(r'C\((.*?)(?:,.*?)?\)\[[ST]\.(.*?)\]', condition)
                
                data_list.append({
                    'model_str': model_str, 'model_type': model_type, 'modality': modality,
                    'dilemma': dilemma, 'attribute': attribute,
                    'condition': condition, 'level': level,
                    'log_or': log_or, 'features': features
                })

# ==========================================
# 3. Visualization Functions
# ==========================================

CHARACTER_HIERARCHY = {
    'Species': {
        'human': ['human'],
        'non-human': ['chick', 'chicken', 'goose', 'pig', 'sheep', 'skunk', 'procupine', 'boar', 'fox', 'wolf', 'turtle', 'frog', 'toad', 'crab', 'cat', 'dog'] # Added dog/cat as common non-humans
    },
    'Color': ['black', 'white', 'yellow'],
    'Profession': {
        'criminal': ['thief'],
        'low': ['blue-collar', 'chef', 'unemployed'],
        'high': ['police', 'doctor', 'teacher', 'white-collar', 'boss', 'soldier', 'artist']
    },
    'Gender': ['male', 'female'],
    'Age': ['infant', 'child', 'teenager', 'middle-age', 'elderly'],
    'Wealth': ['normal', 'poor', 'rich'],
    'Fitness': {
        'normal': ['normal'],
        'unhealthy': ['disability', 'mental disease']
    },
    'Education': ['low-educated', 'well-educated']
}

# 2. Build a Reverse Lookup Map (Feature Value -> Category)
# NOTE: 'normal' is ambiguous (Wealth, Fitness, Appearance). 
# We must define a priority or specific handling. 
# In most L1 regression contexts, 'normal' usually implies Wealth or Fitness baseline.
# We will map duplicates to a generic "Baseline" or pick the most likely one based on context.

FEATURE_TO_CATEGORY = {}

for category, content in CHARACTER_HIERARCHY.items():
    if isinstance(content, list):
        # Direct list (e.g., Color, Gender, Wealth)
        for item in content:
            # Handle 'normal' ambiguity: Prioritize Wealth if conflict, or keep distinct logic
            if item == 'normal' and 'normal' in FEATURE_TO_CATEGORY:
                continue # Skip overwriting if already assigned (First come first served)
            FEATURE_TO_CATEGORY[item] = category
    elif isinstance(content, dict):
        # Nested dict (e.g., Species, Profession)
        for sub_cat, items in content.items():
            for item in items:
                FEATURE_TO_CATEGORY[item] = category
            # Also map the sub-category keys themselves if they appear in results (e.g. 'criminal')
            FEATURE_TO_CATEGORY[sub_cat] = category

# Manually fix specific ambiguous terms if necessary
FEATURE_TO_CATEGORY['normal'] = 'Wealth/Fitness' # Indication of ambiguity

def get_feature_category(feature_name):
    """
    Maps a feature value (e.g., 'thief', 'rich') to its Category (e.g., 'Profession', 'Wealth').
    """
    # Clean the string just in case
    clean_feat = feature_name.strip().lower()
    return FEATURE_TO_CATEGORY.get(clean_feat, 'Other')

DIMENSION_PALETTE = {
    'Care': '#4E79A7',       # Blue
    'Fairness': '#F28E2B',   # Orange
    'Loyalty': '#E15759',    # Red
    'Authority': '#76B7B2',  # Teal
    'Purity': '#59A14F'      # Green
}

def plot_l1_stacked_bar_grouped(df, save_dir):
    print("Generating L1: Stacked Agent Preference (Grouped by Attribute)...")
    l1_data = df[df['level'] == 'Agent Intrinsic (L1)'].copy()
    if l1_data.empty: return

    l1_data = l1_data[l1_data['dilemma'].isin(DILEMMA_DIMENSION_LOGIC.keys())]
    
    # 1. Data Aggregation
    expanded_rows = []
    for _, row in l1_data.iterrows():
        logic = DILEMMA_DIMENSION_LOGIC[row['dilemma']]
        val = row['log_or']
        dim = logic[1] if val > 0 else logic[0]
        magnitude = abs(val)
        
        if not row['features']: continue
        reg_variable, feature_val = row['features'][0]
        
        # Determine Category for Sorting
        cat_key = reg_variable.replace('_', ' ').title()
        if cat_key not in CHARACTER_HIERARCHY: 
             cat_key = get_feature_category(feature_val)

        feature_display = feature_val.replace('_', ' ')

        expanded_rows.append({
            'model_type': row['model_type'],
            'modality': row['modality'],
            'agent_feature': feature_display,
            'category': cat_key,
            'moral_dimension': dim,
            'strength': magnitude
        })
    
    plot_df = pd.DataFrame(expanded_rows)
    
    # Sum strength for stacking (Example: A Doctor in 3 Care dilemmas -> Sum of Care Strength)
    agg_df = plot_df.groupby(
        ['model_type', 'modality', 'category', 'agent_feature', 'moral_dimension']
    )['strength'].sum().reset_index()

    # 2. Sorting Logic (Category -> Feature)
    agg_df = agg_df.sort_values(['category', 'agent_feature'])
    features_ordered = agg_df['agent_feature'].unique()
    models = agg_df['model_type'].unique()
    modalities = ['Text', 'Caption', 'Image']
    dimensions = MORAL_DIMENSIONS # ['Care', 'Fairness', 'Loyalty', 'Authority', 'Purity']

    # 3. Plotting Setup
    # One subplot per Model
    fig, axes = plt.subplots(len(models), 1, figsize=(20, 6 * len(models)), sharex=True)
    if len(models) == 1: axes = [axes]
    
    bar_width = 0.25
    # Offsets for Modalities: Text (Left), Caption (Center), Image (Right)
    offsets = {'Text': -bar_width, 'Caption': 0, 'Image': bar_width}
    # Hatch patterns to distinguish Modality (since Color is used for Dimension)
    hatches = {'Text': '', 'Caption': '///', 'Image': '..'}
    
    for ax, model in zip(axes, models):
        model_data = agg_df[agg_df['model_type'] == model]
        
        # Create X-axis indices
        x_indices = np.arange(len(features_ordered))
        
        # We need to plot bars for each Modality
        for mode in modalities:
            # Prepare bottoms for stacking
            bottoms = np.zeros(len(features_ordered))
            
            # Filter data for this Mode
            mode_data = model_data[model_data['modality'] == mode]
            
            # Stack Dimensions
            for dim in dimensions:
                # Align data to features_ordered
                # We merge with a skeleton DF to ensure all features exist (fill 0)
                skeleton = pd.DataFrame({'agent_feature': features_ordered})
                merged = skeleton.merge(
                    mode_data[mode_data['moral_dimension'] == dim], 
                    on='agent_feature', how='left'
                ).fillna({'strength': 0})
                
                values = merged['strength'].values
                
                # Plot Bar Segment
                ax.bar(
                    x_indices + offsets[mode], 
                    values, 
                    width=bar_width, 
                    bottom=bottoms,
                    color=DIMENSION_PALETTE.get(dim, '#333333'),
                    edgecolor='white',
                    linewidth=0.5,
                    hatch=hatches[mode],
                    label=dim if (mode == 'Text' and x_indices[0] == 0) else "" 
                    # Only add label once for Legend? Actually complex with hatches.
                    # We will build custom legend later.
                )
                
                bottoms += values
        
        # Formatting Subplot
        ax.set_title(f"Model: {model}", fontsize=14, fontweight='bold', pad=10)
        ax.set_ylabel("Total Preference Strength (Stacked LogOR)")
        ax.set_xticks(x_indices)
        ax.set_xticklabels(features_ordered, rotation=45, ha='right', fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        # Add visual separators for Categories
        # We need to find indices where Category changes
        # Re-construct category mapping from ordered list
        cat_map = agg_df[['agent_feature', 'category']].drop_duplicates().set_index('agent_feature')['category']
        current_cat = cat_map[features_ordered[0]]
        for i, feat in enumerate(features_ordered):
            cat = cat_map[feat]
            if cat != current_cat:
                ax.axvline(i - 0.5, color='gray', linestyle=':', alpha=0.5)
                # Add Label for previous category? (Optional, might be cluttered)
                current_cat = cat

    # 4. Global Legend construction
    # Dimension Colors
    dim_handles = [mpatches.Patch(color=DIMENSION_PALETTE[d], label=d) for d in dimensions]
    # Modality Hatches (using Grey as neutral color)
    mod_handles = [
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='', label='Text'),
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='///', label='Caption'),
        mpatches.Patch(facecolor='white', edgecolor='black', hatch='..', label='Image')
    ]
    
    fig.legend(
        handles=dim_handles + mod_handles, 
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.02), 
        ncol=8, 
        title="Dimensions (Colors) & Modalities (Patterns)",
        fontsize=12
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.93) # Space for legend
    plt.savefig(os.path.join(save_dir, 'L1_agent_persona_stacked.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_l2_target_valuation_grouped(df, save_dir):
    # Reuse L2 logic but ensure grouping utilizes Variable Name like L1
    print("Generating L2: Target Valuation (Grouped)...")
    l2_data = df[df['level'] == 'Target Valuation (L2)'].copy()
    if l2_data.empty: return

    def calc_score(row):
        if not row['features']: return np.nan, None, None
        cat, feat = row['features'][0]
        sign = get_target_sign(row['dilemma'], cat) # Check role (e.g. rail_horizontal) sign
        if sign == 0: return np.nan, None, None
        
        # Note: L2 regression features usually look like C(rail_horizontal)[T.human]
        # So 'cat' is the Role (rail_horizontal), 'feat' is the Value (human).
        # We need the 'attribute' column from original data for grouping?
        # Actually, 'attribute' column in df usually holds 'species', 'age' etc.
        # Let's use df['attribute'] as Category.
        
        return sign * row['log_or'], row['attribute'].title(), feat

    res = l2_data.apply(calc_score, axis=1, result_type='expand')
    l2_data['val_score'] = res[0]
    l2_data['category'] = res[1]
    l2_data['feature'] = res[2]
    
    l2_data = l2_data.dropna(subset=['val_score'])
    
    # Calculate global mean for sorting within category
    global_mean = l2_data.groupby('feature')['val_score'].mean().reset_index()
    l2_data = l2_data.merge(global_mean, on='feature', suffixes=('', '_global'))
    l2_data = l2_data.sort_values(['category', 'val_score_global'], ascending=[True, False])
    
    models = sorted(l2_data['model_type'].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 12), sharey=True)
    if len(models) == 1: axes = [axes]
    
    mod_palette = {'Text': '#1f77b4', 'Caption': '#2ca02c', 'Image': '#d62728'}

    for i, model in enumerate(models):
        ax = axes[i]
        subset = l2_data[l2_data['model_type'] == model]
        # Order within this subplot
        order = subset.sort_values(['category', 'val_score_global'], ascending=[True, False])['feature'].unique()
        
        sns.barplot(
            data=subset, y='feature', x='val_score', hue='modality',
            order=order, palette=mod_palette, ax=ax,
            hue_order=['Text', 'Caption', 'Image'],
            errorbar=None
        )
        
        ax.set_title(f"Model: {model}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Valuation Score (+ favors survival)")
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        ax.axvline(0, color='black', linewidth=1)
        if i > 0: ax.set_ylabel("")
        else: ax.set_ylabel("Target Attribute")

    plt.tight_layout()
    plt.suptitle('L2: Target Valuation Comparison', y=1.02, fontsize=16)
    plt.savefig(os.path.join(save_dir, 'L2_target_valuation_grouped.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_l2_category_gap(df, save_dir):
    """
    Plots the L2 Preference Gap (Difference in LogOdds) between High-Value and Low-Value groups.
    Uses strict role-based sign logic to ensure directionality is correct.
    """
    print("Generating L2: Category Preference Gap (Value Hierarchy)...")
    
    # 1. Filter for L2 data (Target Valuation)
    l2_data = df[df['level'] == 'Target Valuation (L2)'].copy()
    if l2_data.empty:
        print("No L2 data found.")
        return

    # 2. Define Categories based on Secondary Labels (Group A vs Group B)
    # Group A: Typically "Protected" or "High Value" (Reference)
    # Group B: Typically "Less Protected" or "Low Value" (Contrast)
    CATEGORY_DEFINITIONS = {
        'Species': {
            'Group_A': ['human'], 
            'Group_B': ['non-human']
        },
        'Age': {
            'Group_A': ['infant', 'child', 'teenager'], # Younger
            'Group_B': ['elderly', 'middle-age']        # Older
        },
        'Gender': {
            'Group_A': ['female'], 
            'Group_B': ['male']
        },
        'Profession': {
            'Group_A': ['low', 'high'], 
            'Group_B': ['criminal']
        },
        'Fitness': {
            'Group_A': ['normal'], 
            'Group_B': ['unhealthy']
        },
        'Wealth': {
            'Group_A': ['rich'], 
            'Group_B': ['poor']
        },
        'Education': {
            'Group_A': ['well-educated'], 
            'Group_B': ['low-educated']
        }
    }

    def map_feature_to_group(feature_val):
        """Maps a feature value (e.g. 'non-human') to its Category and Group."""
        val = str(feature_val).lower().strip()
        for cat, groups in CATEGORY_DEFINITIONS.items():
            if val in groups['Group_A']:
                return cat, 'Group_A'
            if val in groups['Group_B']:
                return cat, 'Group_B'
        return None, None

    # 3. Process Data
    processed_rows = []
    
    for _, row in l2_data.iterrows():
        # L2 features structure: [('variable_name', 'feature_value')]
        # e.g., [('rail_horizontal', 'human')]
        if not row['features']: continue
        
        variable_name, feature_val = row['features'][0]
        
        # --- Apply Strict Sign Logic ---
        # 1. Identify if the variable (role) is a Beneficiary or Victim
        direction = get_target_sign(row['dilemma'], variable_name)
        
        # 2. Skip ambiguous roles (direction 0)
        if direction == 0: 
            continue 
            
        # 3. Adjust Score: 
        # If variable is Positive (Beneficiary), LogOdds > 0 means "Save them" (Favor).
        # If variable is Negative (Victim), LogOdds > 0 means "Sacrifice them" (Disfavor).
        # We want Positive Score to ALWAYS mean "Favor/Value this Character".
        # Therefore:
        # - For Beneficiary (+1): Score = LogOdds * 1
        # - For Victim (-1): Score = LogOdds * -1 (because high LogOdds means high action rate = killing them)
        # Note: In most trolley problems, Action = Kill Victim to Save Beneficiary.
        # So for Victim, High LogOdds -> Kill -> Disfavor. So we negate to get "Favor".
        # For Beneficiary, High LogOdds -> Save -> Favor.
        
        # However, we must double check how 'get_target_sign' was used in the original script.
        # In original script: return sign * row['log_or']
        # Let's stick to that exactly.
        adjusted_score = direction * row['log_or']
        
        # --- Map to Category ---
        cat, group = map_feature_to_group(feature_val)
        
        if cat:
            processed_rows.append({
                'model_type': row['model_type'],
                'modality': row['modality'],
                'category': cat,
                'group': group,
                'score': adjusted_score
            })

    if not processed_rows:
        print("No matching categories found for Gap Plot.")
        return

    proc_df = pd.DataFrame(processed_rows)

    # 4. Calculate Gap: (Group A Mean) - (Group B Mean)
    # A positive Gap means model values Group A > Group B (e.g. Human > Non-human)
    agg_df = proc_df.groupby(['model_type', 'modality', 'category', 'group'])['score'].mean().unstack()
    
    # Handle cases where a group might be missing in a split
    if 'Group_A' not in agg_df.columns: agg_df['Group_A'] = np.nan
    if 'Group_B' not in agg_df.columns: agg_df['Group_B'] = np.nan
    
    agg_df['gap'] = agg_df['Group_A'] - agg_df['Group_B']
    plot_data = agg_df.reset_index()
    plot_data = plot_data.dropna(subset=['gap']) # Only keep complete pairs

    # 5. Visualization
    models = sorted(plot_data['model_type'].unique())
    categories = list(CATEGORY_DEFINITIONS.keys())
    present_cats = [c for c in categories if c in plot_data['category'].unique()]
    
    # Setup Subplots
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 6), sharey=True)
    if len(models) == 1: axes = [axes]
    
    # Palette definition
    palette = MODALITY_PALETTE
    
    for i, model in enumerate(models):
        ax = axes[i]
        subset = plot_data[plot_data['model_type'] == model]
        
        sns.barplot(
            data=subset,
            x='category',
            y='gap',
            hue='modality',
            order=present_cats,
            hue_order=['Text', 'Caption', 'Image'],
            palette=palette,
            ax=ax,
            edgecolor='white',
            linewidth=1,
            errorbar=None # Gap is a single derived value from means, usually no error bar unless bootstrapped
        )
        
        ax.set_title(f"{model}", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("")
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.axhline(0, color='black', linewidth=1)
        
        # Y-Axis Label
        if i == 0:
            ax.set_ylabel("Preference Gap (LogOdds)\n(Positive = Favors High-Value/Protected Group)", fontsize=11)
        else:
            ax.set_ylabel("")
            if ax.get_legend(): ax.get_legend().remove()
            
        ax.set_xticklabels(present_cats, rotation=45, ha='right')

    # Unified Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=3, title="Input Modality")
    if axes[0].get_legend(): axes[0].get_legend().remove()

    plt.suptitle("Impact of Modality on Value Hierarchies (L2 Preference Gap)", y=1.12, fontsize=16)
    plt.tight_layout()

    output_path = os.path.join(save_dir, 'L2_target_gap.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    output_path = os.path.join(save_dir, 'L2_target_gap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved L2 Gap plot to {output_path}")

def plot_l2_category_gap_aggregated(df, save_dir):
    """
    Plots the Aggregated L2 Preference Gap with refined layout:
    - Flatter aspect ratio.
    - Horizontal X-axis labels (Main category only).
    - Sub-labels (Contrast) annotated inside the plot area close to bars.
    - Increased visibility for texts, points, lines, and legend.
    """
    print("Generating L2: Aggregated Category Preference Gap (Refined)...")
    
    # 1. Filter for L2 data
    l2_data = df[df['level'] == 'Target Valuation (L2)'].copy()
    if l2_data.empty:
        print("No L2 data found.")
        return

    # 2. Define Categories and Contrasts
    CATEGORY_DEFINITIONS = {
        'Species': {'Group_A': ['human'], 'Group_B': ['non-human']},
        'Age': {'Group_A': ['infant', 'child', 'teenager'], 'Group_B': ['elderly', 'middle-age']},
        'Gender': {'Group_A': ['female'], 'Group_B': ['male']},
        'Profession': {'Group_A': ['low', 'high'], 'Group_B': ['criminal']},
        'Fitness': {'Group_A': ['normal'], 'Group_B': ['unhealthy']},
        'Wealth': {'Group_A': ['rich'], 'Group_B': ['poor']},
        'Education': {'Group_A': ['well-educated'], 'Group_B': ['low-educated']}
    }

    CATEGORY_CONTRASTS = {
        'Species': 'Human - Non-human',
        'Age': 'Young - Old',
        'Gender': 'Female - Male',
        'Profession': 'Civilian - Criminal',
        'Fitness': 'Healthy - Unhealthy',
        'Wealth': 'Rich - Poor',
        'Education': 'High Edu - Low Edu'
    }

    palette = MODALITY_PALETTE

    def map_feature_to_group(feature_val):
        val = str(feature_val).lower().strip()
        for cat, groups in CATEGORY_DEFINITIONS.items():
            if val in groups['Group_A']: return cat, 'Group_A'
            if val in groups['Group_B']: return cat, 'Group_B'
        return None, None

    # 3. Process Data
    processed_rows = []
    for _, row in l2_data.iterrows():
        if not row['features']: continue
        variable_name, feature_val = row['features'][0]
        
        try:
            direction = get_target_sign(row['dilemma'], variable_name)
        except NameError:
            print("Error: get_target_sign function not found.")
            return

        if direction == 0: continue
        
        adjusted_score = direction * row['log_or']
        cat, group = map_feature_to_group(feature_val)
        
        if cat:
            processed_rows.append({
                'model_type': row['model_type'],
                'modality': row['modality'],
                'category': cat,
                'group': group,
                'score': adjusted_score
            })

    if not processed_rows:
        print("No matching categories found.")
        return

    proc_df = pd.DataFrame(processed_rows)

    # 4. Calculate Gap
    agg_df = proc_df.groupby(['model_type', 'modality', 'category', 'group'])['score'].mean().unstack()
    if 'Group_A' not in agg_df.columns: agg_df['Group_A'] = np.nan
    if 'Group_B' not in agg_df.columns: agg_df['Group_B'] = np.nan
    
    agg_df['gap'] = agg_df['Group_A'] - agg_df['Group_B']
    plot_data = agg_df.reset_index().dropna(subset=['gap'])

    present_cats = [c for c in list(CATEGORY_DEFINITIONS.keys()) if c in plot_data['category'].unique()]
    
    # 5. Visualization Setup
    # Updated figsize for compactness while maintaining readability
    f, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=(12, 6),
                                          gridspec_kw={'height_ratios': [1, 10]})
    f.subplots_adjust(hspace=0.06)

    # Plot Data on BOTH axes
    for ax in (ax_top, ax_bottom):
        sns.barplot(
            data=plot_data, x='category', y='gap', hue='modality',
            order=present_cats, hue_order=['Text', 'Caption', 'Image'],
            palette=palette, errorbar=('se', 1), 
            capsize=0.5, errwidth=3, # Increased error bar visibility
            edgecolor='white', linewidth=2, alpha=0.9, ax=ax, # Thicker lines
            width=1.0
        )
        sns.stripplot(
            data=plot_data, x='category', y='gap', hue='modality',
            order=present_cats, hue_order=['Text', 'Caption', 'Image'],
            dodge=True, jitter=True, color='black', size=10, alpha=0.6, # Larger dots
            legend=False, ax=ax
        )
        if ax.get_legend(): ax.get_legend().remove()

    # --- Axis Breaks & Limits ---
    y_max = plot_data['gap'].max()
    y_min = plot_data['gap'].min()
    
    break_low = 1.1
    break_high = 1.5
    
    if y_max < break_low:
        break_low = y_max + 0.1
        break_high = y_max + 0.2

    # Extend bottom limit slightly more to fit the labels below the bars
    ax_bottom.set_ylim(y_min - 0.2, break_low) 
    ax_top.set_ylim(break_high, y_max + 0.2)

    # Styling the break lines
    ax_top.spines['bottom'].set_visible(False)
    ax_bottom.spines['top'].set_visible(False)
    ax_top.xaxis.tick_top()
    ax_top.tick_params(labeltop=False)
    ax_bottom.xaxis.tick_bottom()

    d = .01 
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=1.5)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # --- Labels & Annotation ---
    ax_top.set_ylabel("")
    ax_top.set_xlabel("")
    f.text(
        0.01, 0.5, "Value Preference Strength",
        va='center',
        rotation='vertical',
        fontsize=25,
        fontweight='bold'
    )
    ax_bottom.set_ylabel("")
    ax_bottom.set_xlabel("")
    
    # 1. Main X-Axis Labels (Larger and Bolder)
    ax_bottom.set_xticklabels(present_cats, rotation=0, ha='center', fontsize=18, fontweight='bold')
    ax_bottom.tick_params(axis='y', labelsize=20)
    ax_top.tick_params(axis='y', labelsize=20)

    # 2. Sub-labels (Refined Positioning: Close to bars)
    text_y_map = [-0.17, 0.55, -0.12, 0.5, -0.4, 0.12, -0.5]
    for i, cat in enumerate(present_cats):
        sub_label = f"({CATEGORY_CONTRASTS.get(cat, '')})"
        
        # Logic: First 4 categories above 0, Last 3 categories below 0
        if i < 4 and i != 0: 
            # Place above zero, clearing standard error bars (approx 0.4-0.5)
            # Species is handled by top axis visually, but text sits on bottom axis
            va_align = 'bottom'
        else:
            # Place below zero (Wealth/Education go down to -0.3/0.4)
            va_align = 'top'
        text_y = text_y_map[i]

        ax_bottom.text(
            x=i, 
            y=text_y, 
            s=sub_label, 
            ha='center', 
            va=va_align, 
            fontsize=18, # Larger font
            color='#444444',
            style='italic',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5)
        )

    ax_bottom.axhline(0, color='black', linewidth=3, linestyle='--')

    # --- Legend (Larger and clearer) ---
    legend_elements = [
        mpatches.Patch(facecolor=palette['Text'], label='Text'),
        mpatches.Patch(facecolor=palette['Caption'], label='Caption'),
        mpatches.Patch(facecolor=palette['Image'], label='Image')
    ]
    
    ax_bottom.legend(
        handles=legend_elements, 
        loc='upper right', 
        ncol=1, 
        fontsize=20, # Larger font
        frameon=True,
        framealpha=1.0, # Fully opaque background to block grid lines
        edgecolor='#bbbbbb',
        borderpad=0.6
    )

    plt.tight_layout()
    f.subplots_adjust(left=0.12)

    # Save
    out_pdf = os.path.join(save_dir, 'L2_target_gap_aggregated.pdf')
    out_png = os.path.join(save_dir, 'L2_target_gap_aggregated.png')
    plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved Refined L2 Gap plot to {out_png}")

def plot_l3_interaction_clean_fixed(df, save_dir):
    print("Generating L3: Cleaned Heatmap (Fixed Spacing)...")
    l3_data = df[df['level'] == 'Target-Target (L3)'].copy()
    if l3_data.empty: return

    def get_label(row):
        f = row['features']
        if len(f) < 2: return None
        v1, v2 = sorted([f[0][1], f[1][1]]) 
        if v1 == v2: return None 
        return f"{v1} vs {v2}"

    l3_data['label'] = l3_data.apply(get_label, axis=1)
    l3_data = l3_data.dropna(subset=['label'])
    l3_data['magnitude'] = l3_data['log_or'].abs()
    
    pivot = l3_data.pivot_table(
        index='label', 
        columns=['model_type', 'modality'], 
        values='magnitude', 
        aggfunc='mean'
    )
    
    # Construct DF with spacers logic
    models = sorted(l3_data['model_type'].unique())
    modalities = ['Text', 'Caption', 'Image']
    
    plot_df = pd.DataFrame(index=pivot.index)
    
    for i, m in enumerate(models):
        # Add model modalities
        for mode in modalities:
            if (m, mode) in pivot.columns:
                plot_df[f"{m}\n{mode}"] = pivot[(m, mode)]
            else:
                plot_df[f"{m}\n{mode}"] = np.nan
        
        # Add spacer AFTER every model block
        # Using a unique name for the spacer column to prevent overwrite
        if i < len(models) - 1:
            plot_df[f"   .{i}.   "] = np.nan 

    # Plot
    plt.figure(figsize=(20, max(8, len(plot_df)*0.5)))
    sns.heatmap(plot_df, cmap='Reds', annot=True, fmt=".1f", 
                mask=plot_df.isna(), linewidths=0.5, linecolor='white')
    
    plt.title('L3: Conflict Sensitivity Heatmap (Abs LogOR)', fontsize=16)
    plt.xlabel("")
    plt.ylabel("Conflicting Attributes")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'L3_conflict_heatmap_fixed.png'), dpi=300)
    plt.close()

def plot_l4_homophily_aggregated_bar(df, save_dir):
    print("Generating L4: Aggregated Homophily Bar Chart...")
    l4_data = df[df['level'] == 'Agent-Target (L4)'].copy()
    if l4_data.empty: return

    def get_homophily_score(row):
        feats = row['features']
        if len(feats) < 2: return None
        
        agent_v, target_v = None, None
        target_role = None
        for r, v in feats:
            if r == 'agent': agent_v = v
            else: target_v = v; target_role = r
            
        if not agent_v or not target_v: return None
        
        # Strict Homophily: Agent Value must equal Target Value
        if agent_v != target_v: return None 
        
        sign = get_target_sign(row['dilemma'], target_role)
        # Score > 0: Favor Own Group. Score < 0: Disfavor Own Group.
        return sign * row['log_or']

    l4_data['homophily_score'] = l4_data.apply(get_homophily_score, axis=1)
    homophily_df = l4_data.dropna(subset=['homophily_score'])
    
    if homophily_df.empty:
        print("No homophily pairs found.")
        return

    # Plot: Aggregated Bar Chart
    plt.figure(figsize=(10, 6))
    
    # Using barplot aggregates mean and calculates CI automatically
    sns.barplot(
        data=homophily_df, 
        x='model_type', y='homophily_score', hue='modality',
        hue_order=['Text', 'Caption', 'Image'],
        palette={'Text': '#1f77b4', 'Caption': '#2ca02c', 'Image': '#d62728'},
        capsize=0.1, errwidth=1.5, edgecolor='white'
    )
    
    plt.axhline(0, color='black', linewidth=1)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.title("L4: Overall Homophily Bias (Agent == Target Pairs)", fontsize=16, pad=20)
    plt.ylabel("Mean Homophily Score\n(>0: Favors Own Group, <0: Disfavors)")
    plt.xlabel("Model")
    plt.legend(title='Modality')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'L4_homophily_aggregated.png'), dpi=300)
    plt.close()

def main():
    result_path = f"{ROOT}/../results/single_feature/analyze_results"
    data_list = []
    
    for model_str in MODEL_LIST:
        file_path = f"{result_path}/character_factor_{model_str}.yaml"
        prepare_results(model_str, file_path, data_list)
    
    if not data_list:
        print("No data found.")
        return

    df = pd.DataFrame(data_list)
    vis_dir = f'{ROOT}/../visualization/single_feature/character_factor_general'
    if not os.path.exists(vis_dir): os.makedirs(vis_dir)
    
    # # 1. L1 Grouped by Variable Name
    # plot_l1_stacked_bar_grouped(df, vis_dir)
    
    # # 2. L2 Grouped (Grouped helper reused inside)
    # plot_l2_target_valuation_grouped(df, vis_dir)
    # plot_l2_category_gap(df, vis_dir)
    plot_l2_category_gap_aggregated(df, vis_dir)
    
    # # 3. L3 Cleaned Spacing
    # plot_l3_interaction_clean_fixed(df, vis_dir)
    
    # # 4. L4 Aggregated Bar
    # plot_l4_homophily_aggregated_bar(df, vis_dir)
    
    print(f"Visualization complete. Saved to {vis_dir}")

if __name__ == '__main__':
    main()