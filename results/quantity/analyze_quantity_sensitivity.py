import argparse
import pandas as pd
import numpy as np
import os
import yaml
import re

from config.constants import ROOT

def parse_quantity_info(q_str):
    """
    Parse 'XvsY' string to extract Cost, Benefit, and Net Benefit.
    Assumption: Format is 'Sacrifice vs Saved' (Cost vs Benefit).
    
    Examples:
        '1vs5' -> Sacrifice 1, Save 5 -> Net Benefit = 4
        '5vs1' -> Sacrifice 5, Save 1 -> Net Benefit = -4
        '1vs1' -> Net Benefit = 0
    """
    try:
        # Handle cases where it might already be a number (rare)
        if isinstance(q_str, (int, float)):
            return 0, 0, 0 

        # Extract all numbers from string
        nums = [int(x) for x in re.findall(r'\d+', str(q_str))]
        
        if len(nums) >= 2:
            cost = nums[0]
            benefit = nums[1]
            net_benefit = benefit - cost
            return cost, benefit, net_benefit
        elif len(nums) == 1:
            # Fallback for old formats if any, though likely not useful for this analysis
            return nums[0], 0, 0
        return 0, 0, 0
    except:
        return 0, 0, 0

def analyze_model(model_name, mode):
    dataset_name = "quantity"
    model_str = f"{model_name}_{mode}"
    
    base_path = os.path.join(ROOT, "..", "results", dataset_name, model_name)
    file_name = f"results_{mode}.yaml"
    file_path = os.path.join(base_path, file_name)
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    print(f"Loading {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading YAML: {e}")
        return

    all_records = []
    total_count = 0
    refusal_count = 0
    
    if data:
        for dilemma_key, instances in data.items():
            if not isinstance(instances, dict): continue
            for inst_key, results_list in instances.items():
                if not isinstance(results_list, list): continue
                for res in results_list:
                    total_count += 1
                    raw_ans = res.get('answer', 0)

                    if raw_ans == 0:
                        refusal_count += 1
                        continue

                    # Convert answer to binary action (1=Yes/Action, 0=No/Inaction)
                    val = 1 if raw_ans == 1 else 0
                    
                    q_str = res.get('quantity_level', '')
                    # [MODIFIED] Use new parsing logic
                    cost, benefit, net_benefit = parse_quantity_info(q_str)
                    
                    # Filter out invalid parses if necessary, but keep 0 (1vs1)
                    if cost > 0 or benefit > 0:
                        all_records.append({
                            'Dilemma': res.get('dilemma', dilemma_key),
                            'Instance': inst_key,
                            'Cost': cost,
                            'Benefit': benefit,
                            'Net_Benefit': net_benefit, # [NEW] Key metric for analysis
                            'Action': val
                        })
    
    # Calculate Refusal Rate
    refusal_rate = refusal_count / total_count if total_count > 0 else 0.0
    print(f"Total Samples: {total_count}, Refusals: {refusal_count} ({refusal_rate:.2%})")
    
    if not all_records:
        print(f"No valid records found for {model_str}.")
        return

    df = pd.DataFrame(all_records)
    
    # --- Statistics Calculation (Updated for Net Benefit) ---
    
    # 1. Instance Level Stats
    # Group by Net_Benefit instead of just Quantity
    df_instance_stats = df.groupby(['Dilemma', 'Instance', 'Net_Benefit'])['Action'].mean().reset_index()
    
    # 2. Per Dilemma Stats
    df_dilemma_stats = df_instance_stats.groupby(['Dilemma', 'Net_Benefit'])['Action'].mean().reset_index()
    
    # 3. Global Stats
    df_global_stats = df_dilemma_stats.groupby(['Net_Benefit'])['Action'].mean().reset_index()

    # 4. Linear Regression Slope (Sensitivity to Net Benefit)
    # A positive slope means the model is sensitive to utility (more benefit = more action).
    # A zero slope means the model ignores the utility trade-off.
    slopes_data = []
    
    for dilemma, group in df_dilemma_stats.groupby('Dilemma'):
        x = group['Net_Benefit'].values
        y = group['Action'].values
        
        # Need at least 2 points to fit a line
        if len(np.unique(x)) >= 2:
            try:
                slope, intercept = np.polyfit(x, y, 1)
                
                # Calculate R-squared
                p = np.poly1d([slope, intercept])
                y_hat = p(x)
                y_bar = np.mean(y)
                ss_tot = np.sum((y - y_bar)**2)
                ss_res = np.sum((y - y_hat)**2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                slopes_data.append({
                    'Dilemma': dilemma,
                    'Slope': float(slope),
                    'Intercept': float(intercept),
                    'R2': float(r_squared),
                    'Points': int(len(x))
                })
            except Exception as e:
                print(f"Error fitting for {dilemma}: {e}")

    # --- Save Results ---
    output_dir = os.path.join(ROOT, "..", "results", dataset_name, "analyze_results")
    os.makedirs(output_dir, exist_ok=True)
    
    output_yaml = os.path.join(output_dir, f"quantity_sensitivity_{model_str}.yaml")
    
    stats_dict = {
        'model': model_name,
        'modality': mode,
        'data_info': {
            'total_samples': total_count,
            'refusals': refusal_count,
            'refusal_rate': float(refusal_rate),
            'valid_samples': len(df)
        },
        'dilemma_stats': df_dilemma_stats.to_dict(orient='records'),
        'global_stats': df_global_stats.to_dict(orient='records'),
        'slopes': slopes_data
    }
    
    with open(output_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(stats_dict, f, sort_keys=False, allow_unicode=True)

    print(f"Analysis saved to {output_yaml}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--mode', type=str, default='text', help="'text' (Text only), 'image' (Image + Text), 'caption' (Image -> Text)")

    args = parser.parse_args()
    
    analyze_model(args.model_name, args.mode)