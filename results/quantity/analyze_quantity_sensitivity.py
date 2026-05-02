import argparse
import pandas as pd
import numpy as np
import os
import yaml
import re
from scipy.optimize import curve_fit

from config.constants import ROOT

def parse_quantity_info(q_str):
    """
    Parse 'XvsY' string to extract Cost, Benefit, and Log Ratio.
    """
    try:
        if isinstance(q_str, (int, float)):
            return 0, 0, 0.0 

        nums = [int(x) for x in re.findall(r'\d+', str(q_str))]
        
        if len(nums) >= 2:
            cost = nums[0]
            benefit = nums[1]
            if cost > 0 and benefit > 0:
                log_ratio = np.log10(benefit / cost)
            else:
                log_ratio = 0.0
            return cost, benefit, log_ratio
        elif len(nums) == 1:
            return nums[0], 0, 0.0
        return 0, 0, 0.0
    except:
        return 0, 0, 0.0

def sigmoid(x, L, x0, k, b):
    x_norm = np.clip(-k * (x - x0), -500, 500)
    return L / (1 + np.exp(x_norm)) + b

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
                    cost, benefit, log_ratio = parse_quantity_info(q_str)
                    
                    # Filter out invalid parses if necessary, but keep 0 (1vs1)
                    if cost > 0 or benefit > 0:
                        all_records.append({
                            'Dilemma': res.get('dilemma', dilemma_key),
                            'Instance': inst_key,
                            'Cost': cost,
                            'Benefit': benefit,
                            'Log_Ratio': log_ratio,
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
    
    # # 1. Instance Level Stats
    # # Group by Net_Benefit instead of just Quantity
    # df_instance_stats = df.groupby(['Dilemma', 'Instance', 'Net_Benefit'])['Action'].mean().reset_index()
    
    # # 2. Per Dilemma Stats
    # df_dilemma_stats = df_instance_stats.groupby(['Dilemma', 'Net_Benefit'])['Action'].mean().reset_index()
    
    # # 3. Global Stats
    # df_global_stats = df_dilemma_stats.groupby(['Net_Benefit'])['Action'].mean().reset_index()
    df_instance_stats = df.groupby(['Dilemma', 'Instance', 'Log_Ratio'])['Action'].mean().reset_index()
    df_dilemma_stats = df_instance_stats.groupby(['Dilemma', 'Log_Ratio'])['Action'].mean().reset_index()
    df_global_stats = df_dilemma_stats.groupby(['Log_Ratio'])['Action'].mean().reset_index()

    # 4. Linear Regression Slope (Sensitivity to Net Benefit)
    # A positive slope means the model is sensitive to utility (more benefit = more action).
    # A zero slope means the model ignores the utility trade-off.
    slopes_data = []
    
    for dilemma, group in df_dilemma_stats.groupby('Dilemma'):
        # x = group['Net_Benefit'].values
        x = group['Log_Ratio'].values
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

    # x_global = df_global_stats['Net_Benefit'].values
    x_global = df_global_stats['Log_Ratio'].values
    y_global = df_global_stats['Action'].values

    global_fit = {'k': None, 'R2': None, 'L': None, 'x0': None, 'b': None}

    if len(np.unique(x_global)) >= 4:
        try:
            p0 = [max(y_global) - min(y_global), 0.0, 2.0, min(y_global)]
            bounds = ([0, -1.5, -20, 0], [1.1, 1.5, 20, 1.1])
            popt, pcov = curve_fit(sigmoid, x_global, y_global, p0=p0, bounds=bounds, maxfev=10000)
            y_fit = sigmoid(x_global, *popt)
            
            ss_tot_g = np.sum((y_global - np.mean(y_global))**2)
            ss_res_g = np.sum((y_global - y_fit)**2)
            r2_g = 1 - (ss_res_g / ss_tot_g) if ss_tot_g != 0 else 0
            

            L_val = float(popt[0])
            k_val = float(popt[2])
            max_slope_val = (L_val * k_val) / 4.0
            
            global_fit = {
                'L': L_val,
                'x0': float(popt[1]),
                'k': k_val,
                'b': float(popt[3]),
                'max_slope': max_slope_val,
                'R2': float(r2_g)
            }
        except Exception as e:
            print(f"Global S-curve fit failed for {model_str}: {e}")

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
        'global_fit': global_fit,
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