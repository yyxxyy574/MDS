import argparse
import pandas as pd
import glob
import yaml
import os
import numpy as np

from config.constants import ROOT, DILEMMA
from results.utils import logistic_regression, load_feature_values, select_reference_value
from baseline.utils import get_characters

def run_hierarchical_analysis(data, characters):
    """
    Perform hierarchical logistic regression in 2 steps (for character attributes):
    1. Main effects model (A + B + C) -> Extract Main Effects
    2. 2-way interaction model (A + B + C + AB + BC + CA) -> Extract 2-way Interactions
    
    Returns:
        pd.DataFrame: A combined dataframe containing selected coefficients.
    """
    
    # Format variable names with coding scheme
    vars_coded = [f"C({char}, Sum)" for char in characters]
    
    # Step 1: Main Effects Formula
    formula_main = "answer ~ " + " + ".join(vars_coded)
    
    # Step 2: 2-way Interaction Formula
    formula_2way = "answer ~ (" + " + ".join(vars_coded) + ") ** 2"
    
    results_main = logistic_regression(data, formula_main)
    results_2way = logistic_regression(data, formula_2way)

    combined_results = []

    # Helper to check if a result dataframe is valid
    def is_valid(df):
        return df is not None and not df.empty and 'Error' not in df.index

    # --- Extract from Model 1 (Main Effects) ---
    if is_valid(results_main):
        for idx, row in results_main.iterrows():
            var_name = str(idx)
            # Main effects and Intercept contain NO colons
            if var_name.count(':') == 0:
                combined_results.append(row)

    # --- Extract from Model 2 (2-way Interactions) ---
    if is_valid(results_2way):
        for idx, row in results_2way.iterrows():
            var_name = str(idx)
            # 2-way interactions contain exactly ONE colon
            if var_name.count(':') == 1:
                combined_results.append(row)

    if not combined_results:
        return None
    
    return pd.DataFrame(combined_results)


def main():
    parser = argparse.ArgumentParser(description="Analyze character factors.")
    parser.add_argument('--model-name', type=str, required=True, help='vlm/llm model name (e.g., models/deepseek-vl-7b-chat).')
    parser.add_argument('--mode', type=str, default='text', help="'text' (Text only), 'image' (Image + Text), 'caption' (Image -> Text)")

    args = parser.parse_args()

    result_path = f"{ROOT}/../results/single_feature/{args.model_name}"
    if not os.path.exists(result_path):
        print(f"Error: Result path {result_path} does not exist.")
        return
    
    result_files = [f for f in glob.glob(f'{result_path}/*_{args.mode}.xlsx')]
    if not result_files:
        print(f"Error: There is no file like *_{args.mode}.xlsx - *.csv in {result_path}.")
    
    analyze_results = {}
    global_feature_data = {}
    os.makedirs(f"{ROOT}/../results/single_feature/analyze_results", exist_ok=True)
    
    output_excel_path = f"{ROOT}/../results/single_feature/analyze_results/character_factor_{args.model_name}_{args.mode}.xlsx"
    
    with pd.ExcelWriter(output_excel_path) as w:
        for file_path in result_files:
            dilemma = file_path.split('/')[-1].rpartition('_')[0]
            print(f"Processing {dilemma}")
            sheets = pd.read_excel(file_path, sheet_name=None)
            
            data_by_feature = {}

            # 1. Aggregate data first
            for sheet_name, sheet_df in sheets.items():
                if 'dimension' not in sheet_df.columns:
                    continue
                    
                dimension = sheet_df['dimension'][0]

                # Clean up header
                result_df = sheet_df.copy()
                result_df.columns = result_df.iloc[1]
                result_df = result_df.iloc[2:].reset_index(drop=True)

                characters = get_characters(dimension, dilemma, sheet_name)

                # Group by feature within this sheet
                for feature, sub_df in result_df.groupby('feature'):
                    if feature not in data_by_feature:
                        data_by_feature[feature] = pd.DataFrame(columns=characters + ['answer'])

                    sub_result_df = sub_df.copy()
                    
                    # Logic to handle 'same_quantity_bias'
                    if len(characters) == 3:
                        same_character = None
                        dilemma_info = DILEMMA[dimension][dilemma][sheet_name]
                        if 'character' in dilemma_info:
                             for character in dilemma_info['character']:
                                if 'same_quantity_bias' in dilemma_info['character'][character]:
                                    same_character = character
                                    break
                        if same_character is not None:
                            sub_result_df = sub_result_df[sub_result_df['agent'] == sub_result_df[same_character]]
                    
                    sub_result_df = sub_result_df[sub_result_df['answer'].isin([-1, 1])]
                    if sub_result_df.empty: 
                        continue
                    sub_result_df['answer'] = sub_result_df['answer'].replace(-1, 0)

                    data_by_feature[feature] = pd.concat([data_by_feature[feature], sub_result_df[characters + ['answer']]], ignore_index=True)

            # 2. Analyze aggregated data for each feature
            for feature, feature_data in data_by_feature.items():
                current_characters = [c for c in feature_data.columns if c != 'answer']
                
                print(f"Analyzing {dilemma} - feature: {feature}")
                
                feature_data['answer'] = pd.to_numeric(feature_data['answer'], errors='coerce').dropna()
                
                if feature_data.empty:
                    continue

                # This helps distinguish between "Neutrality" and "Blind Action"
                action_rate = feature_data['answer'].mean()
                sample_size = len(feature_data)

                # Run Hierarchical Analysis
                fit_results = run_hierarchical_analysis(feature_data, current_characters)
                
                if fit_results is not None:
                    # Write to Excel
                    sheet_name = f"{dilemma}_{feature}"[:31]
                    fit_results.to_excel(w, sheet_name=sheet_name, index=True, header=True, startrow=0)
                    
                    # Convert to dict for YAML
                    res_dict = fit_results.to_dict(orient='index')
                    
                    # We use a special key 'Global_Stats' to store meta-info
                    res_dict['Global_Stats'] = {
                        'Action_Rate': float(action_rate),
                        'Sample_Size': int(sample_size)
                    }
                    
                    analyze_results[f'{dilemma}_{feature}'] = res_dict

                if feature not in global_feature_data:
                    global_feature_data[feature] = feature_data.copy()
                else:
                    global_feature_data[feature] = pd.concat([global_feature_data[feature], feature_data], ignore_index=True)

    yaml_path = f"{ROOT}/../results/single_feature/analyze_results/character_factor_{args.model_name}_{args.mode}.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(analyze_results, f, allow_unicode=True)
    
    print(f"Analysis Done. Saved to {yaml_path}")

if __name__ == '__main__':
    main()