import argparse
import pandas as pd
import glob
import yaml
import os

from config.constants import ROOT
from results.utils import logistic_regression
from visualization.utils import DILEMMA_DIMENSION_MAP

# New helper function to perform hierarchical regression analysis
def run_hierarchical_analysis(data):
    """
    Perform hierarchical logistic regression in 3 steps:
    1. Main effects model (A + B + C) -> Extract Main Effects
    2. 2-way interaction model (A + B + C + AB + BC + CA) -> Extract 2-way Interactions
    3. 3-way interaction model (Full factorial) -> Extract 3-way Interactions
    
    Returns:
        pd.DataFrame: A combined dataframe containing selected coefficients from appropriate models.
    """
    
    # Define variable names for formula construction
    var_pf = "C(personal_force, Treatment(reference=0))"
    var_ih = "C(intention_of_harm, Treatment(reference=0))"
    var_sb = "C(self_benefit, Treatment(reference=0))"

    # Step 1: Main Effects Model
    formula_main = f"answer ~ {var_pf} + {var_ih} + {var_sb}"
    # print(f"Running Main Effects Model: {formula_main}")
    results_main = logistic_regression(data, formula_main)
    
    # Step 2: 2-way Interaction Model
    # Using **2 is shorthand for main effects + all 2-way interactions
    formula_2way = f"answer ~ ({var_pf} + {var_ih} + {var_sb})**2"
    # print(f"Running 2-way Interaction Model: {formula_2way}")
    results_2way = logistic_regression(data, formula_2way)
    
    # Step 3: 3-way Interaction Model (Full Model)
    # Using * is shorthand for main effects + all interactions up to 3-way
    formula_3way = f"answer ~ {var_pf} * {var_ih} * {var_sb}"
    # print(f"Running 3-way Interaction Model: {formula_3way}")
    results_3way = logistic_regression(data, formula_3way)

    combined_results = []

    # Helper to check if a result dataframe is valid
    def is_valid(df):
        return df is not None and not df.empty and 'Error' not in df.index

    # --- Extract from Model 1 (Main Effects) ---
    if is_valid(results_main):
        for idx, row in results_main.iterrows():
            var_name = str(idx)
            # Main effects and Intercept contain NO colons (e.g., "C(personal_force)[T.1]")
            if var_name.count(':') == 0:
                # Add a metadata column to indicate source model (optional, for debugging)
                # row['Source_Model'] = 'Main_Effects' 
                combined_results.append(row)

    # --- Extract from Model 2 (2-way Interactions) ---
    if is_valid(results_2way):
        for idx, row in results_2way.iterrows():
            var_name = str(idx)
            # 2-way interactions contain exactly ONE colon (e.g., "C(A)[T.1]:C(B)[T.1]")
            if var_name.count(':') == 1:
                combined_results.append(row)

    # --- Extract from Model 3 (3-way Interactions) ---
    if is_valid(results_3way):
        for idx, row in results_3way.iterrows():
            var_name = str(idx)
            # 3-way interactions contain exactly TWO colons
            if var_name.count(':') == 2:
                combined_results.append(row)

    if not combined_results:
        return None
    
    # Combine back into a DataFrame with the same structure as the original output
    return pd.DataFrame(combined_results)


def main():
    parser = argparse.ArgumentParser(description="Analyze conceptual foctors.")
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
    
    # Formula definition moved inside the hierarchical analysis function
    # formula = "answer ~ C(personal_force, Treatment(reference=0)) * C(intention_of_harm, Treatment(reference=0)) * C(self_benefit, Treatment(reference=0))"
    
    os.makedirs(f"{ROOT}/../results/single_feature/analyze_results", exist_ok=True)
    analyze_results = {}

    care_dilemmas = set(DILEMMA_DIMENSION_MAP.get('Care vs Care', []))
    print(f"Targeting Care vs Care dilemmas: {care_dilemmas}")

    with pd.ExcelWriter(f"{ROOT}/../results/single_feature/analyze_results/conceptual_factor_{args.model_name}_{args.mode}.xlsx") as w:
        total_data = pd.DataFrame(columns=['personal_force', 'intention_of_harm', 'self_benefit', 'answer'])
        care_data = pd.DataFrame(columns=['personal_force', 'intention_of_harm', 'self_benefit', 'answer'])
        
        for file_path in result_files:
            dilemma = file_path.split('/')[-1].rpartition('_')[0]
            print(f"Processing {dilemma}")
            sheets = pd.read_excel(file_path, sheet_name=None)
            data = pd.DataFrame(columns=['personal_force', 'intention_of_harm', 'self_benefit', 'answer'])

            for sheet_name, sheet_df in sheets.items():
                result_df = sheet_df.copy()
                result_df.columns = result_df.iloc[1]
                result_df = result_df.iloc[2:].reset_index(drop=True)

                result_df = result_df[result_df['answer'].isin([-1, 1])]
                result_df['answer'] = result_df['answer'].replace(-1, 0)

                result_df['personal_force'] = int(sheet_df['personal_force'][0])
                result_df['intention_of_harm'] = int(sheet_df['intention_of_harm'][0])
                result_df['self_benefit'] = int(sheet_df['self_benefit'][0])

                data = pd.concat([data, result_df[['personal_force', 'intention_of_harm', 'self_benefit', 'answer']]], ignore_index=True)

            data = data.astype(int)
            
            # Replaced single logistic_regression call with run_hierarchical_analysis
            # fit_results = logistic_regression(data, formula)
            fit_results = run_hierarchical_analysis(data)
            
            print(f"fit_result for {dilemma}:\n", fit_results)
            if fit_results is not None:
                fit_results.to_excel(w, sheet_name=dilemma, index=True, header=True, startrow=0)
                analyze_results[dilemma] = fit_results.to_dict(orient='index')
            
            total_data = pd.concat([total_data, data], ignore_index=True)

            if dilemma in care_dilemmas:
                care_data = pd.concat([care_data, data], ignore_index=True)
        
        total_data = total_data.astype(int)
        
        # Replaced single logistic_regression call with run_hierarchical_analysis for total data
        # fit_results = logistic_regression(total_data, formula)
        fit_results = run_hierarchical_analysis(total_data)
        
        print("fit_result for all dilemmas:\n", fit_results)
        if fit_results is not None:
            fit_results.to_excel(w, sheet_name="total", index=True, header=True, startrow=0)
            analyze_results['total'] = fit_results.to_dict(orient='index')

        if not care_data.empty:
            care_data = care_data.astype(int)
            fit_results_care = run_hierarchical_analysis(care_data)
            print("fit_result for Care vs Care:\n", fit_results_care)
            if fit_results_care is not None:
                fit_results_care.to_excel(w, sheet_name="care_vs_care", index=True, header=True, startrow=0)
                analyze_results['care_vs_care'] = fit_results_care.to_dict(orient='index')
            
    with open(f"{ROOT}/../results/single_feature/analyze_results/conceptual_factor_{args.model_name}_{args.mode}.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(analyze_results, f, allow_unicode=True)

if __name__ == '__main__':
    main()