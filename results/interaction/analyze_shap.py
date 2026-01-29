import argparse
import yaml
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import itertools
import math
import os
import glob
import warnings
import time
import joblib
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from config.constants import ROOT
from data.interaction.generate import AGENTS

MODEL_PARAMS = {
    'n_estimators': 500,
    'max_depth': 6,
    'random_state': 42
}

def detect_character_count(samples):
    max_config = max([int(s['config']) for s in samples])
    n_characters = math.ceil(math.log(max_config + 1) / math.log(len(AGENTS)))
    return n_characters

def generate_configs(n_characters):
    configs = []
    for config in itertools.product(AGENTS, repeat=n_characters):
        feature_config = {}
        for i in range(n_characters):
            character_name = f'person{i+1}'
            feature_config[character_name] = {
                'color': config[i][0],
                'profession': config[i][1],
                'gender': config[i][2]
            }
        configs.append(feature_config)
    return configs

def config_to_flat_dict(feature_config):
    flat = {}
    for character in sorted(feature_config.keys()):
        char_features = feature_config[character]
        for feat_name, feat_value in char_features.items():
            flat[f"{character}_{feat_name}"] = feat_value
    return flat

def load_and_process_yaml(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    all_rows = []
    
    for dilemma_type, instances in data.items():
        if not isinstance(instances, dict):
            continue
        
        for instance, samples in instances.items():
            if not isinstance(samples, list):
                continue

            n_characters = detect_character_count(samples)
            configs = generate_configs(n_characters)
            
            for sample in samples:
                try:
                    row = {}
                    answer = sample['answer']
                    row['answer'] = answer
                    row['quantity_level'] = sample['quantity_level']
                    
                    config_idx = int(sample['config'])
                    if config_idx >= len(configs):
                        continue
                    
                    feature_config = configs[config_idx]
                    row.update(config_to_flat_dict(feature_config))
                    all_rows.append(row)
                except Exception:
                    continue
            
    df = pd.DataFrame(all_rows)
    if not df.empty:
        cols = [c for c in df.columns if c != 'answer'] + ['answer']
        df = df[cols]
    return df

def run_ml_pipeline(df, model_name, model):
    target_col = 'answer'
    
    X = df.drop(columns=[target_col])
    y = df[target_col].map({-1: 0, 1: 1})

    unique_targets = y.unique()
    if len(unique_targets) < 2:
        raise ValueError(f"Target variable contains only one unique value: {unique_targets}")

    # One-hot encode the base features
    X_encoded = pd.get_dummies(X, prefix_sep='=', dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=MODEL_PARAMS['random_state'], stratify=y
    )

    print(f"      Training {model_name}...")
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"      ✓ Training completed")
    print(f"      - Training accuracy: {train_acc:.4f}")
    print(f"      - Test accuracy: {test_acc:.4f}")

    print(f"      Calculating SHAP Interaction values (this may take time)...")
    
    # Use TreeExplainer
    explainer = shap.TreeExplainer(model)
    
    # Returns matrix of shape (samples, features, features) or (samples, features, features, classes)
    shap_interaction_values = explainer.shap_interaction_values(X_test)
    
    # ================= MODIFIED DIMENSION HANDLING =================
    # Handle list output (common in binary classification: [neg_class, pos_class])
    if isinstance(shap_interaction_values, list):
        shap_interaction_values = shap_interaction_values[1]
    
    # Handle 4D output: It can be (N, M, M, Classes) or (N, Classes, M, M)
    if len(shap_interaction_values.shape) == 4:
        # Case 1: (Samples, Features, Features, Classes) - Standard for many SHAP versions
        if shap_interaction_values.shape[-1] == 2: 
            shap_interaction_values = shap_interaction_values[:, :, :, 1]
        
        # Case 2: (Samples, Classes, Features, Features) - Less common but possible
        elif shap_interaction_values.shape[1] == 2:
             shap_interaction_values = shap_interaction_values[:, 1, :, :]
             
        else:
            # Fallback: Assume last dimension is classes if shape matches 2
            # Or print warning
            print(f"      ⚠️ Warning: Ambiguous SHAP shape {shap_interaction_values.shape}. Attempting default slicing.")
            shap_interaction_values = shap_interaction_values[:, :, :, 1]
            
    # ===============================================================
    
    print(f"      ✓ SHAP interactions calculated (shape: {shap_interaction_values.shape})")
    
    return X_train, X_test, shap_interaction_values, train_acc, test_acc

def save_results(X_test, shap_interaction_values, save_dir, model_name_arg, mode, classifier_name, train_acc, test_acc):
    # shap_interaction_values shape: (N_samples, M_features, M_features)
    
    # 1. Calculate Mean Absolute Importance Matrix (Global Importance)
    mean_abs_interaction = np.abs(shap_interaction_values).mean(axis=0) # Shape (M, M)
    total_importance = np.sum(mean_abs_interaction) # Sum of all cells
    
    feature_names = X_test.columns.tolist()
    n_features = len(feature_names)
    
    # Debug print to verify shapes matching
    if mean_abs_interaction.shape[0] != n_features:
        print(f"      ❌ Critical Error: SHAP matrix shape {mean_abs_interaction.shape} does not match feature count {n_features}")
        return pd.DataFrame() # Return empty to safely skip

    rows = []
    
    # 2. Extract Main Effects (Diagonal)
    for i in range(n_features):
        feat_name = feature_names[i]
        importance = mean_abs_interaction[i, i]
        
        # Calculate direction for Main Effects
        raw_vals = shap_interaction_values[:, i, i]
        feature_vals = X_test.iloc[:, i].values
        
        # Simple correlation to determine direction for main effects
        if np.std(feature_vals) == 0 or np.std(raw_vals) == 0:
            impact_code = "Unknown"
        else:
            corr = np.corrcoef(feature_vals, raw_vals)[0, 1]
            impact_code = "POS" if corr > 0 else "NEG"
            
        direction_str = "Positive (Promotes 1)" if impact_code == "POS" else "Negative (Promotes 0)"
        
        rows.append({
            "Feature": feat_name,
            "Type": "Main Effect",
            "Importance": importance,
            "Direction": direction_str,
            "Impact_Code": impact_code,
            "Component_A": feat_name,
            "Component_B": "-"
        })

    # 3. Extract Interaction Effects (Off-Diagonal)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            # Combined importance of i-j and j-i
            importance = mean_abs_interaction[i, j] * 2 
            
            if importance < 1e-6:
                continue
                
            feat_i = feature_names[i]
            feat_j = feature_names[j]
            interaction_name = f"{feat_i} & {feat_j}"
            
            # Determine direction for Interactions
            raw_vals = shap_interaction_values[:, i, j]
            avg_val = np.mean(raw_vals) * 2
            impact_code = "POS" if avg_val > 0 else "NEG"
            direction_str = "Synergy (+)" if impact_code == "POS" else "Interference (-)"

            rows.append({
                "Feature": interaction_name,
                "Type": "Interaction",
                "Importance": importance,
                "Direction": direction_str,
                "Impact_Code": impact_code,
                "Component_A": feat_i,
                "Component_B": feat_j
            })

    # 4. Create DataFrame and Calculate Normalized Importance
    report_df = pd.DataFrame(rows)
    if not report_df.empty:
        report_df = report_df.sort_values(by="Importance", ascending=False)
        extracted_total = report_df['Importance'].sum()
        report_df['Normalized_Importance'] = report_df['Importance'] / extracted_total if extracted_total > 0 else 0
    
    csv_filename = f"interaction_analysis_{classifier_name}.csv"
    html_filename = f"interaction_report_{classifier_name}.html"
    
    report_df.to_csv(os.path.join(save_dir, csv_filename), index=False)
    
    html_content = generate_html_string(report_df, model_name_arg, mode, classifier_name, train_acc, test_acc)
    with open(os.path.join(save_dir, html_filename), "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"      Top 10 Most Important Effects (Main & Interactions):")
    for idx, row in report_df.head(10).iterrows():
        print(f"      {row['Feature']:<50} | Type: {row['Type']:<12} | Imp: {row['Importance']:>8.4f}")
    
    return report_df

def generate_html_string(df, model_name, mode, classifier_name, train_acc, test_acc):
    if df.empty:
        return "<html><body>No data available</body></html>"

    table_rows = ""
    max_imp = df['Importance'].max() if len(df) > 0 else 1
    
    for rank, (idx, row) in enumerate(df.iterrows(), 1):
        dir_cls = "text-success" if row['Impact_Code'] == "POS" else "text-danger"
        dir_icon = "▲" if row['Impact_Code'] == "POS" else "▼"
        width_pct = (row['Importance'] / max_imp) * 100
        
        if row['Type'] == "Main Effect":
            bar_color = "bg-primary"
            type_badge = "bg-primary"
            row_bg = ""
        else:
            bar_color = "bg-warning"
            type_badge = "bg-warning text-dark"
            row_bg = "background-color: #fffae6;"

        if row['Type'] == "Interaction":
            constituents_html = f"""
                <span class="badge bg-light text-dark border">{row['Component_A']}</span><br>
                <span class="text-muted" style="font-size:0.8em;">+</span><br>
                <span class="badge bg-light text-dark border">{row['Component_B']}</span>
            """
        else:
            constituents_html = f"<span class='badge bg-light text-muted border'>{row['Feature']}</span>"

        table_rows += f"""
            <tr style="{row_bg}">
                <td>{rank}</td>
                <td><div class="fw-bold text-break" style="font-size:0.9em;">{row['Feature']}</div></td>
                <td><span class="badge {type_badge}">{row['Type']}</span></td>
                <td>
                    <div class="d-flex align-items-center">
                        <span class="me-2" style="width:50px;">{row['Importance']:.4f}</span>
                        <div class="progress flex-grow-1" style="height: 6px;">
                            <div class="progress-bar {bar_color}" style="width: {width_pct}%"></div>
                        </div>
                    </div>
                </td>
                <td>{row['Normalized_Importance']:.6f}</td>
                <td class="{dir_cls} fw-bold" style="font-size:0.9em;">{row['Direction']} {dir_icon}</td>
                <td style="font-size:0.85em; line-height:1.1;">{constituents_html}</td>
            </tr>
        """

    return f"""
    <!DOCTYPE html>
    <html lang="zh">
    <head>
        <meta charset="UTF-8">
        <title>SHAP Interaction Analysis Report</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdn.datatables.net/1.13.4/css/dataTables.bootstrap5.min.css" rel="stylesheet">
        <style>
            body {{ background-color: #f4f6f9; font-family: 'Segoe UI', sans-serif; padding: 20px; }}
            .card {{ border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }}
            .badge {{ font-weight: 500; }}
            .breadcrumb {{ background: transparent; padding: 0; margin-bottom: 0; }}
            .accuracy-badge {{ padding: 8px 16px; border-radius: 6px; font-size: 0.9em; }}
        </style>
    </head>
    <body>
    <div class="container-fluid">
        <div class="card">
            <div class="card-header bg-white p-4 border-bottom">
                <nav aria-label="breadcrumb">
                  <ol class="breadcrumb">
                    <li class="breadcrumb-item text-muted">{model_name}</li>
                    <li class="breadcrumb-item text-muted">{mode}</li>
                    <li class="breadcrumb-item active" aria-current="page">{classifier_name}</li>
                  </ol>
                </nav>
                <h3 class="text-primary mt-2">SHAP Interaction Analysis Report (Native)</h3>
                <p class="text-muted">Analysis based on SHAP Interaction Values (Main Effects + 2-Way Interactions)</p>
                <div class="mt-3 d-flex gap-3">
                    <span class="accuracy-badge bg-success-subtle text-success border border-success">
                        Train Acc: {train_acc:.4f}
                    </span>
                    <span class="accuracy-badge bg-info-subtle text-info border border-info">
                        Test Acc: {test_acc:.4f}
                    </span>
                </div>
                <div class="mt-3">
                    <button class="btn btn-sm btn-outline-secondary" onclick="filterTable('')">All Effects</button>
                    <button class="btn btn-sm btn-outline-primary" onclick="filterTable('Main Effect')">Main Effects Only</button>
                    <button class="btn btn-sm btn-outline-warning" onclick="filterTable('Interaction')">Interactions Only</button>
                </div>
            </div>
            <div class="card-body">
                <table id="analysisTable" class="table table-hover align-middle" style="width:100%">
                    <thead class="table-light">
                        <tr><th>#</th><th>Feature / Interaction</th><th>Type</th><th>Importance</th><th>Normalized</th><th>Direction</th><th>Components</th></tr>
                    </thead>
                    <tbody>{table_rows}</tbody>
                </table>
            </div>
        </div>
    </div>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.4/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.4/js/dataTables.bootstrap5.min.js"></script>
    <script>
        $(document).ready(function() {{
            var table = $('#analysisTable').DataTable({{ "order": [[ 3, "desc" ]], "pageLength": 15 }});
            window.filterTable = function(type) {{ table.column(2).search(type).draw(); }};
        }});
    </script>
    </body>
    </html>
    """

def process_single_task(model_name, mode, input_path, out_dir):
    print("\n" + "=" * 80)
    print(f"Processing: {model_name} / {mode}")
    print("=" * 80)
    
    if not os.path.exists(input_path):
        print(f"⚠️  Skipping: File not found - {input_path}")
        return None
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    try:
        df = load_and_process_yaml(input_path)
        
        if df is None or df.empty:
            print("⚠️  Skipping: No valid data found in YAML")
            return None
        
        total_samples = len(df)
        refusals = df[df['answer'] == 0]
        refusal_count = len(refusals)
        refusal_rate = refusal_count / total_samples if total_samples > 0 else 0.0
        
        print(f"   Total samples: {total_samples}")
        print(f"   Refusals: {refusal_count} ({refusal_rate:.2%})")
        
        df_clean = df[df['answer'].isin([-1, 1])].copy()
        valid_samples = len(df_clean)
        
        if valid_samples == 0:
            print("⚠️  Skipping: No valid samples for analysis")
            return None
        
        if df_clean['answer'].nunique() < 2:
            print("⚠️  Skipping: Target contains only one value")
            return None
        
        print(f"   Valid samples: {valid_samples}")
        
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=MODEL_PARAMS['n_estimators'],
                max_depth=MODEL_PARAMS['max_depth'],
                random_state=MODEL_PARAMS['random_state'],
                n_jobs=-1
            ),
            'CatBoost': CatBoostClassifier(
                iterations=MODEL_PARAMS['n_estimators'],
                depth=MODEL_PARAMS['max_depth'],
                random_seed=MODEL_PARAMS['random_state'],
                verbose=False
            ),
            'LightGBM': LGBMClassifier(
                n_estimators=MODEL_PARAMS['n_estimators'],
                max_depth=MODEL_PARAMS['max_depth'],
                random_state=MODEL_PARAMS['random_state'],
                verbose=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=MODEL_PARAMS['n_estimators'],
                max_depth=MODEL_PARAMS['max_depth'],
                random_state=MODEL_PARAMS['random_state'],
                eval_metric='logloss',
                verbosity=0
            )
        }
        
        all_results = {}
        
        for classifier_name, model in models.items():
            print(f"\n   Analyzing: {classifier_name}")
            
            try:
                # Run pipeline with SHAP Interaction Values
                X_train, X_test, shap_interaction_values, train_acc, test_acc = run_ml_pipeline(
                    df_clean, classifier_name, model
                )

                try:
                    # 1. Save Test Data (Useful for further plotting if needed)
                    # Reconstruct y_test based on indices
                    y_source = df_clean['answer'].map({-1: 0, 1: 1})
                    y_test_aligned = y_source.loc[X_test.index]
                    
                    test_data_path = os.path.join(out_dir, f"test_data_{model_name}_{mode}.joblib")
                    joblib.dump({
                        'X_test': X_test,
                        'y_test': y_test_aligned,
                        'feature_names': X_test.columns.tolist()
                    }, test_data_path)

                    # 2. Save SHAP Interaction Matrix
                    shap_save_path = os.path.join(out_dir, f"shap_interactions_{model_name}_{mode}_{classifier_name}.npy")
                    np.save(shap_save_path, shap_interaction_values)
                    print(f"      ✓ Visualization Data Saved: {os.path.basename(shap_save_path)}")
                    
                except Exception as save_e:
                    print(f"      ⚠️ Warning: Failed to save visualization data: {save_e}")
                
                # Process results using the interaction matrix logic
                report_df = save_results(
                    X_test, shap_interaction_values, out_dir, 
                    model_name, mode, classifier_name,
                    train_acc, test_acc
                )
                
                if not report_df.empty:
                    all_results[classifier_name] = {
                        'accuracy': {
                            'train': float(train_acc),
                            'test': float(test_acc)
                        },
                        'effects': report_df.to_dict('records')
                    }
                
            except Exception as e:
                print(f"      ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if all_results:
            analysis_summary = {
                'model_name': model_name,
                'modality': mode,
                'data_info': {
                    'total_samples_loaded': total_samples,
                    'refusal_count': refusal_count,
                    'refusal_rate': float(refusal_rate),
                    'valid_samples_for_shap': valid_samples
                },
                'results': all_results
            }
            
            summary_path = os.path.join(out_dir, "summary.yaml")
            with open(summary_path, 'w', encoding='utf-8') as f:
                yaml.dump(analysis_summary, f, sort_keys=False, allow_unicode=True)
            
            print(f"\n   ✓ Summary saved: {summary_path}")
            
            return all_results
        
        return None
        
    except Exception as e:
        print(f"❌ Error processing {model_name}/{mode}: {e}")
        return None

def get_all_tasks():
    results_dir = f"{ROOT}/../results/interaction"
    
    if not os.path.exists(results_dir):
        return []
    
    tasks = []
    
    for model_folder in os.listdir(results_dir):
        model_path = os.path.join(results_dir, model_folder)
        
        if not os.path.isdir(model_path):
            continue
        
        yaml_files = glob.glob(os.path.join(model_path, "results_*.yaml"))
        
        for yaml_file in yaml_files:
            filename = os.path.basename(yaml_file)
            mode = filename.replace("results_", "").replace(".yaml", "")
            tasks.append((model_folder, mode, yaml_file))
    
    return tasks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, required=True, 
                       help="Model name or 'all' to process all models")
    parser.add_argument('--mode', type=str, default='text', 
                       help="Mode: 'text', 'image', 'caption', or 'all' to process all modes")
    
    args = parser.parse_args()
    
    total_start_time = time.time()
    
    print("=" * 80)
    print("Feature Interaction Analysis - Multi-Model Comparison (Native SHAP)")
    print("=" * 80)

    input_root = f"{ROOT}/../results/interaction"
    output_root = f"{ROOT}/../results/interaction/analyze_results"
    
    if args.model_name.lower() == 'all' and args.mode.lower() == 'all':
        print("\n🔍 Scanning all available models and modes...")
        
        tasks = get_all_tasks()
        
        if not tasks:
            print("❌ No tasks found in results_interaction directory")
            return
        
        print(f"📋 Found {len(tasks)} tasks to process")
        
        success_count = 0
        fail_count = 0
        
        for idx, (model_name, mode, input_path) in enumerate(tasks, 1):
            print(f"\n[{idx}/{len(tasks)}] Processing: {model_name} / {mode}")
            
            out_dir = f"{input_root}/{model_name}/{mode}"
            result = process_single_task(model_name, mode, input_path, out_dir)
            
            if result:
                success_count += 1
            else:
                fail_count += 1
        
        elapsed = time.time() - total_start_time
        print("\n" + "=" * 80)
        print(f"✓ Batch processing completed!")
        print(f"   Total tasks: {len(tasks)}")
        print(f"   Successful: {success_count}")
        print(f"   Failed/Skipped: {fail_count}")
        print(f"   Total time: {elapsed:.2f} seconds")
        print("=" * 80)
        
    elif args.model_name.lower() == 'all':
        print(f"\n🔍 Processing all models for mode: {args.mode}")
        
        results_dir = input_root
        
        if not os.path.exists(results_dir):
            print(f"❌ Error: {results_dir} not found")
            return
        
        models = [d for d in os.listdir(results_dir) 
                 if os.path.isdir(os.path.join(results_dir, d))]
        
        if not models:
            print("❌ No model directories found")
            return
        
        print(f"📋 Found {len(models)} models to process")
        
        success_count = 0
        fail_count = 0
        
        for idx, model_name in enumerate(models, 1):
            print(f"\n[{idx}/{len(models)}] Processing: {model_name}")
            
            input_path = f"{input_root}/{model_name}/results_{args.mode}.yaml"
            out_dir = f"{output_root}/{model_name}/{args.mode}"
            
            result = process_single_task(model_name, args.mode, input_path, out_dir)
            
            if result:
                success_count += 1
            else:
                fail_count += 1
        
        elapsed = time.time() - total_start_time
        print("\n" + "=" * 80)
        print(f"✓ Batch processing completed!")
        print(f"   Total models: {len(models)}")
        print(f"   Successful: {success_count}")
        print(f"   Failed/Skipped: {fail_count}")
        print(f"   Total time: {elapsed:.2f} seconds")
        print("=" * 80)
        
    elif args.mode.lower() == 'all':
        print(f"\n🔍 Processing all modes for model: {args.model_name}")
        
        model_dir = f"{input_root}/{args.model_name}"
        
        if not os.path.exists(model_dir):
            print(f"❌ Error: Model directory not found - {model_dir}")
            return
        
        yaml_files = glob.glob(os.path.join(model_dir, "results_*.yaml"))
        
        if not yaml_files:
            print(f"❌ No YAML files found in {model_dir}")
            return
        
        modes = [os.path.basename(f).replace("results_", "").replace(".yaml", "") 
                for f in yaml_files]
        
        print(f"📋 Found {len(modes)} modes to process: {', '.join(modes)}")
        
        success_count = 0
        fail_count = 0
        
        for idx, mode in enumerate(modes, 1):
            print(f"\n[{idx}/{len(modes)}] Processing: {mode}")
            
            input_path = f"{input_root}/{args.model_name}/results_{mode}.yaml"
            out_dir = f"{output_root}/{args.model_name}/{mode}"
            
            result = process_single_task(args.model_name, mode, input_path, out_dir)
            
            if result:
                success_count += 1
            else:
                fail_count += 1
        
        elapsed = time.time() - total_start_time
        print("\n" + "=" * 80)
        print(f"✓ Batch processing completed!")
        print(f"   Total modes: {len(modes)}")
        print(f"   Successful: {success_count}")
        print(f"   Failed/Skipped: {fail_count}")
        print(f"   Total time: {elapsed:.2f} seconds")
        print("=" * 80)
        
    else:
        input_path = f"{input_root}/{args.model_name}/results_{args.mode}.yaml"
        out_dir = f"{output_root}/{args.model_name}/{args.mode}"
        
        result = process_single_task(args.model_name, args.mode, input_path, out_dir)
        
        if result:
            elapsed = time.time() - total_start_time
            print(f"\n✓ Analysis completed in {elapsed:.2f} seconds")
        else:
            print("\n❌ Analysis failed or skipped")

if __name__ == '__main__':
    main()