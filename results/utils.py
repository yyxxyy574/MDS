import numpy as np
import pandas as pd
import re
# import statsmodels.formula.api as smf
import warnings
import os
import yaml
import patsy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from firthlogist import FirthLogisticRegression # when without shap
import math
import itertools

from config.constants import CHARACTER

def parse_quantity(q_str):
    try:
        if isinstance(q_str, (int, float)):
            return int(q_str)
        nums = [int(x) for x in re.findall(r'\d+', str(q_str))]
        if len(nums) >= 2:
            return nums[1] 
        elif len(nums) == 1:
            return nums[0]
        return 0
    except:
        return 0

def parse_formula(formula):
    target_col = formula.split('~')[0].strip()
    predictors_part = formula.split('~')[1].strip()
    
    terms = [term.strip() for term in predictors_part.split('+')]
    
    main_effects = []
    interactions = []

    def extract_c_variable(part):
        part = part.strip()
        match = re.search(r'C\s*\(\s*([^,)]+)', part)
        if match:
            return match.group(1).strip()
        return None
    
    for term in terms:
        if '*' in term:
            interaction_vars = []
            parts = term.split('*')
            for part in parts:
                var_name = extract_c_variable(part)
                if var_name:
                    interaction_vars.append(var_name)
            
            if len(interaction_vars) >= 2:
                interactions.append(interaction_vars)
                main_effects.extend(interaction_vars)
        else:
            var_name = extract_c_variable(term)
            if var_name:
                main_effects.append(var_name)
    
    main_effects = list(dict.fromkeys(main_effects))
    
    return target_col, main_effects, interactions

def calculate_fishers_exact_pvalue(success_1, failure_1, success_0, failure_0):
    """Calculate Fisher's exact test p-value for 2x2 contingency table"""
    try:
        from scipy.stats import fisher_exact
        
        table = np.array([[success_1, failure_1], 
                         [success_0, failure_0]])
        
        odds_ratio, p_value = fisher_exact(table, alternative='two-sided')
        
        if success_1 == 0 or failure_1 == 0 or success_0 == 0 or failure_0 == 0:
            p_value = min(p_value, 1e-15)
        
        return p_value
        
    except (ImportError, Exception):
        total_n = success_1 + failure_1 + success_0 + failure_0
        
        if success_1 == 0 or failure_1 == 0 or success_0 == 0 or failure_0 == 0:
            return 1e-15
        
        expected_11 = (success_1 + success_0) * (success_1 + failure_1) / total_n
        expected_10 = (failure_1 + failure_0) * (success_1 + failure_1) / total_n
        expected_01 = (success_1 + success_0) * (success_0 + failure_0) / total_n
        expected_00 = (failure_1 + failure_0) * (success_0 + failure_0) / total_n
        
        if expected_11 > 0 and expected_10 > 0 and expected_01 > 0 and expected_00 > 0:
            chi_square = ((success_1 - expected_11)**2 / expected_11 + 
                         (failure_1 - expected_10)**2 / expected_10 + 
                         (success_0 - expected_01)**2 / expected_01 + 
                         (failure_0 - expected_00)**2 / expected_00)
            
            if chi_square > 10.83:
                return 0.001
            elif chi_square > 6.635:
                return 0.01
            elif chi_square > 3.841:
                return 0.05
            else:
                return 0.1
        else:
            return 1e-15

def check_data_conditions(data, target_col, formula_or_main_effects):
    """Check for critical conditions that completely prevent analysis"""
    
    # Extract main effects if formula is passed
    if isinstance(formula_or_main_effects, str):
        try:
            _, main_effects, _ = parse_formula(formula_or_main_effects)
        except:
            main_effects = []
    else:
        main_effects = formula_or_main_effects if formula_or_main_effects else []
    
    # Only check for critical conditions that make analysis impossible
    unique_targets = data[target_col].unique()
    if len(unique_targets) <= 1:
        return "Constant_Target", f"Target variable has only one value: {unique_targets[0]}"
    
    if len(unique_targets) > 2:
        return "Non_Binary_Target", f"Target variable has more than 2 values: {unique_targets}"
    
    if len(data) < 5:  # Very minimal sample size
        return "Insufficient_Sample_Size", f"Sample size too small: {len(data)}"
    
    # Check for missing predictors
    for var in main_effects:
        if var not in data.columns:
            return "Missing_Predictor", f"Predictor variable '{var}' not found in data"
    
    return "Data_OK", "Data conditions allow analysis"

def detect_multicollinearity(X, threshold=0.95):
    """
    Detect multicollinearity using correlation matrix.
    Returns pairs of highly correlated features.
    """
    # Calculate correlation matrix for numeric features only
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return []
    
    corr_matrix = X[numeric_cols].corr().abs()
    
    # Find pairs with correlation above threshold
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    return high_corr_pairs

def apply_l1_feature_selection(X, y, alpha_range=None, max_features_ratio=0.8):
    """
    Apply L1 regularization to select features and reduce multicollinearity.
    
    Parameters:
    - X: Feature matrix (pandas DataFrame)
    - y: Target variable (pandas Series)
    - alpha_range: Range of alpha values to try for L1 regularization
    - max_features_ratio: Maximum ratio of features to keep relative to sample size
    
    Returns:
    - selected_features: List of selected feature names
    - l1_model: Fitted L1 logistic regression model
    """
    if alpha_range is None:
        alpha_range = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    
    # Standardize features for L1 regularization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    # Calculate maximum number of features based on sample size
    max_features = int(min(len(y) * max_features_ratio, len(X.columns) * 0.9))
    
    best_alpha = None
    best_n_features = 0
    best_model = None
    
    # Try different alpha values
    for alpha in alpha_range:
        try:
            # Fit L1 regularized logistic regression
            l1_model = LogisticRegression(
                penalty='l1',
                C=1/alpha,  # C is inverse of regularization strength
                solver='liblinear',
                random_state=42,
                max_iter=1000
            )
            l1_model.fit(X_scaled, y)
            
            # Count non-zero coefficients
            n_selected = np.sum(np.abs(l1_model.coef_[0]) > 1e-10)
            
            # Select this alpha if it gives reasonable number of features
            if 1 <= n_selected <= max_features and n_selected > best_n_features:
                best_alpha = alpha
                best_n_features = n_selected
                best_model = l1_model
                
        except Exception as e:
            print(f"Warning: L1 regularization failed for alpha={alpha}: {e}")
            continue
    
    # If no good alpha found, use the least restrictive one that works
    if best_model is None:
        for alpha in sorted(alpha_range):
            try:
                l1_model = LogisticRegression(
                    penalty='l1',
                    C=1/alpha,
                    solver='liblinear',
                    random_state=42,
                    max_iter=1000
                )
                l1_model.fit(X_scaled, y)
                n_selected = np.sum(np.abs(l1_model.coef_[0]) > 1e-10)
                
                if n_selected >= 1:
                    best_model = l1_model
                    best_alpha = alpha
                    break
            except:
                continue
    
    if best_model is None:
        # If L1 completely fails, return all features
        print("Warning: L1 regularization failed, using all features")
        return list(X.columns), None
    
    # Get selected features
    selected_mask = np.abs(best_model.coef_[0]) > 1e-10
    selected_features = list(X.columns[selected_mask])
    
    print(f"L1 regularization (alpha={best_alpha}) selected {len(selected_features)} features from {len(X.columns)}")
    
    return selected_features, best_model

def rebuild_design_matrix_with_selected_features(data, formula, selected_features):
    """
    Rebuild the design matrix keeping only selected features while preserving
    the original structure for categorical variables and interactions.
    """
    # Parse the original formula
    target_col = formula.split('~')[0].strip()
    rhs_formula = formula.split('~')[1].strip()
    
    # Create original design matrix to get structure
    try:
        y_orig, X_orig = patsy.dmatrices(formula, data, return_type='dataframe')
    except Exception as e:
        raise ValueError(f"Failed to create original design matrix: {e}")
    
    # Find which original columns correspond to selected features
    selected_columns = []
    
    for col in X_orig.columns:
        # Always keep intercept
        if col == 'Intercept':
            selected_columns.append(col)
            continue
        
        # Check if this column should be kept based on selected features
        keep_column = False
        
        for selected_feature in selected_features:
            # Handle various column naming patterns from patsy
            if (selected_feature in col or 
                col.startswith(selected_feature) or
                any(selected_feature.startswith(part) for part in col.split('[')[0].split('(')[1:] if '(' in col)):
                keep_column = True
                break
        
        if keep_column:
            selected_columns.append(col)
    
    # If no features selected (except intercept), add the most important one
    if len(selected_columns) <= 1 and len(selected_features) > 0:
        # Add the first selected feature's columns
        for col in X_orig.columns:
            if selected_features[0] in col and col not in selected_columns:
                selected_columns.append(col)
                break
    
    # Create new design matrix with selected columns
    X_selected = X_orig[selected_columns].copy()
    
    print(f"Reduced design matrix from {len(X_orig.columns)} to {len(X_selected.columns)} columns")
    
    return y_orig, X_selected

def compute_covariance_matrix_from_firth(firth_model, X):
    """
    Compute covariance matrix from Firth regression model.
    Use available information from the fitted model.
    """
    try:
        # The covariance matrix is the inverse of the observed Fisher information matrix
        # For Firth regression, we can approximate it using the standard errors
        bse = firth_model.bse_
        n_params = len(bse)
        
        # Create diagonal covariance matrix using variance (se^2)
        cov_matrix = np.diag(bse ** 2)
        
        return cov_matrix
    
    except Exception as e:
        print(f"Warning: Could not compute covariance matrix: {e}")
        # Return identity matrix as fallback
        n_params = len(firth_model.coef_)
        return np.eye(n_params)

def extract_variable_levels_from_design(design_info):
    """
    Extract variable names and their levels from patsy design info.
    Correctly accesses factor_infos to get categories.
    """
    variable_levels = {}
    
    # Iterate through all factors known to the design matrix
    for factor, info in design_info.factor_infos.items():
        # Check if it has categories (categorical variable)
        if info.type == 'categorical':
            # Extract raw variable name from factor name like "C(wealth, Sum)" -> "wealth"
            match = re.search(r'C\s*\(\s*([^,)]+)', factor.name())
            var_name = match.group(1).strip() if match else factor.name()
            
            variable_levels[var_name] = list(info.categories)
            
    return variable_levels

def extract_sum_coding_omitted_levels(firth_model, design_info, results_list, X_cols, covariance_matrix, n_samples):
    """
    Extract omitted levels for Sum coding from Firth regression results.
    Maintains the constraint that sum of effects = 0 for each factor.
    """
    omitted_results = []
    
    # Get variable levels from design info
    variable_levels = extract_variable_levels_from_design(design_info)
    
    # Create mapping from column names to coefficient indices
    col_to_idx = {col: idx for idx, col in enumerate(X_cols)}
    
    # Group visible effects by variable
    visible_effects = {}
    
    # Process main effects
    for i, col_name in enumerate(X_cols):
        # Match main effects pattern: C(var, Sum)[S.level]
        main_match = re.match(r'C\(([^,)]+),\s*Sum\)\[S\.(.+?)\]$', col_name)
        if main_match:
            var_name = main_match.group(1)
            level = main_match.group(2)
            
            if var_name not in visible_effects:
                visible_effects[var_name] = {'main': {}, 'interactions': {}}
            
            visible_effects[var_name]['main'][level] = {
                'log_odds': firth_model.coef_[i],
                'variance': covariance_matrix[i, i],
                'se': firth_model.bse_[i],
                'idx': i
            }
    
    # Process interaction effects
    interaction_effects = {}
    for i, col_name in enumerate(X_cols):
        # Match interaction pattern: C(var1, Sum)[S.level1]:C(var2, Sum)[S.level2]
        interaction_match = re.match(r'C\(([^,)]+),\s*Sum\)\[S\.(.+?)\]:C\(([^,)]+),\s*Sum\)\[S\.(.+?)\]', col_name)
        if interaction_match:
            var1, level1, var2, level2 = interaction_match.groups()
            
            # Create interaction pair identifier (sorted for consistency)
            var_pair = tuple(sorted([var1, var2]))
            if var_pair not in interaction_effects:
                interaction_effects[var_pair] = {}
            
            # Store with original variable order for reconstruction
            level_combination = (var1, level1, var2, level2)
            interaction_effects[var_pair][level_combination] = {
                'log_odds': firth_model.coef_[i],
                'variance': covariance_matrix[i, i],
                'se': firth_model.bse_[i],
                'col_name': col_name,
                'idx': i
            }
    
    # Calculate omitted main effects
    for var_name, effects_dict in visible_effects.items():
        main_effects = effects_dict['main']
        
        if var_name in variable_levels and main_effects:
            visible_levels = set(main_effects.keys())
            all_var_levels = set(str(level) for level in variable_levels[var_name])
            omitted_levels = all_var_levels - visible_levels
            
            if omitted_levels:
                # For Sum coding: sum of all effects = 0
                sum_visible_log_odds = sum(data['log_odds'] for data in main_effects.values())
                sum_visible_variance = sum(data['variance'] for data in main_effects.values())
                
                for omitted_level in omitted_levels:
                    omitted_log_odds = -sum_visible_log_odds
                    omitted_variance = sum_visible_variance
                    omitted_se = np.sqrt(omitted_variance)
                    
                    # Calculate p-value using z-test
                    z = omitted_log_odds / omitted_se if omitted_se > 0 else 0
                    p_val = 2 * (1 - 0.5 * (1 + np.tanh(abs(z) * np.sqrt(2/np.pi))))
                    
                    # Calculate confidence intervals
                    ci_low = np.exp(omitted_log_odds - 1.96 * omitted_se)
                    ci_high = np.exp(omitted_log_odds + 1.96 * omitted_se)
                    
                    col_name = f"C({var_name}, Sum)[S.{omitted_level}]"
                    omitted_results.append({
                        "Variable": col_name,
                        "log_odds": omitted_log_odds,
                        "p_value": p_val,
                        "Odds_ratio": np.exp(omitted_log_odds),
                        "N_total": n_samples,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "Status": "Firth_Sum_Coding_Omitted"
                    })
    
    # Calculate omitted interaction effects using constraint solving
    for var_pair, effects in interaction_effects.items():
        var1, var2 = var_pair
        
        if var1 in variable_levels and var2 in variable_levels:
            levels1 = [str(level) for level in variable_levels[var1]]
            levels2 = [str(level) for level in variable_levels[var2]]
            
            # Build interaction effects matrix
            interaction_matrix = {}
            known_values = {}
            
            for l1 in levels1:
                interaction_matrix[l1] = {}
                for l2 in levels2:
                    interaction_matrix[l1][l2] = None
            
            # Fill in known effects
            for (v1, lv1, v2, lv2), effect_data in effects.items():
                if v1 == var1 and v2 == var2:
                    interaction_matrix[lv1][lv2] = effect_data['log_odds']
                    known_values[(lv1, lv2)] = effect_data
                elif v1 == var2 and v2 == var1:
                    interaction_matrix[lv2][lv1] = effect_data['log_odds']
                    known_values[(lv2, lv1)] = effect_data
            
            # Find unknown positions
            unknown_positions = []
            for l1 in levels1:
                for l2 in levels2:
                    if interaction_matrix[l1][l2] is None:
                        unknown_positions.append((l1, l2))
            
            if unknown_positions:
                # Set up linear system: row sums = 0, column sums = 0
                equations = []
                constants = []
                
                # Row constraints (each row sums to 0)
                for l1 in levels1:
                    row_equation = [0] * len(unknown_positions)
                    known_sum = 0
                    
                    for l2 in levels2:
                        if (l1, l2) in unknown_positions:
                            idx = unknown_positions.index((l1, l2))
                            row_equation[idx] = 1
                        elif interaction_matrix[l1][l2] is not None:
                            known_sum += interaction_matrix[l1][l2]
                    
                    if any(coef != 0 for coef in row_equation):
                        equations.append(row_equation)
                        constants.append(-known_sum)
                
                # Column constraints (each column sums to 0)
                for l2 in levels2:
                    col_equation = [0] * len(unknown_positions)
                    known_sum = 0
                    
                    for l1 in levels1:
                        if (l1, l2) in unknown_positions:
                            idx = unknown_positions.index((l1, l2))
                            col_equation[idx] = 1
                        elif interaction_matrix[l1][l2] is not None:
                            known_sum += interaction_matrix[l1][l2]
                    
                    if any(coef != 0 for coef in col_equation):
                        equations.append(col_equation)
                        constants.append(-known_sum)
                
                # Solve linear system
                try:
                    if equations:
                        A = np.array(equations)
                        b = np.array(constants)
                        
                        # Use least squares solution
                        solution, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                        
                        # Add omitted interaction effects
                        for idx, (l1, l2) in enumerate(unknown_positions):
                            if idx < len(solution):
                                solved_log_odds = solution[idx]
                                
                                # Estimate variance using average of related effects
                                related_variances = [effect_data['variance'] for effect_data in effects.values()]
                                estimated_variance = np.mean(related_variances) if related_variances else 1.0
                                estimated_se = np.sqrt(estimated_variance)
                                
                                # Calculate p-value
                                z = solved_log_odds / estimated_se if estimated_se > 0 else 0
                                p_val = 2 * (1 - 0.5 * (1 + np.tanh(abs(z) * np.sqrt(2/np.pi))))
                                
                                # Determine correct column name format
                                original_var_order = None
                                for (v1, _, v2, _) in effects.keys():
                                    original_var_order = (v1, v2)
                                    break
                                
                                if original_var_order and original_var_order[0] == var1:
                                    col_name = f"C({var1}, Sum)[S.{l1}]:C({var2}, Sum)[S.{l2}]"
                                else:
                                    col_name = f"C({var2}, Sum)[S.{l2}]:C({var1}, Sum)[S.{l1}]"
                                
                                # Calculate confidence intervals
                                ci_low = np.exp(solved_log_odds - 1.96 * estimated_se)
                                ci_high = np.exp(solved_log_odds + 1.96 * estimated_se)
                                
                                omitted_results.append({
                                    "Variable": col_name,
                                    "log_odds": solved_log_odds,
                                    "p_value": p_val,
                                    "Odds_ratio": np.exp(solved_log_odds),
                                    "N_total": n_samples,
                                    "ci_low": ci_low,
                                    "ci_high": ci_high,
                                    "Status": "Firth_Sum_Coding_Interaction_Omitted"
                                })
                
                except np.linalg.LinAlgError as e:
                    print(f"Warning: Could not solve interaction constraints for {var_pair}: {e}")
    
    return omitted_results

def calculate_logit_and_se(success, failure):
    """
    Use Firth Correction (+0.5) to calculate Logit and approximate standard error.
    Prevents log(0) or division by zero.
    """
    s_c = success + 0.5
    f_c = failure + 0.5
    
    # Logit = ln(p / (1-p)) = ln(s / f)
    logit = np.log(s_c / f_c)
    
    # Variance approx = 1/s + 1/f (based on corrected counts)
    variance = 1/s_c + 1/f_c
    se = np.sqrt(variance)
    
    return logit, se, variance

def calculate_penalized_estimates(data, target_col, main_effects, interactions, formula):
    """
    Calculate penalized maximum likelihood estimates for separation cases
    Using Firth's bias-reduction method 
    """
    results_list = []
    
    # Calculate baseline (intercept)
    total_success = data[target_col].sum()
    total_failure = len(data) - total_success
    base_logit, base_se, _ = calculate_logit_and_se(total_success, total_failure)

    # P-value for intercept (vs 0)
    z_score = base_logit / base_se
    p_value = 2 * (1 - 0.5 * (1 + np.tanh(abs(z_score) * np.sqrt(2/np.pi))))
    
    results_list.append({
        "Variable": "Intercept",
        "log_odds": base_logit,
        "p_value": p_value,
        "Odds_ratio": np.exp(base_logit),
        "N_total": len(data),
        "ci_low": np.exp(base_logit - 1.96 * base_se),
        "ci_high": np.exp(base_logit + 1.96 * base_se),
        "Status": "Manual_Intercept"
    })
    

    try:
        predictors_part = formula.split('~')[1].strip()
        # Use patsy to generate design matrix, so we can get accurate column names (e.g., "C(A, Sum)[S.a]")
        design_matrix = patsy.dmatrix(predictors_part, data, return_type='dataframe')
        design_info = design_matrix.design_info
    except Exception as e:
        print(f"Error creating design matrix with patsy: {e}")
        return results_list
    
    # Extract variable levels using the corrected function
    variable_levels = extract_variable_levels_from_design(design_info)
    
    # Store visible effects to calculate hidden ones later
    visible_sum_effects = {}
    
    # Iterate through each Term in the design matrix
    for term in design_info.terms:
        term_name = term.name()
        term_cols = design_matrix.columns[design_info.slice(term)].tolist()
        
        # Main Effect and Sum Coding
        # Use "Deviation from Mean" method for Sum Coding
        is_main_effect = len(term.factors) == 1
        is_sum_coding = "Sum" in term_name
        
        if is_main_effect and is_sum_coding:
            
            raw_var_name = list(term.factors)[0].name()
            if raw_var_name not in data.columns:
                continue

            level_stats = data.groupby(raw_var_name)[target_col].agg(['sum', 'count'])
            level_stats['failure'] = level_stats['count'] - level_stats['sum']
            
            logits = []
            variances = []
            levels = []
            
            for idx, row in level_stats.iterrows():
                l, s, v = calculate_logit_and_se(row['sum'], row['failure'])
                logits.append(l)
                variances.append(v)
                levels.append(idx)
            
            if len(logits) > 0:
                avg_logit = np.mean(logits)
            else:
                avg_logit = 0
            
            for col_name in term_cols:
                if col_name == 'Intercept':
                    continue

                match = re.search(r'\[S\.(.+?)\]', col_name)
                if not match: 
                    continue
                
                target_level_str = match.group(1)
                
                found_idx = -1
                for i, lvl in enumerate(levels):
                    if str(lvl) == target_level_str:
                        found_idx = i
                        break
                
                if found_idx != -1:
                    # Effect = Level Logit - Average Logit
                    effect_log_odds = logits[found_idx] - avg_logit
                    
                    # SE Calculation for (L_k - Mean)
                    # SE = sqrt( ((k-1)/k)^2 * Var_k + sum((1/k)^2 * Var_j for j!=k) )
                    k = len(levels)
                    var_k = variances[found_idx]
                    sum_other_vars = sum(variances) - var_k
                    
                    var_effect = ((k-1)/k)**2 * var_k + (1/k)**2 * sum_other_vars
                    se_effect = np.sqrt(var_effect)
                    
                    # P-value & CI
                    z = effect_log_odds / se_effect
                    p_val = 2 * (1 - 0.5 * (1 + np.tanh(abs(z) * np.sqrt(2/np.pi))))
                    
                    results_list.append({
                        "Variable": col_name,
                        "log_odds": effect_log_odds,
                        "p_value": p_val,
                        "Odds_ratio": np.exp(effect_log_odds),
                        "N_total": len(data),
                        "ci_low": np.exp(effect_log_odds - 1.96 * se_effect),
                        "ci_high": np.exp(effect_log_odds + 1.96 * se_effect),
                        "Status": "Manual_Sum_Coding_Exact"
                    })
                    
                    # Store for recovery
                    visible_sum_effects[col_name] = {
                        'log_odds': effect_log_odds,
                        'variance': var_effect,
                        'se': se_effect
                    }

        else:
            # Treatment Coding or Interaction Term
            # Directly use Design Matrix columns for 0/1 splitting
            
            for col_name in term_cols:
                if col_name == 'Intercept':
                    continue
                
                col_data = design_matrix[col_name]
                
                # Define groups: feature active (1.0) vs others
                mask_success = (col_data == 1.0)
                
                group_1 = data[mask_success]
                group_0 = data[~mask_success]
                
                if len(group_1) == 0:
                    continue

                s1 = group_1[target_col].sum()
                f1 = len(group_1) - s1
                s0 = group_0[target_col].sum()
                f0 = len(group_0) - s0
                
                l1, _, v1 = calculate_logit_and_se(s1, f1)
                l0, _, v0 = calculate_logit_and_se(s0, f0)
                
                # Effect = Logit(Group1) - Logit(Rest)
                log_odds = l1 - l0
                se = np.sqrt(v1 + v0)
                
                z = log_odds / se
                p_val = 2 * (1 - 0.5 * (1 + np.tanh(abs(z) * np.sqrt(2/np.pi))))
                
                status_label = "Manual_Interaction" if ":" in col_name else "Manual_Approx"
                
                results_list.append({
                    "Variable": col_name,
                    "log_odds": log_odds,
                    "p_value": p_val,
                    "Odds_ratio": np.exp(log_odds),
                    "N_total": len(data),
                    "ci_low": np.exp(log_odds - 1.96 * se),
                    "ci_high": np.exp(log_odds + 1.96 * se),
                    "Status": status_label
                })

                # For Sum coding interaction effects, store information
                if "Sum" in col_name and ":" in col_name:
                    visible_sum_effects[col_name] = {
                        'log_odds': log_odds,
                        'variance': se**2,
                        'se': se
                    }

    return results_list

def manual_separation_analysis(data, formula):
    """
    Wrapper for calculate_penalized_estimates to maintain API compatibility.
    Now redirects to the robust patsy-based implementation.
    """
    target_col = formula.split('~')[0].strip()
    return calculate_penalized_estimates(data, target_col, [], [], formula)

def logistic_regression(data, formula, enable_l1_selection=True, l1_alpha_range=None, multicollinearity_threshold=0.95):
    """
    Perform penalized logistic regression using Firth's method with optional L1 feature selection.
    
    Parameters:
    - data: Input data
    - formula: Regression formula
    - enable_l1_selection: Whether to apply L1 regularization for feature selection
    - l1_alpha_range: Range of alpha values for L1 regularization
    - multicollinearity_threshold: Correlation threshold for detecting multicollinearity
    """
    # Parse target column from formula
    target_col = formula.split('~')[0].strip()
    
    # 1. Basic Data Checks
    condition_status, condition_message = check_data_conditions(data, target_col, formula)
    
    if condition_status != "Data_OK":
        print(f"Data condition check failed: {condition_message}")
        
        # Construct specific error returns to match previous API expectation
        if condition_status == "Constant_Target":
            val = data[target_col].iloc[0]
            log_odds = np.inf if val == 1 else -np.inf
            return pd.DataFrame({
                "log_odds": [log_odds],
                "p_value": [np.nan],
                "Odds_ratio": [np.inf if val == 1 else 0.0],
                "N_total": [len(data)],
                "ci_low": [np.nan], 
                "ci_high": [np.nan],
                "Status": [condition_status]
            }, index=["Intercept"])
        else:
            return pd.DataFrame({
                "log_odds": [np.nan], "p_value": [np.nan], "Odds_ratio": [np.nan],
                "N_total": [len(data)], "ci_low": [np.nan], "ci_high": [np.nan],
                "Status": [condition_status]
            }, index=["Error"])

    # 2. Prepare Design Matrix using Patsy
    try:
        # return_type='dataframe' gives us pandas DataFrames with column names
        y, X = patsy.dmatrices(formula, data, return_type='dataframe')
    except Exception as e:
        print(f"Patsy design matrix error: {e}")
        return pd.DataFrame({"Status": [f"Patsy_Error_{str(e)}"]}, index=["Error"])

    # Store original design info for sum coding reconstruction
    original_design_info = X.design_info
    
    # 3. Apply L1 Feature Selection if enabled
    selected_features = None
    l1_model = None
    status_suffix = "Firth_Exact"
    
    if enable_l1_selection and len(X.columns) > 2:  # Only apply if we have more than intercept + 1 feature
        # Detect multicollinearity
        high_corr_pairs = detect_multicollinearity(X, threshold=multicollinearity_threshold)
        
        if high_corr_pairs:
            print(f"Detected {len(high_corr_pairs)} highly correlated feature pairs (threshold={multicollinearity_threshold})")
            for pair in high_corr_pairs[:5]:  # Show first 5
                print(f"  {pair[0]} <-> {pair[1]}: {pair[2]:.3f}")
        
        # Apply L1 regularization for feature selection
        try:
            # Exclude intercept from feature selection
            X_for_selection = X.drop(columns=['Intercept']) if 'Intercept' in X.columns else X
            
            if len(X_for_selection.columns) > 0:
                selected_features, l1_model = apply_l1_feature_selection(
                    X_for_selection, 
                    y.iloc[:, 0], 
                    alpha_range=l1_alpha_range
                )
                
                if selected_features and len(selected_features) < len(X_for_selection.columns):
                    # Rebuild design matrix with selected features
                    y, X = rebuild_design_matrix_with_selected_features(data, formula, selected_features)
                    status_suffix = "Firth_L1_Selected"
                    print(f"Applied L1 feature selection: {len(selected_features)} features selected")
                else:
                    print("L1 regularization kept all features")
            
        except Exception as e:
            print(f"Warning: L1 feature selection failed: {e}")
            print("Proceeding with original feature set")

    # 4. Fit Firth Logistic Regression
    # fit_intercept=False because patsy already includes 'Intercept' column in X
    firth = FirthLogisticRegression(wald=True, fit_intercept=False) 
    
    try:
        firth.fit(X, y.iloc[:, 0]) # y needs to be Series or 1D array
        print(f"Firth regression converged in {firth.n_iter_} iterations")
    except Exception as e:
        print(f"Firth fitting failed: {e}")
        # Fallback to manual method
        print("Falling back to manual penalized estimation...")
        try:
            results_list = calculate_penalized_estimates(data, target_col, [], [], formula)
            results_df = pd.DataFrame(results_list)
            if not results_df.empty:
                results_df.set_index('Variable', inplace=True)
            return results_df
        except Exception as fallback_error:
            print(f"Manual method also failed: {fallback_error}")
            return pd.DataFrame({
                "log_odds": [np.nan],
                "p_value": [np.nan], 
                "Odds_ratio": [np.nan],
                "N_total": [len(data)],
                "ci_low": [np.nan], 
                "ci_high": [np.nan],
                "Status": [f"All_Methods_Failed"]
            }, index=["Error"])

    # 5. Compute covariance matrix for omitted level reconstruction
    covariance_matrix = compute_covariance_matrix_from_firth(firth, X)

    # 6. Format Results
    results_list = []
    X_cols = list(X.columns)
    
    # Extract coefficients and stats
    # firth.coef_ corresponds to columns in X
    for i, col_name in enumerate(X_cols):
        log_odds = firth.coef_[i]
        se = firth.bse_[i]
        p_val = firth.pvals_[i]
        ci_l, ci_u = firth.ci_[i]
        
        results_list.append({
            "Variable": col_name,
            "log_odds": log_odds,
            "p_value": p_val,
            "Odds_ratio": np.exp(log_odds),
            "N_total": len(data),
            "ci_low": np.exp(ci_l),
            "ci_high": np.exp(ci_u),
            "Status": status_suffix
        })

    # 7. Restore Omitted Levels for Sum Coding (only if we have the original design info)
    # This is crucial for consistency with previous 'analyze_character_factor.py' results
    try:
        # Determine which design info to use
        current_design_info = None
        if hasattr(X, 'design_info') and X.design_info is not None:
            current_design_info = X.design_info
        else:
            current_design_info = original_design_info

        if current_design_info is not None:
            omitted_rows = extract_sum_coding_omitted_levels(
                firth, 
                current_design_info, 
                results_list, 
                X_cols, 
                covariance_matrix,
                n_samples=len(data)
            )
            results_list.extend(omitted_rows)
        else:
             print("Warning: No design_info available, skipping Sum coding reconstruction.")

    except Exception as e:
        print(f"Warning: Failed to reconstruct omitted Sum coding levels: {e}")

    # 8. Final DataFrame Construction
    results_df = pd.DataFrame(results_list)
    if not results_df.empty:
        if results_df['Variable'].duplicated().any():
            print("Warning: Duplicate variables found in results. Keeping first occurrence.")
            # drop_duplicates ensures we don't crash on set_index
            results_df = results_df.drop_duplicates(subset=['Variable'], keep='first')
            
        results_df.set_index('Variable', inplace=True)
    else:
        return pd.DataFrame({"Status": ["No_Results"]}, index=["Error"])

    if enable_l1_selection and selected_features:
        print(f"Final model uses {len(X.columns)} features after L1 selection")

    return results_df

def load_feature_values(features_file):
    if not os.path.exists(features_file):
        print(f"Warning: Features file {features_file} not found")
        return {}
    
    try:
        with open(features_file, 'r', encoding='utf-8') as f:
            features_data = yaml.load(f, Loader=yaml.FullLoader)
        
        feature_values = {}
        for feature, combinations in features_data.items():
            values_set = {}
            for combo in combinations:
                for char, value in combo.items():
                    if char not in values_set:
                        values_set[char] = set()
                    values_set[char].add(value)
            for char in values_set:
                if char not in feature_values:
                    feature_values[char] = {}
                feature_values[char][feature] = sorted(list(values_set[char]))
        
        return feature_values
    except Exception as e:
        print(f"Error loading features file {features_file}: {e}")
        return {}
    
def select_reference_value(values, feature):
    if len(values) == 0:
        return None
    
    if isinstance(CHARACTER[feature], dict):
        preferred_reference = list(CHARACTER[feature].keys())[0]
    else:
        preferred_reference = CHARACTER[feature][0]
    
    if preferred_reference in values:
        return preferred_reference
    else:
        return sorted(values)[0]