"""
Visualize Adult Dataset Preprocessing Step-by-Step

Run this script to see exactly what transformations are applied at each step.
Usage: python visualize_preprocessing.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from utils.datasets import load_dataset, get_dataset_config
import openml

def visualize_adult_preprocessing():
    """Show step-by-step preprocessing transformations."""
    
    print("=" * 80)
    print("ADULT DATASET PREPROCESSING - STEP-BY-STEP VISUALIZATION")
    print("=" * 80)
    
    config = get_dataset_config("adult")
    
    # ============================================================================
    # STEP 1: Load from OpenML
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 1: Load from OpenML (ID: 179)")
    print("=" * 80)
    
    dataset = openml.datasets.get_dataset(config.openml_id)
    X_df, y_series, categorical_indicator, feature_names = dataset.get_data(
        target=dataset.default_target_attribute
    )
    
    print(f"\nOriginal shape: {X_df.shape[0]} rows × {X_df.shape[1]} columns")
    print(f"\nOriginal features:")
    for i, col in enumerate(X_df.columns):
        is_cat = categorical_indicator[i] if i < len(categorical_indicator) else False
        dtype = X_df[col].dtype
        n_unique = X_df[col].nunique()
        print(f"  [{i:2d}] {col:20s} | Type: {'Categorical' if is_cat else 'Numerical':12s} | Unique: {n_unique:3d} | dtype: {str(dtype):10s}")
    
    print(f"\nTarget values: {y_series.value_counts().to_dict()}")
    
    # Show sample of original data
    print(f"\nSample of original data (first 3 rows):")
    print(X_df.head(3).to_string())
    
    # ============================================================================
    # STEP 2: Target Encoding
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Target Encoding")
    print("=" * 80)
    
    y = (y_series == config.target_positive_class).astype(int).values
    print(f"\nTarget encoding: '{config.target_positive_class}' → 1, others → 0")
    print(f"Binary target distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for val, count in zip(unique, counts):
        print(f"  {val}: {count:6d} ({count/len(y)*100:5.2f}%)")
    
    # ============================================================================
    # STEP 3: Missing Value Handling
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 3: Missing Value Handling")
    print("=" * 80)
    
    mask = ~X_df.isnull().any(axis=1)
    n_missing = (~mask).sum()
    print(f"\nRows with missing values: {n_missing} ({n_missing/len(X_df)*100:.2f}%)")
    X_df = X_df[mask].reset_index(drop=True)
    y = y[mask]
    print(f"After dropping: {X_df.shape[0]} rows × {X_df.shape[1]} columns")
    
    # ============================================================================
    # STEP 4: Column Classification
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 4: Column Classification (from OpenML categorical_indicator)")
    print("=" * 80)
    
    categorical_cols = [col for col, is_cat in zip(X_df.columns, categorical_indicator) if is_cat]
    numerical_cols = [col for col, is_cat in zip(X_df.columns, categorical_indicator) if not is_cat]
    
    print(f"\nCategorical columns ({len(categorical_cols)}):")
    for col in categorical_cols:
        n_unique = X_df[col].nunique()
        sample_vals = X_df[col].unique()[:5]
        print(f"  - {col:20s} | {n_unique:2d} unique values | Sample: {list(sample_vals)}")
    
    print(f"\nNumerical columns ({len(numerical_cols)}):")
    for col in numerical_cols:
        print(f"  - {col:20s} | Range: [{X_df[col].min():.1f}, {X_df[col].max():.1f}], Mean: {X_df[col].mean():.2f}")
    
    # ============================================================================
    # STEP 5: Column Dropping
    # ============================================================================
    print("\n" + "=" * 80)
    print(f"STEP 5: Column Dropping (from config: {config.columns_to_drop})")
    print("=" * 80)
    
    print(f"\nDropping columns: {config.columns_to_drop}")
    print(f"Reason: fnlwgt = sampling weight (not a feature)")
    print(f"        education = redundant with education-num")
    
    categorical_cols = [c for c in categorical_cols if c not in config.columns_to_drop]
    numerical_cols = [c for c in numerical_cols if c not in config.columns_to_drop]
    X_df = X_df.drop(columns=[c for c in config.columns_to_drop if c in X_df.columns])
    
    print(f"\nAfter dropping: {X_df.shape[0]} rows × {X_df.shape[1]} columns")
    print(f"Categorical: {len(categorical_cols)} columns")
    print(f"Numerical: {len(numerical_cols)} columns")
    
    # ============================================================================
    # STEP 6: One-Hot Encoding
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 6: One-Hot Encoding")
    print("=" * 80)
    
    print(f"\nBefore encoding: {X_df.shape[1]} columns")
    print(f"Encoding {len(categorical_cols)} categorical columns...")
    
    X_encoded = pd.get_dummies(X_df, columns=categorical_cols, drop_first=False)
    
    print(f"\nAfter encoding: {X_encoded.shape[1]} columns")
    
    # Show breakdown by original column
    print(f"\nOne-hot encoding breakdown:")
    for orig_col in categorical_cols:
        encoded_cols = [c for c in X_encoded.columns if c.startswith(f"{orig_col}_")]
        if encoded_cols:
            print(f"  {orig_col:20s} → {len(encoded_cols):2d} columns: {encoded_cols[:3]}..." if len(encoded_cols) > 3 else f"  {orig_col:20s} → {len(encoded_cols):2d} columns: {encoded_cols}")
    
    # Show numerical columns that remain
    for num_col in numerical_cols:
        if num_col in X_encoded.columns:
            print(f"  {num_col:20s} → unchanged (numerical)")
    
    # ============================================================================
    # STEP 7: Sensitive Feature Selection
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 7: Sensitive Feature Selection")
    print("=" * 80)
    
    sensitive_feature = "sex"
    sensitive_col_name = config.sensitive_features[sensitive_feature]
    
    print(f"\nSensitive feature: '{sensitive_feature}'")
    print(f"Keep column: '{sensitive_col_name}'")
    
    # Find all columns from this sensitive feature
    sensitive_group_cols = [c for c in X_encoded.columns if c.startswith(f"{sensitive_feature}_")]
    print(f"\nAll columns from '{sensitive_feature}': {sensitive_group_cols}")
    
    cols_to_drop_encoded = [c for c in sensitive_group_cols if c != sensitive_col_name]
    print(f"Will drop: {cols_to_drop_encoded}")
    print(f"Will keep: {sensitive_col_name}")
    print(f"\nReason: Avoid multicollinearity (sex_Male = 1 - sex_Female)")
    
    # ============================================================================
    # STEP 8: Prefix-Based Dropping
    # ============================================================================
    print("\n" + "=" * 80)
    print(f"STEP 8: Prefix-Based Dropping (from config: {config.drop_prefixes_after_encoding})")
    print("=" * 80)
    
    for prefix in config.drop_prefixes_after_encoding:
        prefix_cols = [c for c in X_encoded.columns if c.startswith(prefix)]
        cols_to_drop_encoded.extend(prefix_cols)
        print(f"\nDropping all columns with prefix '{prefix}': {len(prefix_cols)} columns")
        print(f"Sample columns: {prefix_cols[:5]}..." if len(prefix_cols) > 5 else f"All columns: {prefix_cols}")
        print(f"Reason: Too many categories ({len(prefix_cols)}), high cardinality, risk of overfitting")
    
    X_encoded = X_encoded.drop(columns=[c for c in cols_to_drop_encoded if c in X_encoded.columns])
    
    print(f"\nAfter dropping: {X_encoded.shape[0]} rows × {X_encoded.shape[1]} columns")
    
    # ============================================================================
    # STEP 9: Find Sensitive Column Index
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 9: Find Sensitive Column Index")
    print("=" * 80)
    
    feature_names_final = list(X_encoded.columns)
    sensitive_col_idx = feature_names_final.index(sensitive_col_name)
    
    print(f"\nSensitive column: '{sensitive_col_name}'")
    print(f"Sensitive column index: {sensitive_col_idx}")
    print(f"\nAll final features ({len(feature_names_final)}):")
    for i, name in enumerate(feature_names_final):
        marker = " <-- SENSITIVE" if i == sensitive_col_idx else ""
        print(f"  [{i:2d}] {name}{marker}")
    
    # ============================================================================
    # STEP 10: Standardization
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 10: Standardization of Numerical Columns")
    print("=" * 80)
    
    from sklearn.preprocessing import StandardScaler
    
    numerical_idx = [feature_names_final.index(c) for c in numerical_cols if c in feature_names_final]
    
    print(f"\nNumerical columns to standardize: {[feature_names_final[i] for i in numerical_idx]}")
    
    if numerical_idx:
        X = X_encoded.values.astype(np.float32)
        
        # Show before standardization
        print(f"\nBefore standardization:")
        for idx in numerical_idx:
            col_name = feature_names_final[idx]
            col_values = X[:, idx]
            print(f"  {col_name:20s} | Mean: {col_values.mean():8.3f}, Std: {col_values.std():8.3f}, Range: [{col_values.min():8.3f}, {col_values.max():8.3f}]")
        
        # Standardize
        scaler = StandardScaler()
        X[:, numerical_idx] = scaler.fit_transform(X[:, numerical_idx])
        
        # Show after standardization
        print(f"\nAfter standardization:")
        for idx in numerical_idx:
            col_name = feature_names_final[idx]
            col_values = X[:, idx]
            print(f"  {col_name:20s} | Mean: {col_values.mean():8.3f}, Std: {col_values.std():8.3f}, Range: [{col_values.min():8.3f}, {col_values.max():8.3f}]")
    else:
        X = X_encoded.values.astype(np.float32)
        print(f"\nNo numerical columns to standardize (all are binary from one-hot encoding)")
    
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    print(f"\nFinal dataset shape: {X.shape[0]} rows × {X.shape[1]} columns")
    print(f"Final target shape: {y.shape[0]} labels")
    print(f"\nFinal feature breakdown:")
    print(f"  - Numerical (standardized): {len(numerical_idx)} columns")
    print(f"  - Binary (one-hot encoded): {X.shape[1] - len(numerical_idx)} columns")
    print(f"\nSensitive feature:")
    print(f"  - Name: {sensitive_col_name}")
    print(f"  - Index: {sensitive_col_idx}")
    print(f"  - Distribution: {X[:, sensitive_col_idx].mean()*100:.2f}% are 1")
    
    # Show sample row
    print(f"\nSample final row (first person):")
    sample_row = X[0]
    print(f"  Shape: {sample_row.shape}")
    print(f"  Non-zero values: {np.count_nonzero(sample_row)} out of {len(sample_row)}")
    print(f"  Sensitive feature value (index {sensitive_col_idx}): {sample_row[sensitive_col_idx]}")
    print(f"  Numerical feature value (education-num, index 0): {sample_row[0]:.3f}")
    
    # Show flipped example
    print(f"\n" + "=" * 80)
    print("COUNTERFACTUAL EXAMPLE: Flipping Sensitive Feature")
    print("=" * 80)
    
    from utils.datasets import create_flipped_data
    
    X_flipped = create_flipped_data(X[:3], sensitive_col_idx)
    
    print(f"\nOriginal sensitive values (first 3 rows): {X[:3, sensitive_col_idx]}")
    print(f"Flipped sensitive values (first 3 rows):   {X_flipped[:3, sensitive_col_idx]}")
    print(f"\nThis is what we use for counterfactual consistency evaluation!")
    
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    visualize_adult_preprocessing()

