"""
Dataset loading and preprocessing utilities.

Configuration-based approach for easy extension to new datasets.
"""

import numpy as np
import pandas as pd
import openml
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Dataset Configuration
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    
    name: str
    openml_id: int
    
    # Sensitive features available for fairness analysis
    sensitive_features: Dict[str, str]  # {feature_name: one_hot_column_to_keep}
    
    # Columns to drop before preprocessing
    columns_to_drop: List[str] = field(default_factory=list)
    
    # Column prefixes to drop after one-hot encoding (e.g., native-country_)
    drop_prefixes_after_encoding: List[str] = field(default_factory=list)
    
    # Target encoding (if needed)
    target_positive_class: Optional[str] = None  # e.g., '>50K' for Adult
    
    # Custom preprocessing function (optional)
    custom_preprocess: Optional[Callable] = None


# =============================================================================
# Dataset Registry
# =============================================================================

DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    
    "adult": DatasetConfig(
        name="Adult Income",
        openml_id=179,
        sensitive_features={
            "sex": "sex_Male",      # Binary: Male=1, Female=0
            "race": "race_White",   # Binary: White=1, Other=0
        },
        columns_to_drop=["fnlwgt", "education"],  # fnlwgt=weight, education redundant with education-num
        drop_prefixes_after_encoding=["native-country_"],  # Too many categories
        target_positive_class=">50K",
    ),
    
    "german_credit": DatasetConfig(
        name="German Credit",
        openml_id=31,
        sensitive_features={
            # personal_status combines gender + marital status
            # We'll handle this specially in preprocessing
            "personal_status": "personal_status_male",  # Will be created during preprocessing
        },
        columns_to_drop=[],
        drop_prefixes_after_encoding=[],
        target_positive_class="good",  # 'good' vs 'bad' credit
    ),
    
    "compas": DatasetConfig(
        name="COMPAS Recidivism",
        openml_id=45038,  # ProPublica COMPAS dataset on OpenML
        sensitive_features={
            "race": "race_African-American",
            "sex": "sex_Male",
        },
        columns_to_drop=[],
        drop_prefixes_after_encoding=[],
        target_positive_class=None,  # Already binary
    ),
    
    # Add more datasets here as needed...
}


def list_available_datasets() -> List[str]:
    """List all available dataset names."""
    return list(DATASET_CONFIGS.keys())


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """Get configuration for a dataset."""
    if dataset_name not in DATASET_CONFIGS:
        available = list_available_datasets()
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")
    return DATASET_CONFIGS[dataset_name]


# =============================================================================
# Dataset Loading
# =============================================================================

def load_dataset(
    dataset_name: str,
    sensitive_feature: str,
    random_state: int = 42,
) -> Dict:
    """
    Load and preprocess a dataset from OpenML.
    
    Note: Returns all data as training data. Cross-validation is used
    during SMAC optimization, so no separate test set is needed.
    
    Parameters
    ----------
    dataset_name : str
        Name of dataset (e.g., "adult", "german_credit")
    sensitive_feature : str
        Which sensitive feature to use for counterfactual evaluation
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    dict with keys:
        - X_train: Feature matrix (numpy array) - all data
        - y_train: Target labels - all data
        - feature_names: List of feature names after preprocessing
        - sensitive_col_idx: Index of sensitive column to flip
        - sensitive_col_name: Name of sensitive column
        - config: DatasetConfig used
    """
    config = get_dataset_config(dataset_name)
    
    # Validate sensitive feature
    if sensitive_feature not in config.sensitive_features:
        available = list(config.sensitive_features.keys())
        raise ValueError(
            f"Invalid sensitive feature '{sensitive_feature}' for {dataset_name}. "
            f"Available: {available}"
        )
    
    print(f"Loading {config.name} from OpenML (ID: {config.openml_id})...")
    
    # Load from OpenML
    dataset = openml.datasets.get_dataset(config.openml_id)
    X_df, y_series, categorical_indicator, feature_names = dataset.get_data(
        target=dataset.default_target_attribute
    )
    
    # Convert target to binary
    if config.target_positive_class is not None:
        y = (y_series == config.target_positive_class).astype(int).values
    else:
        # Assume already numeric
        y = y_series.astype(int).values
    
    print(f"Original features: {list(X_df.columns)}")
    
    # Drop rows with missing values
    mask = ~X_df.isnull().any(axis=1)
    X_df = X_df[mask].reset_index(drop=True)
    y = y[mask]
    
    # Identify categorical and numerical columns
    categorical_cols = [col for col, is_cat in zip(X_df.columns, categorical_indicator) if is_cat]
    numerical_cols = [col for col, is_cat in zip(X_df.columns, categorical_indicator) if not is_cat]
    
    # Drop specified columns
    categorical_cols = [c for c in categorical_cols if c not in config.columns_to_drop]
    numerical_cols = [c for c in numerical_cols if c not in config.columns_to_drop]
    X_df = X_df.drop(columns=[c for c in config.columns_to_drop if c in X_df.columns])
    
    # Apply custom preprocessing if defined
    if config.custom_preprocess is not None:
        X_df, categorical_cols, numerical_cols = config.custom_preprocess(
            X_df, categorical_cols, numerical_cols
        )
    
    # Dataset-specific preprocessing
    if dataset_name == "german_credit" and sensitive_feature == "personal_status":
        # Create binary gender from personal_status
        X_df = _preprocess_german_credit_gender(X_df)
        # Remove personal_status from categorical cols (we've replaced it)
        categorical_cols = [c for c in categorical_cols if c != "personal_status"]
    
    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X_df, columns=categorical_cols, drop_first=False)
    
    # Get the sensitive column name to keep
    sensitive_col_name = config.sensitive_features[sensitive_feature]
    
    # Determine columns to drop after encoding
    cols_to_drop_encoded = []
    
    # Drop other columns from the same sensitive feature group
    if sensitive_feature == "sex":
        cols_to_drop_encoded += [c for c in X_encoded.columns 
                                  if c.startswith('sex_') and c != sensitive_col_name]
    elif sensitive_feature == "race":
        cols_to_drop_encoded += [c for c in X_encoded.columns 
                                  if c.startswith('race_') and c != sensitive_col_name]
    elif sensitive_feature == "personal_status":
        # For German Credit, we created a binary column already
        cols_to_drop_encoded += [c for c in X_encoded.columns 
                                  if c.startswith('personal_status_') and c != sensitive_col_name]
    
    # Drop columns with specified prefixes
    for prefix in config.drop_prefixes_after_encoding:
        cols_to_drop_encoded += [c for c in X_encoded.columns if c.startswith(prefix)]
    
    X_encoded = X_encoded.drop(columns=[c for c in cols_to_drop_encoded if c in X_encoded.columns])
    
    # Get final feature names
    feature_names_final = list(X_encoded.columns)
    print(f"\nFinal features ({len(feature_names_final)}): {feature_names_final}")
    
    # Convert to numpy
    X = X_encoded.values.astype(np.float32)
    
    # Find sensitive column index
    if sensitive_col_name not in feature_names_final:
        raise ValueError(
            f"Sensitive column '{sensitive_col_name}' not found in features. "
            f"Available: {feature_names_final}"
        )
    sensitive_col_idx = feature_names_final.index(sensitive_col_name)
    print(f"\nSensitive feature: {sensitive_col_name} (index {sensitive_col_idx})")
    
    # Standardize numerical columns (on all data - CV will handle splits)
    numerical_idx = [feature_names_final.index(c) for c in numerical_cols if c in feature_names_final]
    
    scaler = StandardScaler()
    if numerical_idx:
        X[:, numerical_idx] = scaler.fit_transform(X[:, numerical_idx])
    
    print(f"\nDataset loaded:")
    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"  Positive class ratio: {y.mean():.2%}")
    
    return {
        'X_train': X,  # All data - CV handles train/test splits
        'y_train': y,
        'feature_names': feature_names_final,
        'sensitive_col_idx': sensitive_col_idx,
        'sensitive_col_name': sensitive_col_name,
        'scaler': scaler,
        'config': config,
    }


# =============================================================================
# Dataset-Specific Preprocessing Helpers
# =============================================================================

def _preprocess_german_credit_gender(X_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary gender feature from personal_status in German Credit dataset.
    
    personal_status values:
    - 'male div/sep': male, divorced/separated
    - 'female div/dep/mar': female, divorced/dependent/married  
    - 'male single': male, single
    - 'male mar/wid': male, married/widowed
    
    Creates: personal_status_male (1 if male, 0 if female)
    """
    X_df = X_df.copy()
    
    # Create binary gender column
    male_categories = ['male div/sep', 'male single', 'male mar/wid']
    X_df['personal_status_male'] = X_df['personal_status'].isin(male_categories).astype(int)
    
    # Drop original column
    X_df = X_df.drop(columns=['personal_status'])
    
    return X_df


# =============================================================================
# Utility Functions
# =============================================================================

def create_flipped_data(X: np.ndarray, sensitive_col_idx: int) -> np.ndarray:
    """
    Create counterfactual data by flipping the sensitive feature.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    sensitive_col_idx : int
        Index of sensitive column to flip
        
    Returns
    -------
    X_flipped : np.ndarray
        Feature matrix with sensitive column flipped (0 <-> 1)
    """
    X_flipped = X.copy()
    X_flipped[:, sensitive_col_idx] = 1 - X_flipped[:, sensitive_col_idx]
    return X_flipped


def create_cv_splits(X: np.ndarray, y: np.ndarray, n_splits: int = 5, random_state: int = 42):
    """
    Create stratified cross-validation splits.
    
    Returns list of (train_idx, test_idx) tuples.
    """
    from sklearn.model_selection import StratifiedKFold
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(cv.split(X, y))


# =============================================================================
# Convenience Function (Backward Compatibility)
# =============================================================================

def load_adult_dataset(
    sensitive_feature: str = "sex",
    random_state: int = 42,
) -> Dict:
    """Load Adult Income dataset. Convenience wrapper around load_dataset()."""
    return load_dataset("adult", sensitive_feature, random_state)


# =============================================================================
# Main - Test Dataset Loading
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test dataset loading")
    parser.add_argument("--dataset", type=str, default="adult", 
                        choices=list_available_datasets(),
                        help="Dataset to load")
    parser.add_argument("--sensitive", type=str, default="sex",
                        help="Sensitive feature to use")
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"Loading dataset: {args.dataset}")
    print(f"Sensitive feature: {args.sensitive}")
    print("=" * 70)
    
    # Load dataset
    data = load_dataset(args.dataset, args.sensitive)
    
    # Display results
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    
    X = data['X_train']
    y = data['y_train']
    feature_names = data['feature_names']
    sensitive_idx = data['sensitive_col_idx']
    sensitive_name = data['sensitive_col_name']
    
    print(f"\nShape: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target distribution: {y.mean():.2%} positive class")
    print(f"\nSensitive feature: '{sensitive_name}' at index {sensitive_idx}")
    print(f"Sensitive feature distribution: {X[:, sensitive_idx].mean():.2%} are 1 (e.g., Male)")
    
    print(f"\n{'='*70}")
    print("FEATURE NAMES")
    print("=" * 70)
    for i, name in enumerate(feature_names):
        marker = " <-- SENSITIVE" if i == sensitive_idx else ""
        print(f"  [{i:2d}] {name}{marker}")
    
    print(f"\n{'='*70}")
    print("SAMPLE DATA (first 5 rows)")
    print("=" * 70)
    
    # Create a nice table
    import pandas as pd
    sample_df = pd.DataFrame(X[:5], columns=feature_names)
    sample_df['TARGET'] = y[:5]
    print(sample_df.to_string())
    
    print(f"\n{'='*70}")
    print("FLIPPED DATA EXAMPLE (first 3 rows)")
    print("=" * 70)
    
    X_flipped = create_flipped_data(X, sensitive_idx)
    print(f"\nOriginal sensitive values: {X[:3, sensitive_idx]}")
    print(f"Flipped sensitive values:  {X_flipped[:3, sensitive_idx]}")
    
    print(f"\n{'='*70}")
    print("COMPARISON WITH IBM PREPROCESSING")
    print("=" * 70)
    print("""
IBM's Adult preprocessing (data.py):
  1. One-hot encode: workclass, marital-status, occupation, relationship, race, sex, native-country
  2. Drop: race_* (except White), sex_Female, native-country_*, fnlwgt, education  
  3. Standardize: age, education-num, capital-gain, capital-loss, hours-per-week
  4. IBM REMOVES sex_Male and race_White from model input (stores as protected vars)

Our preprocessing (Approach 1):
  1. One-hot encode: same categorical columns (from OpenML)
  2. Drop: same columns
  3. Standardize: same numerical columns  
  4. WE KEEP sex_Male in model input (for counterfactual evaluation)
  
Key difference: We train WITH sensitive features, IBM trains WITHOUT them.
""")
