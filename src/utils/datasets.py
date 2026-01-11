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
    
    # Proxy features for counterfactual evaluation (Approach 2)
    # Maps proxy_name -> column_name (for flipping when sensitive features are removed)
    proxy_features: Dict[str, str] = field(default_factory=dict)
    
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
        # Proxy features for Approach 2 (used when sensitive features are removed)
        # relationship_Wife is a strong proxy for sex (Wife implies Female)
        proxy_features={
            "relationship": "relationship_Wife",  # Flip this for counterfactual when sex is removed
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
# Dataset Loading (Unified for Both Approaches)
# =============================================================================

def load_dataset(
    dataset_name: str,
    sensitive_feature: str = "sex",
    approach: int = 1,
    # Approach 2 specific parameters
    remove_sensitive: bool = False,
    sensitive_features_to_remove: Optional[List[str]] = None,
    proxy_feature: Optional[str] = None,
    random_state: int = 42,
) -> Dict:
    """
    Load and preprocess a dataset from OpenML.
    
    Supports two approaches:
    - Approach 1: Keep sensitive features in training, flip them for counterfactual
    - Approach 2: Remove sensitive features from training, use proxy for counterfactual
    
    Parameters
    ----------
    dataset_name : str
        Name of dataset (e.g., "adult", "german_credit")
    sensitive_feature : str
        Which sensitive feature to use (Approach 1) or primary sensitive feature (Approach 2)
    approach : int
        1 = keep sensitive features, 2 = remove sensitive features
    remove_sensitive : bool
        If True, remove sensitive features from training (alternative to approach=2)
    sensitive_features_to_remove : list, optional
        For Approach 2: list of sensitive features to remove (default: all)
    proxy_feature : str, optional
        For Approach 2: proxy feature for counterfactual (e.g., "relationship")
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    dict with keys:
        Common:
        - X_train: Feature matrix (numpy array)
        - y_train: Target labels
        - feature_names: List of feature names
        - config: DatasetConfig used
        - scaler: StandardScaler used
        - approach: Which approach was used (1 or 2)
        
        Approach 1 specific:
        - sensitive_col_idx: Index of sensitive column to flip
        - sensitive_col_name: Name of sensitive column
        
        Approach 2 specific:
        - X_protected: Sensitive features (for SenSeI distance metric)
        - protected_feature_names: Names of protected features
        - proxy_col_idx: Index of proxy column for counterfactual
        - proxy_col_name: Name of proxy column
    """
    # Handle approach parameter (remove_sensitive is a convenience alias)
    if remove_sensitive:
        approach = 2
    
    config = get_dataset_config(dataset_name)
    
    # Validate inputs based on approach
    if approach == 1:
        if sensitive_feature not in config.sensitive_features:
            available = list(config.sensitive_features.keys())
            raise ValueError(
                f"Invalid sensitive feature '{sensitive_feature}' for {dataset_name}. "
                f"Available: {available}"
            )
    else:  # Approach 2
        # Default to removing all sensitive features
        if sensitive_features_to_remove is None:
            sensitive_features_to_remove = list(config.sensitive_features.keys())
        
        # Default proxy feature
        if proxy_feature is None:
            if config.proxy_features:
                proxy_feature = list(config.proxy_features.keys())[0]
            else:
                raise ValueError(
                    f"No proxy features configured for {dataset_name}. "
                    f"Please specify proxy_feature parameter."
                )
        
        if proxy_feature not in config.proxy_features:
            available = list(config.proxy_features.keys())
            raise ValueError(
                f"Invalid proxy feature '{proxy_feature}' for {dataset_name}. "
                f"Available: {available}"
            )
    
    # Print loading info
    print(f"Loading {config.name} from OpenML (ID: {config.openml_id})...")
    if approach == 1:
        print(f"APPROACH 1: Keeping sensitive features in training")
        print(f"  Sensitive feature for counterfactual: {sensitive_feature}")
    else:
        print(f"APPROACH 2: Removing sensitive features from training")
        print(f"  Sensitive features to remove: {sensitive_features_to_remove}")
        print(f"  Proxy feature for counterfactual: {proxy_feature}")
    
    # Load from OpenML
    dataset = openml.datasets.get_dataset(config.openml_id)
    X_df, y_series, categorical_indicator, feature_names = dataset.get_data(
        target=dataset.default_target_attribute
    )
    
    # Convert target to binary
    if config.target_positive_class is not None:
        y = (y_series == config.target_positive_class).astype(int).values
    else:
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
    if dataset_name == "german_credit" and (
        sensitive_feature == "personal_status" or 
        (approach == 2 and "personal_status" in (sensitive_features_to_remove or []))
    ):
        X_df = _preprocess_german_credit_gender(X_df)
        categorical_cols = [c for c in categorical_cols if c != "personal_status"]
    
    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X_df, columns=categorical_cols, drop_first=False)
    
    # Drop columns with specified prefixes (e.g., native-country_)
    cols_to_drop_prefixes = []
    for prefix in config.drop_prefixes_after_encoding:
        cols_to_drop_prefixes += [c for c in X_encoded.columns if c.startswith(prefix)]
    X_encoded = X_encoded.drop(columns=cols_to_drop_prefixes, errors='ignore')
    
    # =========================================================================
    # Approach-specific processing
    # =========================================================================
    
    if approach == 1:
        # APPROACH 1: Keep sensitive feature, drop other related columns
        sensitive_col_name = config.sensitive_features[sensitive_feature]
        
        cols_to_drop_encoded = []
        if sensitive_feature == "sex":
            cols_to_drop_encoded += [c for c in X_encoded.columns 
                                      if c.startswith('sex_') and c != sensitive_col_name]
        elif sensitive_feature == "race":
            cols_to_drop_encoded += [c for c in X_encoded.columns 
                                      if c.startswith('race_') and c != sensitive_col_name]
        elif sensitive_feature == "personal_status":
            cols_to_drop_encoded += [c for c in X_encoded.columns 
                                      if c.startswith('personal_status_') and c != sensitive_col_name]
        
        X_encoded = X_encoded.drop(columns=cols_to_drop_encoded, errors='ignore')
        feature_names_final = list(X_encoded.columns)
        
        # Find sensitive column index
        if sensitive_col_name not in feature_names_final:
            raise ValueError(f"Sensitive column '{sensitive_col_name}' not found.")
        sensitive_col_idx = feature_names_final.index(sensitive_col_name)
        
        print(f"\nSensitive feature: {sensitive_col_name} (index {sensitive_col_idx})")
        
    else:
        # APPROACH 2: Remove sensitive features, keep proxy
        protected_cols = []
        cols_to_remove = []
        
        for sens_feat in sensitive_features_to_remove:
            if sens_feat in config.sensitive_features:
                sens_col = config.sensitive_features[sens_feat]
                if sens_col in X_encoded.columns:
                    protected_cols.append(sens_col)
                    cols_to_remove.append(sens_col)
                
                # Also remove other one-hot columns from same feature
                prefix = sens_feat + "_"
                for col in X_encoded.columns:
                    if col.startswith(prefix) and col not in cols_to_remove:
                        cols_to_remove.append(col)
        
        # Extract protected features BEFORE removing
        X_protected = X_encoded[protected_cols].values.astype(np.float32) if protected_cols else None
        protected_feature_names = protected_cols
        
        # Remove sensitive columns
        X_encoded = X_encoded.drop(columns=cols_to_remove, errors='ignore')
        feature_names_final = list(X_encoded.columns)
        
        # Find proxy column index
        proxy_col_name = config.proxy_features[proxy_feature]
        if proxy_col_name not in feature_names_final:
            raise ValueError(f"Proxy column '{proxy_col_name}' not found.")
        proxy_col_idx = feature_names_final.index(proxy_col_name)
        
        print(f"\nProtected features (for SenSeI): {protected_feature_names}")
        print(f"Proxy column for counterfactual: {proxy_col_name} (index {proxy_col_idx})")
    
    # Convert to numpy
    X = X_encoded.values.astype(np.float32)
    
    # Standardize numerical columns
    numerical_idx = [feature_names_final.index(c) for c in numerical_cols if c in feature_names_final]
    scaler = StandardScaler()
    if numerical_idx:
        X[:, numerical_idx] = scaler.fit_transform(X[:, numerical_idx])
    
    print(f"\nFinal features ({len(feature_names_final)}): {feature_names_final[:10]}...")
    print(f"\nDataset loaded (Approach {approach}):")
    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"  Positive class ratio: {y.mean():.2%}")
    
    # Build result dictionary
    result = {
        'X_train': X,
        'y_train': y,
        'feature_names': feature_names_final,
        'scaler': scaler,
        'config': config,
        'approach': approach,
    }
    
    if approach == 1:
        result.update({
            'sensitive_col_idx': sensitive_col_idx,
            'sensitive_col_name': sensitive_col_name,
        })
    else:
        result.update({
            'X_protected': X_protected,
            'protected_feature_names': protected_feature_names,
            'proxy_col_idx': proxy_col_idx,
            'proxy_col_name': proxy_col_name,
        })
    
    return result


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
