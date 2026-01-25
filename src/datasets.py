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
    # For binary: {feature_name: one_hot_column_to_keep}
    # For multiclass: {feature_name: None} and add to multiclass_sensitive_features
    sensitive_features: Dict[str, Optional[str]]
    
    # Multiclass sensitive features: {feature_name: [list of all one-hot column names]}
    # Used for exhaustive counterfactual evaluation (flip to all other categories)
    multiclass_sensitive_features: Dict[str, List[str]] = field(default_factory=dict)
    
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
            "race": "race_White",   # Binary: White=1, Other=0 (keeps only race_White)
            "race_all": None,       # Multiclass: keeps ALL race columns, exhaustive flip
        },
        # Multiclass sensitive features: feature_name -> list of all one-hot columns
        # These will be kept (not dropped) and used for exhaustive counterfactual
        multiclass_sensitive_features={
            "race_all": [
                "race_White", 
                "race_Black", 
                "race_Asian-Pac-Islander", 
                "race_Amer-Indian-Eskimo", 
                "race_Other"
            ],
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
            # personal_status combines gender + marital status (4 categories)
            # Treated as multiclass like race_all in Adult dataset
            "personal_status": None,  # Multiclass: keeps ALL one-hot columns
        },
        # Multiclass sensitive features: exhaustive counterfactual (flip to all other categories)
        multiclass_sensitive_features={
            "personal_status": [
                "personal_status_male div/sep",      # 50 (5.00%)
                "personal_status_female div/dep/mar", # 310 (31.00%)
                "personal_status_male single",        # 548 (54.80%)
                "personal_status_male mar/wid",       # 92 (9.20%)
            ],
        },
        # No proxy features available
        proxy_features={},
        columns_to_drop=[
            "foreign_worker",  # 96.3% are "yes" - extremely imbalanced, provides little signal
        ],
        drop_prefixes_after_encoding=[],
        target_positive_class="good",  # 'good' vs 'bad' credit (good=700, bad=300)
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
    if approach in (1, 3):
        # Approach 1 and 3 both keep sensitive features
        if sensitive_feature not in config.sensitive_features:
            available = list(config.sensitive_features.keys())
            raise ValueError(
                f"Invalid sensitive feature '{sensitive_feature}' for {dataset_name}. "
                f"Available: {available}"
            )
    elif approach == 2:
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
    elif approach == 3:
        print(f"APPROACH 3: Keeping sensitive features in training (with SenSeI distance learning)")
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
    
    # Dataset-specific preprocessing for German Credit
    # Only convert to binary if NOT using multiclass approach
    if dataset_name == "german_credit" and (
        sensitive_feature == "personal_status" or 
        (approach == 2 and "personal_status" in (sensitive_features_to_remove or []))
    ):
        # Check if this is a multiclass sensitive feature - if so, skip binary conversion
        is_multiclass_personal_status = sensitive_feature in config.multiclass_sensitive_features
        if not is_multiclass_personal_status:
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
    
    if approach in (1, 3):
        # APPROACH 1 & 3: Keep sensitive feature, drop other related columns
        
        # Check if this is a multiclass sensitive feature
        is_multiclass = sensitive_feature in config.multiclass_sensitive_features
        
        if is_multiclass:
            # MULTICLASS: Keep ALL columns for this feature
            multiclass_cols = config.multiclass_sensitive_features[sensitive_feature]
            
            # Find which columns exist in the encoded data
            existing_multiclass_cols = [c for c in multiclass_cols if c in X_encoded.columns]
            
            if len(existing_multiclass_cols) == 0:
                raise ValueError(
                    f"No multiclass columns found for '{sensitive_feature}'. "
                    f"Expected: {multiclass_cols}"
                )
            
            # Don't drop any of the multiclass columns
            # But we need to figure out what prefix to use for dropping other variants
            # For race_all, the prefix is "race_"
            feature_prefix = sensitive_feature.replace("_all", "_")
            
            # No columns to drop for multiclass - we keep all race columns
            feature_names_final = list(X_encoded.columns)
            
            # Find indices of all multiclass columns
            sensitive_col_indices = [feature_names_final.index(c) for c in existing_multiclass_cols]
            sensitive_col_names = existing_multiclass_cols
            
            print(f"\nMulticlass sensitive feature: {sensitive_feature}")
            print(f"  Columns ({len(sensitive_col_indices)}): {sensitive_col_names}")
            print(f"  Indices: {sensitive_col_indices}")
            
        else:
            # BINARY: Keep only one column, drop others
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
            
            print(f"\nBinary sensitive feature: {sensitive_col_name} (index {sensitive_col_idx})")
        
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
        
        # Find proxy column index (legacy - single column)
        proxy_col_name = config.proxy_features[proxy_feature]
        if proxy_col_name not in feature_names_final:
            raise ValueError(f"Proxy column '{proxy_col_name}' not found.")
        proxy_col_idx = feature_names_final.index(proxy_col_name)
        
        # Find relationship columns for improved sex proxy flip
        # These are needed for the sex_proxy flip function
        relationship_cols = {}
        for rel_col in ['relationship_Husband', 'relationship_Wife', 'relationship_Unmarried']:
            if rel_col in feature_names_final:
                relationship_cols[rel_col] = feature_names_final.index(rel_col)
            else:
                relationship_cols[rel_col] = None
        
        print(f"\nProtected features (for SenSeI): {protected_feature_names}")
        print(f"Proxy column for counterfactual: {proxy_col_name} (index {proxy_col_idx})")
        print(f"Relationship columns for sex proxy flip:")
        for col_name, col_idx in relationship_cols.items():
            print(f"  {col_name}: index {col_idx}")
    
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
    
    if approach in (1, 3):
        if is_multiclass:
            result.update({
                'is_multiclass': True,
                'sensitive_col_indices': sensitive_col_indices,  # List of indices
                'sensitive_col_names': sensitive_col_names,      # List of column names
            })
        else:
            result.update({
                'is_multiclass': False,
                'sensitive_col_idx': sensitive_col_idx,
                'sensitive_col_name': sensitive_col_name,
            })
    else:
        result.update({
            'X_protected': X_protected,
            'protected_feature_names': protected_feature_names,
            'proxy_col_idx': proxy_col_idx,
            'proxy_col_name': proxy_col_name,
            # Relationship columns for improved sex proxy flip
            'husband_col_idx': relationship_cols.get('relationship_Husband'),
            'wife_col_idx': relationship_cols.get('relationship_Wife'),
            'unmarried_col_idx': relationship_cols.get('relationship_Unmarried'),
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
    Create counterfactual data by flipping the sensitive feature (binary).
    
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


def create_flipped_data_sex_proxy(
    X: np.ndarray, 
    husband_idx: int,
    wife_idx: int,
    unmarried_idx: int
) -> np.ndarray:
    """
    Create counterfactual data by flipping sex proxy (relationship columns).
    
    Implements a more semantically correct sex flip using relationship columns:
    
    Case 1: Husband=1 → Set Husband=0, Wife=1 (male→female)
    Case 2: Husband=0 → Set Husband=1, and:
            - If Wife=1: Set Wife=0 (female→male)
            - If Unmarried=1: Set Unmarried=0 (unmarried→married male)
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    husband_idx : int
        Index of relationship_Husband column
    wife_idx : int
        Index of relationship_Wife column  
    unmarried_idx : int
        Index of relationship_Unmarried column
        
    Returns
    -------
    X_flipped : np.ndarray
        Feature matrix with sex proxy flipped
    """
    X_flipped = X.copy()
    
    # Case 1: Husband=1 → Wife (male to female)
    is_husband = X[:, husband_idx] == 1
    X_flipped[is_husband, husband_idx] = 0
    X_flipped[is_husband, wife_idx] = 1
    
    # Case 2: Husband=0 → Husband=1
    is_not_husband = X[:, husband_idx] == 0
    X_flipped[is_not_husband, husband_idx] = 1
    
    # If was Wife, set Wife=0
    was_wife = (X[:, wife_idx] == 1) & is_not_husband
    X_flipped[was_wife, wife_idx] = 0
    
    # If was Unmarried, set Unmarried=0
    was_unmarried = (X[:, unmarried_idx] == 1) & is_not_husband
    X_flipped[was_unmarried, unmarried_idx] = 0
    
    return X_flipped


def create_flipped_data_multiclass_exhaustive(
    X: np.ndarray, 
    multiclass_col_indices: List[int]
) -> tuple:
    """
    Create exhaustive counterfactual data for multiclass features (e.g., race).
    
    For each sample, creates flipped versions for ALL other categories.
    This enables testing: "Does changing from any race to any other race affect prediction?"
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    multiclass_col_indices : list of int
        Indices of the one-hot encoded columns for the multiclass feature
        
    Returns
    -------
    flipped_versions : list of np.ndarray
        List of K arrays, where K is number of categories.
        flipped_versions[i] has all samples set to category i.
    original_categories : np.ndarray
        Index (0 to K-1) of original category for each sample.
        Used to know which flips to skip (same category = no flip).
    """
    n_categories = len(multiclass_col_indices)
    
    # Find original category for each sample (which one-hot column is 1)
    multiclass_values = X[:, multiclass_col_indices]
    original_categories = np.argmax(multiclass_values, axis=1)
    
    # Create K versions of the data, each with all samples set to category k
    flipped_versions = []
    for target_cat_idx in range(n_categories):
        X_flipped = X.copy()
        # Set all category columns to 0
        X_flipped[:, multiclass_col_indices] = 0
        # Set target category to 1
        X_flipped[:, multiclass_col_indices[target_cat_idx]] = 1
        flipped_versions.append(X_flipped)
    
    return flipped_versions, original_categories


def counterfactual_consistency_multiclass_exhaustive(
    y_pred_original: np.ndarray,
    y_preds_flipped: List[np.ndarray],
    original_categories: np.ndarray
) -> float:
    """
    Calculate counterfactual consistency for multiclass feature using exhaustive method.
    
    For each sample, compares original prediction to predictions for ALL other categories.
    Consistency = fraction of (sample, other_category) pairs where prediction stayed same.
    
    Parameters
    ----------
    y_pred_original : np.ndarray
        Original predictions (n_samples,)
    y_preds_flipped : list of np.ndarray
        Predictions for each target category, list of K arrays each (n_samples,)
    original_categories : np.ndarray
        Original category index for each sample (n_samples,)
        
    Returns
    -------
    consistency : float
        Fraction of consistent predictions across all valid flips (0 to 1)
    """
    n_samples = len(y_pred_original)
    n_categories = len(y_preds_flipped)
    
    total_consistent = 0
    total_comparisons = 0
    
    for sample_idx in range(n_samples):
        orig_cat = original_categories[sample_idx]
        orig_pred = y_pred_original[sample_idx]
        
        # Compare to ALL other categories (skip same category)
        for target_cat in range(n_categories):
            if target_cat == orig_cat:
                continue  # Skip - not a real flip
            
            flipped_pred = y_preds_flipped[target_cat][sample_idx]
            if orig_pred == flipped_pred:
                total_consistent += 1
            total_comparisons += 1
    
    return total_consistent / total_comparisons if total_comparisons > 0 else 1.0


def create_flipped_data_multi_sensitive_exhaustive(
    X: np.ndarray,
    sensitive_features_info: List[Dict],
) -> tuple:
    """
    Create exhaustive counterfactual data for MULTIPLE sensitive features simultaneously.
    
    For each sample, creates ALL combinations of sensitive feature values except the original.
    E.g., for (sex=Male, race=White), creates:
    - (Male, Black), (Male, Asian), ...
    - (Female, White), (Female, Black), (Female, Asian), ...
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    sensitive_features_info : list of dict
        List of sensitive feature specifications, each containing:
        - 'col_indices': list of int - indices of one-hot columns for this feature
        - 'name': str - name of the feature (for debugging)
        
        Example: [
            {'name': 'sex', 'col_indices': [10]},  # Binary: single column
            {'name': 'race', 'col_indices': [11, 12, 13, 14, 15]},  # Multiclass: 5 columns
        ]
        
    Returns
    -------
    flipped_versions : list of np.ndarray
        List of arrays, one for each combination (excluding original for each sample).
        Each array has shape (n_samples, n_features).
    combination_labels : list of tuple
        List of (feature1_cat, feature2_cat, ...) tuples describing each flipped version.
    original_combinations : np.ndarray
        Array of shape (n_samples,) with index of original combination for each sample.
    """
    import itertools
    
    n_samples = X.shape[0]
    
    # Get number of categories for each feature
    n_categories_per_feature = []
    for feat_info in sensitive_features_info:
        col_indices = feat_info['col_indices']
        if len(col_indices) == 1:
            # Binary feature (single column, 0 or 1)
            n_categories_per_feature.append(2)
        else:
            # Multiclass feature (one-hot encoded)
            n_categories_per_feature.append(len(col_indices))
    
    # Generate all combinations
    all_combinations = list(itertools.product(*[range(n) for n in n_categories_per_feature]))
    n_combinations = len(all_combinations)
    
    # Find original combination for each sample
    original_combinations = np.zeros(n_samples, dtype=int)
    for sample_idx in range(n_samples):
        orig_cats = []
        for feat_info in sensitive_features_info:
            col_indices = feat_info['col_indices']
            if len(col_indices) == 1:
                # Binary: value is 0 or 1
                orig_cats.append(int(X[sample_idx, col_indices[0]]))
            else:
                # Multiclass: find which one-hot is 1
                orig_cats.append(int(np.argmax(X[sample_idx, col_indices])))
        
        # Find index of this combination
        orig_combo = tuple(orig_cats)
        original_combinations[sample_idx] = all_combinations.index(orig_combo)
    
    # Create flipped versions for each combination
    flipped_versions = []
    combination_labels = []
    
    for combo_idx, combo in enumerate(all_combinations):
        X_flipped = X.copy()
        
        # Set each feature to its target category
        for feat_idx, (feat_info, target_cat) in enumerate(zip(sensitive_features_info, combo)):
            col_indices = feat_info['col_indices']
            
            if len(col_indices) == 1:
                # Binary: set to 0 or 1
                X_flipped[:, col_indices[0]] = target_cat
            else:
                # Multiclass: one-hot encode
                X_flipped[:, col_indices] = 0
                X_flipped[:, col_indices[target_cat]] = 1
        
        flipped_versions.append(X_flipped)
        combination_labels.append(combo)
    
    return flipped_versions, combination_labels, original_combinations


def counterfactual_consistency_multi_sensitive_exhaustive(
    y_pred_original: np.ndarray,
    y_preds_flipped: List[np.ndarray],
    original_combinations: np.ndarray,
) -> float:
    """
    Calculate counterfactual consistency for multiple sensitive features.
    
    For each sample, compares original prediction to predictions for ALL other
    combinations of sensitive features.
    
    Parameters
    ----------
    y_pred_original : np.ndarray
        Original predictions (n_samples,)
    y_preds_flipped : list of np.ndarray
        Predictions for each combination, list of arrays each (n_samples,)
    original_combinations : np.ndarray
        Index of original combination for each sample (n_samples,)
        
    Returns
    -------
    consistency : float
        Fraction of consistent predictions across all valid flips (0 to 1)
    """
    n_samples = len(y_pred_original)
    n_combinations = len(y_preds_flipped)
    
    total_consistent = 0
    total_comparisons = 0
    
    for sample_idx in range(n_samples):
        orig_combo = original_combinations[sample_idx]
        orig_pred = y_pred_original[sample_idx]
        
        # Compare to ALL other combinations (skip original)
        for combo_idx in range(n_combinations):
            if combo_idx == orig_combo:
                continue  # Skip - same combination
            
            flipped_pred = y_preds_flipped[combo_idx][sample_idx]
            if orig_pred == flipped_pred:
                total_consistent += 1
            total_comparisons += 1
    
    return total_consistent / total_comparisons if total_comparisons > 0 else 1.0


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


# =============================================================================
# Data Splitting Utilities
# =============================================================================

def split_data(data: Dict, test_size: float = 0.2, val_size: float = 0.25, 
               random_state: int = 42) -> Dict:
    """
    Split data into train/val/test sets with stratification.
    
    Parameters
    ----------
    data : dict
        Data dictionary from load_dataset() containing X_train, y_train
    test_size : float
        Fraction of data for test set (default: 0.2 = 20%)
    val_size : float
        Fraction of remaining data for validation (default: 0.25 = 20% of total)
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Updated data dictionary with X_train, y_train, X_val, y_val, X_test, y_test
    """
    from sklearn.model_selection import train_test_split
    
    X_full = data['X_train'].copy()
    y_full = data['y_train'].copy()
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_full, y_full, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y_full
    )
    
    # Second split: train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, 
        test_size=val_size,
        random_state=random_state,
        stratify=y_temp
    )
    
    # Update data dict
    data['X_train'] = X_train
    data['y_train'] = y_train
    data['X_val'] = X_val
    data['y_val'] = y_val
    data['X_test'] = X_test
    data['y_test'] = y_test
    
    # Handle X_protected for SenSeI (Approach 2)
    if 'X_protected' in data and data['X_protected'] is not None:
        X_prot_full = data['X_protected']
        
        # Split protected features the same way
        X_prot_temp, X_prot_test, _, _ = train_test_split(
            X_prot_full, y_full,
            test_size=test_size,
            random_state=random_state,
            stratify=y_full
        )
        X_prot_train, X_prot_val, _, _ = train_test_split(
            X_prot_temp, y_temp,
            test_size=val_size,
            random_state=random_state,
            stratify=y_temp
        )
        
        # Store split protected features
        data['X_protected_train'] = X_prot_train
        data['X_protected_val'] = X_prot_val
        data['X_protected_test'] = X_prot_test
        
        # Keep original key pointing to training data for backward compatibility
        data['X_protected'] = X_prot_train
    
    return data
