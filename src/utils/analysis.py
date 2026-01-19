"""
Analysis utilities for individual fairness experiments.

This module contains functions for:
- SMAC results caching
- SMAC optimization orchestration
- Optimization statistics and summaries
- Fairness confusion matrix analysis
- Trivial fairness detection
- Case study setup and analysis
"""

import os
import pickle
import numpy as np
from typing import Dict, Any, Tuple, Optional, Callable, List
from IPython.display import display, Image


# =============================================================================
# SMAC Results Caching
# =============================================================================

# Default cache directory (can be overridden)
CACHE_DIR = "../cache"


class CachedConfig:
    """Mock config that provides dict-like access for cached results."""
    
    def __init__(self, config_dict: Dict):
        self._config = config_dict
    
    def __getitem__(self, key):
        return self._config[key]
    
    def __iter__(self):
        return iter(self._config)
    
    def __repr__(self):
        return f"CachedConfig({self._config})"
    
    def get(self, key, default=None):
        return self._config.get(key, default)
    
    def keys(self):
        return self._config.keys()
    
    def values(self):
        return self._config.values()
    
    def items(self):
        return self._config.items()


class CachedRunHistory:
    """Mock runhistory that provides the same interface as SMAC's runhistory."""
    
    def __init__(self, configs: List[Dict], costs: List[tuple]):
        self._configs = [CachedConfig(c) for c in configs]
        self._costs = {i: cost for i, cost in enumerate(costs)}
        self._is_cached = True  # Flag to identify cached results
    
    def get_configs(self):
        return self._configs
    
    def average_cost(self, config):
        idx = self._configs.index(config)
        return self._costs[idx]
    
    def items(self):
        """Return empty iterator - cached results don't have timing info."""
        return iter([])


class CachedSMAC:
    """Mock SMAC facade that provides the same interface for visualization."""
    
    def __init__(self, configs: List[Dict], costs: List[tuple]):
        self.runhistory = CachedRunHistory(configs, costs)


def save_smac_results(results: Dict, dataset_name: str, sensitive_feature: str, 
                      approach: int = 1, cache_dir: str = None):
    """
    Save SMAC results to cache.
    
    Parameters
    ----------
    results : dict
        Dictionary mapping model_type -> smac optimizer
    dataset_name : str
        Name of the dataset
    sensitive_feature : str
        Name of the sensitive feature
    approach : int
        Which approach (1 or 2)
    cache_dir : str, optional
        Directory to save cache (default: ../cache)
    """
    if cache_dir is None:
        cache_dir = CACHE_DIR
    
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_data = {}
    for model_type, smac in results.items():
        configs = smac.runhistory.get_configs()
        configs_as_dicts = [dict(c) for c in configs]
        costs = [smac.runhistory.average_cost(c) for c in configs]
        
        cache_data[model_type] = {
            'configs': configs_as_dicts,
            'costs': costs,
        }
    
    cache_file = f"{cache_dir}/smac_results_{dataset_name}_{sensitive_feature}_approach{approach}.pkl"
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"Results cached to: {cache_file}")
    return cache_file


def load_smac_results(dataset_name: str, sensitive_feature: str, 
                      approach: int = 1, cache_dir: str = None) -> Optional[Dict]:
    """
    Load SMAC results from cache.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    sensitive_feature : str
        Name of the sensitive feature
    approach : int
        Which approach (1 or 2)
    cache_dir : str, optional
        Directory to load cache from (default: ../cache)
        
    Returns
    -------
    dict or None
        Dictionary mapping model_type -> CachedSMAC, or None if not found
    """
    if cache_dir is None:
        cache_dir = CACHE_DIR
    
    cache_file = f"{cache_dir}/smac_results_{dataset_name}_{sensitive_feature}_approach{approach}.pkl"
    
    if not os.path.exists(cache_file):
        return None
    
    with open(cache_file, 'rb') as f:
        cache_data = pickle.load(f)
    
    results = {}
    for model_type, data in cache_data.items():
        results[model_type] = CachedSMAC(data['configs'], data['costs'])
    
    print(f"Results loaded from cache: {cache_file}")
    return results


# =============================================================================
# SMAC Optimization Helpers
# =============================================================================

def run_smac_optimization(data, dataset_name, sensitive_feature, 
                          load_from_cache=True, walltime_limit=300, n_trials=100,
                          output_dir="../smac_output"):
    """
    Run SMAC optimization or load from cache.
    
    Parameters
    ----------
    data : dict
        Data dictionary with X_train, y_train, X_val, y_val
    dataset_name : str
        Name of the dataset
    sensitive_feature : str
        Name of the sensitive feature
    load_from_cache : bool
        Whether to try loading from cache first
    walltime_limit : int
        Time limit for optimization in seconds
    n_trials : int
        Number of trials for optimization
    output_dir : str
        Directory for SMAC output files
        
    Returns
    -------
    dict : Results dictionary with 'rf' and 'mlp' SMAC objects
    """
    # Import run_optimization from main module
    from main import run_optimization
    
    results = None
    
    if load_from_cache:
        print(f"Loading results from cache for {dataset_name}/{sensitive_feature}...")
        results = load_smac_results(dataset_name, sensitive_feature, approach=1)
        if results is not None:
            print(f" Loaded from cache!")
            return results
        else:
            print("  Cache not found! Running optimization...")
    
    print(f"Running SMAC optimization for {dataset_name}/{sensitive_feature}...")
    results = {}
    
    for model_type in ["rf", "mlp"]:
        print(f"\n{'='*60}")
        print(f"Optimizing {model_type.upper()} (Sensitive: {sensitive_feature})...")
        print(f"{'='*60}")
        
        smac = run_optimization(
            model_type=model_type,
            data=data,
            walltime_limit=walltime_limit,
            n_trials=n_trials,
            output_dir=f"{output_dir}/{dataset_name}_{sensitive_feature}"
        )
        results[model_type] = smac
    
    print("\n" + "="*60)
    print(" Optimization complete!")
    print("="*60)
    
    # Save results to cache
    save_smac_results(results, dataset_name, sensitive_feature, approach=1)
    
    return results


def get_optimization_stats(smac, model_name) -> Dict[str, Any]:
    """
    Extract statistics from SMAC optimization run.
    
    Parameters
    ----------
    smac : SMAC optimizer
        SMAC optimizer object
    model_name : str
        Name of the model
        
    Returns
    -------
    dict
        Statistics dictionary
    """
    from main import get_pareto_front
    
    runhistory = smac.runhistory
    
    # Number of configurations evaluated
    n_configs = len(runhistory.get_configs())
    
    # Get Pareto front size
    pareto_configs, pareto_costs = get_pareto_front(smac)
    n_incumbents = len(pareto_configs)
    
    # Check if this is a cached result (no timing info available)
    is_cached = getattr(runhistory, '_is_cached', False)
    
    if is_cached:
        # Cached results don't have timing information
        return {
            'Model': model_name.upper(),
            'Configs Evaluated': n_configs,
            'Pareto Front Size': n_incumbents,
            'Total Time (s)': "(cached)",
            'Avg Time/Config (s)': "(cached)",
        }
    
    # Calculate total time and average time per config
    total_time = 0
    for key, value in runhistory.items():
        if value.time is not None:
            total_time += value.time
    
    avg_time = total_time / n_configs if n_configs > 0 else 0
    
    return {
        'Model': model_name.upper(),
        'Configs Evaluated': n_configs,
        'Pareto Front Size': n_incumbents,
        'Total Time (s)': f"{total_time:.1f}",
        'Avg Time/Config (s)': f"{avg_time:.2f}",
    }


def print_optimization_summary(results, dataset_name, sensitive_feature):
    """
    Print optimization summary table and insights.
    
    Parameters
    ----------
    results : dict
        Dictionary mapping model_type -> smac optimizer
    dataset_name : str
        Name of the dataset
    sensitive_feature : str
        Name of the sensitive feature
    """
    import pandas as pd
    from main import get_pareto_front
    
    # Build summary table
    summary_data = []
    for model_type, smac in results.items():
        stats = get_optimization_stats(smac, model_type)
        summary_data.append(stats)
    
    summary_df = pd.DataFrame(summary_data)
    
    print("=" * 70)
    print(f"OPTIMIZATION SUMMARY: {dataset_name.upper()} / {sensitive_feature.upper()}")
    print("=" * 70)
    display(summary_df)
    
    # Additional insights
    print("\n" + "=" * 70)
    print("PARETO EFFICIENCY")
    print("=" * 70)
    for model_type, smac in results.items():
        n_configs = len(smac.runhistory.get_configs())
        pareto_configs, _ = get_pareto_front(smac)
        pareto_pct = len(pareto_configs) / n_configs * 100 if n_configs > 0 else 0
        print(f"{model_type.upper()}: {len(pareto_configs)}/{n_configs} configs are Pareto-optimal ({pareto_pct:.1f}%)")


# =============================================================================
# Fairness Confusion Matrix Analysis
# =============================================================================

def analyze_fairness_confusion_matrix(results, data, dataset_name, sensitive_feature, output_dir,
                                       selection_method='knee'):
    """
    Analyze fairness-accuracy confusion matrix for selected Pareto configurations.
    
    Trains models on training set, evaluates on validation set.
    
    Parameters
    ----------
    results : dict
        Dictionary mapping model_type -> smac optimizer
    data : dict
        Data dictionary with train/val splits
    dataset_name : str
        Name of the dataset
    sensitive_feature : str
        Name of the sensitive feature
    output_dir : str
        Directory to save plots
    selection_method : str
        Method to select Pareto point: 'knee' (balanced), 'best_accuracy', 'utopia', 'weighted'
        
    Returns
    -------
    dict
        FCM results and plot paths for each model
    """
    from main import get_pareto_front, create_rf_model, create_mlp_model
    from utils.plotting import compute_fairness_confusion_matrix, plot_fairness_confusion_matrix, print_fairness_confusion_summary
    
    is_multiclass = 'sensitive_col_indices' in data
    all_results = {}
    
    # Describe selection method
    method_descriptions = {
        'knee': 'Knee-Optimal (Balanced)',
        'best_accuracy': 'Best Accuracy',
        'utopia': 'Closest to Utopia',
        'weighted': 'Weighted (50-50)',
    }
    method_desc = method_descriptions.get(selection_method, selection_method)
    
    for model_type in ["rf", "mlp"]:
        if model_type not in results:
            continue
            
        print(f"\n{'='*70}")
        print(f"Analyzing {model_type.upper()} - {method_desc} Configuration")
        print(f"{'='*70}")
        
        # Get config from Pareto front based on selection method
        configs, costs = get_pareto_front(results[model_type])
        
        if selection_method == 'best_accuracy':
            selected_idx = np.argmin(costs[:, 0])  # Lowest error = best accuracy
            selected_config = configs[selected_idx]
            accuracy = 1 - costs[selected_idx, 0]
            consistency = 1 - costs[selected_idx, 1]
        else:
            # Use balanced selection (knee, utopia, or weighted)
            selected_idx, info = select_balanced_pareto_point(configs, costs, method=selection_method)
            selected_config = configs[selected_idx]
            accuracy = info['accuracy']
            consistency = info['consistency']
        
        print(f"Selection method: {method_desc}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Consistency: {consistency:.4f}")
        
        # Create and train the model on TRAINING set
        if model_type == "rf":
            model = create_rf_model(selected_config)
        else:
            model = create_mlp_model(selected_config)
        
        model.fit(data['X_train'], data['y_train'])
        
        # Compute fairness confusion matrix on VALIDATION set
        if is_multiclass:
            fcm_results = compute_fairness_confusion_matrix(
                model, data['X_val'], data['y_val'],
                sensitive_col_indices=data['sensitive_col_indices'],
                is_multiclass=True
            )
        else:
            fcm_results = compute_fairness_confusion_matrix(
                model, data['X_val'], data['y_val'],
                sensitive_col_idx=data['sensitive_col_idx'],
                is_multiclass=False
            )
        
        # Print summary
        print_fairness_confusion_summary(fcm_results, f"{model_type.upper()} ({method_desc})")
        
        # Plot and save
        method_suffix = selection_method.replace(' ', '_').lower()
        fcm_paths = plot_fairness_confusion_matrix(
            fcm_results,
            model_name=f"{model_type.upper()} ({method_desc})",
            output_dir=output_dir,
            filename=f"fairness_confusion_{dataset_name}_{sensitive_feature}_{model_type}_{method_suffix}.png",
            formats=["notebook", "latex"]
        )
        
        # Display
        display(Image(filename=fcm_paths["notebook"]))
        
        all_results[model_type] = {
            'config': selected_config,
            'accuracy': accuracy,
            'consistency': consistency,
            'selection_method': selection_method,
            'fcm': fcm_results,
            'paths': fcm_paths
        }
    
    return all_results


# =============================================================================
# Trivial Fairness Analysis
# =============================================================================

def analyze_trivial_fairness(results, data, model_type='mlp'):
    """
    Investigate potentially degenerate 'trivially fair' configurations.
    
    Compares the fairest config with ReLU activation to diagnose vanishing gradients.
    
    Parameters
    ----------
    results : dict
        SMAC results dictionary
    data : dict
        Data dictionary with X_train, y_train, X_val, y_val
    model_type : str
        Model type to analyze (default: 'mlp')
    
    Returns
    -------
    dict
        Analysis results
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import balanced_accuracy_score
    from main import get_pareto_front
    
    if model_type not in results:
        print(f"No {model_type} results found.")
        return None
    
    # Get the fairest MLP config from Pareto front
    configs, costs = get_pareto_front(results[model_type])
    best_fair_idx = np.argmin(costs[:, 1])  # Lowest inconsistency
    fair_config = configs[best_fair_idx]
    
    print("="*70)
    print("TRIVIALLY FAIR MLP ANALYSIS")
    print("="*70)
    
    # Check if this is actually a degenerate case
    accuracy = 1 - costs[best_fair_idx, 0]
    consistency = 1 - costs[best_fair_idx, 1]
    
    print(f"\nFairest configuration from Pareto front:")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Consistency: {consistency:.1%}")
    
    if accuracy > 0.6:
        print(f"\n This config has reasonable accuracy ({accuracy:.1%}).")
        print(f"  Not a degenerate 'trivial fairness' case.")
        return {'is_degenerate': False, 'accuracy': accuracy, 'consistency': consistency}
    
    # Build config dict for sklearn
    fairest_config = {
        'hidden_layer_sizes': tuple([fair_config["n_neurons"]] * fair_config["n_hidden_layers"]),
        'activation': fair_config["activation"],
        'solver': fair_config["solver"],
        'alpha': fair_config["alpha"],
        'learning_rate_init': fair_config["learning_rate_init"],
        'max_iter': 500,
        'early_stopping': True,
        'random_state': 42
    }
    
    print(f"\nConfiguration: {fair_config['activation']} activation, {fair_config['solver']} solver")
    print(f"Learning rate: {fair_config['learning_rate_init']:.4f}, Alpha: {fair_config['alpha']:.4f}")
    
    # Get data splits
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    
    # Train with original config
    model_original = MLPClassifier(**fairest_config)
    model_original.fit(X_train, y_train)
    y_pred_original = model_original.predict(X_val)
    acc_original = balanced_accuracy_score(y_val, y_pred_original)
    
    print(f"\n--- With {fair_config['activation'].upper()} activation ---")
    print(f"Unique predictions: {np.unique(y_pred_original)}", end="")
    if len(np.unique(y_pred_original)) == 1:
        print("  Predicts only ONE class!")
    else:
        print("  Predicts both classes")
    print(f"Balanced Accuracy: {acc_original:.1%}")
    print(f"Training iterations: {model_original.n_iter_}")
    
    # Test with ReLU as a fix
    relu_config = fairest_config.copy()
    relu_config['activation'] = 'relu'
    
    model_relu = MLPClassifier(**relu_config)
    model_relu.fit(X_train, y_train)
    y_pred_relu = model_relu.predict(X_val)
    acc_relu = balanced_accuracy_score(y_val, y_pred_relu)
    
    print(f"\n--- With ReLU activation (same other params) ---")
    print(f"Unique predictions: {np.unique(y_pred_relu)}", end="")
    if len(np.unique(y_pred_relu)) == 1:
        print("  Predicts only ONE class!")
    else:
        print("  Predicts both classes")
    print(f"Balanced Accuracy: {acc_relu:.1%}")
    print(f"Training iterations: {model_relu.n_iter_}")
    
    # Conclusion
    print(f"\n" + "="*70)
    print("CONCLUSION:")
    print("="*70)
    
    if acc_relu > acc_original + 0.1:
        print(f" {fair_config['activation'].title()} activation caused vanishing gradients")
        print(f" Network never learned → constant predictions → 'trivial' 100% fairness")
        print(f" ReLU fixes this: accuracy improves from {acc_original:.1%} to {acc_relu:.1%}")
        print(f"\nHigh fairness + low accuracy = likely a degenerate solution!")
    else:
        print(f" Activation function doesn't appear to be the issue")
        print(f" May be caused by other hyperparameters (alpha, learning rate)")
    
    return {
        'is_degenerate': len(np.unique(y_pred_original)) == 1,
        'original_activation': fair_config['activation'],
        'acc_original': acc_original,
        'acc_relu': acc_relu,
        'config': fair_config
    }


# Pareto Point Selection

def select_balanced_pareto_point(configs, costs, method='knee'):
    """
    Select a balanced point from the Pareto front.
    
    Parameters
    ----------
    configs : list
        Pareto configurations
    costs : np.ndarray
        Cost matrix (n_points, 2) where columns are [error, inconsistency]
    method : str
        'knee' - maximum curvature point (recommended)
        'utopia' - closest to ideal point  
        'weighted' - weighted sum with w=0.5
        
    Returns
    -------
    int : index of selected point
    dict : selection info with accuracy, consistency, and method used
    """
    # Convert costs to objectives (higher is better)
    accuracy = 1 - costs[:, 0]
    consistency = 1 - costs[:, 1]
    
    # Handle edge case: only one point
    if len(configs) == 1:
        return 0, {
            'method': method,
            'accuracy': accuracy[0],
            'consistency': consistency[0],
        }
    
    # Normalize to [0, 1] for comparison
    acc_range = accuracy.max() - accuracy.min()
    con_range = consistency.max() - consistency.min()
    
    acc_norm = (accuracy - accuracy.min()) / (acc_range + 1e-10)
    con_norm = (consistency - consistency.min()) / (con_range + 1e-10)
    
    if method == 'knee':
        # Find point with maximum perpendicular distance from line connecting extremes
        # This is the "knee" or "elbow" point where trade-off is most balanced
        # Distance = (acc_norm + con_norm - 1) / sqrt(2) for line from (0,1) to (1,0)
        distances = (acc_norm + con_norm - 1) / np.sqrt(2)
        best_idx = np.argmax(distances)
        
    elif method == 'utopia':
        # Distance to utopia point (1, 1) in normalized space
        distances = np.sqrt((1 - acc_norm)**2 + (1 - con_norm)**2)
        best_idx = np.argmin(distances)
        
    elif method == 'weighted':
        # Equal weights (0.5 each)
        scores = 0.5 * acc_norm + 0.5 * con_norm
        best_idx = np.argmax(scores)
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'knee', 'utopia', or 'weighted'.")
    
    return best_idx, {
        'method': method,
        'accuracy': accuracy[best_idx],
        'consistency': consistency[best_idx],
        'all_accuracies': accuracy,
        'all_consistencies': consistency,
    }


# Case Study Setup

def setup_case_study_analysis(results, data, dataset_name, sensitive_feature,
                              selection_method='best_accuracy'):
    """
    Setup for case study analysis: train models and compute counterfactual predictions.
    
    Parameters
    ----------
    results : dict
        SMAC results dictionary
    data : dict
        Data dictionary with train/val splits
    dataset_name : str
        Name of the dataset
    sensitive_feature : str
        Name of the sensitive feature
    selection_method : str
        Method to select Pareto point: 'best_accuracy' (default), 'knee', 'utopia', 'weighted'
        
    Returns
    -------
    dict
        All computed values needed for case studies
    """
    from datasets import create_flipped_data_multiclass_exhaustive, create_flipped_data
    from main import get_pareto_front, create_rf_model, create_mlp_model
    
    # Describe selection method
    method_descriptions = {
        'knee': 'Knee-Optimal (Balanced)',
        'best_accuracy': 'Best Accuracy',
        'utopia': 'Closest to Utopia',
        'weighted': 'Weighted (50-50)',
    }
    method_desc = method_descriptions.get(selection_method, selection_method)
    
    # Get configs from Pareto front
    rf_configs, rf_costs = get_pareto_front(results['rf'])
    mlp_configs, mlp_costs = get_pareto_front(results['mlp'])
    
    # Select config based on method
    if selection_method == 'best_accuracy':
        rf_idx = np.argmin(rf_costs[:, 0])
        mlp_idx = np.argmin(mlp_costs[:, 0])
        rf_accuracy = 1 - rf_costs[rf_idx, 0]
        rf_consistency = 1 - rf_costs[rf_idx, 1]
        mlp_accuracy = 1 - mlp_costs[mlp_idx, 0]
        mlp_consistency = 1 - mlp_costs[mlp_idx, 1]
    else:
        rf_idx, rf_info = select_balanced_pareto_point(rf_configs, rf_costs, method=selection_method)
        mlp_idx, mlp_info = select_balanced_pareto_point(mlp_configs, mlp_costs, method=selection_method)
        rf_accuracy = rf_info['accuracy']
        rf_consistency = rf_info['consistency']
        mlp_accuracy = mlp_info['accuracy']
        mlp_consistency = mlp_info['consistency']
    
    rf_config = rf_configs[rf_idx]
    mlp_config = mlp_configs[mlp_idx]
    
    print(f"Selected Models ({method_desc} from Pareto Front):")
    print(f"  RF:  Accuracy={rf_accuracy:.4f}, Consistency={rf_consistency:.4f}")
    print(f"  MLP: Accuracy={mlp_accuracy:.4f}, Consistency={mlp_consistency:.4f}")
    
    # Create and train models on TRAINING set
    rf_model = create_rf_model(rf_config)
    mlp_model = create_mlp_model(mlp_config)
    
    rf_model.fit(data['X_train'], data['y_train'])
    mlp_model.fit(data['X_train'], data['y_train'])
    
    # Use VALIDATION set for analysis
    X_val = data['X_val']
    y_val = data['y_val']
    
    # Original predictions and probabilities
    rf_pred_orig = rf_model.predict(X_val)
    rf_proba_orig = rf_model.predict_proba(X_val)[:, 1]
    mlp_pred_orig = mlp_model.predict(X_val)
    mlp_proba_orig = mlp_model.predict_proba(X_val)[:, 1]
    
    # Determine if multiclass
    is_multiclass = 'sensitive_col_indices' in data
    
    # Initialize result dict
    cs = {
        'rf_model': rf_model,
        'mlp_model': mlp_model,
        'X_val': X_val,
        'y_val': y_val,
        'rf_pred_orig': rf_pred_orig,
        'rf_proba_orig': rf_proba_orig,
        'mlp_pred_orig': mlp_pred_orig,
        'mlp_proba_orig': mlp_proba_orig,
        'is_multiclass': is_multiclass,
        'sensitive_feature': sensitive_feature,
    }
    
    if is_multiclass:
        flipped_versions, original_categories = create_flipped_data_multiclass_exhaustive(
            X_val, data['sensitive_col_indices']
        )
        n_categories = len(data['sensitive_col_indices'])
        sens_names = data['sensitive_col_names']
        
        # Compute predictions for ALL flipped versions
        rf_preds_flipped = [rf_model.predict(fv) for fv in flipped_versions]
        rf_probas_flipped = [rf_model.predict_proba(fv)[:, 1] for fv in flipped_versions]
        mlp_preds_flipped = [mlp_model.predict(fv) for fv in flipped_versions]
        mlp_probas_flipped = [mlp_model.predict_proba(fv)[:, 1] for fv in flipped_versions]
        
        # Compute TRUE multiclass consistency
        rf_inconsistent = np.zeros(len(X_val), dtype=bool)
        mlp_inconsistent = np.zeros(len(X_val), dtype=bool)
        
        for target_cat in range(n_categories):
            is_actual_flip = (original_categories != target_cat)
            rf_inconsistent |= (is_actual_flip & (rf_pred_orig != rf_preds_flipped[target_cat]))
            mlp_inconsistent |= (is_actual_flip & (mlp_pred_orig != mlp_preds_flipped[target_cat]))
        
        # Probability analysis
        rf_proba_changes = np.array([np.abs(rf_probas_flipped[i] - rf_proba_orig) for i in range(n_categories)])
        mlp_proba_changes = np.array([np.abs(mlp_probas_flipped[i] - mlp_proba_orig) for i in range(n_categories)])
        
        rf_proba_change = rf_proba_changes.max(axis=0)
        mlp_proba_change = mlp_proba_changes.max(axis=0)
        rf_max_flip_target = rf_proba_changes.argmax(axis=0)
        mlp_max_flip_target = mlp_proba_changes.argmax(axis=0)
        
        cs.update({
            'original_categories': original_categories,
            'n_categories': n_categories,
            'sens_names': sens_names,
            'rf_preds_flipped': rf_preds_flipped,
            'rf_probas_flipped': rf_probas_flipped,
            'mlp_preds_flipped': mlp_preds_flipped,
            'mlp_probas_flipped': mlp_probas_flipped,
            'rf_max_flip_target': rf_max_flip_target,
            'mlp_max_flip_target': mlp_max_flip_target,
        })
        
        print(f"\nUsing MULTICLASS counterfactual ({n_categories} categories)")
        print(f"Categories: {sens_names}")
        
    else:
        # Binary case
        X_val_flipped = create_flipped_data(X_val, data['sensitive_col_idx'])
        
        rf_pred_flip = rf_model.predict(X_val_flipped)
        rf_proba_flip = rf_model.predict_proba(X_val_flipped)[:, 1]
        mlp_pred_flip = mlp_model.predict(X_val_flipped)
        mlp_proba_flip = mlp_model.predict_proba(X_val_flipped)[:, 1]
        
        rf_inconsistent = rf_pred_orig != rf_pred_flip
        mlp_inconsistent = mlp_pred_orig != mlp_pred_flip
        rf_proba_change = np.abs(rf_proba_flip - rf_proba_orig)
        mlp_proba_change = np.abs(mlp_proba_flip - mlp_proba_orig)
        
        cs.update({
            'X_val_flipped': X_val_flipped,
            'rf_pred_flip': rf_pred_flip,
            'rf_proba_flip': rf_proba_flip,
            'mlp_pred_flip': mlp_pred_flip,
            'mlp_proba_flip': mlp_proba_flip,
            'sensitive_col_idx': data['sensitive_col_idx'],
            'sensitive_col_name': data.get('sensitive_col_name', sensitive_feature),
        })
        
        print(f"\nUsing BINARY counterfactual (feature: {sensitive_feature})")
    
    cs.update({
        'rf_inconsistent': rf_inconsistent,
        'mlp_inconsistent': mlp_inconsistent,
        'rf_proba_change': rf_proba_change,
        'mlp_proba_change': mlp_proba_change,
    })
    
    # Summary
    print(f"\n{'='*60}")
    print("Counterfactual Consistency on VALIDATION SET:")
    print(f"  RF:  {100*(1-rf_inconsistent.mean()):.1f}% consistent ({rf_inconsistent.sum():,} inconsistent)")
    print(f"  MLP: {100*(1-mlp_inconsistent.mean()):.1f}% consistent ({mlp_inconsistent.sum():,} inconsistent)")
    
    return cs


# Case Study Functions


def case_study_prediction_flip(cs):
    """
    CASE 1: Find the most extreme prediction flip when sensitive attribute changes.
    
    Parameters
    ----------
    cs : dict
        Case study data from setup_case_study_analysis()
    """
    X_val = cs['X_val']
    y_val = cs['y_val']
    is_multiclass = cs['is_multiclass']
    rf_inconsistent = cs['rf_inconsistent']
    rf_proba_orig = cs['rf_proba_orig']
    rf_pred_orig = cs['rf_pred_orig']
    rf_proba_change = cs['rf_proba_change']
    
    rf_flip_indices = np.where(rf_inconsistent)[0]
    
    print("="*70)
    print("CASE: Prediction FLIPS When Sensitive Attribute Changes (Most Extreme)")
    print("="*70)
    
    if len(rf_flip_indices) == 0:
        print("No prediction flips found!")
        return
    
    flip_proba_changes = rf_proba_change[rf_flip_indices]
    most_extreme_local_idx = np.argmax(flip_proba_changes)
    idx = rf_flip_indices[most_extreme_local_idx]
    
    print(f"Sample {idx}: True label = {y_val[idx]}")
    
    if is_multiclass:
        orig_cat = cs['original_categories'][idx]
        target_cat = cs['rf_max_flip_target'][idx]
        sens_names = cs['sens_names']
        print(f"  Original ({sens_names[orig_cat]}):  P(class=1) = {rf_proba_orig[idx]:.4f} → Pred = {rf_pred_orig[idx]}")
        print(f"  Flipped to {sens_names[target_cat]}:  P(class=1) = {cs['rf_probas_flipped'][target_cat][idx]:.4f} → Pred = {cs['rf_preds_flipped'][target_cat][idx]}")
        print(f"  ΔP = {cs['rf_probas_flipped'][target_cat][idx] - rf_proba_orig[idx]:+.4f}")
        print(f"  Direction: {sens_names[orig_cat]} → {sens_names[target_cat]}")
    else:
        sens_name = cs['sensitive_col_name']
        orig_val = "1" if X_val[idx, cs['sensitive_col_idx']] == 1 else "0"
        flip_val = "0" if orig_val == "1" else "1"
        print(f"  Original ({sens_name}={orig_val}):  P(class=1) = {rf_proba_orig[idx]:.4f} → Pred = {rf_pred_orig[idx]}")
        print(f"  Flipped ({sens_name}={flip_val}):   P(class=1) = {cs['rf_proba_flip'][idx]:.4f} → Pred = {cs['rf_pred_flip'][idx]}")
        print(f"  ΔP = {cs['rf_proba_flip'][idx] - rf_proba_orig[idx]:+.4f}")


def case_study_consistent_sample(cs):
    """
    CASE 2: Find a high-confidence sample that stays consistent.
    
    Parameters
    ----------
    cs : dict
        Case study data from setup_case_study_analysis()
    """
    X_val = cs['X_val']
    y_val = cs['y_val']
    is_multiclass = cs['is_multiclass']
    rf_inconsistent = cs['rf_inconsistent']
    rf_proba_orig = cs['rf_proba_orig']
    rf_pred_orig = cs['rf_pred_orig']
    rf_proba_change = cs['rf_proba_change']
    
    rf_consistent_indices = np.where(~rf_inconsistent)[0]
    
    print("="*70)
    print("CASE: Prediction STAYS CONSISTENT (Robust to Change)")
    print("="*70)
    
    if len(rf_consistent_indices) == 0:
        print("No consistent samples found!")
        return
    
    consistent_proba = rf_proba_orig[rf_consistent_indices]
    high_conf_local_idx = np.argmax(np.abs(consistent_proba - 0.5))
    idx = rf_consistent_indices[high_conf_local_idx]
    
    print(f"Sample {idx}: True label = {y_val[idx]}")
    
    if is_multiclass:
        orig_cat = cs['original_categories'][idx]
        target_cat = cs['rf_max_flip_target'][idx]
        sens_names = cs['sens_names']
        print(f"  Original ({sens_names[orig_cat]}):  P(class=1) = {rf_proba_orig[idx]:.4f} → Pred = {rf_pred_orig[idx]}")
        print(f"  Max change flip ({sens_names[target_cat]}): P(class=1) = {cs['rf_probas_flipped'][target_cat][idx]:.4f} → Pred = {cs['rf_preds_flipped'][target_cat][idx]}")
        print(f"  Max |ΔP| = {rf_proba_change[idx]:.4f} (prediction unchanged)")
        print(f"  Model robust across ALL {cs['n_categories']} categories")
    else:
        sens_name = cs['sensitive_col_name']
        sens_idx = cs['sensitive_col_idx']
        orig_val = int(X_val[idx, sens_idx])
        flip_val = 1 - orig_val
        print(f"  Original ({sens_name}={orig_val}):  P(class=1) = {rf_proba_orig[idx]:.4f} → Pred = {rf_pred_orig[idx]}")
        print(f"  Flipped ({sens_name}={flip_val}):   P(class=1) = {cs['rf_proba_flip'][idx]:.4f} → Pred = {cs['rf_pred_flip'][idx]}")
        print(f"  |ΔP| = {rf_proba_change[idx]:.4f} (prediction unchanged)")
        print(f"  Model relies on other features, not {sens_name}")


def case_study_edge_cases(cs):
    """
    CASE 3: Analyze samples near the decision boundary.
    
    Parameters
    ----------
    cs : dict
        Case study data from setup_case_study_analysis()
    """
    X_val = cs['X_val']
    is_multiclass = cs['is_multiclass']
    rf_inconsistent = cs['rf_inconsistent']
    rf_proba_orig = cs['rf_proba_orig']
    
    print("="*70)
    print("CASE: Edge Cases Near Decision Boundary")
    print("="*70)
    
    boundary_dist = np.abs(rf_proba_orig - 0.5)
    near_boundary = boundary_dist < 0.1
    
    n_near_boundary = near_boundary.sum()
    n_near_boundary_inconsistent = (near_boundary & rf_inconsistent).sum()
    
    print(f"Samples within 10% of boundary: {n_near_boundary:,}")
    print(f"Of these, inconsistent: {n_near_boundary_inconsistent:,}")
    
    if n_near_boundary > 0:
        boundary_inconsistency_rate = n_near_boundary_inconsistent / n_near_boundary
        overall_inconsistency_rate = rf_inconsistent.mean()
        
        print(f"Inconsistency rate near boundary: {100*boundary_inconsistency_rate:.1f}%")
        print(f"Inconsistency rate overall: {100*overall_inconsistency_rate:.1f}%")
        
        if overall_inconsistency_rate > 0:
            vulnerability_ratio = boundary_inconsistency_rate / overall_inconsistency_rate
            print(f"\n Edge cases are {vulnerability_ratio:.1f}x more vulnerable!")
        
        if is_multiclass:
            print(f"\n(Evaluated across all {cs['n_categories']} {cs['sensitive_feature']} categories)")
    
    print(f"\n Analysis on VALIDATION set ({len(X_val):,} samples)")


def case_study_model_comparison(cs):
    """
    CASE 4: Compare RF vs MLP fairness behavior.
    
    Parameters
    ----------
    cs : dict
        Case study data from setup_case_study_analysis()
    """
    rf_inconsistent = cs['rf_inconsistent']
    mlp_inconsistent = cs['mlp_inconsistent']
    
    print("="*70)
    print("CASE: RF vs MLP - Different Models, Different Fairness")
    print("="*70)
    
    rf_only = rf_inconsistent & ~mlp_inconsistent
    mlp_only = ~rf_inconsistent & mlp_inconsistent
    both = rf_inconsistent & mlp_inconsistent
    neither = ~rf_inconsistent & ~mlp_inconsistent
    
    print(f"Only RF unfair:  {rf_only.sum():,}")
    print(f"Only MLP unfair: {mlp_only.sum():,}")
    print(f"Both unfair:     {both.sum():,}")
    print(f"Both fair:       {neither.sum():,}")
    print(f"\nModels can be unfair to DIFFERENT individuals!")


def case_study_probability_swings(cs):
    """
    CASE 5: Analyze largest probability swings with directional information.
    
    Parameters
    ----------
    cs : dict
        Case study data from setup_case_study_analysis()
    """
    X_val = cs['X_val']
    is_multiclass = cs['is_multiclass']
    rf_inconsistent = cs['rf_inconsistent']
    mlp_inconsistent = cs['mlp_inconsistent']
    rf_proba_orig = cs['rf_proba_orig']
    mlp_proba_orig = cs['mlp_proba_orig']
    rf_proba_change = cs['rf_proba_change']
    mlp_proba_change = cs['mlp_proba_change']
    
    print("="*70)
    print("CASE: Largest Probability Swings")
    print("="*70)
    
    rf_max_idx = np.argmax(rf_proba_change)
    mlp_max_idx = np.argmax(mlp_proba_change)
    
    if is_multiclass:
        sens_names = cs['sens_names']
        rf_target_cat = cs['rf_max_flip_target'][rf_max_idx]
        mlp_target_cat = cs['mlp_max_flip_target'][mlp_max_idx]
        rf_orig_cat = cs['original_categories'][rf_max_idx]
        mlp_orig_cat = cs['original_categories'][mlp_max_idx]
        
        print(f"RF  max swing: Sample {rf_max_idx}")
        print(f"    Direction: {sens_names[rf_orig_cat]} → {sens_names[rf_target_cat]}")
        print(f"    P: {rf_proba_orig[rf_max_idx]:.4f} → {cs['rf_probas_flipped'][rf_target_cat][rf_max_idx]:.4f} (Δ={rf_proba_change[rf_max_idx]:.4f})")
        print(f"    Prediction flipped: {'Yes ' if rf_inconsistent[rf_max_idx] else 'No '}")
        
        print(f"\nMLP max swing: Sample {mlp_max_idx}")
        print(f"    Direction: {sens_names[mlp_orig_cat]} → {sens_names[mlp_target_cat]}")
        print(f"    P: {mlp_proba_orig[mlp_max_idx]:.4f} → {cs['mlp_probas_flipped'][mlp_target_cat][mlp_max_idx]:.4f} (Δ={mlp_proba_change[mlp_max_idx]:.4f})")
        print(f"    Prediction flipped: {'Yes ' if mlp_inconsistent[mlp_max_idx] else 'No '}")
    else:
        print(f"RF  max swing: Sample {rf_max_idx}")
        print(f"    P: {rf_proba_orig[rf_max_idx]:.4f} → {cs['rf_proba_flip'][rf_max_idx]:.4f} (Δ={rf_proba_change[rf_max_idx]:.4f})")
        print(f"    Prediction flipped: {'Yes ' if rf_inconsistent[rf_max_idx] else 'No '}")
        
        print(f"\nMLP max swing: Sample {mlp_max_idx}")
        print(f"    P: {mlp_proba_orig[mlp_max_idx]:.4f} → {cs['mlp_proba_flip'][mlp_max_idx]:.4f} (Δ={mlp_proba_change[mlp_max_idx]:.4f})")
        print(f"    Prediction flipped: {'Yes ' if mlp_inconsistent[mlp_max_idx] else 'No '}")
    


def case_study_directional_analysis(cs):
    """
    CASE 6: Aggregate directional analysis of probability changes.
    
    Parameters
    ----------
    cs : dict
        Case study data from setup_case_study_analysis()
    """
    X_val = cs['X_val']
    is_multiclass = cs['is_multiclass']
    rf_proba_orig = cs['rf_proba_orig']
    mlp_proba_orig = cs['mlp_proba_orig']
    rf_proba_change = cs['rf_proba_change']
    mlp_proba_change = cs['mlp_proba_change']
    
    print("="*70)
    print("CASE: Aggregate Directional Analysis")
    print("="*70)
    
    if is_multiclass:
        rf_signed_change = np.array([
            cs['rf_probas_flipped'][cs['rf_max_flip_target'][i]][i] - rf_proba_orig[i] 
            for i in range(len(rf_proba_orig))
        ])
        mlp_signed_change = np.array([
            cs['mlp_probas_flipped'][cs['mlp_max_flip_target'][i]][i] - mlp_proba_orig[i] 
            for i in range(len(mlp_proba_orig))
        ])
        
        print(f"\nOverall Average Probability Changes:")
        print(f"  RF:  Mean ΔP = {rf_signed_change.mean():+.4f}, Mean |ΔP| = {rf_proba_change.mean():.4f}")
        print(f"  MLP: Mean ΔP = {mlp_signed_change.mean():+.4f}, Mean |ΔP| = {mlp_proba_change.mean():.4f}")
        
        sens_names = cs['sens_names']
        original_categories = cs['original_categories']
        
        print(f"\nRF breakdown by original {cs['sensitive_feature']}:")
        for cat_idx, cat_name in enumerate(sens_names):
            mask = original_categories == cat_idx
            if mask.sum() > 0:
                mean_signed = rf_signed_change[mask].mean()
                mean_abs = rf_proba_change[mask].mean()
                direction = "↑" if mean_signed > 0 else "↓"
                print(f"  {cat_name:<20}: Mean ΔP = {mean_signed:+.4f} {direction}, |ΔP| = {mean_abs:.4f} (n={mask.sum():,})")
        
        print(f"\nMLP breakdown by original {cs['sensitive_feature']}:")
        for cat_idx, cat_name in enumerate(sens_names):
            mask = original_categories == cat_idx
            if mask.sum() > 0:
                mean_signed = mlp_signed_change[mask].mean()
                mean_abs = mlp_proba_change[mask].mean()
                direction = "↑" if mean_signed > 0 else "↓"
                print(f"  {cat_name:<20}: Mean ΔP = {mean_signed:+.4f} {direction}, |ΔP| = {mean_abs:.4f} (n={mask.sum():,})")
    
    else:
        rf_signed_change = cs['rf_proba_flip'] - rf_proba_orig
        mlp_signed_change = cs['mlp_proba_flip'] - mlp_proba_orig
        
        print(f"\nOverall Average Probability Changes:")
        print(f"  RF:  Mean ΔP = {rf_signed_change.mean():+.4f}, Mean |ΔP| = {rf_proba_change.mean():.4f}")
        print(f"  MLP: Mean ΔP = {mlp_signed_change.mean():+.4f}, Mean |ΔP| = {mlp_proba_change.mean():.4f}")
        print(f"  (positive ΔP = flipping 0→1 INCREASES probability on average)")
        
        sens_idx = cs['sensitive_col_idx']
        sens_name = cs['sensitive_col_name']
        orig_vals = X_val[:, sens_idx]
        
        mask_0 = orig_vals == 0
        mask_1 = orig_vals == 1
        
        print(f"\nRF breakdown by original {sens_name}:")
        print(f"  {sens_name}=0 → 1: Mean ΔP = {rf_signed_change[mask_0].mean():+.4f}, |ΔP| = {rf_proba_change[mask_0].mean():.4f} (n={mask_0.sum():,})")
        print(f"  {sens_name}=1 → 0: Mean ΔP = {rf_signed_change[mask_1].mean():+.4f}, |ΔP| = {rf_proba_change[mask_1].mean():.4f} (n={mask_1.sum():,})")
        
        print(f"\nMLP breakdown by original {sens_name}:")
        print(f"  {sens_name}=0 → 1: Mean ΔP = {mlp_signed_change[mask_0].mean():+.4f}, |ΔP| = {mlp_proba_change[mask_0].mean():.4f} (n={mask_0.sum():,})")
        print(f"  {sens_name}=1 → 0: Mean ΔP = {mlp_signed_change[mask_1].mean():+.4f}, |ΔP| = {mlp_proba_change[mask_1].mean():.4f} (n={mask_1.sum():,})")
    
    print(f"\n Analysis on VALIDATION set ({len(X_val):,} samples)")

