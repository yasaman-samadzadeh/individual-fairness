"""
Ablation Studies and Deep Analysis Script

This script performs comprehensive ablation studies:
1. Hyperparameter Ablation: Impact of individual hyperparameters
2. SMAC Configuration Ablation: Impact of optimization settings  
3. Model Comparison: RF vs MLP performance analysis
4. Fairness Analysis: Counterfactual consistency analysis
5. Sensitivity Analysis: Robustness to different settings

Usage:
    python ablation_analysis.py --study hyperparameter --model rf
    python ablation_analysis.py --study smac --model rf
    python ablation_analysis.py --study comparison
"""

import sys
import os
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import time
import json
import argparse
from pathlib import Path

# Project imports
from utils.datasets import load_dataset, get_dataset_config
from utils.individual_fairness import counterfactual_consistency
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create output directories
os.makedirs('ablation_results', exist_ok=True)
os.makedirs('ablation_results/plots', exist_ok=True)
os.makedirs('ablation_results/tables', exist_ok=True)


# =============================================================================
# Helper Functions
# =============================================================================

def evaluate_configuration(
    model_type: str,
    config_dict: Dict,
    X: np.ndarray,
    y: np.ndarray,
    sensitive_col_idx: int,
    n_cv_splits: int = 5,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Evaluate a single configuration using cross-validation.
    
    Returns:
        dict with 'accuracy', 'consistency', 'error', 'inconsistency'
    """
    accuracy_scores = []
    consistency_scores = []
    
    cv = StratifiedKFold(n_splits=n_cv_splits, shuffle=True, random_state=random_state)
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Create model
        if model_type == 'rf':
            model = RandomForestClassifier(
                n_estimators=config_dict.get('n_estimators', 100),
                max_depth=config_dict.get('max_depth', None),
                min_samples_split=config_dict.get('min_samples_split', 2),
                min_samples_leaf=config_dict.get('min_samples_leaf', 1),
                criterion=config_dict.get('criterion', 'gini'),
                max_features=config_dict.get('max_features', 'sqrt'),
                random_state=random_state,
                n_jobs=-1
            )
        elif model_type == 'mlp':
            n_neurons = config_dict.get('n_neurons', 64)
            n_layers = config_dict.get('n_hidden_layers', 2)
            hidden_sizes = tuple([n_neurons] * n_layers)
            
            model = MLPClassifier(
                hidden_layer_sizes=hidden_sizes,
                activation=config_dict.get('activation', 'relu'),
                solver=config_dict.get('solver', 'adam'),
                alpha=config_dict.get('alpha', 1e-4),
                learning_rate_init=config_dict.get('learning_rate_init', 1e-3),
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=random_state
            )
        
        # Train and evaluate
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = balanced_accuracy_score(y_test, y_pred)
        accuracy_scores.append(acc)
        
        # Counterfactual consistency
        X_test_flipped = X_test.copy()
        X_test_flipped[:, sensitive_col_idx] = 1 - X_test_flipped[:, sensitive_col_idx]
        y_pred_flipped = model.predict(X_test_flipped)
        consistency = counterfactual_consistency(y_pred, y_pred_flipped)
        consistency_scores.append(consistency)
    
    mean_acc = np.mean(accuracy_scores)
    mean_cons = np.mean(consistency_scores)
    
    return {
        'accuracy': mean_acc,
        'consistency': mean_cons,
        'error': 1.0 - mean_acc,
        'inconsistency': 1.0 - mean_cons
    }


# =============================================================================
# 1. Hyperparameter Ablation Studies
# =============================================================================

def hyperparameter_ablation_rf(
    X: np.ndarray,
    y: np.ndarray,
    sensitive_col_idx: int,
    base_config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Ablation study for Random Forest hyperparameters.
    Tests each hyperparameter individually while keeping others at default.
    """
    if base_config is None:
        base_config = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'criterion': 'gini',
            'max_features': 'sqrt'
        }
    
    results = []
    
    # Test n_estimators
    print("Testing n_estimators...")
    for n_est in [10, 50, 100, 150, 200]:
        config = base_config.copy()
        config['n_estimators'] = n_est
        metrics = evaluate_configuration('rf', config, X, y, sensitive_col_idx)
        results.append({
            'hyperparameter': 'n_estimators',
            'value': n_est,
            **metrics
        })
    
    # Test max_depth
    print("Testing max_depth...")
    for max_d in [3, 5, 10, 15, 20, None]:
        config = base_config.copy()
        config['max_depth'] = max_d
        metrics = evaluate_configuration('rf', config, X, y, sensitive_col_idx)
        results.append({
            'hyperparameter': 'max_depth',
            'value': max_d if max_d else 'None',
            **metrics
        })
    
    # Test min_samples_split
    print("Testing min_samples_split...")
    for mss in [2, 5, 10, 15, 20]:
        config = base_config.copy()
        config['min_samples_split'] = mss
        metrics = evaluate_configuration('rf', config, X, y, sensitive_col_idx)
        results.append({
            'hyperparameter': 'min_samples_split',
            'value': mss,
            **metrics
        })
    
    # Test criterion
    print("Testing criterion...")
    for crit in ['gini', 'entropy']:
        config = base_config.copy()
        config['criterion'] = crit
        metrics = evaluate_configuration('rf', config, X, y, sensitive_col_idx)
        results.append({
            'hyperparameter': 'criterion',
            'value': crit,
            **metrics
        })
    
    # Test max_features
    print("Testing max_features...")
    for mf in ['sqrt', 'log2', None]:
        config = base_config.copy()
        config['max_features'] = mf
        metrics = evaluate_configuration('rf', config, X, y, sensitive_col_idx)
        results.append({
            'hyperparameter': 'max_features',
            'value': mf if mf else 'None',
            **metrics
        })
    
    return pd.DataFrame(results)


def hyperparameter_ablation_mlp(
    X: np.ndarray,
    y: np.ndarray,
    sensitive_col_idx: int,
    base_config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Ablation study for MLP hyperparameters.
    """
    if base_config is None:
        base_config = {
            'n_hidden_layers': 2,
            'n_neurons': 64,
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 1e-4,
            'learning_rate_init': 1e-3
        }
    
    results = []
    
    # Test n_hidden_layers
    print("Testing n_hidden_layers...")
    for n_layers in [1, 2, 3]:
        config = base_config.copy()
        config['n_hidden_layers'] = n_layers
        metrics = evaluate_configuration('mlp', config, X, y, sensitive_col_idx)
        results.append({
            'hyperparameter': 'n_hidden_layers',
            'value': n_layers,
            **metrics
        })
    
    # Test n_neurons
    print("Testing n_neurons...")
    for n_neurons in [16, 32, 64, 128, 256]:
        config = base_config.copy()
        config['n_neurons'] = n_neurons
        metrics = evaluate_configuration('mlp', config, X, y, sensitive_col_idx)
        results.append({
            'hyperparameter': 'n_neurons',
            'value': n_neurons,
            **metrics
        })
    
    # Test activation
    print("Testing activation...")
    for act in ['relu', 'tanh', 'logistic']:
        config = base_config.copy()
        config['activation'] = act
        metrics = evaluate_configuration('mlp', config, X, y, sensitive_col_idx)
        results.append({
            'hyperparameter': 'activation',
            'value': act,
            **metrics
        })
    
    # Test solver
    print("Testing solver...")
    for solv in ['adam', 'sgd']:
        config = base_config.copy()
        config['solver'] = solv
        metrics = evaluate_configuration('mlp', config, X, y, sensitive_col_idx)
        results.append({
            'hyperparameter': 'solver',
            'value': solv,
            **metrics
        })
    
    # Test alpha (regularization)
    print("Testing alpha (regularization)...")
    for alpha in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        config = base_config.copy()
        config['alpha'] = alpha
        metrics = evaluate_configuration('mlp', config, X, y, sensitive_col_idx)
        results.append({
            'hyperparameter': 'alpha',
            'value': alpha,
            **metrics
        })
    
    return pd.DataFrame(results)


# =============================================================================
# 2. Visualization Functions
# =============================================================================

def plot_hyperparameter_ablation(
    df: pd.DataFrame,
    model_type: str,
    output_path: str
):
    """
    Plot hyperparameter ablation results.
    """
    hyperparams = df['hyperparameter'].unique()
    n_hyperparams = len(hyperparams)
    
    fig, axes = plt.subplots(2, n_hyperparams, figsize=(5*n_hyperparams, 10))
    
    if n_hyperparams == 1:
        axes = axes.reshape(2, 1)
    
    for idx, hp in enumerate(hyperparams):
        hp_df = df[df['hyperparameter'] == hp].copy()
        
        # Sort values properly handling mixed types (int/float vs string/None)
        # Strategy: Always create a sortable column, never sort the original 'value' column directly
        
        # Step 1: Handle None values - convert to string
        hp_df['value_clean'] = hp_df['value'].apply(lambda x: 'None' if x is None else x)
        
        # Step 2: Create a sortable column based on value types
        # Try numeric conversion
        numeric_vals = pd.to_numeric(hp_df['value_clean'], errors='coerce')
        
        # Check if all non-None values are numeric
        none_mask = hp_df['value_clean'] == 'None'
        non_none_numeric = numeric_vals[~none_mask].notna().all() if (~none_mask).any() else False
        
        if non_none_numeric and numeric_vals.notna().any():
            # All non-None values are numeric - sort numerically
            hp_df['sort_col'] = numeric_vals
            # Put None as infinity (last)
            hp_df.loc[none_mask, 'sort_col'] = float('inf')
            hp_df = hp_df.sort_values('sort_col')
        else:
            # Mixed types or all categorical - sort as strings
            hp_df['sort_col'] = hp_df['value_clean'].astype(str)
            hp_df = hp_df.sort_values('sort_col')
        
        # Convert to string for plotting (always use string for display)
        hp_df['value_display'] = hp_df['value_clean'].astype(str)
        
        # Plot accuracy vs hyperparameter value
        ax1 = axes[0, idx]
        ax1.plot(hp_df['value_display'], hp_df['accuracy'], 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel(f'{hp}', fontsize=12)
        ax1.set_ylabel('Balanced Accuracy', fontsize=12)
        ax1.set_title(f'{hp} vs Accuracy', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot consistency vs hyperparameter value
        ax2 = axes[1, idx]
        ax2.plot(hp_df['value_display'], hp_df['consistency'], 's-', linewidth=2, markersize=8, color='orange')
        ax2.set_xlabel(f'{hp}', fontsize=12)
        ax2.set_ylabel('Counterfactual Consistency', fontsize=12)
        ax2.set_title(f'{hp} vs Consistency', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
    
    plt.suptitle(f'{model_type.upper()} Hyperparameter Ablation Study', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_path}")


def create_summary_table(rf_ablation_df: pd.DataFrame, mlp_ablation_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary table of hyperparameter impacts.
    """
    summary = []
    
    for model_type, df in [('RF', rf_ablation_df), ('MLP', mlp_ablation_df)]:
        for hp in df['hyperparameter'].unique():
            hp_df = df[df['hyperparameter'] == hp]
            
            summary.append({
                'Model': model_type,
                'Hyperparameter': hp,
                'Accuracy_Range': f"{hp_df['accuracy'].min():.4f} - {hp_df['accuracy'].max():.4f}",
                'Accuracy_Std': hp_df['accuracy'].std(),
                'Consistency_Range': f"{hp_df['consistency'].min():.4f} - {hp_df['consistency'].max():.4f}",
                'Consistency_Std': hp_df['consistency'].std(),
                'Best_Value_Acc': hp_df.loc[hp_df['accuracy'].idxmax(), 'value'],
                'Best_Value_Cons': hp_df.loc[hp_df['consistency'].idxmax(), 'value']
            })
    
    return pd.DataFrame(summary)


# =============================================================================
# 3. Model Comparison Analysis
# =============================================================================

def compare_models_deep(
    rf_smac,  # SMAC optimizer for RF
    mlp_smac,  # SMAC optimizer for MLP
    output_dir: str = 'ablation_results'
) -> Dict:
    """
    Deep comparison between RF and MLP models.
    """
    from main import get_pareto_front, get_all_costs
    
    comparison = {}
    
    # Extract Pareto fronts
    rf_configs, rf_costs = get_pareto_front(rf_smac)
    mlp_configs, mlp_costs = get_pareto_front(mlp_smac)
    
    # Get all costs
    rf_all_costs = get_all_costs(rf_smac)
    mlp_all_costs = get_all_costs(mlp_smac)
    
    # Convert to scores
    rf_scores = 1 - rf_costs
    mlp_scores = 1 - mlp_costs
    
    # Best configurations
    rf_best_acc_idx = np.argmin(rf_costs[:, 0])
    rf_best_cons_idx = np.argmin(rf_costs[:, 1])
    mlp_best_acc_idx = np.argmin(mlp_costs[:, 0])
    mlp_best_cons_idx = np.argmin(mlp_costs[:, 1])
    
    comparison['rf'] = {
        'n_evaluated': len(rf_all_costs),
        'n_pareto': len(rf_costs),
        'best_accuracy': 1 - rf_costs[rf_best_acc_idx, 0],
        'best_consistency': 1 - rf_costs[rf_best_cons_idx, 1],
        'pareto_front': rf_scores.tolist()
    }
    
    comparison['mlp'] = {
        'n_evaluated': len(mlp_all_costs),
        'n_pareto': len(mlp_costs),
        'best_accuracy': 1 - mlp_costs[mlp_best_acc_idx, 0],
        'best_consistency': 1 - mlp_costs[mlp_best_cons_idx, 1],
        'pareto_front': mlp_scores.tolist()
    }
    
    # Calculate dominance
    comparison['rf_dominates_mlp'] = _check_dominance(rf_scores, mlp_scores)
    comparison['mlp_dominates_rf'] = _check_dominance(mlp_scores, rf_scores)
    
    # Calculate hypervolume (area under Pareto front)
    comparison['rf_hypervolume'] = _calculate_hypervolume(rf_scores)
    comparison['mlp_hypervolume'] = _calculate_hypervolume(mlp_scores)
    
    return comparison


def _check_dominance(front1: np.ndarray, front2: np.ndarray) -> float:
    """
    Check what percentage of front2 is dominated by front1.
    """
    if len(front2) == 0:
        return 0.0
    
    dominated_count = 0
    for point2 in front2:
        for point1 in front1:
            # point1 dominates point2 if better in all objectives
            if np.all(point1 >= point2) and np.any(point1 > point2):
                dominated_count += 1
                break
    return dominated_count / len(front2)


def _calculate_hypervolume(pareto_scores: np.ndarray, ref_point: Tuple[float, float] = (0.5, 0.5)) -> float:
    """
    Calculate hypervolume (area under Pareto front) as a measure of quality.
    """
    if len(pareto_scores) == 0:
        return 0.0
    
    # Sort by first objective
    sorted_scores = pareto_scores[np.argsort(pareto_scores[:, 0])]
    
    # Calculate area using trapezoidal rule
    area = 0.0
    prev_x = ref_point[0]
    
    for point in sorted_scores:
        x, y = point[0], point[1]
        area += (x - prev_x) * (y - ref_point[1])
        prev_x = x
    
    return area


# =============================================================================
# Main Execution
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run ablation studies')
    parser.add_argument('--study', type=str, choices=['hyperparameter', 'comparison', 'all'],
                       default='all', help='Type of ablation study to run')
    parser.add_argument('--model', type=str, choices=['rf', 'mlp', 'both'],
                       default='both', help='Model to test')
    parser.add_argument('--dataset', type=str, default='adult',
                       help='Dataset to use')
    parser.add_argument('--sensitive', type=str, default='sex',
                       help='Sensitive feature')
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    data = load_dataset(args.dataset, sensitive_feature=args.sensitive)
    X = data['X_train']
    y = data['y_train']
    sensitive_col_idx = data['sensitive_col_idx']
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Run hyperparameter ablation
    if args.study in ['hyperparameter', 'all']:
        if args.model in ['rf', 'both']:
            print("\n" + "="*60)
            print("RF Hyperparameter Ablation")
            print("="*60)
            
            rf_ablation_df = hyperparameter_ablation_rf(X, y, sensitive_col_idx)
            rf_ablation_df.to_csv('ablation_results/tables/rf_hyperparameter_ablation.csv', index=False)
            print("\nRF Ablation Results:")
            print(rf_ablation_df)
            
            plot_hyperparameter_ablation(
                rf_ablation_df,
                'rf',
                'ablation_results/plots/rf_hyperparameter_ablation.png'
            )
        
        if args.model in ['mlp', 'both']:
            print("\n" + "="*60)
            print("MLP Hyperparameter Ablation")
            print("="*60)
            
            mlp_ablation_df = hyperparameter_ablation_mlp(X, y, sensitive_col_idx)
            mlp_ablation_df.to_csv('ablation_results/tables/mlp_hyperparameter_ablation.csv', index=False)
            print("\nMLP Ablation Results:")
            print(mlp_ablation_df)
            
            plot_hyperparameter_ablation(
                mlp_ablation_df,
                'mlp',
                'ablation_results/plots/mlp_hyperparameter_ablation.png'
            )
        
        # Create summary table
        if args.model == 'both':
            summary_df = create_summary_table(rf_ablation_df, mlp_ablation_df)
            summary_df.to_csv('ablation_results/tables/hyperparameter_summary.csv', index=False)
            print("\nSummary Table:")
            print(summary_df.to_string())
    
    # Run model comparison
    if args.study in ['comparison', 'all']:
        print("\n" + "="*60)
        print("Model Comparison Analysis")
        print("="*60)
        print("Note: This requires existing SMAC optimization results.")
        print("Run main.py first to generate results, then load them here.")
        
        # This would load existing results - implementation depends on how you store them
        # For now, just show the structure
        print("\nTo run comparison, load SMAC results and call:")
        print("comparison = compare_models_deep(rf_smac, mlp_smac)")


if __name__ == "__main__":
    main()

