"""
Multi-Objective Fairness Optimization with SMAC

Approach 1: Standard models with sensitive features included.
Evaluate using counterfactual consistency on the sensitive feature directly.

Models: RandomForest, MLP (separate SMAC runs for each)
Objectives: Accuracy (maximize) + Counterfactual Consistency (maximize)
"""

from __future__ import annotations

import os
import warnings
from typing import Optional, Dict, List
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    Float,
    Integer,
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.facade.abstract_facade import AbstractFacade
from smac.multi_objective.parego import ParEGO

from utils.datasets import load_dataset, create_flipped_data, list_available_datasets, get_dataset_config
from utils.individual_fairness import counterfactual_consistency


# =============================================================================
# Configuration Spaces for Each Model
# =============================================================================

def get_rf_configspace() -> ConfigurationSpace:
    """Configuration space for Random Forest."""
    cs = ConfigurationSpace(seed=42)
    
    cs.add_hyperparameters([
        Integer("n_estimators", (10, 200), default=100),
        Integer("max_depth", (3, 20), default=10),
        Integer("min_samples_split", (2, 20), default=2),
        Integer("min_samples_leaf", (1, 10), default=1),
        Categorical("criterion", ["gini", "entropy"], default="gini"),
        Categorical("max_features", ["sqrt", "log2", None], default="sqrt"),
    ])
    
    return cs


def get_mlp_configspace() -> ConfigurationSpace:
    """Configuration space for MLP."""
    cs = ConfigurationSpace(seed=42)
    
    cs.add_hyperparameters([
        Integer("n_hidden_layers", (1, 3), default=2),
        Integer("n_neurons", (16, 256), log=True, default=64),
        Categorical("activation", ["relu", "tanh", "logistic"], default="relu"),
        Categorical("solver", ["adam", "sgd"], default="adam"),
        Float("alpha", (1e-5, 1e-1), log=True, default=1e-4),  # L2 regularization
        Float("learning_rate_init", (1e-4, 1e-1), log=True, default=1e-3),
    ])
    
    return cs


# =============================================================================
# Model Creation
# =============================================================================

def create_rf_model(config: Configuration) -> RandomForestClassifier:
    """Create Random Forest model from configuration."""
    return RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_split=config["min_samples_split"],
        min_samples_leaf=config["min_samples_leaf"],
        criterion=config["criterion"],
        max_features=config["max_features"],
        random_state=42,
        n_jobs=-1,
    )


def create_mlp_model(config: Configuration) -> MLPClassifier:
    """Create MLP model from configuration."""
    hidden_layer_sizes = tuple([config["n_neurons"]] * config["n_hidden_layers"])
    
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=config["activation"],
        solver=config["solver"],
        alpha=config["alpha"],
        learning_rate_init=config["learning_rate_init"],
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )


# =============================================================================
# Training and Evaluation Pipeline
# =============================================================================

class FairnessPipeline:
    """
    Pipeline for multi-objective optimization with fairness.
    
    Trains models using cross-validation and evaluates both
    accuracy and counterfactual consistency.
    """
    
    def __init__(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        sensitive_col_idx: int,
        n_cv_splits: int = 5,
    ):
        """
        Parameters
        ----------
        model_type : str
            "rf" for Random Forest, "mlp" for MLP
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        sensitive_col_idx : int
            Index of sensitive column for counterfactual
        n_cv_splits : int
            Number of cross-validation splits
        """
        self.model_type = model_type
        self.X = X
        self.y = y
        self.sensitive_col_idx = sensitive_col_idx
        self.n_cv_splits = n_cv_splits
        
        # Pre-compute CV splits for consistency
        cv = StratifiedKFold(n_splits=n_cv_splits, shuffle=True, random_state=42)
        self.cv_splits = list(cv.split(X, y))
    
    @property
    def configspace(self) -> ConfigurationSpace:
        if self.model_type == "rf":
            return get_rf_configspace()
        elif self.model_type == "mlp":
            return get_mlp_configspace()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _create_model(self, config: Configuration):
        if self.model_type == "rf":
            return create_rf_model(config)
        elif self.model_type == "mlp":
            return create_mlp_model(config)
    
    def train(self, config: Configuration, seed: int = 0) -> Dict[str, float]:
        """
        Train and evaluate a configuration using cross-validation.
        
        Returns
        -------
        dict with two objectives (both to minimize for SMAC):
            - error: 1 - balanced_accuracy (lower = better)
            - inconsistency: 1 - counterfactual_consistency (lower = better)
        """
        accuracy_scores = []
        consistency_scores = []
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            for train_idx, test_idx in self.cv_splits:
                # Split data
                X_train, X_test = self.X[train_idx], self.X[test_idx]
                y_train, y_test = self.y[train_idx], self.y[test_idx]
                
                # Create and train model
                model = self._create_model(config)
                model.fit(X_train, y_train)
                
                # Evaluate accuracy
                y_pred = model.predict(X_test)
                acc = balanced_accuracy_score(y_test, y_pred)
                accuracy_scores.append(acc)
                
                # Evaluate counterfactual consistency
                X_test_flipped = create_flipped_data(X_test, self.sensitive_col_idx)
                y_pred_flipped = model.predict(X_test_flipped)
                consistency = counterfactual_consistency(y_pred, y_pred_flipped)
                consistency_scores.append(consistency)
        
        # Return objectives (SMAC minimizes, so return 1 - score)
        return {
            "error": 1.0 - np.mean(accuracy_scores),
            "inconsistency": 1.0 - np.mean(consistency_scores),
        }


# =============================================================================
# SMAC Optimization
# =============================================================================

def run_optimization(
    model_type: str,
    data: Dict,
    walltime_limit: int = 300,
    n_trials: int = 100,
    output_dir: str = "smac_output",
) -> AbstractFacade:
    """
    Run SMAC multi-objective optimization for a model.
    
    Parameters
    ----------
    model_type : str
        "rf" or "mlp"
    data : dict
        Data dictionary from load_adult_dataset()
    walltime_limit : int
        Time limit in seconds
    n_trials : int
        Maximum number of configurations to try
    output_dir : str
        Directory to save SMAC output
        
    Returns
    -------
    smac : AbstractFacade
        SMAC optimizer with results
    """
    print(f"\n{'='*60}")
    print(f"Running SMAC optimization for {model_type.upper()}")
    print(f"{'='*60}")
    
    # Create pipeline
    pipeline = FairnessPipeline(
        model_type=model_type,
        X=data['X_train'],
        y=data['y_train'],
        sensitive_col_idx=data['sensitive_col_idx'],
    )
    
    # Define scenario
    objectives = ["error", "inconsistency"]
    
    scenario = Scenario(
        pipeline.configspace,
        objectives=objectives,
        walltime_limit=walltime_limit,
        n_trials=n_trials,
        n_workers=1,
        output_directory=os.path.join(output_dir, model_type),
    )
    
    # Create SMAC
    initial_design = HPOFacade.get_initial_design(scenario, n_configs=5)
    multi_objective_algorithm = ParEGO(scenario)
    
    smac = HPOFacade(
        scenario,
        pipeline.train,
        initial_design=initial_design,
        multi_objective_algorithm=multi_objective_algorithm,
        overwrite=True,
    )
    
    # Optimize
    incumbents = smac.optimize()
    
    # Print results
    print(f"\nOptimization complete for {model_type.upper()}")
    print(f"Number of configurations evaluated: {len(smac.runhistory.get_configs())}")
    
    return smac


# =============================================================================
# Pareto Front Extraction and Plotting
# =============================================================================

def get_pareto_front(smac: AbstractFacade) -> tuple:
    """
    Extract Pareto front from SMAC runhistory.
    
    Returns
    -------
    configs : list
        Pareto-optimal configurations
    costs : np.ndarray
        Costs for Pareto-optimal configurations (shape: n_configs x 2)
    """
    configs = smac.runhistory.get_configs()
    
    # Get average costs for each configuration
    costs = []
    for config in configs:
        cost = smac.runhistory.average_cost(config)
        costs.append(cost)
    
    costs = np.array(costs)
    
    # Find Pareto front
    is_pareto = np.ones(len(costs), dtype=bool)
    
    for i, c in enumerate(costs):
        if is_pareto[i]:
            # Check if any other point dominates this one
            is_pareto[is_pareto] = np.any(costs[is_pareto] < c, axis=1) | np.all(costs[is_pareto] == c, axis=1)
            is_pareto[i] = True
    
    pareto_configs = [configs[i] for i in range(len(configs)) if is_pareto[i]]
    pareto_costs = costs[is_pareto]
    
    # Sort by first objective
    sort_idx = np.argsort(pareto_costs[:, 0])
    pareto_costs = pareto_costs[sort_idx]
    pareto_configs = [pareto_configs[i] for i in sort_idx]
    
    return pareto_configs, pareto_costs


def get_all_costs(smac: AbstractFacade) -> np.ndarray:
    """Get costs for all evaluated configurations."""
    configs = smac.runhistory.get_configs()
    costs = [smac.runhistory.average_cost(config) for config in configs]
    return np.array(costs)


def plot_pareto_comparison(
    results: Dict[str, AbstractFacade],
    output_path: str = "plots/pareto_comparison.png",
):
    """
    Plot Pareto fronts for multiple models.
    
    Parameters
    ----------
    results : dict
        Dictionary mapping model_type -> smac optimizer
    output_path : str
        Path to save the plot
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {'rf': 'blue', 'mlp': 'orange'}
    markers = {'rf': 'o', 'mlp': 's'}
    
    # Plot 1: All points with Pareto fronts
    ax1 = axes[0]
    
    for model_type, smac in results.items():
        # All points
        all_costs = get_all_costs(smac)
        ax1.scatter(
            all_costs[:, 0], all_costs[:, 1],
            c=colors[model_type], marker=markers[model_type],
            alpha=0.3, s=30, label=f'{model_type.upper()} (all)'
        )
        
        # Pareto front
        _, pareto_costs = get_pareto_front(smac)
        ax1.scatter(
            pareto_costs[:, 0], pareto_costs[:, 1],
            c=colors[model_type], marker=markers[model_type],
            s=100, edgecolors='black', linewidths=2,
            label=f'{model_type.upper()} (Pareto)'
        )
        ax1.plot(pareto_costs[:, 0], pareto_costs[:, 1], 
                 c=colors[model_type], linestyle='--', alpha=0.7)
    
    ax1.set_xlabel('Error (1 - Balanced Accuracy)', fontsize=12)
    ax1.set_ylabel('Inconsistency (1 - Counterfactual Consistency)', fontsize=12)
    ax1.set_title('All Configurations with Pareto Fronts', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Just Pareto fronts (cleaner view)
    ax2 = axes[1]
    
    for model_type, smac in results.items():
        _, pareto_costs = get_pareto_front(smac)
        
        # Convert to accuracy/consistency for easier interpretation
        accuracy = 1 - pareto_costs[:, 0]
        consistency = 1 - pareto_costs[:, 1]
        
        ax2.scatter(
            accuracy, consistency,
            c=colors[model_type], marker=markers[model_type],
            s=100, edgecolors='black', linewidths=2,
            label=f'{model_type.upper()}'
        )
        ax2.plot(accuracy, consistency, 
                 c=colors[model_type], linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('Balanced Accuracy', fontsize=12)
    ax2.set_ylabel('Counterfactual Consistency', fontsize=12)
    ax2.set_title('Pareto Fronts (Higher = Better)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to: {output_path}")


def print_pareto_summary(results: Dict[str, AbstractFacade]):
    """Print summary of Pareto-optimal configurations."""
    
    print("\n" + "="*60)
    print("PARETO FRONT SUMMARY")
    print("="*60)
    
    for model_type, smac in results.items():
        configs, costs = get_pareto_front(smac)
        
        print(f"\n{model_type.upper()} Pareto Front ({len(configs)} configurations):")
        print("-" * 50)
        
        for i, (config, cost) in enumerate(zip(configs, costs)):
            accuracy = 1 - cost[0]
            consistency = 1 - cost[1]
            print(f"  Config {i+1}: Accuracy={accuracy:.4f}, Consistency={consistency:.4f}")
        
        # Best accuracy
        best_acc_idx = np.argmin(costs[:, 0])
        print(f"\n  Best Accuracy: {1-costs[best_acc_idx, 0]:.4f} (Consistency: {1-costs[best_acc_idx, 1]:.4f})")
        
        # Best consistency
        best_cons_idx = np.argmin(costs[:, 1])
        print(f"  Best Consistency: {1-costs[best_cons_idx, 1]:.4f} (Accuracy: {1-costs[best_cons_idx, 0]:.4f})")


# =============================================================================
# Main Entry Point
# =============================================================================

def main(
    dataset_name: str = "adult",
    sensitive_feature: str = "sex",
    walltime_limit: int = 300,
    n_trials: int = 100,
):
    """
    Run the full experiment.
    
    Parameters
    ----------
    dataset_name : str
        Name of dataset to use (e.g., "adult", "german_credit")
    sensitive_feature : str
        Which sensitive feature to use (depends on dataset)
    walltime_limit : int
        Time limit per model in seconds
    n_trials : int
        Max configurations per model
    """
    # Get dataset config to show available sensitive features
    config = get_dataset_config(dataset_name)
    
    print("="*60)
    print("APPROACH 1: Standard Models with Sensitive Features")
    print("="*60)
    print(f"Dataset: {config.name} (OpenML ID: {config.openml_id})")
    print(f"Sensitive feature: {sensitive_feature}")
    print(f"Time limit per model: {walltime_limit}s")
    print(f"Max trials per model: {n_trials}")
    
    # Load data
    print(f"\nLoading {config.name} dataset...")
    data = load_dataset(dataset_name, sensitive_feature=sensitive_feature)
    
    # Run optimization for each model
    results = {}
    
    for model_type in ["rf", "mlp"]:
        smac = run_optimization(
            model_type=model_type,
            data=data,
            walltime_limit=walltime_limit,
            n_trials=n_trials,
        )
        results[model_type] = smac
    
    # Plot and summarize results
    output_path = f"plots/pareto_{dataset_name}_{sensitive_feature}.png"
    plot_pareto_comparison(results, output_path=output_path)
    print_pareto_summary(results)
    
    return results


if __name__ == "__main__":
    import argparse
    
    # Get available datasets for help text
    available_datasets = list_available_datasets()
    
    parser = argparse.ArgumentParser(
        description="Multi-objective fairness optimization with SMAC"
    )
    parser.add_argument(
        "--dataset", type=str, default="adult",
        choices=available_datasets,
        help=f"Dataset to use (default: adult). Available: {available_datasets}"
    )
    parser.add_argument(
        "--sensitive", type=str, default="sex",
        help="Sensitive feature to use (default: sex). Options depend on dataset."
    )
    parser.add_argument(
        "--walltime", type=int, default=300,
        help="Time limit per model in seconds (default: 300)"
    )
    parser.add_argument(
        "--n-trials", type=int, default=100,
        help="Max configurations per model (default: 100)"
    )
    parser.add_argument(
        "--list-datasets", action="store_true",
        help="List available datasets and their sensitive features"
    )
    
    args = parser.parse_args()
    
    # List datasets mode
    if args.list_datasets:
        print("Available datasets:")
        print("-" * 50)
        for name in available_datasets:
            cfg = get_dataset_config(name)
            print(f"\n{name}:")
            print(f"  Name: {cfg.name}")
            print(f"  OpenML ID: {cfg.openml_id}")
            print(f"  Sensitive features: {list(cfg.sensitive_features.keys())}")
    else:
        results = main(
            dataset_name=args.dataset,
            sensitive_feature=args.sensitive,
            walltime_limit=args.walltime,
            n_trials=args.n_trials,
        )
