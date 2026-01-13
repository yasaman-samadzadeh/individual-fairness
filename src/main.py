"""
Multi-Objective Fairness Optimization with SMAC

Two approaches for individual fairness evaluation:

Approach 1: Standard models WITH sensitive features included.
- Train RF/MLP with all features (including sex, race)
- Evaluate counterfactual consistency by flipping sensitive features directly

Approach 2: Standard models + SenSeI WITHOUT sensitive features.
- Train RF/MLP/SenSeI without sensitive features
- Use sensitive features only for SenSeI's fair distance metric
- Evaluate counterfactual consistency on proxy features (e.g., relationship_Wife)

Models: RandomForest, MLP, SenSeI (PyTorch neural network)
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
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.facade.abstract_facade import AbstractFacade
from smac.multi_objective.parego import ParEGO

from utils.datasets import (
    load_dataset, 
    create_flipped_data,
    create_flipped_data_multiclass_exhaustive,
    counterfactual_consistency_multiclass_exhaustive,
    list_available_datasets, 
    get_dataset_config
)
from utils.individual_fairness import counterfactual_consistency
from utils.plotting import (
    get_pareto_front,
    get_pareto_indices,
    plot_pareto_comparison,
    plot_parallel_coordinates,
    plot_mds_projection,
    print_pareto_summary,
    generate_all_visualizations,
    extract_config_data,
    compute_fairness_confusion_matrix,
    plot_fairness_confusion_matrix,
    print_fairness_confusion_summary,
)

# Check if inFairness (SenSeI) is available
try:
    import torch
    import torch.nn.functional as F
    from inFairness.fairalgo import SenSeI
    from inFairness.distances import LogisticRegSensitiveSubspace, SquaredEuclideanDistance
    SENSEI_AVAILABLE = True
except ImportError:
    SENSEI_AVAILABLE = False
    print("Warning: inFairness not installed. SenSeI model will not be available.")
    print("Install with: pip install inFairness")


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


def get_configspace(model_type: str) -> ConfigurationSpace:
    """
    Get configuration space for a given model type.
    
    Parameters
    ----------
    model_type : str
        One of "rf", "mlp", or "sensei"
    
    Returns
    -------
    ConfigurationSpace
        The configuration space for the model
    """
    if model_type == "rf":
        return get_rf_configspace()
    elif model_type == "mlp":
        return get_mlp_configspace()
    elif model_type == "sensei":
        return get_sensei_configspace()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


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
    
    Supports both binary and multiclass sensitive features:
    - Binary: flip single column (0 <-> 1)
    - Multiclass: exhaustive flip to all other categories, average consistency
    """
    
    def __init__(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        sensitive_col_idx: Optional[int] = None,
        sensitive_col_indices: Optional[List[int]] = None,
        is_multiclass: bool = False,
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
        sensitive_col_idx : int, optional
            Index of sensitive column for binary counterfactual
        sensitive_col_indices : list of int, optional
            Indices of sensitive columns for multiclass counterfactual
        is_multiclass : bool
            If True, use exhaustive multiclass counterfactual
        n_cv_splits : int
            Number of cross-validation splits
        """
        self.model_type = model_type
        self.X = X
        self.y = y
        self.is_multiclass = is_multiclass
        self.n_cv_splits = n_cv_splits
        
        if is_multiclass:
            if sensitive_col_indices is None:
                raise ValueError("sensitive_col_indices required for multiclass")
            self.sensitive_col_indices = sensitive_col_indices
        else:
            if sensitive_col_idx is None:
                raise ValueError("sensitive_col_idx required for binary")
            self.sensitive_col_idx = sensitive_col_idx
        
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
                if self.is_multiclass:
                    # Multiclass: exhaustive flip to all categories
                    flipped_versions, orig_cats = create_flipped_data_multiclass_exhaustive(
                        X_test, self.sensitive_col_indices
                    )
                    # Get predictions for each flipped version
                    y_preds_flipped = [model.predict(X_flip) for X_flip in flipped_versions]
                    consistency = counterfactual_consistency_multiclass_exhaustive(
                        y_pred, y_preds_flipped, orig_cats
                    )
                else:
                    # Binary: simple flip
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
# Approach 2: Pipeline WITHOUT Sensitive Features
# =============================================================================

class FairnessPipelineApproach2:
    """
    Pipeline for Approach 2: Models trained WITHOUT sensitive features.
    
    Counterfactual consistency is evaluated on a proxy feature instead
    of the sensitive feature directly.
    """
    
    def __init__(
        self,
        model_type: str,
        X: np.ndarray,
        y: np.ndarray,
        proxy_col_idx: int,  # Proxy column for counterfactual
        n_cv_splits: int = 5,
    ):
        """
        Parameters
        ----------
        model_type : str
            "rf" for Random Forest, "mlp" for MLP
        X : np.ndarray
            Feature matrix (WITHOUT sensitive features)
        y : np.ndarray
            Target labels
        proxy_col_idx : int
            Index of proxy column for counterfactual (e.g., relationship_Wife)
        n_cv_splits : int
            Number of cross-validation splits
        """
        self.model_type = model_type
        self.X = X
        self.y = y
        self.proxy_col_idx = proxy_col_idx
        self.n_cv_splits = n_cv_splits
        
        # Pre-compute CV splits
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
        Train and evaluate using proxy-based counterfactual consistency.
        """
        accuracy_scores = []
        consistency_scores = []

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            for train_idx, test_idx in self.cv_splits:
                X_train, X_test = self.X[train_idx], self.X[test_idx]
                y_train, y_test = self.y[train_idx], self.y[test_idx]
                
                model = self._create_model(config)
                model.fit(X_train, y_train)
                
                # Evaluate accuracy
                y_pred = model.predict(X_test)
                acc = balanced_accuracy_score(y_test, y_pred)
                accuracy_scores.append(acc)
                
                # Evaluate counterfactual consistency on PROXY column
                X_test_flipped = create_flipped_data(X_test, self.proxy_col_idx)
                y_pred_flipped = model.predict(X_test_flipped)
                consistency = counterfactual_consistency(y_pred, y_pred_flipped)
                consistency_scores.append(consistency)
        
        return {
            "error": 1.0 - np.mean(accuracy_scores),
            "inconsistency": 1.0 - np.mean(consistency_scores),
        }


# =============================================================================
# SenSeI Model (Individual Fairness In-Training)
# =============================================================================

def get_sensei_configspace() -> ConfigurationSpace:
    """Configuration space for SenSeI neural network.
    
    Based on IBM's inFairness example parameters, optimized for faster training.
    
    Note: Reduced ranges compared to IBM defaults to make HPO tractable.
    The auditor_nsteps is the main cost driver - reduced significantly.
    """
    cs = ConfigurationSpace(seed=42)
    
    cs.add_hyperparameters([
        Integer("n_hidden_layers", (1, 2), default=2),       # Reduced from (1,3)
        Integer("n_neurons", (50, 150), log=True, default=100),  # Narrowed range
        Float("learning_rate", (5e-4, 5e-3), log=True, default=1e-3),  # Narrowed
        Float("rho", (2.0, 15.0), log=True, default=5.0),    # Reduced from (1,25)
        Float("eps", (0.05, 0.2), log=True, default=0.1),    # Narrowed
        Integer("auditor_nsteps", (10, 30), default=20),     # REDUCED from (10,100)!
        Float("auditor_lr", (1e-3, 5e-2), log=True, default=1e-2),  # Higher LR
        Integer("batch_size", (64, 256), log=True, default=128),  # Larger batches
        Integer("epochs", (3, 8), default=5),                # REDUCED from (5,20)!
    ])
    
    return cs


class SenSeIPipeline:
    """
    Pipeline for SenSeI (Sensitive Set Invariance) model.
    
    SenSeI is an in-training individual fairness algorithm that learns
    to be invariant to changes in sensitive attributes.
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_protected: np.ndarray,
        proxy_col_idx: int,
        n_cv_splits: int = 5,
    ):
        """
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (WITHOUT sensitive features)
        y : np.ndarray
            Target labels
        X_protected : np.ndarray
            Sensitive features for learning fair distance metric
        proxy_col_idx : int
            Index of proxy column for counterfactual evaluation
        n_cv_splits : int
            Number of CV splits
        """
        if not SENSEI_AVAILABLE:
            raise ImportError(
                "inFairness not installed. Install with: pip install inFairness"
            )
        
        self.X = X
        self.y = y
        self.X_protected = X_protected
        self.proxy_col_idx = proxy_col_idx
        self.n_cv_splits = n_cv_splits
        
        cv = StratifiedKFold(n_splits=n_cv_splits, shuffle=True, random_state=42)
        self.cv_splits = list(cv.split(X, y))
        
        # Device selection: CUDA (NVIDIA) > MPS (Mac) > CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS (Mac GPU) acceleration")
        else:
            self.device = torch.device("cpu")
            print("Warning: No GPU available, using CPU (will be slow)")

    @property
    def configspace(self) -> ConfigurationSpace:
        return get_sensei_configspace()
    
    def train(self, config: Configuration, seed: int = 0) -> Dict[str, float]:
        """Train SenSeI model with cross-validation.
        
        Implementation follows IBM's inFairness example:
        https://github.com/IBM/inFairness/blob/main/examples/adult-income-prediction/
        """
        accuracy_scores = []
        consistency_scores = []
        
        for train_idx, test_idx in self.cv_splits:
            X_train = self.X[train_idx]
            X_test = self.X[test_idx]
            y_train = self.y[train_idx]
            y_test = self.y[test_idx]
            X_prot_train = self.X_protected[train_idx]
            
            # Convert to torch tensors
            X_train_t = torch.FloatTensor(X_train).to(self.device)
            y_train_t = torch.LongTensor(y_train).to(self.device)
            X_test_t = torch.FloatTensor(X_test).to(self.device)
            X_prot_train_t = torch.FloatTensor(X_prot_train).to(self.device)
            
            # Build network architecture (same as IBM: FC layers with ReLU)
            n_features = X_train.shape[1]
            hidden_sizes = [config["n_neurons"]] * config["n_hidden_layers"]
            output_size = 2  # Binary classification
            
            layers = []
            prev_size = n_features
            for h_size in hidden_sizes:
                layers.extend([
                    torch.nn.Linear(prev_size, h_size),
                    torch.nn.ReLU(),
                ])
                prev_size = h_size
            layers.append(torch.nn.Linear(prev_size, output_size))
            
            network = torch.nn.Sequential(*layers).to(self.device)
            
            # ============================================================
            # Distance metrics (following IBM's approach)
            # ============================================================
            
            # Input space distance: LogisticRegSensitiveSubspace
            # This learns to ignore variations in sensitive attributes
            distance_x = LogisticRegSensitiveSubspace()
            distance_x.fit(X_train_t, data_SensitiveAttrs=X_prot_train_t)
            distance_x.to(self.device)
            
            # Output space distance: SquaredEuclideanDistance
            distance_y = SquaredEuclideanDistance()
            distance_y.fit(num_dims=output_size)
            distance_y.to(self.device)
            
            # ============================================================
            # Create SenSeI model (with all required parameters)
            # ============================================================
            sensei = SenSeI(
                network=network,
                distance_x=distance_x,
                distance_y=distance_y,
                loss_fn=F.cross_entropy,
                rho=config["rho"],
                eps=config["eps"],
                auditor_nsteps=config["auditor_nsteps"],
                auditor_lr=config["auditor_lr"],
            )
            
            # ============================================================
            # Training loop (following IBM's approach)
            # ============================================================
            optimizer = torch.optim.Adam(network.parameters(), lr=config["learning_rate"])
            batch_size = config["batch_size"]
            n_epochs = config["epochs"]
            
            sensei.train()  # Set to training mode
            
            for epoch in range(n_epochs):
                # Shuffle training data
                perm = torch.randperm(len(X_train_t))
                
                for i in range(0, len(X_train_t), batch_size):
                    idx = perm[i:i+batch_size]
                    X_batch = X_train_t[idx]
                    y_batch = y_train_t[idx]
                    
                    optimizer.zero_grad()
                    # IBM's approach: result = fairalgo(x, y), result.loss.backward()
                    result = sensei(X_batch, y_batch)
                    result.loss.backward()
                    optimizer.step()
            
            # ============================================================
            # Evaluation
            # ============================================================
            network.eval()
            with torch.no_grad():
                logits = network(X_test_t)
                y_pred = logits.argmax(dim=1).cpu().numpy()
                
                # Counterfactual on proxy (following IBM's spouse_consistency)
                X_test_flipped = create_flipped_data(X_test, self.proxy_col_idx)
                X_test_flipped_t = torch.FloatTensor(X_test_flipped).to(self.device)
                logits_flipped = network(X_test_flipped_t)
                y_pred_flipped = logits_flipped.argmax(dim=1).cpu().numpy()
            
            acc = balanced_accuracy_score(y_test, y_pred)
            consistency = counterfactual_consistency(y_pred, y_pred_flipped)
            
            accuracy_scores.append(acc)
            consistency_scores.append(consistency)
        
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
    approach: int = 1,
) -> AbstractFacade:
    """
    Run SMAC multi-objective optimization for a model.
    
    Parameters
    ----------
    model_type : str
        "rf", "mlp", or "sensei"
    data : dict
        Data dictionary from load_dataset() or load_dataset_approach2()
    walltime_limit : int
        Time limit in seconds
    n_trials : int
        Maximum number of configurations to try
    output_dir : str
        Directory to save SMAC output
    approach : int
        1 = with sensitive features, 2 = without sensitive features

    Returns
    -------
    smac : AbstractFacade
        SMAC optimizer with results
    """
    print(f"\n{'='*60}")
    print(f"Running SMAC optimization for {model_type.upper()} (Approach {approach})")
    print(f"{'='*60}")
    
    # Create appropriate pipeline based on approach
    if approach == 1:
        # Approach 1: With sensitive features (binary or multiclass)
        is_multiclass = data.get('is_multiclass', False)
        
        if is_multiclass:
            print(f"Using MULTICLASS counterfactual (exhaustive, {len(data['sensitive_col_indices'])} categories)")
            pipeline = FairnessPipeline(
                model_type=model_type,
                X=data['X_train'],
                y=data['y_train'],
                sensitive_col_indices=data['sensitive_col_indices'],
                is_multiclass=True,
            )
        else:
            print(f"Using BINARY counterfactual (single column flip)")
            pipeline = FairnessPipeline(
                model_type=model_type,
                X=data['X_train'],
                y=data['y_train'],
                sensitive_col_idx=data['sensitive_col_idx'],
                is_multiclass=False,
            )
    else:
        # Approach 2: Without sensitive features
        if model_type == "sensei":
            if not SENSEI_AVAILABLE:
                raise ImportError("SenSeI requires inFairness. Install with: pip install inFairness")
            pipeline = SenSeIPipeline(
                X=data['X_train'],
                y=data['y_train'],
                X_protected=data['X_protected'],
                proxy_col_idx=data['proxy_col_idx'],
            )
        else:
            pipeline = FairnessPipelineApproach2(
                model_type=model_type,
                X=data['X_train'],
                y=data['y_train'],
                proxy_col_idx=data['proxy_col_idx'],
            )
    
    # Define scenario
    objectives = ["error", "inconsistency"]
    
    scenario = Scenario(
        pipeline.configspace,
        objectives=objectives,
        walltime_limit=walltime_limit,
        n_trials=n_trials,
        n_workers=1,
        output_directory=os.path.join(output_dir, f"{model_type}_approach{approach}"),
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
# Main Entry Points
# =============================================================================

def main(
    dataset_name: str = "adult",
    sensitive_feature: str = "sex",
    walltime_limit: int = 300,
    n_trials: int = 100,
):
    """
    Run Approach 1: Standard models WITH sensitive features.
    
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
            approach=1,
        )
        results[model_type] = smac
    
    # Plot and summarize results
    generate_all_visualizations(results, dataset_name, sensitive_feature)
    print_pareto_summary(results)
    
    return results, data


def main_approach2(
    dataset_name: str = "adult",
    sensitive_features_to_remove: List[str] = None,
    proxy_feature: str = "relationship",
    walltime_limit: int = 300,
    n_trials: int = 100,
    include_sensei: bool = True,
):
    """
    Run Approach 2: Models WITHOUT sensitive features + SenSeI comparison.
    
    Parameters
    ----------
    dataset_name : str
        Name of dataset to use
    sensitive_features_to_remove : list
        Sensitive features to remove from training (default: ["sex", "race"])
    proxy_feature : str
        Proxy feature to use for counterfactual evaluation (default: "relationship")
    walltime_limit : int
        Time limit per model in seconds
    n_trials : int
        Max configurations per model
    include_sensei : bool
        Whether to include SenSeI model (requires inFairness)
    """
    if sensitive_features_to_remove is None:
        sensitive_features_to_remove = ["sex", "race"]
    
    config = get_dataset_config(dataset_name)
    
    print("="*60)
    print("APPROACH 2: Models WITHOUT Sensitive Features")
    print("="*60)
    print(f"Dataset: {config.name} (OpenML ID: {config.openml_id})")
    print(f"Sensitive features REMOVED: {sensitive_features_to_remove}")
    print(f"Proxy feature for counterfactual: {proxy_feature}")
    print(f"Time limit per model: {walltime_limit}s")
    print(f"Max trials per model: {n_trials}")
    
    if include_sensei and not SENSEI_AVAILABLE:
        print("\nWarning: SenSeI not available (inFairness not installed)")
        include_sensei = False
    
    # Load data for Approach 2 (using unified load_dataset with approach=2)
    print(f"\nLoading {config.name} dataset (Approach 2)...")
    data = load_dataset(
        dataset_name,
        approach=2,
        sensitive_features_to_remove=sensitive_features_to_remove,
        proxy_feature=proxy_feature,
    )
    
    # Run optimization for each model
    results = {}
    
    # Standard models without sensitive features
    for model_type in ["rf", "mlp"]:
        smac = run_optimization(
            model_type=model_type,
            data=data,
            walltime_limit=walltime_limit,
            n_trials=n_trials,
            approach=2,
        )
        results[model_type] = smac
    
    # SenSeI (if available)
    if include_sensei:
        smac = run_optimization(
            model_type="sensei",
            data=data,
            walltime_limit=walltime_limit,
            n_trials=n_trials,
            approach=2,
        )
        results["sensei"] = smac
    
    # Plot and summarize results
    generate_all_visualizations(
        results, 
        dataset_name, 
        f"proxy_{proxy_feature}_approach2"
    )
    print_pareto_summary(results)
    
    return results, data


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
        results, data = main(
            dataset_name=args.dataset,
            sensitive_feature=args.sensitive,
            walltime_limit=args.walltime,
            n_trials=args.n_trials,
        )
