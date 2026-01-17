"""
Plotting utilities for fairness optimization experiments.

Supports two output formats:
- 'notebook': Interactive display with titles, suitable for Jupyter notebooks
- 'latex': Publication-ready without titles, suitable for LaTeX reports

Directory structure:
- plots/notebook/  - Plots with titles for interactive viewing
- plots/latex/     - Clean plots without titles for publications
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Literal, Tuple, List
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler

from smac.facade.abstract_facade import AbstractFacade


# =============================================================================
# Constants and Type Definitions
# =============================================================================

PlotFormat = Literal["notebook", "latex"]

# Default settings for each format
FORMAT_SETTINGS = {
    "notebook": {
        "dpi": 150,
        "show_title": True,
        "show_suptitle": True,
        "figsize_scale": 1.0,
        "font_scale": 1.0,
        "subdir": "notebook",
    },
    "latex": {
        "dpi": 300,
        "show_title": False,
        "show_suptitle": False,
        "figsize_scale": 0.9,  # Slightly smaller for papers
        "font_scale": 1.1,     # Larger fonts for readability
        "subdir": "latex",
    },
}

# Color schemes
MODEL_COLORS = {
    'rf': '#1f77b4',      # Blue
    'mlp': '#ff7f0e',     # Orange
    'sensei': '#2ca02c',  # Green
}

MODEL_MARKERS = {
    'rf': 'o',
    'mlp': 's', 
    'sensei': '^',
}

MODEL_LABELS = {
    'rf': 'Random Forest',
    'mlp': 'MLP',
    'sensei': 'SenSeI',
}


# =============================================================================
# Helper Functions
# =============================================================================

def _get_light_color(hex_color: str, factor: float = 0.3) -> tuple:
    """Create a lighter version of a color by blending with white."""
    rgb = mcolors.to_rgb(hex_color)
    light_rgb = tuple(1 - factor * (1 - c) for c in rgb)
    return light_rgb


def _ensure_output_dirs(base_dir: str) -> Tuple[str, str]:
    """Create and return paths for notebook and latex subdirectories."""
    notebook_dir = os.path.join(base_dir, "notebook")
    latex_dir = os.path.join(base_dir, "latex")
    os.makedirs(notebook_dir, exist_ok=True)
    os.makedirs(latex_dir, exist_ok=True)
    return notebook_dir, latex_dir


def _get_output_path(base_dir: str, filename: str, format: PlotFormat) -> str:
    """Get the appropriate output path based on format."""
    subdir = FORMAT_SETTINGS[format]["subdir"]
    output_dir = os.path.join(base_dir, subdir)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, filename)


def get_pareto_front(smac: AbstractFacade) -> Tuple[List, np.ndarray]:
    """
    Extract Pareto-optimal configurations from SMAC results.
    
    Returns configurations sorted by first objective (ascending cost).
    
    Returns
    -------
    pareto_configs : list
        List of Pareto-optimal Configuration objects
    pareto_costs : np.ndarray
        Costs for Pareto-optimal configs (shape: n_pareto x 2)
    """
    configs = smac.runhistory.get_configs()
    
    # Get average costs for each configuration
    costs = []
    for config in configs:
        cost = smac.runhistory.average_cost(config)
        costs.append(cost)
    
    costs = np.array(costs)
    
    # Find Pareto front (non-dominated solutions)
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
    pareto_configs = [pareto_configs[i] for i in sort_idx]
    pareto_costs = pareto_costs[sort_idx]
    
    return pareto_configs, pareto_costs


def get_pareto_indices(smac: AbstractFacade) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract Pareto-optimal indices from SMAC runhistory.
    
    Returns
    -------
    pareto_indices : np.ndarray
        Indices of Pareto-optimal configurations in the config list
    all_costs : np.ndarray
        Costs for all configurations (shape: n_configs x 2)
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
            is_pareto[is_pareto] = np.any(costs[is_pareto] < c, axis=1) | np.all(costs[is_pareto] == c, axis=1)
            is_pareto[i] = True
    
    pareto_indices = np.where(is_pareto)[0]
    
    return pareto_indices, costs


def extract_config_data(
    smac: AbstractFacade, 
    model_type: str,
    get_configspace_fn: callable,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract hyperparameter values and costs from SMAC runhistory.
    
    Parameters
    ----------
    smac : AbstractFacade
        SMAC optimizer with results
    model_type : str
        "rf" or "mlp"
    get_configspace_fn : callable
        Function to get configspace for the model type
    
    Returns
    -------
    hp_names : list
        Hyperparameter names
    hp_values : np.ndarray
        Hyperparameter values for each config (n_configs x n_hyperparams)
    costs : np.ndarray
        Costs for each config (n_configs x 2)
    pareto_mask : np.ndarray
        Boolean mask indicating Pareto-optimal configs
    """
    configs = smac.runhistory.get_configs()
    
    # Get hyperparameter names from first config
    hp_names = list(configs[0].keys())
    
    # Get configspace
    cs = get_configspace_fn(model_type)
    
    # Extract hyperparameter values
    hp_values = []
    for config in configs:
        values = []
        for hp_name in hp_names:
            val = config[hp_name]
            # Convert categorical to numeric
            if isinstance(val, str):
                hp = cs.get_hyperparameter(hp_name)
                if hasattr(hp, 'choices'):
                    choices = list(hp.choices)
                    val = choices.index(val) if val in choices else 0
            elif val is None:
                val = 0
            values.append(float(val))
        hp_values.append(values)
    
    hp_values = np.array(hp_values)
    
    # Get costs
    costs = np.array([smac.runhistory.average_cost(config) for config in configs])
    
    # Get Pareto mask
    pareto_indices, _ = get_pareto_indices(smac)
    pareto_mask = np.zeros(len(configs), dtype=bool)
    pareto_mask[pareto_indices] = True
    
    return hp_names, hp_values, costs, pareto_mask


# =============================================================================
# Main Plotting Functions
# =============================================================================

def plot_pareto_comparison(
    results: Dict[str, AbstractFacade],
    output_dir: str = "plots",
    filename: str = "pareto_comparison.png",
    formats: List[PlotFormat] = ["notebook", "latex"],
) -> Dict[str, str]:
    """
    Plot Pareto fronts for multiple models.
    
    Creates a figure with:
    - Top row: Individual subplots for each model
    - Bottom row: Combined plot with all models overlaid
    
    Parameters
    ----------
    results : dict
        Dictionary mapping model_type -> smac optimizer
    output_dir : str
        Base directory to save plots
    filename : str
        Filename for the plot (without path)
    formats : list
        List of formats to generate ("notebook", "latex", or both)
    
    Returns
    -------
    saved_paths : dict
        Dictionary mapping format -> saved file path
    """
    _ensure_output_dirs(output_dir)
    saved_paths = {}
    
    n_models = len(results)
    model_names = list(results.keys())
    
    for format in formats:
        settings = FORMAT_SETTINGS[format]
        
        # Create figure
        figsize = (5 * n_models * settings["figsize_scale"], 
                   10 * settings["figsize_scale"])
        fig = plt.figure(figsize=figsize)
        
        # Create grid
        gs = fig.add_gridspec(2, n_models, height_ratios=[1, 1.2], hspace=0.25, wspace=0.3)
        individual_axes = [fig.add_subplot(gs[0, i]) for i in range(n_models)]
        combined_ax = fig.add_subplot(gs[1, :])
        
        def plot_model_data(ax, model_type, smac, show_legend=True, title=None):
            """Plot data for a single model on the given axis."""
            color = MODEL_COLORS.get(model_type, '#7f7f7f')
            light_color = _get_light_color(color)
            marker = MODEL_MARKERS.get(model_type, 'o')
            
            # Get Pareto data
            pareto_indices, all_costs = get_pareto_indices(smac)
            non_pareto_indices = np.setdiff1d(np.arange(len(all_costs)), pareto_indices)
            pareto_configs, pareto_costs = get_pareto_front(smac)
            
            # Plot non-Pareto points
            if len(non_pareto_indices) > 0:
                non_pareto_costs = all_costs[non_pareto_indices]
                ax.scatter(
                    1 - non_pareto_costs[:, 0], 1 - non_pareto_costs[:, 1],
                    c=[light_color], marker=marker,
                    alpha=0.6, s=50, edgecolors=color, linewidths=0.8,
                    label=f'{model_type.upper()} (dominated)' if show_legend else None,
                    zorder=1
                )
            
            # Plot Pareto front
            if len(pareto_costs) > 0:
                pareto_accuracy = 1 - pareto_costs[:, 0]
                pareto_consistency = 1 - pareto_costs[:, 1]
                
                sort_idx = np.argsort(pareto_accuracy)
                ax.plot(pareto_accuracy[sort_idx], pareto_consistency[sort_idx], 
                        c=color, linestyle='--', linewidth=2.5, alpha=0.9, zorder=2)
                ax.scatter(
                    pareto_accuracy, pareto_consistency,
                    c=color, marker=marker,
                    s=130, edgecolors='black', linewidths=2,
                    label=f'{model_type.upper()} (Pareto Front)' if show_legend else None,
                    zorder=3
                )
            
            font_scale = settings["font_scale"]
            ax.set_xlabel('Balanced Accuracy', fontsize=11 * font_scale)
            ax.set_ylabel('Counterfactual Consistency', fontsize=11 * font_scale)
            ax.grid(True, alpha=0.3)
            
            if title and settings["show_title"]:
                ax.set_title(title, fontsize=12 * font_scale, fontweight='bold')
            
            if show_legend:
                ax.legend(loc='lower left', fontsize=9 * font_scale)
        
        # Plot individual models
        for i, model_type in enumerate(model_names):
            plot_model_data(
                individual_axes[i], model_type, results[model_type],
                show_legend=True,
                title=f'{model_type.upper()} Pareto Front'
            )
        
        # Plot combined
        for model_type, smac in results.items():
            plot_model_data(combined_ax, model_type, smac, show_legend=True)
        
        if settings["show_title"]:
            combined_ax.set_title('Combined Pareto Front Comparison', 
                                  fontsize=14 * settings["font_scale"], fontweight='bold')
        combined_ax.legend(loc='lower left', fontsize=10 * settings["font_scale"], ncol=2)
        
        # Save
        output_path = _get_output_path(output_dir, filename, format)
        plt.savefig(output_path, dpi=settings["dpi"], bbox_inches='tight')
        plt.close()
        
        saved_paths[format] = output_path
        print(f"[{format}] Pareto plot saved to: {output_path}")
    
    return saved_paths


def plot_parallel_coordinates(
    smac: AbstractFacade,
    model_type: str,
    get_configspace_fn: callable,
    output_dir: str = "plots",
    filename: str = None,
    color_by: str = "error",
    formats: List[PlotFormat] = ["notebook", "latex"],
) -> Dict[str, str]:
    """
    Create a parallel coordinate plot of hyperparameters and objectives.
    
    Parameters
    ----------
    smac : AbstractFacade
        SMAC optimizer with results
    model_type : str
        "rf" or "mlp"
    get_configspace_fn : callable
        Function to get configspace for the model type
    output_dir : str
        Base directory to save plots
    filename : str
        Filename (defaults to parallel_coords_{model_type}.png)
    color_by : str
        What to color lines by: "error", "inconsistency", or "pareto"
    formats : list
        List of formats to generate
    
    Returns
    -------
    saved_paths : dict
        Dictionary mapping format -> saved file path
    """
    if filename is None:
        filename = f"parallel_coords_{model_type}.png"
    
    _ensure_output_dirs(output_dir)
    saved_paths = {}
    
    # Extract data
    hp_names, hp_values, costs, pareto_mask = extract_config_data(
        smac, model_type, get_configspace_fn
    )
    
    # Get configspace for categorical info
    cs = get_configspace_fn(model_type)
    categorical_info = {}
    for hp_name in hp_names:
        hp = cs.get_hyperparameter(hp_name)
        if hasattr(hp, 'choices'):
            categorical_info[hp_name] = list(hp.choices)
        else:
            categorical_info[hp_name] = None
    
    # Normalize values
    scaler = MinMaxScaler()
    hp_values_norm = scaler.fit_transform(hp_values)
    
    # Add objectives
    accuracy = 1 - costs[:, 0]
    consistency = 1 - costs[:, 1]
    acc_norm = (accuracy - accuracy.min()) / (accuracy.max() - accuracy.min() + 1e-10)
    cons_norm = (consistency - consistency.min()) / (consistency.max() - consistency.min() + 1e-10)
    
    all_values = np.column_stack([acc_norm, cons_norm, hp_values_norm])
    all_names = ["Accuracy", "Consistency"] + hp_names
    
    # Color setup
    if color_by == "error":
        color_values = costs[:, 0]
        cmap = cm.RdYlGn_r
        cbar_label = "Error (1 - Accuracy)"
    elif color_by == "inconsistency":
        color_values = costs[:, 1]
        cmap = cm.RdYlGn_r
        cbar_label = "Inconsistency (1 - Consistency)"
    else:
        color_values = pareto_mask.astype(float)
        cmap = cm.coolwarm
        cbar_label = "Pareto Optimal"
    
    norm = Normalize(vmin=color_values.min(), vmax=color_values.max())
    
    for format in formats:
        settings = FORMAT_SETTINGS[format]
        
        figsize = (16 * settings["figsize_scale"], 7 * settings["figsize_scale"])
        fig, ax = plt.subplots(figsize=figsize)
        
        x_positions = np.arange(len(all_names))
        sort_idx = np.argsort(color_values)[::-1]
        
        for idx in sort_idx:
            color = cmap(norm(color_values[idx]))
            alpha = 0.8 if pareto_mask[idx] else 0.3
            linewidth = 2.5 if pareto_mask[idx] else 1.0
            zorder = 10 if pareto_mask[idx] else 1
            ax.plot(x_positions, all_values[idx], 
                    color=color, alpha=alpha, linewidth=linewidth, zorder=zorder)
        
        # Draw axes and labels
        font_scale = settings["font_scale"]
        for i, name in enumerate(all_names):
            ax.axvline(x=i, color='black', linewidth=1.5, alpha=0.7)
            
            if name in categorical_info and categorical_info[name] is not None:
                categories = categorical_info[name]
                n_categories = len(categories)
                
                for j, cat_name in enumerate(categories):
                    y_pos = j / (n_categories - 1) if n_categories > 1 else 0.5
                    ax.annotate(
                        str(cat_name),
                        xy=(i + 0.08, y_pos),
                        fontsize=9 * font_scale,
                        fontweight='semibold',
                        color='#1a3d5c',
                        alpha=0.95,
                        va='center',
                        ha='left',
                        bbox=dict(
                            boxstyle='round,pad=0.2',
                            facecolor='white',
                            edgecolor='#cccccc',
                            alpha=0.85,
                            linewidth=0.5
                        )
                    )
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(all_names, rotation=45, ha='right', fontsize=10 * font_scale)
        ax.set_ylabel('Normalized Value', fontsize=12 * font_scale)
        
        if settings["show_title"]:
            ax.set_title(f'Parallel Coordinate Plot - {model_type.upper()}\n'
                        f'(Bold lines = Pareto optimal, Faint = Dominated)', 
                        fontsize=14 * font_scale, fontweight='bold')
        
        # Colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label(cbar_label, fontsize=11 * font_scale)
        
        ax.set_ylim(-0.08, 1.08)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(['Min', 'Mid', 'Max'])
        
        plt.tight_layout()
        output_path = _get_output_path(output_dir, filename, format)
        plt.savefig(output_path, dpi=settings["dpi"], bbox_inches='tight')
        plt.close()
        
        saved_paths[format] = output_path
        print(f"[{format}] Parallel coords saved to: {output_path}")
    
    return saved_paths


def plot_mds_projection(
    smac: AbstractFacade,
    model_type: str,
    get_configspace_fn: callable,
    output_dir: str = "plots",
    filename: str = None,
    formats: List[PlotFormat] = ["notebook", "latex"],
) -> Dict[str, str]:
    """
    Create MDS projection of hyperparameter configurations to 2D.
    
    Parameters
    ----------
    smac : AbstractFacade
        SMAC optimizer with results
    model_type : str
        "rf" or "mlp"
    get_configspace_fn : callable
        Function to get configspace for the model type
    output_dir : str
        Base directory to save plots
    filename : str
        Filename (defaults to mds_projection_{model_type}.png)
    formats : list
        List of formats to generate
    
    Returns
    -------
    saved_paths : dict
        Dictionary mapping format -> saved file path
    """
    from scipy.interpolate import RBFInterpolator
    
    if filename is None:
        filename = f"mds_projection_{model_type}.png"
    
    _ensure_output_dirs(output_dir)
    saved_paths = {}
    
    # Extract data
    hp_names, hp_values, costs, pareto_mask = extract_config_data(
        smac, model_type, get_configspace_fn
    )
    
    # Normalize and apply MDS
    scaler = MinMaxScaler()
    hp_values_norm = scaler.fit_transform(hp_values)
    
    n_configs = len(hp_values_norm)
    if n_configs < 3:
        print(f"Warning: Not enough configurations ({n_configs}) for MDS projection")
        return saved_paths
    
    mds = MDS(n_components=2, random_state=42, dissimilarity='euclidean', normalized_stress='auto')
    coords_2d = mds.fit_transform(hp_values_norm)
    
    # Grid for interpolation
    margin = 0.1
    x_min, x_max = coords_2d[:, 0].min() - margin, coords_2d[:, 0].max() + margin
    y_min, y_max = coords_2d[:, 1].min() - margin, coords_2d[:, 1].max() + margin
    
    grid_resolution = 100
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution)
    )
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    
    objectives = [
        ("Error", costs[:, 0], cm.YlOrRd),
        ("Inconsistency", costs[:, 1], cm.YlOrRd),
    ]
    
    for format in formats:
        settings = FORMAT_SETTINGS[format]
        
        figsize = (14 * settings["figsize_scale"], 6 * settings["figsize_scale"])
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        font_scale = settings["font_scale"]
        
        for ax, (obj_name, obj_values, cmap) in zip(axes, objectives):
            # Interpolate background
            try:
                rbf = RBFInterpolator(coords_2d, obj_values, kernel='thin_plate_spline', smoothing=0.1)
                zz = rbf(grid_points).reshape(xx.shape)
                zz = np.clip(zz, obj_values.min(), obj_values.max())
                
                im = ax.pcolormesh(xx, yy, zz, cmap=cmap, alpha=0.7, shading='auto')
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label(f'Predicted {obj_name}', fontsize=10 * font_scale)
            except Exception as e:
                print(f"Warning: Could not create background interpolation: {e}")
            
            # Plot points
            non_pareto = ~pareto_mask
            if non_pareto.sum() > 0:
                ax.scatter(
                    coords_2d[non_pareto, 0], coords_2d[non_pareto, 1],
                    c='white', s=60, alpha=0.9, edgecolors='gray', linewidths=1,
                    marker='o', label='Non-Pareto', zorder=5
                )
            
            if pareto_mask.sum() > 0:
                ax.scatter(
                    coords_2d[pareto_mask, 0], coords_2d[pareto_mask, 1],
                    c='red', s=120, edgecolors='black', linewidths=2,
                    marker='s', label='Pareto Optimal', zorder=10
                )
            
            ax.set_xlabel('MDS-X', fontsize=11 * font_scale)
            ax.set_ylabel('MDS-Y', fontsize=11 * font_scale)
            
            if settings["show_title"]:
                ax.set_title(f'{model_type.upper()} - {obj_name}', 
                            fontsize=12 * font_scale, fontweight='bold')
            
            ax.legend(loc='upper right', fontsize=9 * font_scale)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        
        if settings["show_suptitle"]:
            plt.suptitle(f'MDS Projection of Hyperparameter Space - {model_type.upper()}\n'
                        f'(Background = Interpolated Performance, Red Squares = Pareto Optimal)', 
                        fontsize=13 * font_scale, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        output_path = _get_output_path(output_dir, filename, format)
        plt.savefig(output_path, dpi=settings["dpi"], bbox_inches='tight')
        plt.close()
        
        saved_paths[format] = output_path
        print(f"[{format}] MDS projection saved to: {output_path}")
    
    return saved_paths


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


def generate_all_visualizations(
    results: Dict[str, AbstractFacade],
    get_configspace_fn: callable,
    dataset_name: str,
    sensitive_feature: str,
    output_dir: str = "plots",
    formats: List[PlotFormat] = ["notebook", "latex"],
) -> Dict[str, Dict[str, str]]:
    """
    Generate all visualizations for the experiment results.
    
    Parameters
    ----------
    results : dict
        Dictionary mapping model_type -> smac optimizer
    get_configspace_fn : callable
        Function to get configspace for model types
    dataset_name : str
        Name of the dataset
    sensitive_feature : str
        Name of the sensitive feature
    output_dir : str
        Base directory to save plots
    formats : list
        List of formats to generate
    
    Returns
    -------
    all_paths : dict
        Nested dict: {plot_type: {format: path}}
    """
    all_paths = {}
    prefix = f"{dataset_name}_{sensitive_feature}"
    
    # 1. Pareto comparison
    all_paths['pareto'] = plot_pareto_comparison(
        results, 
        output_dir=output_dir,
        filename=f"pareto_{prefix}.png",
        formats=formats
    )
    
    # 2. Parallel coordinates for each model
    for model_type, smac in results.items():
        key = f"parallel_coords_{model_type}"
        all_paths[key] = plot_parallel_coordinates(
            smac, model_type, get_configspace_fn,
            output_dir=output_dir,
            filename=f"parallel_coords_{prefix}_{model_type}.png",
            color_by="error",
            formats=formats
        )
    
    # 3. MDS projections for each model
    for model_type, smac in results.items():
        key = f"mds_{model_type}"
        all_paths[key] = plot_mds_projection(
            smac, model_type, get_configspace_fn,
            output_dir=output_dir,
            filename=f"mds_projection_{prefix}_{model_type}.png",
            formats=formats
        )
    
    print(f"\nAll visualizations saved to: {output_dir}/")
    print(f"  - Notebook versions: {output_dir}/notebook/")
    print(f"  - LaTeX versions: {output_dir}/latex/")
    
    return all_paths


# =============================================================================
# Fairness-Accuracy Confusion Matrix Analysis
# =============================================================================

def compute_fairness_confusion_matrix(
    model,
    X: np.ndarray,
    y: np.ndarray,
    sensitive_col_idx: int = None,
    sensitive_col_indices: List[int] = None,
    is_multiclass: bool = False,
) -> Dict:
    """
    Compute the fairness-accuracy confusion matrix.
    
    Creates a 2x2 matrix analyzing the relationship between:
    - Correct vs Incorrect predictions
    - Consistent vs Inconsistent predictions (under counterfactual)
    
    Parameters
    ----------
    model : sklearn estimator
        Trained model with predict() method
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        True labels
    sensitive_col_idx : int, optional
        Index of sensitive column for binary counterfactual
    sensitive_col_indices : list of int, optional
        Indices of sensitive columns for multiclass counterfactual
    is_multiclass : bool
        If True, use exhaustive multiclass counterfactual
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'matrix': 2x2 numpy array [[correct_consistent, correct_inconsistent],
                                     [wrong_consistent, wrong_inconsistent]]
        - 'counts': dict with named counts
        - 'percentages': dict with named percentages
        - 'metrics': dict with derived metrics
    """
    # Import here to avoid circular imports
    from datasets import (
        create_flipped_data,
        create_flipped_data_multiclass_exhaustive,
    )
    
    # Get predictions on original data
    y_pred = model.predict(X)
    
    # Determine correctness
    is_correct = (y_pred == y)
    
    # Determine consistency based on counterfactual type
    if is_multiclass:
        if sensitive_col_indices is None:
            raise ValueError("sensitive_col_indices required for multiclass")
        
        # For multiclass: check if prediction is consistent across ALL flips
        # Get all flipped versions and original categories
        flipped_versions, original_categories = create_flipped_data_multiclass_exhaustive(
            X, sensitive_col_indices
        )
        
        n_categories = len(sensitive_col_indices)
        all_consistent = np.ones(len(X), dtype=bool)
        
        # For each target category, check consistency
        # Skip when target == original (that's not a flip)
        for target_cat in range(n_categories):
            X_flipped = flipped_versions[target_cat]
            y_pred_flipped = model.predict(X_flipped)
            
            # Only check consistency for samples that are actually flipped
            # (i.e., original category != target category)
            is_actual_flip = (original_categories != target_cat)
            
            # For samples that were flipped, check if prediction changed
            flipped_consistent = (y_pred == y_pred_flipped)
            
            # Update all_consistent: if this was an actual flip and prediction changed, mark as inconsistent
            all_consistent &= (~is_actual_flip | flipped_consistent)
        
        is_consistent = all_consistent
    else:
        if sensitive_col_idx is None:
            raise ValueError("sensitive_col_idx required for binary")
        
        # For binary: simple flip
        X_flipped = create_flipped_data(X, sensitive_col_idx)
        y_pred_flipped = model.predict(X_flipped)
        is_consistent = (y_pred == y_pred_flipped)
    
    # Compute the 2x2 matrix
    correct_consistent = np.sum(is_correct & is_consistent)
    correct_inconsistent = np.sum(is_correct & ~is_consistent)
    wrong_consistent = np.sum(~is_correct & is_consistent)
    wrong_inconsistent = np.sum(~is_correct & ~is_consistent)
    
    total = len(y)
    
    matrix = np.array([
        [correct_consistent, correct_inconsistent],
        [wrong_consistent, wrong_inconsistent]
    ])
    
    counts = {
        'correct_consistent': int(correct_consistent),
        'correct_inconsistent': int(correct_inconsistent),
        'wrong_consistent': int(wrong_consistent),
        'wrong_inconsistent': int(wrong_inconsistent),
        'total': int(total),
        'total_correct': int(np.sum(is_correct)),
        'total_wrong': int(np.sum(~is_correct)),
        'total_consistent': int(np.sum(is_consistent)),
        'total_inconsistent': int(np.sum(~is_consistent)),
    }
    
    percentages = {k: 100 * v / total for k, v in counts.items() if k != 'total'}
    percentages['total'] = 100.0
    
    # Derived metrics
    metrics = {
        'accuracy': counts['total_correct'] / total,
        'consistency': counts['total_consistent'] / total,
        'p_consistent_given_correct': (
            correct_consistent / counts['total_correct'] 
            if counts['total_correct'] > 0 else 0
        ),
        'p_consistent_given_wrong': (
            wrong_consistent / counts['total_wrong']
            if counts['total_wrong'] > 0 else 0
        ),
        'p_correct_given_consistent': (
            correct_consistent / counts['total_consistent']
            if counts['total_consistent'] > 0 else 0
        ),
        'p_correct_given_inconsistent': (
            correct_inconsistent / counts['total_inconsistent']
            if counts['total_inconsistent'] > 0 else 0
        ),
    }
    
    return {
        'matrix': matrix,
        'counts': counts,
        'percentages': percentages,
        'metrics': metrics,
        'is_correct': is_correct,
        'is_consistent': is_consistent,
    }


def plot_fairness_confusion_matrix(
    results: Dict,
    model_name: str = "Model",
    output_dir: str = "plots",
    filename: str = None,
    formats: List[PlotFormat] = ["notebook", "latex"],
) -> Dict[str, str]:
    """
    Plot the fairness-accuracy confusion matrix as a heatmap.
    
    Parameters
    ----------
    results : dict
        Output from compute_fairness_confusion_matrix()
    model_name : str
        Name of the model for the title
    output_dir : str
        Base directory to save plots
    filename : str
        Filename (defaults to fairness_confusion_{model_name}.png)
    formats : list
        List of formats to generate
    
    Returns
    -------
    saved_paths : dict
        Dictionary mapping format -> saved file path
    """
    if filename is None:
        filename = f"fairness_confusion_{model_name.lower().replace(' ', '_')}.png"
    
    _ensure_output_dirs(output_dir)
    saved_paths = {}
    
    matrix = results['matrix']
    counts = results['counts']
    metrics = results['metrics']
    
    for format in formats:
        settings = FORMAT_SETTINGS[format]
        font_scale = settings["font_scale"]
        
        fig, ax = plt.subplots(figsize=(8 * settings["figsize_scale"], 
                                         6 * settings["figsize_scale"]))
        
        # Create heatmap
        im = ax.imshow(matrix, cmap='Blues', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Count', fontsize=11 * font_scale)
        
        # Labels
        row_labels = ['Correct\nPrediction', 'Wrong\nPrediction']
        col_labels = ['Consistent', 'Inconsistent']
        
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(col_labels, fontsize=12 * font_scale)
        ax.set_yticklabels(row_labels, fontsize=12 * font_scale)
        
        # Add text annotations
        total = counts['total']
        annotations = [
            [f"{matrix[0,0]}\n({100*matrix[0,0]/total:.1f}%)\n✅ Ideal",
             f"{matrix[0,1]}\n({100*matrix[0,1]/total:.1f}%)\n⚠️ Unfair"],
            [f"{matrix[1,0]}\n({100*matrix[1,0]/total:.1f}%)\n🔸 Fair Error",
             f"{matrix[1,1]}\n({100*matrix[1,1]/total:.1f}%)\n❌ Worst"],
        ]
        
        for i in range(2):
            for j in range(2):
                text_color = 'white' if matrix[i, j] > matrix.max() / 2 else 'black'
                ax.text(j, i, annotations[i][j],
                       ha='center', va='center',
                       fontsize=11 * font_scale,
                       color=text_color,
                       fontweight='bold')
        
        ax.set_xlabel('Counterfactual Consistency', fontsize=13 * font_scale)
        ax.set_ylabel('Prediction Correctness', fontsize=13 * font_scale)
        
        if settings["show_title"]:
            ax.set_title(f'Fairness-Accuracy Confusion Matrix - {model_name}\n'
                        f'Accuracy: {metrics["accuracy"]:.1%} | '
                        f'Consistency: {metrics["consistency"]:.1%}',
                        fontsize=14 * font_scale, fontweight='bold')
        
        plt.tight_layout()
        output_path = _get_output_path(output_dir, filename, format)
        plt.savefig(output_path, dpi=settings["dpi"], bbox_inches='tight')
        plt.close()
        
        saved_paths[format] = output_path
        print(f"[{format}] Fairness confusion matrix saved to: {output_path}")
    
    return saved_paths


def print_fairness_confusion_summary(results: Dict, model_name: str = "Model"):
    """
    Print a detailed summary of the fairness confusion matrix.
    
    Parameters
    ----------
    results : dict
        Output from compute_fairness_confusion_matrix()
    model_name : str
        Name of the model
    """
    counts = results['counts']
    percentages = results['percentages']
    metrics = results['metrics']
    
    print(f"\n{'='*70}")
    print(f"FAIRNESS-ACCURACY CONFUSION MATRIX: {model_name}")
    print(f"{'='*70}")
    
    print(f"\n📊 Matrix (Counts):")
    print(f"{'':20} {'Consistent':>15} {'Inconsistent':>15} {'Total':>10}")
    print(f"{'-'*60}")
    print(f"{'Correct Prediction':20} {counts['correct_consistent']:>15} "
          f"{counts['correct_inconsistent']:>15} {counts['total_correct']:>10}")
    print(f"{'Wrong Prediction':20} {counts['wrong_consistent']:>15} "
          f"{counts['wrong_inconsistent']:>15} {counts['total_wrong']:>10}")
    print(f"{'-'*60}")
    print(f"{'Total':20} {counts['total_consistent']:>15} "
          f"{counts['total_inconsistent']:>15} {counts['total']:>10}")
    
    print(f"\n📈 Key Metrics:")
    print(f"  • Accuracy:    {metrics['accuracy']:.2%}")
    print(f"  • Consistency: {metrics['consistency']:.2%}")
    
    print(f"\n🔍 Conditional Probabilities:")
    print(f"  • P(Consistent | Correct):   {metrics['p_consistent_given_correct']:.2%}")
    print(f"  • P(Consistent | Wrong):     {metrics['p_consistent_given_wrong']:.2%}")
    print(f"  • P(Correct | Consistent):   {metrics['p_correct_given_consistent']:.2%}")
    print(f"  • P(Correct | Inconsistent): {metrics['p_correct_given_inconsistent']:.2%}")
    
    print(f"\n💡 Interpretation:")
    
    # Check if unfairness correlates with errors
    if metrics['p_consistent_given_correct'] > metrics['p_consistent_given_wrong']:
        diff = metrics['p_consistent_given_correct'] - metrics['p_consistent_given_wrong']
        print(f"  → Correct predictions are {diff:.1%} more likely to be consistent")
        print(f"    (Fairness and accuracy are positively correlated)")
    else:
        diff = metrics['p_consistent_given_wrong'] - metrics['p_consistent_given_correct']
        print(f"  → Wrong predictions are {diff:.1%} more likely to be consistent")
        print(f"    (Potential trade-off between fairness and accuracy)")
    
    # Check the "right for wrong reasons" proportion
    if counts['total_correct'] > 0:
        unfair_correct_rate = counts['correct_inconsistent'] / counts['total_correct']
        print(f"  → {unfair_correct_rate:.1%} of correct predictions are 'right but unfair'")
    
    print(f"{'='*70}\n")


# =============================================================================
# Notebook Display Helpers
# =============================================================================

def plot_sensitive_distribution(data, dataset_name, sensitive_feature):
    """
    Plot the distribution of sensitive feature(s) vs target variable with disparity analysis.
    
    Parameters
    ----------
    data : dict
        Data dictionary containing X_train, y_train, and sensitive feature info
    dataset_name : str
        Name of the dataset (e.g., 'adult')
    sensitive_feature : str
        Name of the sensitive feature (e.g., 'sex', 'race')
    """
    X_train = data['X_train']
    y_train = data['y_train']
    
    # Determine if binary or multiclass based on which keys exist
    is_multiclass = 'sensitive_col_indices' in data
    
    # Get target label names (customize based on dataset)
    target_names = {0: 'Income ≤50K', 1: 'Income >50K'} if dataset_name == 'adult' else {0: 'Class 0', 1: 'Class 1'}
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if is_multiclass:
        # Multiclass sensitive feature (e.g., race)
        sens_indices = data['sensitive_col_indices']
        sens_names = data.get('sensitive_col_names', [f'Cat_{i}' for i in range(len(sens_indices))])
        n_categories = len(sens_names)
        
        # Determine which category each sample belongs to
        categories = np.argmax(X_train[:, sens_indices], axis=1)
        
        # Count samples per category and target
        category_target_counts = {}
        for cat_idx, cat_name in enumerate(sens_names):
            mask = (categories == cat_idx)
            pos_count = (y_train[mask] == 1).sum()
            neg_count = (y_train[mask] == 0).sum()
            category_target_counts[cat_name] = {'positive': pos_count, 'negative': neg_count, 'total': mask.sum()}
        
        # Plot 1: Stacked bar chart (absolute counts)
        cat_names = list(category_target_counts.keys())
        pos_counts = [category_target_counts[c]['positive'] for c in cat_names]
        neg_counts = [category_target_counts[c]['negative'] for c in cat_names]
        
        x = np.arange(len(cat_names))
        width = 0.6
        
        axes[0].bar(x, neg_counts, width, label=target_names[0], color='#3498db', alpha=0.8)
        axes[0].bar(x, pos_counts, width, bottom=neg_counts, label=target_names[1], color='#e74c3c', alpha=0.8)
        
        axes[0].set_xlabel(f'Sensitive Feature: {sensitive_feature.title()}', fontsize=11)
        axes[0].set_ylabel('Count', fontsize=11)
        axes[0].set_title('Distribution by Sensitive Group (Absolute)', fontsize=12, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(cat_names, rotation=45, ha='right')
        axes[0].legend()
        
        # Add count labels
        for i, (neg, pos) in enumerate(zip(neg_counts, pos_counts)):
            axes[0].annotate(f'{neg+pos:,}', (i, neg+pos+50), ha='center', fontsize=9)
        
        # Plot 2: Base rate (percentage positive) per group
        base_rates = [100 * category_target_counts[c]['positive'] / category_target_counts[c]['total'] 
                      for c in cat_names]
        
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(cat_names)))
        axes[1].bar(x, base_rates, width, color=colors, edgecolor='black', linewidth=0.5)
        
        axes[1].axhline(y=100*y_train.mean(), color='black', linestyle='--', linewidth=2, 
                        label=f'Overall rate: {100*y_train.mean():.1f}%')
        axes[1].set_xlabel(f'Sensitive Feature: {sensitive_feature.title()}', fontsize=11)
        axes[1].set_ylabel(f'% {target_names[1]}', fontsize=11)
        axes[1].set_title('Base Rate Disparity (% Positive Outcome)', fontsize=12, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(cat_names, rotation=45, ha='right')
        axes[1].legend()
        axes[1].set_ylim(0, max(base_rates) * 1.2)
        
        # Add percentage labels
        for i, rate in enumerate(base_rates):
            axes[1].annotate(f'{rate:.1f}%', (i, rate + 1), ha='center', fontsize=10, fontweight='bold')
        
        # For summary
        grp_names = cat_names
        counts = category_target_counts
        sens_name = sensitive_feature

    else:
        # Binary sensitive feature (e.g., sex)
        sens_idx = data['sensitive_col_idx']
        sens_name = data.get('sensitive_col_name', sensitive_feature)
        
        # Get group labels (customize based on feature)
        if 'sex' in sensitive_feature.lower():
            group_names = {0: 'Female', 1: 'Male'}
        else:
            group_names = {0: f'{sens_name}=0', 1: f'{sens_name}=1'}
        
        # Count samples per group and target
        group0_mask = (X_train[:, sens_idx] == 0)
        group1_mask = (X_train[:, sens_idx] == 1)
        
        counts = {
            group_names[0]: {'positive': int((y_train[group0_mask] == 1).sum()), 
                             'negative': int((y_train[group0_mask] == 0).sum()),
                             'total': int(group0_mask.sum())},
            group_names[1]: {'positive': int((y_train[group1_mask] == 1).sum()), 
                             'negative': int((y_train[group1_mask] == 0).sum()),
                             'total': int(group1_mask.sum())}
        }
        
        # Plot 1: Grouped bar chart
        grp_names = list(counts.keys())
        x = np.arange(len(grp_names))
        width = 0.35
        
        neg_counts = [counts[g]['negative'] for g in grp_names]
        pos_counts = [counts[g]['positive'] for g in grp_names]
        
        bars1 = axes[0].bar(x - width/2, neg_counts, width, label=target_names[0], color='#3498db', alpha=0.8)
        bars2 = axes[0].bar(x + width/2, pos_counts, width, label=target_names[1], color='#e74c3c', alpha=0.8)
        
        axes[0].set_xlabel(f'Sensitive Feature: {sens_name}', fontsize=11)
        axes[0].set_ylabel('Count', fontsize=11)
        axes[0].set_title('Distribution by Sensitive Group', fontsize=12, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(grp_names)
        axes[0].legend()
        
        # Add count labels
        for bar in bars1:
            axes[0].annotate(f'{int(bar.get_height()):,}', 
                             (bar.get_x() + bar.get_width()/2, bar.get_height()),
                             ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            axes[0].annotate(f'{int(bar.get_height()):,}', 
                             (bar.get_x() + bar.get_width()/2, bar.get_height()),
                             ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Base rate comparison
        base_rates = [100 * counts[g]['positive'] / counts[g]['total'] for g in grp_names]
        
        colors = ['#9b59b6', '#1abc9c']
        axes[1].bar(x, base_rates, width=0.5, color=colors, edgecolor='black', linewidth=1)
        
        axes[1].axhline(y=100*y_train.mean(), color='black', linestyle='--', linewidth=2, 
                        label=f'Overall rate: {100*y_train.mean():.1f}%')
        axes[1].set_xlabel(f'Sensitive Feature: {sens_name}', fontsize=11)
        axes[1].set_ylabel(f'% {target_names[1]}', fontsize=11)
        axes[1].set_title('Base Rate Disparity', fontsize=12, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(grp_names)
        axes[1].legend()
        axes[1].set_ylim(0, max(base_rates) * 1.3)
        
        # Add percentage labels
        for i, rate in enumerate(base_rates):
            axes[1].annotate(f'{rate:.1f}%', (i, rate + 1.5), ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("DISTRIBUTION SUMMARY")
    print("="*60)
    
    if is_multiclass:
        print(f"\nSensitive Feature: {sensitive_feature} ({n_categories} categories)")
    else:
        print(f"\nSensitive Feature: {sens_name}")
    
    print(f"{'Group':<15} {'Count':>10} {'% of Data':>12} {'Base Rate':>12}")
    print("-"*50)
    for grp_name in grp_names:
        c = counts[grp_name]
        pct_data = 100 * c['total'] / len(y_train)
        base_rate = 100 * c['positive'] / c['total']
        print(f"{grp_name:<15} {c['total']:>10,} {pct_data:>11.1f}% {base_rate:>11.1f}%")
    
    print("-"*50)
    print(f"{'OVERALL':<15} {len(y_train):>10,} {'100.0%':>12} {100*y_train.mean():>11.1f}%")
    
    # Calculate disparity
    rates = [counts[g]['positive'] / counts[g]['total'] for g in grp_names]
    disparity_ratio = max(rates) / min(rates)
    disparity_diff = 100 * (max(rates) - min(rates))
    
    print(f"\n📊 Disparity Analysis:")
    print(f"   • Highest base rate: {100*max(rates):.1f}%")
    print(f"   • Lowest base rate:  {100*min(rates):.1f}%")
    print(f"   • Disparity ratio:   {disparity_ratio:.2f}x")
    print(f"   • Absolute gap:      {disparity_diff:.1f} percentage points")


def plot_and_display_pareto(results, dataset_name, sensitive_feature, output_dir, 
                            get_pareto_front_fn=None):
    """
    Generate and display Pareto comparison plot.
    
    Parameters
    ----------
    results : dict
        Dictionary mapping model_type -> smac optimizer
    dataset_name : str
        Name of the dataset
    sensitive_feature : str
        Name of the sensitive feature
    output_dir : str
        Directory to save plots
    get_pareto_front_fn : callable, optional
        Function to get Pareto front (imported if not provided)
        
    Returns
    -------
    dict
        Paths to saved plots
    """
    pareto_filename = f"pareto_{dataset_name}_{sensitive_feature}.png"
    pareto_paths = plot_pareto_comparison(
        results, 
        output_dir=output_dir,
        filename=pareto_filename,
        formats=["notebook", "latex"]
    )
    
    return pareto_paths


def plot_parallel_coords_all_models(results, dataset_name, sensitive_feature, output_dir,
                                    get_configspace_fn=None):
    """
    Generate parallel coordinate plots for ALL models in results.
    
    Works for any number of models (rf, mlp, sensei, etc.)
    
    Parameters
    ----------
    results : dict
        Dictionary mapping model_type -> smac optimizer
    dataset_name : str
        Name of the dataset
    sensitive_feature : str
        Name of the sensitive feature
    output_dir : str
        Directory to save plots
    get_configspace_fn : callable
        Function to get configspace for a model type
        
    Returns
    -------
    dict
        Paths to all generated plots
    """
    paths = {}
    
    for model_type in results.keys():
        filename = f"parallel_coords_{dataset_name}_{sensitive_feature}_{model_type}.png"
        model_paths = plot_parallel_coordinates(
            results[model_type], 
            model_type,
            get_configspace_fn=get_configspace_fn,
            output_dir=output_dir,
            filename=filename,
            color_by='error',
            formats=["notebook", "latex"]
        )
        paths[model_type] = model_paths
    
    return paths


def plot_mds_all_models(results, dataset_name, sensitive_feature, output_dir,
                        get_configspace_fn=None):
    """
    Generate MDS projection plots for ALL models in results.
    
    Works for any number of models (rf, mlp, sensei, etc.)
    
    Parameters
    ----------
    results : dict
        Dictionary mapping model_type -> smac optimizer
    dataset_name : str
        Name of the dataset
    sensitive_feature : str
        Name of the sensitive feature
    output_dir : str
        Directory to save plots
    get_configspace_fn : callable
        Function to get configspace for a model type
        
    Returns
    -------
    dict
        Paths to all generated plots
    """
    paths = {}
    
    for model_type in results.keys():
        filename = f"mds_projection_{dataset_name}_{sensitive_feature}_{model_type}.png"
        model_paths = plot_mds_projection(
            results[model_type], 
            model_type,
            get_configspace_fn=get_configspace_fn,
            output_dir=output_dir,
            filename=filename,
            formats=["notebook", "latex"]
        )
        paths[model_type] = model_paths
    
    return paths

