"""Utility modules for individual fairness experiments."""

from utils.datasets import (
    load_dataset,
    create_flipped_data,
    list_available_datasets,
    get_dataset_config,
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
    compute_fairness_confusion_matrix,
    plot_fairness_confusion_matrix,
    print_fairness_confusion_summary,
)

__all__ = [
    # datasets
    "load_dataset",
    "create_flipped_data",
    "list_available_datasets",
    "get_dataset_config",
    # individual_fairness
    "counterfactual_consistency",
    # plotting
    "get_pareto_front",
    "get_pareto_indices",
    "plot_pareto_comparison",
    "plot_parallel_coordinates",
    "plot_mds_projection",
    "print_pareto_summary",
    "generate_all_visualizations",
    # fairness confusion matrix
    "compute_fairness_confusion_matrix",
    "plot_fairness_confusion_matrix",
    "print_fairness_confusion_summary",
]

