"""Utility modules for individual fairness experiments."""

from datasets import (
    load_dataset,
    create_flipped_data,
    create_flipped_data_sex_proxy,
    list_available_datasets,
    get_dataset_config,
    split_data,
)
from metrics import counterfactual_consistency
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
    # Notebook display helpers
    plot_sensitive_distribution,
    plot_and_display_pareto,
    plot_parallel_coords_all_models,
    plot_mds_all_models,
)
from utils.analysis import (
    # Caching
    save_smac_results,
    load_smac_results,
    CachedSMAC,
    CachedConfig,
    # Optimization helpers
    run_smac_optimization,
    get_optimization_stats,
    print_optimization_summary,
    analyze_fairness_confusion_matrix,
    analyze_trivial_fairness,
    setup_case_study_analysis,
    case_study_prediction_flip,
    case_study_consistent_sample,
    case_study_edge_cases,
    case_study_model_comparison,
    case_study_probability_swings,
    case_study_directional_analysis,
)

__all__ = [
    # datasets
    "load_dataset",
    "create_flipped_data",
    "create_flipped_data_sex_proxy",
    "list_available_datasets",
    "get_dataset_config",
    "split_data",
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
    # notebook display helpers
    "plot_sensitive_distribution",
    "plot_and_display_pareto",
    "plot_parallel_coords_all_models",
    "plot_mds_all_models",
    # analysis - caching
    "save_smac_results",
    "load_smac_results",
    "CachedSMAC",
    "CachedConfig",
    # analysis - optimization helpers
    "run_smac_optimization",
    "get_optimization_stats",
    "print_optimization_summary",
    "analyze_fairness_confusion_matrix",
    "analyze_trivial_fairness",
    "setup_case_study_analysis",
    "case_study_prediction_flip",
    "case_study_consistent_sample",
    "case_study_edge_cases",
    "case_study_model_comparison",
    "case_study_probability_swings",
    "case_study_directional_analysis",
]

