"""
Individual Fairness Metrics Module

Provides metrics for evaluating individual fairness, particularly
counterfactual consistency which measures if predictions change
when sensitive attributes are flipped.
"""

import numpy as np
from typing import List, Union, Callable


def counterfactual_consistency(
    y_pred_original: np.ndarray,
    y_pred_flipped: np.ndarray,
) -> float:
    """
    Calculate counterfactual consistency (individual fairness metric).
    
    Measures how consistent a model's predictions are when sensitive attributes
    are flipped/perturbed. A score of 1.0 means the model is perfectly consistent
    (changing sensitive attributes never changes the prediction).
    
    Parameters
    ----------
    y_pred_original : np.ndarray
        Predictions on the original test data
    y_pred_flipped : np.ndarray
        Predictions on the test data with sensitive attributes flipped
        
    Returns
    -------
    float
        Consistency score between 0 and 1. Higher is better (more fair).
        
    Example
    -------
    >>> y_pred_orig = model.predict(X_test)
    >>> y_pred_flip = model.predict(X_test_flipped)
    >>> score = counterfactual_consistency(y_pred_orig, y_pred_flip)
    >>> print(f"Consistency: {score:.2%}")  # e.g., "Consistency: 95.50%"
    """
    y_pred_original = np.asarray(y_pred_original).ravel()
    y_pred_flipped = np.asarray(y_pred_flipped).ravel()
    
    if len(y_pred_original) != len(y_pred_flipped):
        raise ValueError(
            f"Prediction arrays must have same length. "
            f"Got {len(y_pred_original)} and {len(y_pred_flipped)}"
        )
    
    # Calculate proportion of predictions that stayed the same
    score = np.mean(y_pred_original == y_pred_flipped)
    return float(score)


def counterfactual_inconsistency(
    y_pred_original: np.ndarray,
    y_pred_flipped: np.ndarray,
) -> float:
    """
    Calculate counterfactual inconsistency (1 - consistency).
    
    This is useful for optimization where lower is better.
    
    Returns
    -------
    float
        Inconsistency score between 0 and 1. Lower is better (more fair).
    """
    return 1.0 - counterfactual_consistency(y_pred_original, y_pred_flipped)
