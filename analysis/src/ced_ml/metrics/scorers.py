"""
Custom scikit-learn scorers for model selection.

Provides scorers optimized for clinical operating points (e.g., fixed specificity).
"""

from collections.abc import Callable

import numpy as np
from sklearn.metrics import make_scorer, roc_curve


def tpr_at_fpr_score(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.05) -> float:
    """
    Calculate TPR (sensitivity) at a target FPR (1 - specificity).

    Args:
        y_true: Binary labels (0/1)
        y_score: Predicted probabilities or decision scores
        target_fpr: Target false positive rate (e.g., 0.05 for 95% specificity)

    Returns:
        Maximum TPR achievable at or below target_fpr
        Returns 0.0 if no threshold achieves target_fpr
        Returns np.nan if only one class present in y_true

    Notes:
        Single-class cases return NaN to match behavior of other metrics
        (AUROC, PR-AUC, etc.) for consistency in cross-validation.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    # Check for single-class case
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        return np.nan

    fpr, tpr, _ = roc_curve(y_true, y_score)
    valid = fpr <= target_fpr
    return float(np.max(tpr[valid])) if np.any(valid) else 0.0


def make_tpr_at_fpr_scorer(target_fpr: float) -> Callable:
    """
    Create a scorer for TPR at a specific FPR threshold.

    Args:
        target_fpr: Target false positive rate (e.g., 0.05 for 95% specificity)

    Returns:
        sklearn-compatible scorer

    Examples:
        >>> scorer = make_tpr_at_fpr_scorer(target_fpr=0.05)
        >>> from sklearn.model_selection import cross_val_score
        >>> scores = cross_val_score(clf, X, y, cv=5, scoring=scorer)
    """
    return make_scorer(
        lambda y, p, **kwargs: tpr_at_fpr_score(
            y, p, target_fpr=target_fpr
        ),  # noqa: ARG005 - sklearn API
        needs_proba=True,
        greater_is_better=True,
    )


def get_scorer(scoring: str, target_fpr: float | None = None) -> Callable:
    """
    Get scorer by name, supporting custom scorers.

    Args:
        scoring: Scorer name (e.g., "roc_auc", "tpr_at_fpr")
        target_fpr: Target FPR for custom scorers (default: 0.05 for 95% specificity)

    Returns:
        Scorer callable (either custom or sklearn string name)

    Examples:
        >>> scorer = get_scorer("tpr_at_fpr", target_fpr=0.05)
        >>> scorer = get_scorer("roc_auc")  # Standard sklearn scorer
    """
    if scoring == "tpr_at_fpr":
        if target_fpr is None:
            target_fpr = 0.05  # Default to 95% specificity
        return make_tpr_at_fpr_scorer(target_fpr)

    # Return sklearn string name for standard scorers
    return scoring
