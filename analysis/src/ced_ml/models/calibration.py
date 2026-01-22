"""
Calibration wrappers and utilities.

This module provides:
- Prevalence-adjusted probability calibration
- Calibration metrics (intercept, slope, ECE)
- sklearn CalibratedClassifierCV wrapper utilities

Note: Prevalence adjustment functions are imported from prevalence.py
      to avoid code duplication and ensure consistency.
"""

import logging

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Import prevalence adjustment utilities from canonical module
from .prevalence import (
    PrevalenceAdjustedModel,
    adjust_probabilities_for_prevalence,
)

logger = logging.getLogger(__name__)

__all__ = [
    "calibration_intercept_slope",
    "calib_intercept_metric",
    "calib_slope_metric",
    "expected_calibration_error",
    "adjust_probabilities_for_prevalence",  # Re-exported for backward compatibility
    "PrevalenceAdjustedModel",  # Re-exported for backward compatibility
    "get_calibrated_estimator_param_name",
    "get_calibrated_cv_param_name",
    "maybe_calibrate_estimator",
]


def calibration_intercept_slope(y_true: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """
    Compute calibration intercept and slope using logistic regression on logit scale.

    These indicate how well-calibrated probabilities are:
    - Intercept ~0 indicates probabilities match observed proportions
    - Slope ~1 indicates correct ordering/ranking

    Reference:
        Van Calster et al. (2016). Calibration of risk prediction models.
        Medical Decision Making.

    Args:
        y_true: True binary labels (0/1)
        p: Predicted probabilities

    Returns:
        (intercept, slope) tuple
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)

    # Filter valid values
    mask = np.isfinite(p) & np.isfinite(y)
    y = y[mask]
    p = p[mask]

    # Clip probabilities to avoid log(0)
    eps = 1e-7
    p_clipped = np.clip(p, eps, 1 - eps)
    log_odds = np.log(p_clipped / (1 - p_clipped))

    # Need both classes for calibration
    if len(np.unique(y)) < 2:
        return np.nan, np.nan

    # Fit logistic regression on log-odds
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(log_odds.reshape(-1, 1), y)
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


def calib_intercept_metric(y: np.ndarray, p: np.ndarray) -> float:
    """Compute calibration intercept metric for bootstrap CIs."""
    a, _ = calibration_intercept_slope(y, p)
    return float(a)


def calib_slope_metric(y: np.ndarray, p: np.ndarray) -> float:
    """Compute calibration slope metric for bootstrap CIs."""
    _, b = calibration_intercept_slope(y, p)
    return float(b)


def expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the difference between predicted probabilities and observed outcomes
    across probability bins.

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        n_bins: Number of bins for grouping predictions

    Returns:
        ECE value (lower is better calibrated)
    """
    y = np.asarray(y_true).astype(float)
    p = np.asarray(y_pred).astype(float)

    # Filter valid (remove NaN/inf before converting to int)
    mask = np.isfinite(p) & np.isfinite(y)
    y = y[mask].astype(int)
    p = p[mask]

    if len(y) == 0:
        return np.nan

    # Create bins
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (p >= bin_edges[i]) & (p < bin_edges[i + 1])
        if i == n_bins - 1:
            in_bin = in_bin | (p == 1.0)

        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            avg_pred = np.mean(p[in_bin])
            avg_true = np.mean(y[in_bin])
            ece += np.abs(avg_pred - avg_true) * prop_in_bin

    return float(ece)


# Note: adjust_probabilities_for_prevalence and PrevalenceAdjustedModel
# are now imported from prevalence.py (see top of file) to avoid code duplication


def get_calibrated_estimator_param_name() -> str:
    """
    Detect parameter name for base estimator in CalibratedClassifierCV.

    Different sklearn versions use different names ('estimator' vs 'base_estimator').

    Returns:
        'estimator' or 'base_estimator'
    """
    tmp = CalibratedClassifierCV(LinearSVC())
    params = tmp.get_params().keys()
    if "estimator" in params:
        return "estimator"
    if "base_estimator" in params:
        return "base_estimator"
    raise ValueError("Could not determine base estimator parameter name")


def get_calibrated_cv_param_name() -> str:
    """
    Detect CV parameter name in CalibratedClassifierCV.

    Returns:
        'cv' (standard name)
    """
    tmp = CalibratedClassifierCV(LinearSVC())
    params = tmp.get_params().keys()
    if "cv" in params:
        return "cv"
    raise ValueError("Could not determine cv parameter name")


def maybe_calibrate_estimator(
    estimator, model_name: str, calibrate: bool, method: str = "sigmoid", cv: int = 3
):
    """
    Optional calibration wrapper for LR/RF (SVM is already calibrated).

    This is applied consistently in CV and final training when enabled.

    Args:
        estimator: Base sklearn estimator
        model_name: Model name (e.g., 'RF', 'LR_EN', 'LinSVM_cal')
        calibrate: Whether to apply calibration
        method: Calibration method ('sigmoid' or 'isotonic')
        cv: Number of CV folds for calibration

    Returns:
        Calibrated or original estimator
    """
    if not calibrate:
        return estimator

    # Don't calibrate SVM (already calibrated)
    if model_name == "LinSVM_cal":
        return estimator

    # Don't double-calibrate
    if isinstance(estimator, CalibratedClassifierCV):
        return estimator

    try:
        kwargs = {"method": str(method), "cv": int(cv)}
        param_name = get_calibrated_estimator_param_name()
        kwargs[param_name] = estimator
        return CalibratedClassifierCV(**kwargs)
    except Exception as e:
        logger.warning(
            f"Failed to calibrate {model_name} with method={method}, cv={cv}: {e}. "
            "Returning original estimator without calibration."
        )
        return estimator
