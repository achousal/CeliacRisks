"""
Stratified bootstrap confidence intervals for binary classification metrics.

This module provides functions for computing bootstrap confidence intervals
using stratified resampling to maintain case/control ratios. Supports both
percentile and BCa (bias-corrected and accelerated) methods.
"""

from collections.abc import Callable
from typing import Literal

import numpy as np

# Check for scipy.stats.bootstrap availability (scipy >= 1.7)
try:
    from scipy.stats import bootstrap as scipy_bootstrap

    HAS_SCIPY_BOOTSTRAP = True
except ImportError:
    HAS_SCIPY_BOOTSTRAP = False


# Type alias for CI method
CIMethod = Literal["percentile", "bca"]


def _safe_metric(metric_fn: Callable, y: np.ndarray, p: np.ndarray) -> float:
    """
    Safely compute metric, returning NaN on failure.

    Args:
        metric_fn: Metric function that takes (y_true, y_pred)
        y: True labels
        p: Predicted probabilities

    Returns:
        Metric value, or NaN if computation failed
    """
    try:
        return metric_fn(y, p)
    except Exception:
        return np.nan


def _percentile_ci(vals: list[float], alpha: float = 0.05) -> tuple[float, float]:
    """Compute CI using simple percentile method."""
    lower_pct = 100 * (alpha / 2)
    upper_pct = 100 * (1 - alpha / 2)
    return (float(np.percentile(vals, lower_pct)), float(np.percentile(vals, upper_pct)))


def _bca_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    Compute BCa (bias-corrected and accelerated) bootstrap CI using scipy.

    BCa provides better coverage than percentile method for small samples
    and skewed distributions (Efron & Tibshirani, 1993).

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        metric_fn: Metric function
        n_boot: Number of bootstrap iterations
        seed: Random seed
        alpha: Significance level (default: 0.05 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound)

    Raises:
        ImportError: If scipy >= 1.7 is not available
    """
    if not HAS_SCIPY_BOOTSTRAP:
        raise ImportError(
            "BCa bootstrap requires scipy >= 1.7. " "Install with: pip install 'scipy>=1.7'"
        )

    # scipy.stats.bootstrap expects data as tuple of arrays
    # and a statistic function that takes resampled arrays
    def statistic(*data_tuple, axis):
        # data_tuple contains resampled (y_true, y_pred)
        # axis=-1 means samples are along last axis
        y_t, y_p = data_tuple
        # Handle vectorized case (axis=-1)
        if axis == -1 and y_t.ndim > 1:
            # Compute metric for each bootstrap sample
            results = []
            for i in range(y_t.shape[0]):
                results.append(metric_fn(y_t[i], y_p[i]))
            return np.array(results)
        return metric_fn(y_t, y_p)

    rng = np.random.default_rng(seed)
    result = scipy_bootstrap(
        (y_true, y_pred),
        statistic=statistic,
        n_resamples=n_boot,
        method="BCa",
        confidence_level=1 - alpha,
        random_state=rng,
        paired=True,
    )
    return (float(result.confidence_interval.low), float(result.confidence_interval.high))


def stratified_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable,
    n_boot: int = 1000,
    seed: int = 0,
    min_valid_frac: float = 0.1,
    method: CIMethod = "percentile",
) -> tuple[float, float]:
    """
    Compute stratified bootstrap confidence interval for a metric.

    Performs stratified resampling (maintaining case/control ratio) and computes
    95% CI using the specified method. If fewer than `max(20, n_boot * min_valid_frac)`
    valid bootstrap samples are obtained, returns (NaN, NaN).

    Args:
        y_true: True binary labels (0/1)
        y_pred: Predicted probabilities [0, 1]
        metric_fn: Function that takes (y_true, y_pred) and returns a scalar
        n_boot: Number of bootstrap iterations (default: 1000)
        seed: Random seed for reproducibility (default: 0)
        min_valid_frac: Minimum fraction of valid samples required (default: 0.1)
        method: CI method - "percentile" (default) or "bca" (bias-corrected).
                BCa provides better coverage for small samples but requires scipy >= 1.7.

    Returns:
        Tuple of (lower_bound, upper_bound) for 95% CI, or (NaN, NaN) if insufficient
        valid samples

    Raises:
        ValueError: If fewer than 2 cases or 2 controls in y_true
        ValueError: If method is not "percentile" or "bca"
        ImportError: If method="bca" but scipy >= 1.7 is not available

    Examples:
        >>> from sklearn.metrics import roc_auc_score
        >>> y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        >>> y_pred = np.array([0.1, 0.2, 0.7, 0.8, 0.9, 0.3, 0.6, 0.4])
        >>> ci_lower, ci_upper = stratified_bootstrap_ci(
        ...     y_true, y_pred, roc_auc_score, n_boot=100, seed=42
        ... )
        >>> 0 <= ci_lower <= ci_upper <= 1
        True
        >>> # Using BCa method (requires scipy >= 1.7)
        >>> ci_lower, ci_upper = stratified_bootstrap_ci(
        ...     y_true, y_pred, roc_auc_score, n_boot=100, seed=42, method="bca"
        ... )
    """
    if method not in ("percentile", "bca"):
        raise ValueError(f"method must be 'percentile' or 'bca', got '{method}'")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Validate inputs
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} elements, "
            f"y_pred has {len(y_pred)} elements"
        )

    # Identify cases and controls
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]

    if len(pos) < 2 or len(neg) < 2:
        raise ValueError(
            f"Insufficient samples for stratified bootstrap: "
            f"{len(pos)} cases, {len(neg)} controls (need >= 2 each)"
        )

    # For BCa, use scipy's implementation (non-stratified but more accurate CI)
    if method == "bca":
        return _bca_ci(y_true, y_pred, metric_fn, n_boot, seed)

    # Percentile method with stratified resampling
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        i_pos = rng.choice(pos, size=len(pos), replace=True)
        i_neg = rng.choice(neg, size=len(neg), replace=True)
        idx = np.concatenate([i_pos, i_neg])
        v = _safe_metric(metric_fn, y_true[idx], y_pred[idx])
        if np.isfinite(v):
            vals.append(v)

    # Check minimum valid samples threshold
    min_valid = max(20, int(n_boot * min_valid_frac))
    if len(vals) < min_valid:
        return (np.nan, np.nan)

    return _percentile_ci(vals)


def stratified_bootstrap_diff_ci(
    y_true: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    metric_fn: Callable,
    n_boot: int = 500,
    seed: int = 0,
    min_valid_frac: float = 0.1,
    method: CIMethod = "percentile",
) -> tuple[float, float, float]:
    """
    Compute stratified bootstrap CI for difference between two models.

    Computes the metric difference (model1 - model2) on the full sample and
    on stratified bootstrap resamples, then returns the full-sample difference
    and its 95% CI.

    Args:
        y_true: True binary labels (0/1)
        p1: Predictions from model 1
        p2: Predictions from model 2
        metric_fn: Function that takes (y_true, y_pred) and returns a scalar
        n_boot: Number of bootstrap iterations (default: 500)
        seed: Random seed for reproducibility (default: 0)
        min_valid_frac: Minimum fraction of valid samples required (default: 0.1)
        method: CI method - "percentile" (default) or "bca" (bias-corrected).
                BCa provides better coverage for small samples but requires scipy >= 1.7.

    Returns:
        Tuple of (diff_full, lower_bound, upper_bound) where:
        - diff_full: Full-sample difference (model1 - model2)
        - lower_bound: 2.5th percentile of bootstrap distribution
        - upper_bound: 97.5th percentile of bootstrap distribution

        If insufficient valid samples, returns (diff_full, NaN, NaN)

    Raises:
        ValueError: If fewer than 2 cases or 2 controls in y_true, or if
                    array lengths don't match
        ValueError: If method is not "percentile" or "bca"
        ImportError: If method="bca" but scipy >= 1.7 is not available

    Examples:
        >>> from sklearn.metrics import roc_auc_score
        >>> y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        >>> p1 = np.array([0.1, 0.2, 0.7, 0.8, 0.9, 0.3, 0.6, 0.4])
        >>> p2 = np.array([0.2, 0.3, 0.6, 0.7, 0.8, 0.4, 0.5, 0.5])
        >>> diff, ci_lower, ci_upper = stratified_bootstrap_diff_ci(
        ...     y_true, p1, p2, roc_auc_score, n_boot=100, seed=42
        ... )
        >>> isinstance(diff, float)
        True
    """
    if method not in ("percentile", "bca"):
        raise ValueError(f"method must be 'percentile' or 'bca', got '{method}'")

    y_true = np.asarray(y_true).astype(int)
    p1 = np.asarray(p1).astype(float)
    p2 = np.asarray(p2).astype(float)

    # Validate inputs
    if not (len(y_true) == len(p1) == len(p2)):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, p1={len(p1)}, p2={len(p2)}")

    # Compute full-sample difference
    diff_full = float(metric_fn(y_true, p1) - metric_fn(y_true, p2))

    # Identify cases and controls
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]

    if len(pos) < 2 or len(neg) < 2:
        raise ValueError(
            f"Insufficient samples for stratified bootstrap: "
            f"{len(pos)} cases, {len(neg)} controls (need >= 2 each)"
        )

    # For BCa on difference, use scipy's implementation
    if method == "bca":
        if not HAS_SCIPY_BOOTSTRAP:
            raise ImportError(
                "BCa bootstrap requires scipy >= 1.7. " "Install with: pip install 'scipy>=1.7'"
            )

        def diff_statistic(*data_tuple, axis):
            y_t, pred1, pred2 = data_tuple
            if axis == -1 and y_t.ndim > 1:
                results = []
                for i in range(y_t.shape[0]):
                    m1 = metric_fn(y_t[i], pred1[i])
                    m2 = metric_fn(y_t[i], pred2[i])
                    results.append(m1 - m2)
                return np.array(results)
            return metric_fn(y_t, pred1) - metric_fn(y_t, pred2)

        rng = np.random.default_rng(seed)
        result = scipy_bootstrap(
            (y_true, p1, p2),
            statistic=diff_statistic,
            n_resamples=n_boot,
            method="BCa",
            confidence_level=0.95,
            random_state=rng,
            paired=True,
        )
        return (
            diff_full,
            float(result.confidence_interval.low),
            float(result.confidence_interval.high),
        )

    # Percentile method with stratified resampling
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n_boot):
        i_pos = rng.choice(pos, size=len(pos), replace=True)
        i_neg = rng.choice(neg, size=len(neg), replace=True)
        idx = np.concatenate([i_pos, i_neg])

        m1 = _safe_metric(metric_fn, y_true[idx], p1[idx])
        m2 = _safe_metric(metric_fn, y_true[idx], p2[idx])
        if np.isfinite(m1) and np.isfinite(m2):
            diffs.append(m1 - m2)

    # Check minimum valid samples threshold
    min_valid = max(20, int(n_boot * min_valid_frac))
    if len(diffs) < min_valid:
        return (diff_full, np.nan, np.nan)

    ci_low, ci_high = _percentile_ci(diffs)
    return (diff_full, ci_low, ci_high)
