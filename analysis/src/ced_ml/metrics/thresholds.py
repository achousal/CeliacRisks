"""Threshold selection utilities for binary classification.

Provides multiple strategies for choosing decision thresholds:
- Max F1/F-beta score
- Youden's J statistic
- Fixed specificity/precision targets
- Control-based quantile thresholds

All functions operate on true labels (y_true) and predicted probabilities (p).
"""

from typing import Any, Dict, Optional, Tuple, TypedDict

import numpy as np

# ============================================================================
# ThresholdBundle: Standardized threshold data structure for plotting
# ============================================================================


class ThresholdMetrics(TypedDict, total=False):
    """Metrics computed at a specific threshold.

    Used for plotting threshold markers on ROC curves and risk distributions.
    """

    threshold: float
    fpr: float
    tpr: float
    sensitivity: float
    specificity: float
    precision: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int


class ThresholdBundle(TypedDict, total=False):
    """Standardized container for all threshold-related data.

    This bundle provides a single, consistent interface for passing threshold
    information to plotting functions. All plotting functions should accept
    this bundle and extract what they need.

    Keys:
        youden: ThresholdMetrics at Youden's J optimal threshold
        spec_target: ThresholdMetrics at target specificity threshold
        dca: ThresholdMetrics at DCA zero-crossing threshold (if computed)
        target_spec: The actual specificity target value (e.g., 0.95)
        youden_threshold: Raw threshold value for Youden
        spec_target_threshold: Raw threshold value for specificity target
        dca_threshold: Raw threshold value for DCA zero-crossing

    Usage:
        bundle = compute_threshold_bundle(y_true, y_pred, target_spec=0.95)
        plot_roc_curve(..., threshold_bundle=bundle)
        plot_risk_distribution(..., threshold_bundle=bundle)
    """

    youden: ThresholdMetrics
    spec_target: ThresholdMetrics
    dca: ThresholdMetrics
    target_spec: float
    youden_threshold: float
    spec_target_threshold: float
    dca_threshold: float


from sklearn.metrics import (  # noqa: E402
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)


def threshold_max_f1(y_true: np.ndarray, p: np.ndarray) -> float:
    """Find threshold that maximizes F1-score.

    Args:
        y_true: True binary labels (0/1)
        p: Predicted probabilities [0, 1]

    Returns:
        Optimal threshold (float in [0, 1])

    Notes:
        - Uses precision-recall curve to efficiently scan all thresholds
        - Falls back to 0.5 if no valid threshold found
        - F1 = 2 * (precision * recall) / (precision + recall)
    """
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)

    # Handle empty arrays
    if len(y_true) == 0 or len(p) == 0:
        return 0.5

    prec, rec, thr = precision_recall_curve(y_true, p)
    if thr.size == 0:
        return 0.5

    prec_t = prec[:-1]
    rec_t = rec[:-1]
    denom = prec_t + rec_t

    f1 = np.zeros_like(denom, dtype=float)
    np.divide(2.0 * prec_t * rec_t, denom, out=f1, where=(denom > 0))

    i = int(np.nanargmax(f1))
    return float(thr[i])


def threshold_max_fbeta(y_true: np.ndarray, p: np.ndarray, beta: float = 1.0) -> float:
    """Find threshold that maximizes F-beta score.

    Args:
        y_true: True binary labels (0/1)
        p: Predicted probabilities [0, 1]
        beta: Beta parameter (beta > 1 emphasizes recall, < 1 emphasizes precision)

    Returns:
        Optimal threshold (float in [0, 1])

    Notes:
        - F-beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
        - beta=1 is equivalent to F1-score
        - beta=2 weights recall twice as much as precision
    """
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    beta = float(beta)
    if beta <= 0:
        beta = 1.0

    prec, rec, thr = precision_recall_curve(y_true, p)
    if thr.size == 0:
        return 0.5
    prec_t = prec[:-1]
    rec_t = rec[:-1]

    b2 = beta * beta
    denom = (b2 * prec_t) + rec_t
    f = np.zeros_like(denom, dtype=float)
    np.divide((1.0 + b2) * prec_t * rec_t, denom, out=f, where=(denom > 0))
    i = int(np.nanargmax(f))
    return float(thr[i])


def threshold_youden(y_true: np.ndarray, p: np.ndarray) -> float:
    """Find threshold that maximizes Youden's J statistic (TPR - FPR).

    Args:
        y_true: True binary labels (0/1)
        p: Predicted probabilities [0, 1]

    Returns:
        Optimal threshold (float in [0, 1])

    Notes:
        - Youden's J = sensitivity + specificity - 1 = TPR - FPR
        - Maximizes the vertical distance from the ROC curve to the diagonal
        - Falls back to 0.5 if no valid threshold found
    """
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)

    # Handle empty arrays
    if len(y_true) == 0 or len(p) == 0:
        return 0.5

    fpr, tpr, thr = roc_curve(y_true, p)
    J = tpr - fpr
    if thr.size == 0:
        return 0.5
    i = int(np.nanargmax(J))
    th = float(thr[i])
    if not np.isfinite(th):
        th = 0.5
    return th


def threshold_for_specificity(
    y_true: np.ndarray, p: np.ndarray, target_spec: float = 0.90
) -> float:
    """Find threshold achieving target specificity with highest sensitivity.

    Args:
        y_true: True binary labels (0/1)
        p: Predicted probabilities [0, 1]
        target_spec: Target specificity (0-1), default 0.90

    Returns:
        Threshold achieving target specificity (float in [0, 1])

    Notes:
        - Selects lowest threshold (highest sensitivity) among those meeting specificity target
        - Falls back to closest specificity if target unattainable
        - For clinical screening: high specificity (e.g., 0.95) minimizes false positives
    """
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    fpr, tpr, thr = roc_curve(y_true, p)
    spec = 1.0 - fpr
    ok = spec >= target_spec
    if np.any(ok):
        j = int(np.argmax(tpr[ok]))
        th = thr[ok][j]
    else:
        j = int(np.argmin(np.abs(spec - target_spec)))
        th = thr[j]
    if not np.isfinite(th):
        th = float(np.max(p) + 1e-12)
    return float(th)


def threshold_for_precision(y_true: np.ndarray, p: np.ndarray, target_ppv: float) -> float:
    """Find the LOWEST threshold achieving precision >= target_ppv.

    Args:
        y_true: True binary labels (0/1)
        p: Predicted probabilities [0, 1]
        target_ppv: Target precision/PPV (0-1)

    Returns:
        Threshold achieving target precision (float in [0, 1])

    Notes:
        - Falls back to max-F1 if target precision unattainable
        - Lower threshold = more inclusive predictions
        - Precision = TP / (TP + FP) = PPV
    """
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    target_ppv = float(target_ppv)
    if not (0 < target_ppv <= 1):
        return threshold_max_f1(y_true, p)

    prec, rec, thr = precision_recall_curve(y_true, p)
    if thr.size == 0:
        return 0.5

    prec_t = prec[:-1]
    thr_t = thr

    ok = np.where(prec_t >= target_ppv)[0]
    if ok.size == 0:
        return threshold_max_f1(y_true, p)

    # Want lowest threshold (most inclusive) among those achieving target
    idx = int(ok[-1])  # in PR curve, thresholds typically increase with index
    th = float(thr_t[idx])
    if not np.isfinite(th):
        th = threshold_max_f1(y_true, p)
    return th


def threshold_from_controls(p_controls: np.ndarray, target_spec: float) -> float:
    """Find threshold from control quantile to achieve target specificity.

    Args:
        p_controls: Predicted probabilities for control samples only
        target_spec: Target specificity (0-1)

    Returns:
        Threshold from control distribution (float in [0, 1])

    Notes:
        - Uses control quantile matching target specificity
        - Spec = 0.95 -> 95th percentile of control predictions
        - Assumes controls and cases are from same distribution (independence)
    """
    pc = np.asarray(p_controls, dtype=float)
    pc = pc[np.isfinite(pc)]
    if pc.size == 0:
        return 0.5
    q = float(target_spec)
    try:
        thr = float(np.quantile(pc, q, method="higher"))
    except TypeError:
        # Older numpy versions use 'interpolation' parameter
        thr = float(np.quantile(pc, q, interpolation="higher"))
    except Exception:
        thr = float(np.quantile(pc, q))
    if not np.isfinite(thr):
        thr = float(np.max(pc) + 1e-12)
    return thr


def binary_metrics_at_threshold(y_true: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, Any]:
    """Compute classification metrics at a specific threshold.

    Args:
        y_true: True binary labels (0/1)
        p: Predicted probabilities [0, 1]
        thr: Classification threshold (predictions >= thr -> positive)

    Returns:
        Dictionary containing:
        - threshold: Applied threshold
        - precision: TP / (TP + FP)
        - sensitivity: TP / (TP + FN) = recall = TPR
        - f1: F1-score
        - specificity: TN / (TN + FP)
        - fpr: False positive rate = 1 - specificity
        - tpr: True positive rate = sensitivity
        - tp, fp, tn, fn: Confusion matrix counts

    Notes:
        - Uses zero_division=0 for precision/recall when no positive predictions
        - Specificity = np.nan if no negative samples
    """
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    y_hat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    prec = precision_score(y_true, y_hat, zero_division=0)
    rec = recall_score(y_true, y_hat, zero_division=0)
    f1 = f1_score(y_true, y_hat, zero_division=0)
    spec = (tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    fpr = 1.0 - spec if not np.isnan(spec) else np.nan
    return {
        "threshold": float(thr),
        "precision": float(prec),
        "sensitivity": float(rec),
        "f1": float(f1),
        "specificity": float(spec),
        "fpr": float(fpr) if not np.isnan(fpr) else np.nan,
        "tpr": float(rec),  # TPR = sensitivity = recall
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def top_risk_capture(y_true: np.ndarray, p: np.ndarray, frac: float = 0.01) -> Dict[str, Any]:
    """Analyze risk capture in top fraction of predictions.

    Args:
        y_true: True binary labels (0/1)
        p: Predicted probabilities [0, 1]
        frac: Fraction of samples to select (0-1), default 0.01 (top 1%)

    Returns:
        Dictionary containing:
        - frac: Input fraction
        - n_top: Number of samples in top fraction
        - cases_in_top: Number of positive cases in top fraction
        - controls_in_top: Number of negative samples in top fraction
        - case_capture: Proportion of all cases in top fraction

    Notes:
        - Useful for screening scenarios: "Top 1% captures 20% of cases"
        - case_capture = sensitivity at a threshold set by risk score rank
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    n = len(y)
    if n == 0:
        return {
            "frac": float(frac),
            "n_top": 0,
            "cases_in_top": 0,
            "controls_in_top": 0,
            "case_capture": np.nan,
        }
    k = int(np.ceil(frac * n))
    k = max(1, min(n, k))
    idx = np.argsort(p)[::-1][:k]
    cases_in_top = int(y[idx].sum())
    controls_in_top = int(k - cases_in_top)
    total_cases = int(y.sum())
    capture = (cases_in_top / total_cases) if total_cases > 0 else np.nan
    return {
        "frac": float(frac),
        "n_top": int(k),
        "cases_in_top": int(cases_in_top),
        "controls_in_top": int(controls_in_top),
        "case_capture": float(capture) if np.isfinite(capture) else np.nan,
    }


def choose_threshold_objective(
    y_true: np.ndarray,
    p: np.ndarray,
    objective: str,
    fbeta: float = 1.0,
    fixed_spec: float = 0.90,
    fixed_ppv: float = 0.5,
) -> Tuple[str, float]:
    """Select threshold based on specified objective.

    Args:
        y_true: True binary labels (0/1)
        p: Predicted probabilities [0, 1]
        objective: One of ['max_f1', 'max_fbeta', 'youden', 'fixed_spec', 'fixed_ppv']
        fbeta: Beta parameter for F-beta score (default 1.0)
        fixed_spec: Target specificity for 'fixed_spec' objective (default 0.90)
        fixed_ppv: Target precision for 'fixed_ppv' objective (default 0.5)

    Returns:
        Tuple of (objective_name, threshold)

    Notes:
        - Provides unified interface for multiple threshold strategies
        - Falls back to max_f1 for unknown objectives
        - Objective choice should match clinical/business requirements

    Examples:
        >>> y = np.array([0, 0, 1, 1])
        >>> p = np.array([0.1, 0.4, 0.6, 0.9])
        >>> name, thr = choose_threshold_objective(y, p, "max_f1")
        >>> name
        'max_f1'
    """
    obj = (objective or "max_f1").strip().lower()
    if obj == "max_f1":
        return ("max_f1", threshold_max_f1(y_true, p))
    if obj == "max_fbeta":
        return ("max_fbeta", threshold_max_fbeta(y_true, p, beta=fbeta))
    if obj == "youden":
        return ("youden", threshold_youden(y_true, p))
    if obj == "fixed_spec":
        return (
            "fixed_spec",
            threshold_for_specificity(y_true, p, target_spec=float(fixed_spec)),
        )
    if obj == "fixed_ppv":
        return (
            "fixed_ppv",
            threshold_for_precision(y_true, p, target_ppv=float(fixed_ppv)),
        )
    # fallback
    return ("max_f1", threshold_max_f1(y_true, p))


def compute_threshold_bundle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_spec: float = 0.95,
    dca_threshold: Optional[float] = None,
) -> ThresholdBundle:
    """Compute all standard thresholds and metrics in a single call.

    This factory function creates a ThresholdBundle containing Youden's J threshold,
    the target specificity threshold, and optionally the DCA threshold with their
    associated metrics. Use this to ensure consistent threshold handling across
    all plotting functions.

    Args:
        y_true: True binary labels (0/1)
        y_pred: Predicted probabilities [0, 1]
        target_spec: Target specificity (default 0.95)
        dca_threshold: Optional pre-computed DCA zero-crossing threshold

    Returns:
        ThresholdBundle with all threshold data needed for plotting

    Example:
        >>> bundle = compute_threshold_bundle(y_true, y_pred, target_spec=0.95)
        >>> plot_roc_curve(..., threshold_bundle=bundle)
        >>> plot_risk_distribution(..., threshold_bundle=bundle)
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(float)

    # Compute Youden threshold and metrics
    youden_thr = threshold_youden(y_true, y_pred)
    youden_metrics = binary_metrics_at_threshold(y_true, y_pred, youden_thr)

    # Compute specificity target threshold and metrics
    spec_thr = threshold_for_specificity(y_true, y_pred, target_spec=target_spec)
    spec_metrics = binary_metrics_at_threshold(y_true, y_pred, spec_thr)

    bundle: ThresholdBundle = {
        "youden": {
            "threshold": youden_thr,
            "fpr": youden_metrics["fpr"],
            "tpr": youden_metrics["tpr"],
            "sensitivity": youden_metrics["sensitivity"],
            "specificity": youden_metrics["specificity"],
            "precision": youden_metrics["precision"],
            "f1": youden_metrics["f1"],
            "tp": youden_metrics["tp"],
            "fp": youden_metrics["fp"],
            "tn": youden_metrics["tn"],
            "fn": youden_metrics["fn"],
        },
        "spec_target": {
            "threshold": spec_thr,
            "fpr": spec_metrics["fpr"],
            "tpr": spec_metrics["tpr"],
            "sensitivity": spec_metrics["sensitivity"],
            "specificity": spec_metrics["specificity"],
            "precision": spec_metrics["precision"],
            "f1": spec_metrics["f1"],
            "tp": spec_metrics["tp"],
            "fp": spec_metrics["fp"],
            "tn": spec_metrics["tn"],
            "fn": spec_metrics["fn"],
        },
        "target_spec": target_spec,
        "youden_threshold": youden_thr,
        "spec_target_threshold": spec_thr,
    }

    # Add DCA threshold if provided
    if dca_threshold is not None:
        dca_metrics = binary_metrics_at_threshold(y_true, y_pred, dca_threshold)
        bundle["dca"] = {
            "threshold": dca_threshold,
            "fpr": dca_metrics["fpr"],
            "tpr": dca_metrics["tpr"],
            "sensitivity": dca_metrics["sensitivity"],
            "specificity": dca_metrics["specificity"],
            "precision": dca_metrics["precision"],
            "f1": dca_metrics["f1"],
            "tp": dca_metrics["tp"],
            "fp": dca_metrics["fp"],
            "tn": dca_metrics["tn"],
            "fn": dca_metrics["fn"],
        }
        bundle["dca_threshold"] = dca_threshold

    return bundle
