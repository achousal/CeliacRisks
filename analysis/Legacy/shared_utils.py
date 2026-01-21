#!/usr/bin/env python3
"""
shared_utils.py - Shared Utilities for Celiac ML Pipeline

Contains utility functions used across training (celiacML_faith.py)
and post-processing (postprocess_compare.py) to avoid code duplication.

Sections:
  1. File/Directory Utilities
  2. Metric Computation
  3. Threshold Selection
  4. Calibration
  5. Plotting Utilities (metadata application, LOESS helpers)
  6. Decision Curve Analysis (DCA)
  7. Clinical Plotting (ROC, PR, Calibration, Risk Distribution) - NEW

Version: 1.2.0
Created: 2026-01-17
Updated: 2026-01-17 (Phase 3: consolidated plotting functions from postprocess_compare.py)
"""

import os
import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
import statsmodels.api as sm
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# =============================================================================
# 1. FILE/DIRECTORY UTILITIES
# =============================================================================

def _mkdir(p: str) -> str:
    """
    Create directory if it doesn't exist.

    Args:
        p: Path to directory

    Returns:
        The same path (for chaining)
    """
    os.makedirs(p, exist_ok=True)
    return p


# =============================================================================
# 2. METRIC COMPUTATION
# =============================================================================

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


def prob_metrics(
    y: np.ndarray,
    p: np.ndarray,
    include_logloss: bool = True,
    include_youden: bool = True,
    include_alpha: bool = True,
    alpha_target_spec: float = 0.95,
) -> dict:
    """
    Compute standard probability-based classification metrics.

    Args:
        y: True binary labels (0/1)
        p: Predicted probabilities
        include_logloss: If True, include LogLoss (requires clipping for stability)
        include_youden: If True, include Youden's J statistic
        include_alpha: If True, include Alpha (sensitivity at target specificity)
        alpha_target_spec: Target specificity for Alpha metric (default 0.95)

    Returns:
        Dictionary with AUROC, PR_AUC, Brier, and optionally LogLoss, Youden, Alpha
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)

    metrics = {
        "AUROC": float(roc_auc_score(y, p)),
        "PR_AUC": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
    }

    if include_logloss:
        # Clip probabilities to avoid log(0) in log_loss
        p_clipped = np.clip(p, 1e-15, 1 - 1e-15)
        metrics["LogLoss"] = float(log_loss(y, p_clipped))

    if include_youden:
        metrics["Youden"] = youden_j(y, p)

    if include_alpha:
        metrics["Alpha"] = alpha_specificity_sensitivity(y, p, target_spec=alpha_target_spec)

    return metrics


def format_ci(lo: float, hi: float, decimals: int = 3) -> str:
    """
    Format confidence interval as [lo, hi] with specified precision.

    Args:
        lo: Lower bound
        hi: Upper bound
        decimals: Number of decimal places

    Returns:
        Formatted string [lo, hi] or empty string if bounds not finite
    """
    if not np.isfinite(lo) or not np.isfinite(hi):
        return ""
    return f"[{lo:.{decimals}f}, {hi:.{decimals}f}]"


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    """AUROC metric (shorthand for sklearn)."""
    return float(roc_auc_score(y, p))


def prauc(y: np.ndarray, p: np.ndarray) -> float:
    """PR-AUC metric (shorthand for sklearn)."""
    return float(average_precision_score(y, p))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Brier score (shorthand for sklearn)."""
    return float(brier_score_loss(y, p))


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    """Log loss with numerical stability clipping."""
    p_clipped = np.clip(p, 1e-15, 1 - 1e-15)
    return float(log_loss(y, p_clipped))


def youden_j(y: np.ndarray, p: np.ndarray) -> float:
    """
    Compute Youden's J statistic (max TPR - FPR).

    J statistic represents the maximum vertical distance between the ROC curve
    and the diagonal (random classifier). Higher is better (0-1 range).

    Args:
        y: True binary labels (0/1)
        p: Predicted probabilities

    Returns:
        Maximum Youden's J value across all possible thresholds
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)

    fpr, tpr, _ = roc_curve(y, p)
    J = tpr - fpr
    return float(np.nanmax(J))


def alpha_specificity_sensitivity(y: np.ndarray, p: np.ndarray, target_spec: float = 0.95) -> float:
    """
    Compute sensitivity (TPR) at a target specificity (Alpha metric).

    This metric evaluates model performance at high-specificity operating points,
    useful for clinical screening where false positives must be minimized.

    Args:
        y: True binary labels (0/1)
        p: Predicted probabilities
        target_spec: Target specificity level (default 0.95 for 95% specificity)

    Returns:
        Sensitivity achieved at or above the target specificity
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)

    fpr, tpr, _ = roc_curve(y, p)
    spec = 1.0 - fpr

    # Find thresholds meeting or exceeding target specificity
    ok = spec >= target_spec
    if np.any(ok):
        # Among thresholds meeting target spec, pick the one with highest sensitivity
        return float(np.max(tpr[ok]))
    else:
        # If target specificity unattainable, return max sensitivity at closest specificity
        closest_idx = int(np.argmin(np.abs(spec - target_spec)))
        return float(tpr[closest_idx])


def compute_metrics_with_cis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 500,
    seed: int = 0,
    suffix: str = "",
    include_cis: bool = True,
    include_youden: bool = True,
    include_alpha: bool = True,
    alpha_target_spec: float = 0.95,
) -> Dict[str, Any]:
    """
    Compute standard metrics (AUROC, PR_AUC, Brier, Youden, Alpha) with optional bootstrap CIs.

    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        n_boot: Bootstrap iterations
        seed: Random seed
        suffix: Suffix for metric keys (e.g., "_oof", "_test")
        include_cis: If True, compute bootstrap CIs
        include_youden: If True, include Youden's J statistic
        include_alpha: If True, include Alpha (sensitivity at target specificity)
        alpha_target_spec: Target specificity for Alpha metric (default 0.95)

    Returns:
        Dictionary with metrics and optionally CIs
    """
    metrics = prob_metrics(
        y_true,
        y_pred,
        include_logloss=False,
        include_youden=include_youden,
        include_alpha=include_alpha,
        alpha_target_spec=alpha_target_spec,
    )

    result = {
        f"AUROC{suffix}": float(metrics["AUROC"]),
        f"PR_AUC{suffix}": float(metrics["PR_AUC"]),
        f"Brier{suffix}": float(metrics["Brier"]),
    }

    if include_youden:
        result[f"Youden{suffix}"] = float(metrics["Youden"])

    if include_alpha:
        result[f"Alpha{suffix}"] = float(metrics["Alpha"])

    if include_cis and n_boot > 0:
        ci_auc = stratified_bootstrap_ci(y_true, y_pred, auroc, n_boot=n_boot, seed=seed)
        ci_pr = stratified_bootstrap_ci(y_true, y_pred, prauc, n_boot=n_boot, seed=seed)
        ci_br = stratified_bootstrap_ci(y_true, y_pred, brier, n_boot=n_boot, seed=seed)
        ci_youden = stratified_bootstrap_ci(y_true, y_pred, youden_j, n_boot=n_boot, seed=seed)

        result[f"AUROC{suffix}_95CI"] = format_ci(ci_auc[0], ci_auc[1], decimals=3)
        result[f"PR_AUC{suffix}_95CI"] = format_ci(ci_pr[0], ci_pr[1], decimals=3)
        result[f"Brier{suffix}_95CI"] = format_ci(ci_br[0], ci_br[1], decimals=4)
        result[f"Youden{suffix}_95CI"] = format_ci(ci_youden[0], ci_youden[1], decimals=3)

        if include_alpha:
            # Alpha has target specificity, so we need a wrapper function
            alpha_fn = lambda y, p: alpha_specificity_sensitivity(y, p, target_spec=alpha_target_spec)
            ci_alpha = stratified_bootstrap_ci(y_true, y_pred, alpha_fn, n_boot=n_boot, seed=seed)
            result[f"Alpha{suffix}_95CI"] = format_ci(ci_alpha[0], ci_alpha[1], decimals=3)

    return result


def stratified_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable,
    n_boot: int = 1000,
    seed: int = 0
) -> Tuple[float, float]:
    """
    Compute stratified bootstrap confidence interval for a metric.

    Performs stratified resampling (maintaining case/control ratio) and computes
    95% CI using percentile method.

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        metric_fn: Function that takes (y_true, y_pred) and returns a scalar
        n_boot: Number of bootstrap iterations
        seed: Random seed for reproducibility

    Returns:
        Tuple of (lower_bound, upper_bound) for 95% CI
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    if len(pos) < 2 or len(neg) < 2:
        return (np.nan, np.nan)

    vals = []
    for _ in range(n_boot):
        i_pos = rng.choice(pos, size=len(pos), replace=True)
        i_neg = rng.choice(neg, size=len(neg), replace=True)
        idx = np.concatenate([i_pos, i_neg])
        v = _safe_metric(metric_fn, y_true[idx], y_pred[idx])
        if np.isfinite(v):
            vals.append(v)

    if len(vals) < max(20, n_boot // 10):
        return (np.nan, np.nan)

    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def stratified_bootstrap_diff_ci(
    y_true: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    metric_fn: Callable,
    n_boot: int = 500,
    seed: int = 0
) -> Tuple[float, float, float]:
    """
    Compute stratified bootstrap CI for difference between two models.

    Args:
        y_true: True binary labels
        p1: Predictions from model 1
        p2: Predictions from model 2
        metric_fn: Function that takes (y_true, y_pred) and returns a scalar
        n_boot: Number of bootstrap iterations
        seed: Random seed

    Returns:
        Tuple of (diff_full, lower_bound, upper_bound) where:
        - diff_full: Full-sample difference (model1 - model2)
        - lower_bound: 2.5th percentile of bootstrap distribution
        - upper_bound: 97.5th percentile of bootstrap distribution
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true).astype(int)
    p1 = np.asarray(p1).astype(float)
    p2 = np.asarray(p2).astype(float)

    diff_full = float(metric_fn(y_true, p1) - metric_fn(y_true, p2))

    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    if len(pos) < 2 or len(neg) < 2:
        return (diff_full, np.nan, np.nan)

    diffs = []
    for _ in range(n_boot):
        i_pos = rng.choice(pos, size=len(pos), replace=True)
        i_neg = rng.choice(neg, size=len(neg), replace=True)
        idx = np.concatenate([i_pos, i_neg])

        m1 = _safe_metric(metric_fn, y_true[idx], p1[idx])
        m2 = _safe_metric(metric_fn, y_true[idx], p2[idx])
        if np.isfinite(m1) and np.isfinite(m2):
            diffs.append(m1 - m2)

    if len(diffs) < max(20, n_boot // 10):
        return (diff_full, np.nan, np.nan)

    return (diff_full, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


# =============================================================================
# 3. THRESHOLD SELECTION
# =============================================================================
"""
Clinical threshold selection utilities for classification models.

Implements multiple strategies:
- Max F1-score / F-beta
- Youden's J statistic (max sensitivity + specificity)
- Fixed specificity/precision targets
- Control-based thresholds
- Top-risk capture analysis
"""


def threshold_max_f1(y_true: np.ndarray, p: np.ndarray) -> float:
    """Find threshold that maximizes F1-score."""
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)

    prec, rec, thr = precision_recall_curve(y_true, p)
    if thr.size == 0:
        return 0.5

    prec_t = prec[:-1]
    rec_t  = rec[:-1]
    denom  = prec_t + rec_t

    f1 = np.zeros_like(denom, dtype=float)
    np.divide(2.0 * prec_t * rec_t, denom, out=f1, where=(denom > 0))

    i = int(np.nanargmax(f1))
    return float(thr[i])


def threshold_max_fbeta(y_true: np.ndarray, p: np.ndarray, beta: float = 1.0) -> float:
    """Find threshold that maximizes F-beta score."""
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    beta = float(beta)
    if beta <= 0:
        beta = 1.0

    prec, rec, thr = precision_recall_curve(y_true, p)
    if thr.size == 0:
        return 0.5
    prec_t = prec[:-1]
    rec_t  = rec[:-1]

    b2 = beta * beta
    denom = (b2 * prec_t) + rec_t
    f = np.zeros_like(denom, dtype=float)
    np.divide((1.0 + b2) * prec_t * rec_t, denom, out=f, where=(denom > 0))
    i = int(np.nanargmax(f))
    return float(thr[i])


def threshold_youden(y_true: np.ndarray, p: np.ndarray) -> float:
    """Find threshold that maximizes Youden's J statistic (TPR - FPR)."""
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    fpr, tpr, thr = roc_curve(y_true, p)
    J = tpr - fpr
    if thr.size == 0:
        return 0.5
    i = int(np.nanargmax(J))
    th = float(thr[i])
    if not np.isfinite(th):
        th = 0.5
    return th


def threshold_for_specificity(y_true: np.ndarray, p: np.ndarray, target_spec: float = 0.90) -> float:
    """Find threshold achieving target specificity with highest sensitivity."""
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
    """
    Find the LOWEST threshold achieving precision >= target_ppv.
    If unattainable, falls back to max-F1.
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
    thr_t  = thr

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
    """Find threshold from control quantile to achieve target specificity."""
    pc = np.asarray(p_controls, dtype=float)
    pc = pc[np.isfinite(pc)]
    if pc.size == 0:
        return 0.5
    q = float(target_spec)
    try:
        thr = float(np.quantile(pc, q, method="higher"))
    except TypeError:
        thr = float(np.quantile(pc, q, interpolation="higher"))
    except Exception:
        thr = float(np.quantile(pc, q))
    if not np.isfinite(thr):
        thr = float(np.max(pc) + 1e-12)
    return thr


def binary_metrics_at_threshold(y_true: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, Any]:
    """Compute classification metrics at a specific threshold."""
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    y_hat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    prec = precision_score(y_true, y_hat, zero_division=0)
    rec  = recall_score(y_true, y_hat, zero_division=0)
    f1   = f1_score(y_true, y_hat, zero_division=0)
    spec = (tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    return {
        "threshold": float(thr),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "specificity": float(spec),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def top_risk_capture(y_true: np.ndarray, p: np.ndarray, frac: float = 0.01) -> Dict[str, Any]:
    """Analyze risk capture in top fraction of predictions."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)
    n = len(y)
    if n == 0:
        return {"frac": float(frac), "n_top": 0, "cases_in_top": 0, "controls_in_top": 0, "case_capture": np.nan}
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
    y_tr: np.ndarray,
    p_oof: np.ndarray,
    objective: str,
    fbeta: float = 1.0,
    fixed_spec: float = 0.90,
    fixed_ppv: float = 0.5,
) -> Tuple[str, float]:
    """
    Select threshold based on specified objective.

    Args:
        y_tr: True labels
        p_oof: Predicted probabilities
        objective: One of ['max_f1', 'max_fbeta', 'youden', 'fixed_spec', 'fixed_ppv']
        fbeta: Beta parameter for F-beta score
        fixed_spec: Target specificity (0-1)
        fixed_ppv: Target precision (0-1)

    Returns:
        Tuple of (objective_name, threshold)
    """
    obj = (objective or "max_f1").strip().lower()
    if obj == "max_f1":
        return ("max_f1", threshold_max_f1(y_tr, p_oof))
    if obj == "max_fbeta":
        return ("max_fbeta", threshold_max_fbeta(y_tr, p_oof, beta=fbeta))
    if obj == "youden":
        return ("youden", threshold_youden(y_tr, p_oof))
    if obj == "fixed_spec":
        return ("fixed_spec", threshold_for_specificity(y_tr, p_oof, target_spec=float(fixed_spec)))
    if obj == "fixed_ppv":
        return ("fixed_ppv", threshold_for_precision(y_tr, p_oof, target_ppv=float(fixed_ppv)))
    # fallback
    return ("max_f1", threshold_max_f1(y_tr, p_oof))


# =============================================================================
# 4. CALIBRATION
# =============================================================================
"""
Probability calibration utilities.

Implements:
- Prevalence adjustment for probability recalibration
- Calibration metrics (intercept, slope, ECE)
- PrevalenceAdjustedModel wrapper class
"""


def _prevalence(y: np.ndarray) -> float:
    """Compute prevalence (case proportion) from binary labels."""
    y = np.asarray(y).astype(int)
    return float(np.mean(y)) if y.size else np.nan


def _clip_prob(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Clip probabilities to [eps, 1-eps] to avoid log(0)."""
    p = np.asarray(p, dtype=float)
    return np.clip(p, eps, 1.0 - eps)


def _logit_np(p: np.ndarray) -> np.ndarray:
    """Compute logit (log-odds) from probabilities."""
    eps = 1e-9
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _inv_logit_np(z: np.ndarray) -> np.ndarray:
    """Compute inverse logit (sigmoid) from log-odds."""
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


def calibration_intercept_slope(y_true: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    """
    Compute calibration intercept and slope using logistic regression on logit scale.

    These indicate how well-calibrated probabilities are:
    - Intercept ~0 indicates probabilities match observed proportions
    - Slope ~1 indicates correct ordering/ranking
    """
    y = np.asarray(y_true).astype(int)
    p = _clip_prob(p)
    x = np.log(p / (1.0 - p))
    try:
        X = sm.add_constant(x)
        m = sm.Logit(y, X).fit(disp=0)
        return float(m.params[0]), float(m.params[1])
    except Exception:
        return (np.nan, np.nan)


def calib_intercept_metric(y: np.ndarray, p: np.ndarray) -> float:
    """Compute calibration intercept metric for bootstrap CIs."""
    a, _ = calibration_intercept_slope(y, p)
    return float(a)


def calib_slope_metric(y: np.ndarray, p: np.ndarray) -> float:
    """Compute calibration slope metric for bootstrap CIs."""
    _, b = calibration_intercept_slope(y, p)
    return float(b)


def adjust_probabilities_for_prevalence(
    probs: np.ndarray,
    sample_prev: float,
    target_prev: float,
) -> np.ndarray:
    """
    Apply intercept shift so that predicted probabilities reflect target prevalence.

    Uses the method: P(Y=1|X,prev_new) = sigmoid(logit(p) + logit(prev_new) - logit(prev_old))

    Args:
        probs: Raw probabilities from the classifier.
        sample_prev: Observed prevalence in the training sample.
        target_prev: Target prevalence (e.g., population-level).

    Returns:
        Adjusted probabilities reflecting target prevalence.
    """
    if not np.isfinite(sample_prev) or not np.isfinite(target_prev):
        return probs
    if not (0.0 < sample_prev < 1.0) or not (0.0 < target_prev < 1.0):
        return probs
    delta = np.log(target_prev / (1.0 - target_prev)) - np.log(sample_prev / (1.0 - sample_prev))
    logits = _logit_np(probs)
    return np.clip(_inv_logit_np(logits + delta), 1e-9, 1.0 - 1e-9)


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE is the weighted average of the absolute difference between
    predicted probability and expected frequency across bins.

    Reference:
        Naeini et al. (2015). Obtaining Well Calibrated Probabilities
        Using Bayesian Binning. AAAI.

    Args:
        y_true: True binary labels
        p: Predicted probabilities
        n_bins: Number of probability bins

    Returns:
        ECE value (lower is better, 0 = perfect calibration)
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (p >= bin_boundaries[i]) & (p < bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            avg_pred = np.mean(p[in_bin])
            avg_true = np.mean(y[in_bin])
            ece += np.abs(avg_pred - avg_true) * prop_in_bin

    return float(ece)


class PrevalenceAdjustedModel(BaseEstimator):
    """
    Wraps a fitted classifier and applies a prevalence shift to predict_proba outputs.

    This ensures the serialized artifact produces the same adjusted probabilities
    that were evaluated within this script.
    """

    def __init__(self, base_model, sample_prevalence: float, target_prevalence: float):
        self.base_model = base_model
        self.sample_prevalence = float(sample_prevalence)
        self.target_prevalence = float(target_prevalence)
        self.classes_ = getattr(base_model, "classes_", None)

    def _can_adjust(self) -> bool:
        return (
            np.isfinite(self.sample_prevalence)
            and np.isfinite(self.target_prevalence)
            and 0.0 < self.sample_prevalence < 1.0
            and 0.0 < self.target_prevalence < 1.0
        )

    def _adjust_binary_probs(self, probs: np.ndarray) -> np.ndarray:
        if probs.ndim != 2 or probs.shape[1] != 2:
            return probs
        pos = probs[:, 1]
        adj_pos = adjust_probabilities_for_prevalence(pos, self.sample_prevalence, self.target_prevalence)
        adj_pos = np.clip(adj_pos, 1e-9, 1.0 - 1e-9)
        adj = np.column_stack([1.0 - adj_pos, adj_pos])
        return adj

    def predict_proba(self, X):
        base_probs = self.base_model.predict_proba(X)
        base_probs = np.asarray(base_probs, dtype=float)
        if not self._can_adjust():
            return base_probs
        if base_probs.ndim == 1:
            adj = adjust_probabilities_for_prevalence(base_probs, self.sample_prevalence, self.target_prevalence)
            return np.column_stack([1.0 - adj, adj])
        if base_probs.shape[1] == 2:
            return self._adjust_binary_probs(base_probs)
        return base_probs

    def predict(self, X):
        probs = self.predict_proba(X)
        if probs.ndim == 1:
            probs = np.column_stack([1.0 - probs, probs])
        idx = np.argmax(probs, axis=1)
        if self.classes_ is not None and len(self.classes_) == probs.shape[1]:
            classes = np.asarray(self.classes_)
            return classes[idx]
        return idx

    def get_base_model(self):
        return self.base_model

    def __getattr__(self, name):
        return getattr(self.base_model, name)


# =============================================================================
# 5. PLOTTING UTILITIES
# =============================================================================

def _apply_plot_metadata(fig, meta_lines: Optional[Sequence[str]] = None) -> float:
    """
    Apply metadata text to bottom of figure, return required bottom margin.

    Args:
        fig: matplotlib figure object
        meta_lines: sequence of metadata strings to display

    Returns:
        Required bottom margin as fraction of figure height (0.0 to 1.0)
    """
    lines = [str(line) for line in (meta_lines or []) if line]
    if not lines:
        return 0.10  # Default minimum bottom margin

    # Position metadata at very bottom with fixed offset from edge
    fig.text(0.5, 0.005, "\n".join(lines), ha="center", va="bottom", fontsize=8, wrap=True)

    # Calculate required bottom margin: base + space per line
    # Each line ~0.015 height + small padding
    required_bottom = 0.10 + (0.018 * len(lines))
    return min(required_bottom, 0.30)  # Cap at 30% to avoid excessive margin


def _compute_recalibration(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Compute calibration intercept and slope via Logistic Regression on log-odds.

    This computes:
    - Intercept: calibration-in-the-large (should be ~0 for well-calibrated)
    - Slope: calibration slope (should be ~1 for well-calibrated)

    Reference:
        Van Calster et al. (2016). Calibration of risk prediction models.
        In: Medical Decision Making.

    Args:
        y_true: True binary labels (0/1)
        y_pred: Predicted probabilities

    Returns:
        Tuple of (intercept, slope)
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(float)
    mask = np.isfinite(p) & np.isfinite(y)
    y = y[mask]
    p = p[mask]

    eps = 1e-7
    p_clipped = np.clip(p, eps, 1 - eps)
    log_odds = np.log(p_clipped / (1 - p_clipped))

    if len(np.unique(y)) < 2:
        return np.nan, np.nan

    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(log_odds.reshape(-1, 1), y)
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


def _binned_logits(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
    bin_strategy: str = "quantile",
    min_bin_size: int = 30,
    merge_tail: bool = True
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute binned logits for calibration plot in logit space with proper binomial CIs.

    Logits are log-odds of probabilities: log(p/(1-p)). This creates a calibration curve
    with predicted logits on x-axis and observed logits on y-axis.

    Reference:
        Austin & Steyerberg (2019). The Integrated Calibration Index (ICI).
        Statistics in Medicine.

    Args:
        y_true: True binary labels (0/1)
        y_pred: Predicted probabilities
        n_bins: Number of bins for grouping predictions
        bin_strategy: 'uniform' (equal width) or 'quantile' (equal size)
        min_bin_size: Minimum number of samples per bin
        merge_tail: If True, merge small bins with adjacent bins

    Returns:
        Tuple of (xs, ys, ys_lo, ys_hi, sizes) where:
        - xs: predicted log-odds (bin centers)
        - ys: observed log-odds (empirical event rates)
        - ys_lo: lower CI bound (log-odds)
        - ys_hi: upper CI bound (log-odds)
        - sizes: bin sizes
        Returns (None, None, None, None, None) if insufficient data
    """
    from statsmodels.stats.proportion import proportion_confint

    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if len(y) == 0:
        return None, None, None, None, None

    # Create initial bins
    if bin_strategy == "quantile":
        quantiles = np.linspace(0, 100, int(n_bins) + 1)
        bins = np.percentile(p, quantiles)
        bins = np.unique(bins)
        if len(bins) < 3:
            bins = np.linspace(0, 1, int(n_bins) + 1)
    else:
        bins = np.linspace(0, 1, int(n_bins) + 1)

    eps = 1e-7
    idx = np.digitize(p, bins) - 1
    idx = np.clip(idx, 0, len(bins) - 2)

    # First pass: compute bin sizes and merge small bins
    bin_masks = []
    for i in range(len(bins) - 1):
        m = idx == i
        if np.any(m):
            bin_masks.append(m)

    # Merge bins at tail if too small
    if merge_tail and len(bin_masks) > 0:
        merged_masks = []
        i = 0
        while i < len(bin_masks):
            current_mask = bin_masks[i]
            current_size = current_mask.sum()

            # If current bin is too small and not the last, try to merge with next
            while current_size < min_bin_size and i < len(bin_masks) - 1:
                i += 1
                current_mask = current_mask | bin_masks[i]
                current_size = current_mask.sum()

            merged_masks.append(current_mask)
            i += 1

        bin_masks = merged_masks

    # Second pass: compute statistics for merged bins
    xs, ys, ys_lo, ys_hi, sizes = [], [], [], [], []

    for m in bin_masks:
        y_bin = y[m]
        p_bin = p[m]
        n_bin = int(m.sum())
        n_events = int(y_bin.sum())

        # X-axis: mean predicted probability (no CI needed)
        pred_mean = float(np.mean(p_bin))
        pred_mean = float(np.clip(pred_mean, eps, 1 - eps))
        xs.append(np.log(pred_mean / (1 - pred_mean)))

        # Y-axis: observed event rate with Wilson binomial CI
        if n_events == 0:
            # Handle zero events: use continuity correction
            obs_rate = 0.5 / n_bin
            ci_lo, ci_hi = proportion_confint(0, n_bin, method='wilson')
        elif n_events == n_bin:
            # Handle all events: use continuity correction
            obs_rate = (n_bin - 0.5) / n_bin
            ci_lo, ci_hi = proportion_confint(n_bin, n_bin, method='wilson')
        else:
            obs_rate = n_events / n_bin
            ci_lo, ci_hi = proportion_confint(n_events, n_bin, method='wilson')

        # Clip to avoid log(0) and convert to log-odds
        obs_rate = float(np.clip(obs_rate, eps, 1 - eps))
        ci_lo = float(np.clip(ci_lo, eps, 1 - eps))
        ci_hi = float(np.clip(ci_hi, eps, 1 - eps))

        ys.append(np.log(obs_rate / (1 - obs_rate)))
        ys_lo.append(np.log(ci_lo / (1 - ci_lo)))
        ys_hi.append(np.log(ci_hi / (1 - ci_hi)))
        sizes.append(n_bin)

    if len(xs) < 2:
        return None, None, None, None, None

    # Sort by predicted log-odds
    order = np.argsort(xs)
    return (
        np.array(xs, dtype=float)[order],
        np.array(ys, dtype=float)[order],
        np.array(ys_lo, dtype=float)[order],
        np.array(ys_hi, dtype=float)[order],
        np.array(sizes, dtype=int)[order],
    )


# =============================================================================
# 4. DECISION CURVE ANALYSIS (DCA)
# =============================================================================

def _dca_thresholds(min_thr: float, max_thr: float, step: float) -> np.ndarray:
    """
    Generate array of threshold values for DCA analysis.

    Args:
        min_thr: Minimum threshold (will be clamped to 0.0001)
        max_thr: Maximum threshold (will be clamped to 0.999)
        step: Step size between thresholds

    Returns:
        Array of threshold values
    """
    min_thr = max(1e-4, float(min_thr))
    max_thr = min(0.999, float(max_thr))
    step = max(1e-4, float(step))
    if min_thr >= max_thr:
        return np.array([min_thr, max_thr])
    n = int(np.floor((max_thr - min_thr) / step)) + 1
    return np.linspace(min_thr, max_thr, n)


def _parse_dca_report_points(s: str) -> List[float]:
    """
    Parse comma-separated string of DCA report thresholds.

    Args:
        s: Comma-separated string like "0.005,0.01,0.02,0.05"

    Returns:
        List of float thresholds between 0 and 1
    """
    pts = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = float(tok)
            if 0.0 < v < 1.0:
                pts.append(v)
        except Exception:
            continue
    return pts


def decision_curve_analysis(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    prevalence_adjustment: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compute Decision Curve Analysis (DCA) for clinical utility assessment.

    DCA evaluates the net benefit of using a prediction model vs. treating all
    or treating none, across a range of threshold probabilities.

    Net Benefit = (TP/n) - (FP/n) * (threshold / (1 - threshold))

    Reference:
        Vickers AJ, Elkin EB (2006). Decision curve analysis: a novel method
        for evaluating prediction models. Med Decis Making.

    Args:
        y_true: True binary labels (0/1)
        y_pred_prob: Predicted probabilities
        thresholds: Array of threshold probabilities (default: 0.001 to 0.10)
        prevalence_adjustment: If provided, adjust for different prevalence

    Returns:
        DataFrame with columns: threshold, net_benefit_model, net_benefit_all,
        net_benefit_none, relative_utility
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred_prob).astype(float)
    n = len(y)

    if n == 0:
        return pd.DataFrame()

    if thresholds is None:
        thresholds = _dca_thresholds(0.001, 0.10, 0.001)

    prevalence = np.mean(y)
    if prevalence_adjustment is not None:
        prevalence = prevalence_adjustment

    results = []
    for t in thresholds:
        if t <= 0 or t >= 1:
            continue

        # Classify based on threshold
        y_pred_binary = (p >= t).astype(int)

        # Calculate TP, FP, TN, FN
        tp = np.sum((y_pred_binary == 1) & (y == 1))
        fp = np.sum((y_pred_binary == 1) & (y == 0))
        tn = np.sum((y_pred_binary == 0) & (y == 0))
        fn = np.sum((y_pred_binary == 0) & (y == 1))

        # Net benefit of the model
        # NB = (TP/n) - (FP/n) * (t / (1-t))
        odds = t / (1 - t)
        nb_model = (tp / n) - (fp / n) * odds

        # Net benefit of treating all (always predict positive)
        # NB_all = prevalence - (1-prevalence) * (t / (1-t))
        nb_all = prevalence - (1 - prevalence) * odds

        # Net benefit of treating none is 0
        nb_none = 0.0

        # Relative utility: how much of the maximum benefit is captured
        # relative_utility = NB_model / max(NB_all, 0) when NB_all > 0
        if nb_all > 0:
            relative_utility = nb_model / nb_all
        else:
            relative_utility = np.nan

        results.append({
            "threshold": t,
            "threshold_pct": t * 100,
            "net_benefit_model": nb_model,
            "net_benefit_all": nb_all,
            "net_benefit_none": nb_none,
            "relative_utility": relative_utility,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "n_treat": int(tp + fp),
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        })

    return pd.DataFrame(results)


def compute_dca_summary(dca_df: pd.DataFrame, report_points: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Compute summary statistics from DCA results.

    Args:
        dca_df: DataFrame from decision_curve_analysis()
        report_points: Key thresholds for summary (e.g., [0.005, 0.01, 0.02, 0.05])

    Returns:
        Dictionary with DCA summary metrics
    """
    if dca_df.empty:
        return {"dca_computed": False}

    summary = {
        "dca_computed": True,
        "n_thresholds": len(dca_df),
        "threshold_range": f"{dca_df['threshold'].min():.3f}-{dca_df['threshold'].max():.3f}",
    }

    # Find where model beats "treat all"
    model_beats_all = dca_df[dca_df["net_benefit_model"] > dca_df["net_benefit_all"]]
    if len(model_beats_all) > 0:
        summary["model_beats_all_from"] = float(model_beats_all["threshold"].min())
        summary["model_beats_all_to"] = float(model_beats_all["threshold"].max())
        summary["model_beats_all_range"] = f"{summary['model_beats_all_from']:.3f}-{summary['model_beats_all_to']:.3f}"
    else:
        summary["model_beats_all_from"] = np.nan
        summary["model_beats_all_to"] = np.nan
        summary["model_beats_all_range"] = "Never"

    # Find where model beats "treat none" (has positive net benefit)
    model_beats_none = dca_df[dca_df["net_benefit_model"] > 0]
    if len(model_beats_none) > 0:
        summary["model_beats_none_from"] = float(model_beats_none["threshold"].min())
        summary["model_beats_none_to"] = float(model_beats_none["threshold"].max())
    else:
        summary["model_beats_none_from"] = np.nan
        summary["model_beats_none_to"] = np.nan

    # Area under the net benefit curve (approximation of overall utility)
    thresholds = dca_df["threshold"].values
    nb_model = dca_df["net_benefit_model"].values
    nb_all = dca_df["net_benefit_all"].values

    # Compute integrated net benefit (area under curve)
    if len(thresholds) > 1:
        summary["integrated_nb_model"] = float(np.trapezoid(nb_model, thresholds))
        summary["integrated_nb_all"] = float(np.trapezoid(nb_all, thresholds))

        # Net benefit improvement over treat-all
        nb_improvement = nb_model - nb_all
        summary["integrated_nb_improvement"] = float(np.trapezoid(nb_improvement, thresholds))

    # Key clinical thresholds
    report_points = report_points or [0.005, 0.01, 0.02, 0.05]
    for key_t in report_points:
        row = dca_df[np.isclose(dca_df["threshold"], key_t, atol=0.001)]
        if len(row) > 0:
            r = row.iloc[0]
            summary[f"nb_model_at_{key_t:.1%}".replace(".", "p")] = float(r["net_benefit_model"])
            summary[f"nb_all_at_{key_t:.1%}".replace(".", "p")] = float(r["net_benefit_all"])

    return summary


def save_dca_results(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    out_dir: str,
    prefix: str = "",
    thresholds: Optional[np.ndarray] = None,
    report_points: Optional[List[float]] = None,
    prevalence_adjustment: Optional[float] = None,
    meta_lines: Optional[Sequence[str]] = None,
    plot_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute and save DCA results to files.

    Args:
        y_true: True labels
        y_pred_prob: Predicted probabilities
        out_dir: Output directory
        prefix: Filename prefix
        thresholds: Array of threshold values (default: 0.001 to 0.10)
        report_points: Key thresholds for summary (e.g., [0.005, 0.01, 0.02, 0.05])
        prevalence_adjustment: Optional prevalence for adjustment
        meta_lines: Optional metadata lines to annotate plots
        plot_dir: Optional separate directory for plots (default: same as out_dir)

    Returns:
        DCA summary dictionary
    """
    os.makedirs(out_dir, exist_ok=True)

    # Compute DCA
    dca_df = decision_curve_analysis(
        y_true,
        y_pred_prob,
        thresholds=thresholds,
        prevalence_adjustment=prevalence_adjustment,
    )

    if dca_df.empty:
        return {"dca_computed": False, "error": "Empty DCA results"}

    # Save full DCA table
    csv_path = os.path.join(out_dir, f"{prefix}dca_curve.csv")
    dca_df.to_csv(csv_path, index=False)

    # Compute summary
    summary = compute_dca_summary(dca_df, report_points=report_points)
    if prevalence_adjustment is not None:
        summary["prevalence_adjustment"] = float(prevalence_adjustment)
    summary["dca_csv_path"] = csv_path

    # Save summary
    json_path = os.path.join(out_dir, f"{prefix}dca_summary.json")
    with open(json_path, "w") as f:
        # Convert any numpy types for JSON serialization
        json_summary = {}
        for k, v in summary.items():
            if isinstance(v, (np.integer, np.floating)):
                json_summary[k] = float(v) if np.isfinite(v) else None
            else:
                json_summary[k] = v
        json.dump(json_summary, f, indent=2)

    summary["dca_json_path"] = json_path

    # Try to generate plot
    try:
        plot_base = plot_dir or out_dir
        os.makedirs(plot_base, exist_ok=True)
        plot_path = os.path.join(plot_base, f"{prefix}dca_plot.png")
        _plot_dca(dca_df, plot_path, meta_lines=meta_lines)
        summary["dca_plot_path"] = plot_path
    except Exception as e:
        summary["dca_plot_error"] = str(e)

    return summary


def _plot_dca(dca_df: pd.DataFrame, out_path: str, meta_lines: Optional[Sequence[str]] = None) -> None:
    """
    Generate DCA plot.

    Args:
        dca_df: DataFrame from decision_curve_analysis()
        out_path: Path to save plot
        meta_lines: Optional metadata lines to display at bottom
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    thresholds = dca_df["threshold_pct"].values
    nb_model = dca_df["net_benefit_model"].values
    nb_all = dca_df["net_benefit_all"].values
    nb_none = dca_df["net_benefit_none"].values

    ax.plot(thresholds, nb_model, color="steelblue", linestyle="-", linewidth=2, label="Model")
    ax.plot(thresholds, nb_all, "r--", linewidth=1.5, label="Treat All")
    ax.plot(thresholds, nb_none, "k:", linewidth=1.5, label="Treat None")

    # Shade region where model is better than alternatives
    ax.fill_between(
        thresholds,
        np.maximum(nb_all, nb_none),
        nb_model,
        where=(nb_model > np.maximum(nb_all, nb_none)),
        alpha=0.2,
        color="steelblue",
        label="Model Benefit",
    )

    ax.set_xlabel("Threshold Probability (%)", fontsize=12)
    ax.set_ylabel("Net Benefit", fontsize=12)
    ax.set_title("Decision Curve Analysis", fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Set reasonable y-axis limits
    y_min = min(nb_model.min(), nb_all.min(), -0.01)
    y_max = max(nb_model.max(), nb_all.max(), 0.01) * 1.1
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(thresholds.min(), thresholds.max())

    bottom_margin = _apply_plot_metadata(fig, meta_lines)
    plt.subplots_adjust(left=0.15, right=0.9, top=0.8, bottom=bottom_margin)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.8)
    plt.close()


def _plot_dca_curve(y_true, y_pred, out_path, title, subtitle="", max_pt=0.20, step=0.005, split_ids=None, meta_lines=None):
    """
    Generate DCA plot from raw predictions with multi-split averaging.

    Computes Decision Curve Analysis across threshold range, with optional
    multi-split averaging and confidence bands (following _plot_roc_curve pattern).

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        out_path: Path to save plot
        title: Plot title
        subtitle: Optional subtitle
        max_pt: Maximum threshold (default: 0.20)
        step: Threshold step size (default: 0.005)
        split_ids: Optional array of split IDs for multi-split averaging
        meta_lines: Optional metadata lines to display at bottom
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[PLOT] DCA plot failed to import dependencies: {e}")
        return

    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(float)
    mask = np.isfinite(p) & np.isfinite(y)
    y = y[mask]
    p = p[mask]

    if len(y) == 0:
        return

    # Generate threshold array
    thresholds = np.arange(0.0005, max_pt + step, step)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Multi-split handling
    if split_ids is not None:
        split_ids = np.asarray(split_ids)[mask]
        unique_splits = pd.Series(split_ids).dropna().unique().tolist()
    else:
        unique_splits = []

    if len(unique_splits) > 1:
        # Compute DCA per split and average
        nb_model_curves = []
        nb_all_curves = []
        nb_none_curves = []

        for sid in unique_splits:
            m = split_ids == sid
            y_s = y[m]
            p_s = p[m]
            if len(np.unique(y_s)) < 2 or len(y_s) < 2:
                continue

            dca_df = decision_curve_analysis(y_s, p_s, thresholds=thresholds)
            if not dca_df.empty:
                nb_model_curves.append(dca_df["net_benefit_model"].values)
                nb_all_curves.append(dca_df["net_benefit_all"].values)
                nb_none_curves.append(dca_df["net_benefit_none"].values)

        if nb_model_curves:
            nb_model_curves = np.vstack(nb_model_curves)
            nb_all_curves = np.vstack(nb_all_curves)
            nb_none_curves = np.vstack(nb_none_curves)

            nb_model_mean = np.mean(nb_model_curves, axis=0)
            nb_model_sd = np.std(nb_model_curves, axis=0)
            nb_model_lo = np.percentile(nb_model_curves, 2.5, axis=0)
            nb_model_hi = np.percentile(nb_model_curves, 97.5, axis=0)

            nb_all_mean = np.mean(nb_all_curves, axis=0)
            nb_none_mean = np.mean(nb_none_curves, axis=0)

            thr = thresholds[:len(nb_model_mean)]

            # Plot with confidence bands
            ax.fill_between(thr, nb_model_lo, nb_model_hi, color="steelblue", alpha=0.15, label="95% CI")
            ax.fill_between(thr, np.maximum(0, nb_model_mean - nb_model_sd), np.minimum(1, nb_model_mean + nb_model_sd),
                            color="steelblue", alpha=0.30, label="±1 SD")
            ax.plot(thr, nb_model_mean, color="steelblue", linestyle="-", linewidth=2, label="Model")
            ax.plot(thr, nb_all_mean, "r--", linewidth=1.5, label="Treat All")
            ax.plot(thr, nb_none_mean, "k:", linewidth=1.5, label="Treat None")

            # Shade region where model is better
            ax.fill_between(thr, np.maximum(nb_all_mean, nb_none_mean), nb_model_mean,
                            where=(nb_model_mean > np.maximum(nb_all_mean, nb_none_mean)),
                            alpha=0.2, color="steelblue", label="Model Benefit")
        else:
            # Fallback to single curve if all splits fail
            dca_df = decision_curve_analysis(y, p, thresholds=thresholds)
            if not dca_df.empty:
                thr = dca_df["threshold"].values
                ax.plot(thr, dca_df["net_benefit_model"].values, color="steelblue", linestyle="-", linewidth=2, label="Model")
                ax.plot(thr, dca_df["net_benefit_all"].values, "r--", linewidth=1.5, label="Treat All")
                ax.plot(thr, dca_df["net_benefit_none"].values, "k:", linewidth=1.5, label="Treat None")
                ax.fill_between(thr, np.maximum(dca_df["net_benefit_all"].values, dca_df["net_benefit_none"].values),
                                dca_df["net_benefit_model"].values,
                                where=(dca_df["net_benefit_model"].values > np.maximum(dca_df["net_benefit_all"].values, dca_df["net_benefit_none"].values)),
                                alpha=0.2, color="steelblue", label="Model Benefit")
    else:
        # Single split or no split_ids
        dca_df = decision_curve_analysis(y, p, thresholds=thresholds)
        if not dca_df.empty:
            thr = dca_df["threshold"].values
            ax.plot(thr, dca_df["net_benefit_model"].values, color="steelblue", linestyle="-", linewidth=2, label="Model")
            ax.plot(thr, dca_df["net_benefit_all"].values, "r--", linewidth=1.5, label="Treat All")
            ax.plot(thr, dca_df["net_benefit_none"].values, "k:", linewidth=1.5, label="Treat None")
            ax.fill_between(thr, np.maximum(dca_df["net_benefit_all"].values, dca_df["net_benefit_none"].values),
                            dca_df["net_benefit_model"].values,
                            where=(dca_df["net_benefit_model"].values > np.maximum(dca_df["net_benefit_all"].values, dca_df["net_benefit_none"].values)),
                            alpha=0.2, color="steelblue", label="Model Benefit")

    # Compute y-range to include all curves (treat all, treat none, model)
    y_min = 0
    y_max = 0

    if len(unique_splits) > 1 and nb_model_curves:
        y_min = min(y_min, np.nanmin(nb_model_lo), np.nanmin(nb_all_mean), np.nanmin(nb_none_mean))
        y_max = max(y_max, np.nanmax(nb_model_hi), np.nanmax(nb_all_mean), np.nanmax(nb_none_mean))
    elif not dca_df.empty:
        y_min = min(y_min, dca_df["net_benefit_model"].min(),
                    dca_df["net_benefit_all"].min(), dca_df["net_benefit_none"].min())
        y_max = max(y_max, dca_df["net_benefit_model"].max(),
                    dca_df["net_benefit_all"].max(), dca_df["net_benefit_none"].max())

    # Add 10% padding
    y_range = y_max - y_min
    if y_range > 0:
        y_min_padded = y_min - 0.1 * y_range
        y_max_padded = y_max + 0.1 * y_range
    else:
        y_min_padded = -0.1
        y_max_padded = 0.1
    ax.set_ylim([y_min_padded, y_max_padded])

    ax.set_xlabel("Threshold Probability", fontsize=12)
    ax.set_ylabel("Net Benefit", fontsize=12)
    if subtitle:
        ax.set_title(f"{title}\n{subtitle}", fontsize=12)
    else:
        ax.set_title(title, fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    bottom_margin = _apply_plot_metadata(fig, meta_lines)
    plt.subplots_adjust(left=0.15, right=0.9, top=0.8, bottom=bottom_margin)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.8)
    plt.close()


# =============================================================================
# DCA UTILITY FUNCTIONS
# =============================================================================

def find_dca_zero_crossing(dca_curve_path: str) -> Optional[float]:
    """
    Find DCA zero-crossing threshold from DCA curve CSV.

    Loads a DCA curve CSV file and identifies where the model net benefit
    line crosses zero (net_benefit_model = 0). Uses linear interpolation
    between points to find the crossing.

    If no zero-crossing exists (model benefit always positive/negative),
    returns the threshold nearest to zero-crossing, or None if impossible.

    Args:
        dca_curve_path: Path to DCA curve CSV file with columns:
            threshold, net_benefit_model, net_benefit_all, net_benefit_none

    Returns:
        Threshold value where zero-crossing occurs, or nearest to zero if
        no crossing found. Returns None if:
        - File doesn't exist
        - File can't be parsed
        - Data is invalid

    Example:
        >>> dca_threshold = find_dca_zero_crossing("dca_curve.csv")
        >>> if dca_threshold:
        ...     print(f"DCA optimal threshold: {dca_threshold:.4f}")
    """
    if not os.path.exists(dca_curve_path):
        return None

    try:
        dca_df = pd.read_csv(dca_curve_path)
        if "net_benefit_model" not in dca_df.columns:
            return None

        # Get net benefit and threshold values
        nb = dca_df["net_benefit_model"].values
        thr = dca_df["threshold"].values

        if len(nb) == 0 or len(thr) == 0:
            return None

        # Find sign changes (where curve crosses zero)
        sign_changes = np.where(np.diff(np.sign(nb)))[0]

        if len(sign_changes) > 0:
            # Use first crossing (where model benefit becomes positive)
            idx = sign_changes[0]
            if idx + 1 < len(thr):
                # Linear interpolation between crossing points
                thr_lo, thr_hi = thr[idx], thr[idx + 1]
                nb_lo, nb_hi = nb[idx], nb[idx + 1]

                # Interpolate to find zero
                if abs(nb_hi - nb_lo) > 1e-10:
                    crossing = thr_lo + (0 - nb_lo) * (thr_hi - thr_lo) / (nb_hi - nb_lo)
                    return float(crossing)
        else:
            # No zero-crossing in data range
            # Return threshold closest to zero net benefit as fallback
            abs_nb = np.abs(nb)
            closest_idx = np.nanargmin(abs_nb)
            closest_threshold = thr[closest_idx]
            closest_nb = nb[closest_idx]

            # Only use this fallback if closest point is reasonably close to zero
            if abs(closest_nb) < 0.05:  # Within 5% of zero is acceptable
                return float(closest_threshold)

    except Exception:
        pass

    return None


def load_spec95_threshold(metrics_csv_path: str, split: str = "test") -> Optional[float]:
    """
    Load 95% specificity threshold from metrics CSV.

    Extracts the threshold value that achieves 95% specificity on controls,
    as computed during training (stored in test_metrics.csv).

    Args:
        metrics_csv_path: Path to test_metrics.csv with columns including:
            thr_train_oof_spec95_ctrl (or thr_{split}_oof_spec95_ctrl)
        split: Which split to load ("test", "val", "train"), default "test"

    Returns:
        Threshold value, or None if:
        - File doesn't exist
        - Column not found
        - File can't be parsed

    Example:
        >>> spec95_thr = load_spec95_threshold("test_metrics.csv")
        >>> if spec95_thr:
        ...     print(f"95% specificity threshold: {spec95_thr:.3f}")
    """
    if not os.path.exists(metrics_csv_path):
        return None

    try:
        df = pd.read_csv(metrics_csv_path)

        # Column name format: thr_train_oof_spec95_ctrl or thr_{split}_oof_spec95_ctrl
        col_name = f"thr_train_oof_spec95_ctrl"

        if col_name in df.columns and len(df) > 0:
            val = df[col_name].iloc[0]
            if pd.notna(val) and np.isfinite(val):
                return float(val)

    except Exception:
        return None

    return None


# =============================================================================
# 7. CLINICAL PLOTTING
# =============================================================================
"""
Clinical ML visualization functions.

Unified plotting functions for:
- ROC curves (single and multi-split aggregation)
- Precision-Recall curves
- Calibration curves (probability-space + log-odds, 3-panel or 5-panel)
- Risk distribution histograms
- DCA curves

Consolidates plotting from both celiacML_faith.py and postprocess_compare.py
with postprocess_compare.py versions as canonical (more feature-complete).
"""

def _plot_roc_curve(y_true, y_pred, out_path, title, subtitle="", split_ids=None, meta_lines=None,
                   youden_threshold=None, alpha_threshold=None, metrics_at_thresholds=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
    except Exception as e:
        print(f"[PLOT] ROC plot failed to import dependencies: {e}")
        return

    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(float)
    mask = np.isfinite(p) & np.isfinite(y)
    y = y[mask]
    p = p[mask]
    if len(y) == 0:
        return

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6)

    if split_ids is not None:
        split_ids = np.asarray(split_ids)[mask]
        unique_splits = pd.Series(split_ids).dropna().unique().tolist()
    else:
        unique_splits = []

    if len(unique_splits) > 1:
        base_fpr = np.linspace(0, 1, 120)
        tprs = []
        aucs = []
        for sid in unique_splits:
            m = split_ids == sid
            y_s = y[m]
            p_s = p[m]
            if len(np.unique(y_s)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_s, p_s)
            tpr_i = np.interp(base_fpr, fpr, tpr)
            tpr_i[0] = 0.0
            tprs.append(tpr_i)
            aucs.append(roc_auc_score(y_s, p_s))

        if tprs:
            tprs = np.vstack(tprs)
            tpr_mean = np.mean(tprs, axis=0)
            tpr_sd = np.std(tprs, axis=0)
            tpr_lo = np.nanpercentile(tprs, 2.5, axis=0)
            tpr_hi = np.nanpercentile(tprs, 97.5, axis=0)
            auc_mean = float(np.mean(aucs))
            auc_sd = float(np.std(aucs))

            ax.fill_between(base_fpr, tpr_lo, tpr_hi, color="steelblue", alpha=0.15, label="95% CI")
            ax.fill_between(base_fpr, np.maximum(0, tpr_mean - tpr_sd), np.minimum(1, tpr_mean + tpr_sd),
                            color="steelblue", alpha=0.30, label="±1 SD")
            ax.plot(base_fpr, tpr_mean, color="steelblue", linewidth=2,
                    label=f"AUC = {auc_mean:.3f} ± {auc_sd:.3f}")
        else:
            fpr, tpr, _ = roc_curve(y, p)
            auc = roc_auc_score(y, p)
            ax.plot(fpr, tpr, color="steelblue", linewidth=2, label=f"AUC = {auc:.3f}")
    else:
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ax.plot(fpr, tpr, color="steelblue", linewidth=2, label=f"AUC = {auc:.3f}")

    # Add Youden and alpha threshold points if available
    if metrics_at_thresholds is not None:
        if youden_threshold is not None and 'youden' in metrics_at_thresholds:
            m = metrics_at_thresholds['youden']
            fpr_youden = m.get('fpr', None)
            tpr_youden = m.get('tpr', None)
            if fpr_youden is not None and tpr_youden is not None and 0 <= fpr_youden <= 1 and 0 <= tpr_youden <= 1:
                ax.scatter([fpr_youden], [tpr_youden], s=100, color='green',
                          marker='o', edgecolors='darkgreen', linewidths=2,
                          label='Youden', zorder=5)

        if alpha_threshold is not None and 'alpha' in metrics_at_thresholds:
            m = metrics_at_thresholds['alpha']
            fpr_alpha = m.get('fpr', None)
            tpr_alpha = m.get('tpr', None)
            if fpr_alpha is not None and tpr_alpha is not None and 0 <= fpr_alpha <= 1 and 0 <= tpr_alpha <= 1:
                ax.scatter([fpr_alpha], [tpr_alpha], s=100, color='purple',
                          marker='D', edgecolors='darkviolet', linewidths=2,
                          label='Alpha threshold', zorder=5)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    if subtitle:
        ax.set_title(f"{title}\n{subtitle}", fontsize=12)
    else:
        ax.set_title(title, fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.2)
    bottom_margin = _apply_plot_metadata(fig, meta_lines)
    plt.subplots_adjust(left=0.15, right=0.9, top=0.8, bottom=bottom_margin)
    plt.savefig(out_path, dpi=150, pad_inches=0.1)
    plt.close()


def _plot_pr_curve(y_true, y_pred, out_path, title, subtitle="", split_ids=None, meta_lines=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve
    except Exception as e:
        print(f"[PLOT] PR plot failed to import dependencies: {e}")
        return

    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(float)
    mask = np.isfinite(p) & np.isfinite(y)
    y = y[mask]
    p = p[mask]
    if len(y) == 0:
        return

    baseline = np.mean(y)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.axhline(y=baseline, color="k", linestyle="--", linewidth=1, alpha=0.6,
               label=f"Prevalence = {baseline:.4f}")

    if split_ids is not None:
        split_ids = np.asarray(split_ids)[mask]
        unique_splits = pd.Series(split_ids).dropna().unique().tolist()
    else:
        unique_splits = []

    if len(unique_splits) > 1:
        base_recall = np.linspace(0, 1, 120)
        precisions = []
        aps = []
        for sid in unique_splits:
            m = split_ids == sid
            y_s = y[m]
            p_s = p[m]
            if len(np.unique(y_s)) < 2:
                continue
            precision, recall, _ = precision_recall_curve(y_s, p_s)
            precision_i = np.interp(base_recall, recall[::-1], precision[::-1])
            precisions.append(precision_i)
            aps.append(average_precision_score(y_s, p_s))

        if precisions:
            precisions = np.vstack(precisions)
            prec_mean = np.mean(precisions, axis=0)
            prec_sd = np.std(precisions, axis=0)
            prec_lo = np.nanpercentile(precisions, 2.5, axis=0)
            prec_hi = np.nanpercentile(precisions, 97.5, axis=0)
            ap_mean = float(np.mean(aps))
            ap_sd = float(np.std(aps))

            ax.fill_between(base_recall, np.clip(prec_lo, 0, 1), np.clip(prec_hi, 0, 1),
                            color="steelblue", alpha=0.15, label="95% CI")
            ax.fill_between(base_recall, np.clip(prec_mean - prec_sd, 0, 1), np.clip(prec_mean + prec_sd, 0, 1),
                            color="steelblue", alpha=0.30, label="±1 SD")
            ax.plot(base_recall, prec_mean, color="steelblue", linewidth=2,
                    label=f"AP = {ap_mean:.3f} ± {ap_sd:.3f}")
        else:
            precision, recall, _ = precision_recall_curve(y, p)
            ap = average_precision_score(y, p)
            ax.plot(recall, precision, color="steelblue", linewidth=2, label=f"AP = {ap:.3f}")
    else:
        precision, recall, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ax.plot(recall, precision, color="steelblue", linewidth=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    if subtitle:
        ax.set_title(f"{title}\n{subtitle}", fontsize=12)
    else:
        ax.set_title(title, fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.2)
    bottom_margin = _apply_plot_metadata(fig, meta_lines)
    plt.subplots_adjust(left=0.15, right=0.9, top=0.8, bottom=bottom_margin)
    plt.savefig(out_path, dpi=150, pad_inches=0.1)
    plt.close()


def _plot_prob_calibration_panel(ax, y, p, bins, bin_centers, actual_n_bins, bin_strategy,
                                   split_ids=None, unique_splits=None, panel_title="", variable_sizes=True):
    """Helper to plot a single probability-space calibration panel.

    Args:
        ax: Matplotlib axis to plot on
        y: True labels
        p: Predicted probabilities
        bins: Bin edges
        bin_centers: Center points of bins
        actual_n_bins: Number of bins
        bin_strategy: 'uniform' or 'quantile'
        split_ids: Optional split identifiers
        unique_splits: List of unique split IDs
        panel_title: Title for this panel
        variable_sizes: If True, circle sizes vary with bin sample counts; if False, all circles same size
    """
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration", alpha=0.7)

    if unique_splits is not None and len(unique_splits) > 1:
        curves = []
        counts_all = []
        for sid in unique_splits:
            m_split = split_ids == sid
            y_s = y[m_split]
            p_s = p[m_split]
            bin_idx = np.digitize(p_s, bins) - 1
            bin_idx = np.clip(bin_idx, 0, actual_n_bins - 1)
            obs = []
            counts = []
            for i in range(actual_n_bins):
                m = bin_idx == i
                obs.append(np.nan if m.sum() == 0 else y_s[m].mean())
                counts.append(int(m.sum()))
            curves.append(obs)
            counts_all.append(counts)
        curves = np.array(curves, dtype=float)
        counts_all = np.array(counts_all, dtype=float)
        obs_mean = np.nanmean(curves, axis=0)
        obs_sd = np.nanstd(curves, axis=0)
        obs_lo = np.nanpercentile(curves, 2.5, axis=0)
        obs_hi = np.nanpercentile(curves, 97.5, axis=0)
        mean_counts = np.nanmean(counts_all, axis=0)
        sum_counts = np.nansum(counts_all, axis=0)

        ax.fill_between(bin_centers, np.clip(obs_lo, 0, 1), np.clip(obs_hi, 0, 1),
                        color="steelblue", alpha=0.15, label="95% CI")
        ax.fill_between(bin_centers, np.clip(obs_mean - obs_sd, 0, 1), np.clip(obs_mean + obs_sd, 0, 1),
                        color="steelblue", alpha=0.30, label="±1 SD")

        valid = ~np.isnan(obs_mean) & (sum_counts > 0)
        # Only use variable marker sizes for uniform binning; quantile binning gets fixed sizes
        if bin_strategy == "quantile":
            scatter_sizes = 50  # Fixed size for quantile binning
        elif variable_sizes:
            # Use variable marker sizes based on aggregate counts (like celiacML_faith.py)
            scatter_sizes = np.clip(sum_counts[valid] * 1, 5, 300)
        else:
            # Fixed marker size for all points
            scatter_sizes = 30
        ax.scatter(bin_centers[valid], obs_mean[valid], s=scatter_sizes, color="steelblue",
                   alpha=0.6, edgecolors="darkblue", linewidths=0.5)
        ax.plot(bin_centers, obs_mean, color="steelblue", linewidth=2, alpha=0.5,
                label=f"Mean (n={len(curves)} splits)")
    else:
        bin_idx = np.digitize(p, bins) - 1
        bin_idx = np.clip(bin_idx, 0, actual_n_bins - 1)
        obs = []
        pred_means = []
        sizes = []
        for i in range(actual_n_bins):
            m = bin_idx == i
            if m.sum() == 0:
                obs.append(np.nan)
                pred_means.append(np.nan)
                sizes.append(0)
            else:
                obs.append(y[m].mean())
                pred_means.append(p[m].mean())
                sizes.append(int(m.sum()))
        obs = np.array(obs)
        pred_means = np.array(pred_means)
        sizes = np.array(sizes)
        valid = ~np.isnan(obs)

        # Only use variable marker sizes for uniform binning; quantile binning gets fixed sizes
        if bin_strategy == "quantile":
            scatter_sizes = 60  # Fixed size for quantile binning
        elif variable_sizes:
            scatter_sizes = np.clip(sizes[valid] * 3, 30, 500)
        else:
            scatter_sizes = 60
        ax.scatter(pred_means[valid], obs[valid], s=scatter_sizes, color="steelblue",
                   alpha=0.6, edgecolors="darkblue", linewidths=0.5)
        ax.plot(pred_means[valid], obs[valid], color="steelblue", linewidth=1.5, alpha=0.5)

    bin_label = "quantile" if bin_strategy == "quantile" else "uniform"
    if panel_title:
        title_text = panel_title
    else:
        title_text = f"Calibration ({bin_label} bins, k={actual_n_bins})"
    ax.set_title(title_text, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted probability", fontsize=11)
    ax.set_ylabel("Expected frequency", fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect("equal")

    # Add size legend for uniform binning with multiple splits
    if bin_strategy == "uniform" and unique_splits is not None and len(unique_splits) > 1:
        from matplotlib.lines import Line2D
        size_handles = []
        size_labels = []
        for sample_count in [10, 50, 100, 200]:
            size = np.clip(sample_count * 1, 5, 300)
            markersize = np.sqrt(size) / 2  # Convert area to radius
            handle = Line2D([0], [0], marker='o', color='w',
                           markerfacecolor='steelblue', markersize=markersize,
                           markeredgecolor='darkblue', linewidth=0.5)
            size_handles.append(handle)
            size_labels.append(f'{sample_count}')

        size_legend = ax.legend(size_handles, size_labels,
                               title='Bin size (n)',
                               loc='center left',
                               bbox_to_anchor=(1.05, 0.5),
                               frameon=True,
                               fontsize=8)
        ax.add_artist(size_legend)
        # Re-add main legend (was overwritten by size legend)
        ax.legend(loc="upper left", fontsize=9)
    else:
        ax.legend(loc="upper left", fontsize=9)



def _plot_logit_calibration_panel(ax, y, p, n_bins, bin_strategy, split_ids, unique_splits, panel_title,
                                   lowess, calib_intercept, calib_slope, eps=1e-7):
    """Helper to plot a single logit-space calibration panel.

    Args:
        ax: Matplotlib axis to plot on
        y: True labels
        p: Predicted probabilities
        n_bins: Number of bins for binning
        bin_strategy: 'uniform' or 'quantile'
        split_ids: Optional split identifiers
        unique_splits: List of unique split IDs
        panel_title: Title for this panel
        lowess: LOESS function from statsmodels (or None)
        calib_intercept: Calibration intercept (alpha) from logistic recalibration
        calib_slope: Calibration slope (beta) from logistic recalibration
        eps: Small epsilon for clipping probabilities (default 1e-7)
    """
    # Clip probabilities for numerical stability
    p_clipped = np.clip(p, eps, 1 - eps)

    # Create bins based on strategy
    if bin_strategy == "quantile":
        quantiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(p, quantiles)
        bins = np.unique(bins)
        if len(bins) < 3:
            bins = np.linspace(0, 1, n_bins + 1)
    else:
        bins = np.linspace(0, 1, n_bins + 1)

    actual_n_bins = len(bins) - 1

    # Convert to logit space
    logit_pred = np.log(p_clipped / (1 - p_clipped))

    loess_ok = False
    loess_x = None
    loess_logit_y = None

    # Initialize axis ranges with default values
    logit_range_x = [-5, 5]
    logit_range_y = [-5, 5]

    # Multi-split logit calibration aggregation (using fixed probability bins)
    if unique_splits is not None and len(unique_splits) > 1:
        # Define fixed probability bins (same as probability-space calibration)
        prob_x_bins = []  # Predicted probabilities per split/bin (aggregate in prob space)
        prob_y_bins = []  # Observed frequencies per split/bin (aggregate in prob space)
        bin_sizes_per_split = []  # Track bin sample counts for marker sizing

        for sid in unique_splits:
            m_split = split_ids == sid
            y_s = y[m_split]
            p_s = p[m_split]

            # Bin predictions in probability space
            bin_idx = np.digitize(p_s, bins) - 1
            bin_idx = np.clip(bin_idx, 0, actual_n_bins - 1)

            # Compute observed frequency per bin (BEFORE logit transform)
            prob_x_per_split = []
            prob_y_per_split = []
            bin_sizes_per_bin = []
            for i in range(actual_n_bins):
                m_bin = bin_idx == i
                if m_bin.sum() == 0:
                    prob_x_per_split.append(np.nan)
                    prob_y_per_split.append(np.nan)
                    bin_sizes_per_bin.append(0)
                else:
                    # Mean predicted probability in bin (probability scale)
                    pred_mean = np.mean(p_s[m_bin])
                    prob_x_per_split.append(pred_mean)

                    # Apply Jeffreys smoothing to observed frequency to avoid 0/1
                    # Prevents extreme logits when bins are sparse: smoothed_rate = (k + 0.5) / (n + 1)
                    n_in_bin = m_bin.sum()
                    k_in_bin = np.sum(y_s[m_bin])
                    obs_freq_smoothed = (k_in_bin + 0.5) / (n_in_bin + 1)
                    prob_y_per_split.append(obs_freq_smoothed)
                    bin_sizes_per_bin.append(n_in_bin)

            prob_x_bins.append(prob_x_per_split)
            prob_y_bins.append(prob_y_per_split)
            bin_sizes_per_split.append(bin_sizes_per_bin)

        # Aggregate across splits in PROBABILITY SPACE (not logit space!)
        prob_x_bins = np.array(prob_x_bins, dtype=float)  # shape: (n_splits, n_bins)
        prob_y_bins = np.array(prob_y_bins, dtype=float)  # shape: (n_splits, n_bins)
        bin_sizes_per_split = np.array(bin_sizes_per_split, dtype=int)  # shape: (n_splits, n_bins)

        # Aggregate predicted and observed probabilities across splits
        prob_x_mean = np.nanmean(prob_x_bins, axis=0)
        prob_y_mean = np.nanmean(prob_y_bins, axis=0)
        prob_y_lo = np.nanpercentile(prob_y_bins, 2.5, axis=0)
        prob_y_hi = np.nanpercentile(prob_y_bins, 97.5, axis=0)
        prob_y_sd = np.nanstd(prob_y_bins, axis=0)

        # Aggregate bin sizes across splits (mean per bin)
        bin_sizes_mean = np.nanmean(bin_sizes_per_split, axis=0)

        # NOW convert aggregated probabilities to logit space (stable!)
        logit_x_mean = np.log(np.clip(prob_x_mean, eps, 1 - eps) / (1 - np.clip(prob_x_mean, eps, 1 - eps)))
        logit_y_mean = np.log(np.clip(prob_y_mean, eps, 1 - eps) / (1 - np.clip(prob_y_mean, eps, 1 - eps)))
        logit_y_lo = np.log(np.clip(prob_y_lo, eps, 1 - eps) / (1 - np.clip(prob_y_lo, eps, 1 - eps)))
        logit_y_hi = np.log(np.clip(prob_y_hi, eps, 1 - eps) / (1 - np.clip(prob_y_hi, eps, 1 - eps)))

        # For SD: compute logit of each smoothed split value, then take SD in logit space
        logit_curves_smooth = np.log(np.clip(prob_y_bins, eps, 1 - eps) / (1 - np.clip(prob_y_bins, eps, 1 - eps)))
        logit_y_sd = np.nanstd(logit_curves_smooth, axis=0)

        # Plot aggregated logit calibration bands
        valid_logit = ~np.isnan(logit_x_mean) & ~np.isnan(logit_y_mean)
        if valid_logit.sum() > 0:
            ax.fill_between(
                logit_x_mean[valid_logit],
                logit_y_lo[valid_logit],
                logit_y_hi[valid_logit],
                color="steelblue", alpha=0.15, label="95% CI"
            )
            ax.fill_between(
                logit_x_mean[valid_logit],
                np.clip(logit_y_mean[valid_logit] - logit_y_sd[valid_logit], -20, 20),
                np.clip(logit_y_mean[valid_logit] + logit_y_sd[valid_logit], -20, 20),
                color="steelblue", alpha=0.30, label="±1 SD"
            )

            # Plot line connecting bin centers
            ax.plot(
                logit_x_mean[valid_logit],
                logit_y_mean[valid_logit],
                '-', color="steelblue", linewidth=2, alpha=0.5, zorder=4
            )

            # Compute marker sizes: only variable for uniform binning
            if bin_strategy == "quantile":
                marker_sizes = 6  # Fixed size for quantile binning
            else:
                valid_sizes = bin_sizes_mean[valid_logit]
                if len(valid_sizes) > 0 and valid_sizes.max() > 0:
                    min_size, max_size = 4, 16  # Marker size range
                    norm_sizes = (valid_sizes - valid_sizes.min()) / (valid_sizes.max() - valid_sizes.min() + 1e-7)
                    marker_sizes = min_size + norm_sizes * (max_size - min_size)
                else:
                    marker_sizes = 6

            # Plot markers with variable/fixed sizes
            ax.scatter(
                logit_x_mean[valid_logit],
                logit_y_mean[valid_logit],
                s=marker_sizes**2,  # s parameter needs squared size for scatter
                marker='o',
                color="steelblue", alpha=0.8,
                edgecolors='darkblue', linewidth=0.5,
                label=f"Mean logit calib (n={len(unique_splits)} splits)",
                zorder=5
            )

            # Add legend for dot sizes
            if len(valid_sizes) > 0 and valid_sizes.max() > 0:
                size_range = [int(valid_sizes.min()), int(valid_sizes.max())]
                size_legend_text = f"Dot size ∝ sample n: {size_range[0]}–{size_range[1]}"
                ax.text(0.98, 0.02, size_legend_text, transform=ax.transAxes,
                        fontsize=8, ha='right', va='bottom',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            loess_ok = True  # Skip LOESS when multi-split aggregation is used

        # Determine axis ranges from aggregated data
        if valid_logit.sum() > 0:
            logit_range_x = [
                np.nanpercentile(logit_x_mean[valid_logit], 1) - 0.5,
                np.nanpercentile(logit_x_mean[valid_logit], 99) + 0.5
            ]
            logit_range_y = [
                np.nanpercentile(logit_y_lo[valid_logit], 1) - 0.5,
                np.nanpercentile(logit_y_hi[valid_logit], 99) + 0.5
            ]
        else:
            logit_range_x = [-5, 5]
            logit_range_y = [-5, 5]

    try:
        if lowess is None or len(y) < 20:
            raise RuntimeError("lowess unavailable or insufficient data")

        # Skip LOESS if multi-split aggregation already computed
        if unique_splits is not None and len(unique_splits) > 1 and loess_ok:
            raise RuntimeError("Multi-split aggregation computed, skip LOESS")

        # Sort by predicted probability for LOESS
        sort_idx = np.argsort(p_clipped)
        p_sorted = p_clipped[sort_idx].astype(np.float64)
        y_sorted = y[sort_idx].astype(np.float64)  # Ensure float64 for LOESS

        # Apply LOESS to estimate local probability P(Y=1|p)
        # With 16% prevalence and 1500 samples, frac=0.3 gives windows of ~450 samples (~75 positives)
        # Note: lowess(endog, exog) where endog=y-values, exog=x-values
        # IMPORTANT: it=0 to avoid statsmodels bug where robust iterations collapse to zero
        loess_result = lowess(y_sorted, p_sorted, frac=0.3, return_sorted=True, it=0)

        loess_p = loess_result[:, 0]  # Predicted probabilities (x-axis)
        loess_prob = loess_result[:, 1]  # Smoothed observed probabilities

        # Check high-probability tail for boundary effects
        high_prob_mask = loess_p > 0.9
        if high_prob_mask.sum() > 0:
            pass

        # CRITICAL: LOESS can overshoot [0,1] bounds at boundaries (e.g., 1.068 at high-prob tail)
        # This breaks log-odds calculation: log(1.068/(1-1.068)) = log(negative) = NaN
        # Strategy: Truncate the curve where LOESS overshoots instead of clipping
        # This preserves the shape in valid regions and removes boundary artifacts

        # Define valid probability range for log-odds conversion
        # Using slightly tighter bounds to ensure numerical stability
        clip_lower = 0.001   # Corresponds to log-odds ≈ -6.9
        clip_upper = 0.999   # Corresponds to log-odds ≈ +6.9

        # Find points where LOESS output is valid (within probability bounds)
        # This removes the overshooting tail instead of clipping it
        valid_loess_mask = (loess_prob >= clip_lower) & (loess_prob <= clip_upper)

        if valid_loess_mask.sum() < 10:
            # If too few valid points, fall back to simple clipping
            loess_prob = np.clip(loess_prob, clip_lower, clip_upper)
            loess_p_clipped = np.clip(loess_p, clip_lower, clip_upper)
            loess_logit_y = np.log(loess_prob / (1 - loess_prob))
            loess_x = np.log(loess_p_clipped / (1 - loess_p_clipped))
        else:
            # Truncate to valid region (removes boundary overshoot)
            loess_prob = loess_prob[valid_loess_mask]
            loess_p = loess_p[valid_loess_mask]

            # Also ensure x-axis values are in valid range
            loess_p = np.clip(loess_p, clip_lower, clip_upper)

            # Convert to log-odds (no artificial ceiling now)
            loess_logit_y = np.log(loess_prob / (1 - loess_prob))
            loess_x = np.log(loess_p / (1 - loess_p))

        # Validate output
        valid_mask = np.isfinite(loess_x) & np.isfinite(loess_logit_y)

        if valid_mask.sum() > 5:
            loess_x = loess_x[valid_mask]
            loess_logit_y = loess_logit_y[valid_mask]
            loess_ok = True
        else:
            loess_ok = False
    except Exception as e:
        loess_ok = False

    # Determine axis ranges based on actual data
    # (Skip if multi-split aggregation already set ranges)
    if not (unique_splits is not None and len(unique_splits) > 1):
        logit_min = np.percentile(logit_pred, 1)
        logit_max = np.percentile(logit_pred, 99)
        logit_range_x = [logit_min - 0.5, logit_max + 0.5]

        # Y-axis range should accommodate both LOESS and recalibration line
        logit_range_y = list(logit_range_x)  # Start with same as x-axis

        if loess_ok and loess_logit_y is not None:
            loess_min = np.percentile(loess_logit_y, 1)
            loess_max = np.percentile(loess_logit_y, 99)
            logit_range_y = [min(logit_range_y[0], loess_min - 0.5),
                            max(logit_range_y[1], loess_max + 0.5)]

    # Plot ideal calibration line
    ax.plot(logit_range_x, logit_range_x, "k--", linewidth=1.5, alpha=0.7, label="Ideal (α=0, β=1)")

    # Plot recalibration line if available
    if calib_intercept is not None and calib_slope is not None and np.isfinite(calib_intercept) and np.isfinite(calib_slope):
        recal_x = np.array(logit_range_x)
        recal_y = calib_intercept + calib_slope * recal_x
        ax.plot(recal_x, recal_y, "r-", linewidth=2, alpha=0.8,
                label=f"Recalibration (α={calib_intercept:.2f}, β={calib_slope:.2f})")
        # Extend y-range if recalibration line goes outside
        logit_range_y = [min(logit_range_y[0], recal_y.min() - 0.5),
                        max(logit_range_y[1], recal_y.max() + 0.5)]

    # Compute binned observations with binomial CIs (skip for multi-split, already computed above)
    if not (unique_splits is not None and len(unique_splits) > 1):
        binned_result = _binned_logits(y, p, n_bins=n_bins, bin_strategy=bin_strategy,
                                         min_bin_size=30, merge_tail=True)
        bx, by, by_lo, by_hi, bin_sizes = binned_result

        # Plot observed calibration (LOESS or binned)
        method_label = ""
        if loess_ok and loess_x is not None and loess_logit_y is not None:
            # Plot LOESS curve on top of CI band
            label = "LOESS (smoothed)"
            ax.plot(loess_x, loess_logit_y, "steelblue", linewidth=2.5, alpha=0.9, label=label, zorder=5)
            method_label = "LOESS"

            # Overlay binned observations with binomial CIs
            if bx is not None and by is not None and by_lo is not None and by_hi is not None:
                yerr_lo = by - by_lo
                yerr_hi = by_hi - by
                yerr = np.vstack([yerr_lo, yerr_hi])

                # Compute marker sizes scaled by bin sample numbers
                if bin_sizes is not None and len(bin_sizes) > 0:
                    min_size, max_size = 4, 16  # Marker size range
                    norm_sizes = (bin_sizes - bin_sizes.min()) / (bin_sizes.max() - bin_sizes.min() + 1e-7)
                    marker_sizes = min_size + norm_sizes * (max_size - min_size)
                else:
                    marker_sizes = 7

                # Plot error bars first with fixed size
                ax.errorbar(
                    bx, by,
                    yerr=yerr,
                    fmt='none', capsize=4, capthick=1.5,
                    color='darkorange', ecolor='darkorange', alpha=0.8,
                    zorder=9
                )

                # Plot markers with variable sizes
                scatter = ax.scatter(
                    bx, by,
                    s=marker_sizes**2,  # s parameter needs squared size for scatter
                    marker='o',
                    color='darkorange', alpha=0.8,
                    edgecolors='darkred', linewidth=0.5,
                    label=f'Binned observations (n={len(bx)} bins, Wilson CI)',
                    zorder=10
                )

                # Add legend for dot sizes
                if bin_sizes is not None and len(bin_sizes) > 0:
                    size_range = [int(bin_sizes.min()), int(bin_sizes.max())]
                    size_legend_text = f"Dot size ∝ sample n: {size_range[0]}–{size_range[1]}"
                    ax.text(0.98, 0.02, size_legend_text, transform=ax.transAxes,
                            fontsize=8, ha='right', va='bottom',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                # Extend y-range for binned CIs
                logit_range_y = [min(logit_range_y[0], by_lo.min() - 0.3),
                                max(logit_range_y[1], by_hi.max() + 0.3)]
        else:
            # Fallback: only binned data available
            if bx is not None and by is not None and by_lo is not None and by_hi is not None:
                yerr_lo = by - by_lo
                yerr_hi = by_hi - by
                yerr = np.vstack([yerr_lo, yerr_hi])

                # Compute marker sizes scaled by bin sample numbers
                if bin_sizes is not None and len(bin_sizes) > 0:
                    min_size, max_size = 4, 16  # Marker size range
                    norm_sizes = (bin_sizes - bin_sizes.min()) / (bin_sizes.max() - bin_sizes.min() + 1e-7)
                    marker_sizes = min_size + norm_sizes * (max_size - min_size)
                else:
                    marker_sizes = 7

                # Plot error bars first with fixed size
                ax.errorbar(
                    bx, by,
                    yerr=yerr,
                    fmt='none', capsize=4, capthick=1.5,
                    color='steelblue', ecolor='steelblue', alpha=0.8,
                    linewidth=2,
                    zorder=4
                )

                # Plot line connecting bin centers
                ax.plot(bx, by, '-', color='steelblue', linewidth=2, alpha=0.6, zorder=4)

                # Plot markers with variable sizes
                scatter = ax.scatter(
                    bx, by,
                    s=marker_sizes**2,  # s parameter needs squared size for scatter
                    marker='o',
                    color='steelblue', alpha=0.8,
                    edgecolors='darkblue', linewidth=0.5,
                    label=f'Binned logits (n={len(bx)} bins, Wilson CI)',
                    zorder=5
                )

                # Add legend for dot sizes
                if bin_sizes is not None and len(bin_sizes) > 0:
                    size_range = [int(bin_sizes.min()), int(bin_sizes.max())]
                    size_legend_text = f"Dot size ∝ sample n: {size_range[0]}–{size_range[1]}"
                    ax.text(0.98, 0.02, size_legend_text, transform=ax.transAxes,
                            fontsize=8, ha='right', va='bottom',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                method_label = "Binned"
                # Extend y-range for binned data
                logit_range_y = [min(logit_range_y[0], by_lo.min() - 0.5),
                                max(logit_range_y[1], by_hi.max() + 0.5)]
    else:
        # Multi-split mode: aggregated bands already plotted
        method_label = "Multi-split aggregated"

    ax.set_title(panel_title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted logit: logit(p̂)", fontsize=11)
    ylabel = f"Empirical logit ({method_label})" if method_label else "Empirical logit"
    ax.set_ylabel(ylabel, fontsize=11)

    # Add size legend for uniform binning with multiple splits
    if bin_strategy == "uniform" and unique_splits is not None and len(unique_splits) > 1:
        from matplotlib.lines import Line2D
        size_handles = []
        size_labels = []
        for sample_count in [10, 50, 100, 200]:
            size = np.clip(sample_count * 1, 5, 300)
            markersize = np.sqrt(size) / 2  # Convert area to radius
            handle = Line2D([0], [0], marker='o', color='w',
                           markerfacecolor='steelblue', markersize=markersize,
                           markeredgecolor='darkblue', linewidth=0.5)
            size_handles.append(handle)
            size_labels.append(f'{sample_count}')

        size_legend = ax.legend(size_handles, size_labels,
                               title='Bin size (n)',
                               loc='center left',
                               bbox_to_anchor=(1.05, 0.5),
                               frameon=True,
                               fontsize=8)
        ax.add_artist(size_legend)
        # Re-add main legend (was overwritten by size legend)
        ax.legend(loc="upper left", fontsize=9)
    else:
        ax.legend(loc="upper left", fontsize=9)

    ax.grid(True, alpha=0.3)
    ax.set_xlim(logit_range_x)
    ax.set_ylim(logit_range_y)
    # Don't force equal aspect - let the plot expand to fill available space
    # Equal aspect would squeeze the logit plot in multi-panel layouts



def _plot_calibration_curve(y_true, y_pred, out_path, title, subtitle="", n_bins=10, split_ids=None, meta_lines=None,
                            bin_strategy="uniform", calib_intercept=None, calib_slope=None, four_panel=False):
    """Generate 4-panel calibration plot.

    Always generates a 2x2 layout:
        Panel 1 (top-left): Calibration curve with quantile binning
        Panel 2 (top-right): Calibration curve with uniform binning
        Panel 3 (middle-left): Logit calibration curve with quantile binning
        Panel 4 (middle-right): Logit calibration curve with uniform binning

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        out_path: Output file path
        title: Plot title
        subtitle: Plot subtitle
        n_bins: Number of bins for calibration curve
        split_ids: Optional split identifiers for multi-split aggregation
        meta_lines: Metadata lines for plot annotation
        bin_strategy: Ignored (both quantile and uniform always shown)
        calib_intercept: Calibration intercept (alpha) from logistic recalibration
        calib_slope: Calibration slope (beta) from logistic recalibration
        four_panel: Deprecated parameter (always True now, kept for backward compatibility)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[PLOT] Calibration plot failed to import dependencies: {e}")
        return

    lowess = None
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess
        lowess = _lowess
    except Exception:
        lowess = None

    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(float)
    mask = np.isfinite(p) & np.isfinite(y)
    y = y[mask]
    p = p[mask]
    if len(y) == 0:
        return

    # Clip probabilities for numerical stability
    eps = 1e-7
    p_clipped = np.clip(p, eps, 1 - eps)

    if split_ids is not None:
        split_ids = np.asarray(split_ids)[mask]
        unique_splits = pd.Series(split_ids).dropna().unique().tolist()
    else:
        unique_splits = []

    # Create figure layout: Always 2x2 panels
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])  # Top-left: Calibration quantile
    ax2 = fig.add_subplot(gs[0, 1])  # Top-right: Calibration uniform
    ax3 = fig.add_subplot(gs[1, 0])  # Bottom-left: Logit quantile
    ax4 = fig.add_subplot(gs[1, 1])  # Bottom-right: Logit uniform

    # ========== Panel 1 (top-left): Probability-space calibration curve with quantile binning ==========
    quantiles = np.linspace(0, 100, n_bins + 1)
    bins_quantile = np.percentile(p, quantiles)
    bins_quantile = np.unique(bins_quantile)
    if len(bins_quantile) < 3:
        bins_quantile = np.linspace(0, 1, n_bins + 1)
    actual_n_bins_q = len(bins_quantile) - 1
    bin_centers_q = (bins_quantile[:-1] + bins_quantile[1:]) / 2

    # Compute actual bin sizes from the quantile bins for the probability-space plot
    bin_idx_q = np.digitize(p, bins_quantile) - 1
    bin_idx_q = np.clip(bin_idx_q, 0, actual_n_bins_q - 1)
    bin_sizes_q = np.array([int((bin_idx_q == i).sum()) for i in range(actual_n_bins_q)])

    # Compute per-bin sample counts (only count non-zero bins for statistics)
    nonzero_sizes = bin_sizes_q[bin_sizes_q > 0]
    if len(nonzero_sizes) > 0:
        mean_size = int(np.mean(nonzero_sizes))
        min_size = int(np.min(nonzero_sizes))
        max_size = int(np.max(nonzero_sizes))
        if min_size == max_size:
            bin_size_str = f"n={mean_size}/bin"
        else:
            bin_size_str = f"n≈{mean_size}/bin (range {min_size}–{max_size})"
    else:
        bin_size_str = ""

    panel_title_1 = f"Calibration (quantile bins)\nk={actual_n_bins_q}, {bin_size_str}"
    if subtitle:
        panel_title_1 = f"{subtitle} – Calibration (quantile bins)\nk={actual_n_bins_q}, {bin_size_str}"

    _plot_prob_calibration_panel(
        ax1, y, p, bins_quantile, bin_centers_q, actual_n_bins_q, "quantile",
        split_ids=split_ids, unique_splits=unique_splits, panel_title=panel_title_1,
        variable_sizes=False
    )

    # ========== Panel 2 (top-right): Probability-space calibration curve with uniform binning ==========
    bins_uniform = np.linspace(0, 1, n_bins + 1)
    actual_n_bins_u = len(bins_uniform) - 1
    bin_centers_u = (bins_uniform[:-1] + bins_uniform[1:]) / 2

    panel_title_2 = f"Calibration (uniform bins)\nk={actual_n_bins_u}"
    if subtitle:
        panel_title_2 = f"{subtitle} – Calibration (uniform bins)\nk={actual_n_bins_u}"

    _plot_prob_calibration_panel(
        ax2, y, p, bins_uniform, bin_centers_u, actual_n_bins_u, "uniform",
        split_ids=split_ids, unique_splits=unique_splits, panel_title=panel_title_2,
        variable_sizes=True
    )

    # ========== Panel 3 (bottom-left): Log-odds calibration with quantile binning ==========
    logit_title_q = "Logit calibration (quantile bins)"
    _plot_logit_calibration_panel(
        ax3, y, p, n_bins, "quantile",
        split_ids, unique_splits, logit_title_q,
        lowess, calib_intercept, calib_slope, eps=eps
    )

    # ========== Panel 4 (bottom-right): Log-odds calibration with uniform binning ==========
    logit_title_u = "Logit calibration (uniform bins)"
    _plot_logit_calibration_panel(
        ax4, y, p, n_bins, "uniform",
        split_ids, unique_splits, logit_title_u,
        lowess, calib_intercept, calib_slope, eps=eps
    )

    # Add title at the top
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    # Combined tight_layout and metadata application handled by _apply_plot_metadata
    bottom_margin = _apply_plot_metadata(fig, meta_lines)
    # Generous margins to prevent overlap
    plt.subplots_adjust(left=0.10, right=0.95, top=0.92, bottom=bottom_margin)
    plt.savefig(out_path, dpi=150, pad_inches=0.1)
    plt.close()


def compute_distribution_stats(scores: np.ndarray) -> Dict[str, float]:
    """
    Compute summary statistics for a distribution of scores.

    Args:
        scores: Array of numeric scores

    Returns:
        Dictionary with keys: mean, median, iqr, sd
    """
    scores = np.asarray(scores).astype(float)
    scores = scores[np.isfinite(scores)]

    if len(scores) == 0:
        return {"mean": np.nan, "median": np.nan, "iqr": np.nan, "sd": np.nan}

    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)

    return {
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "iqr": float(q3 - q1),
        "sd": float(np.std(scores))
    }


def _plot_risk_distribution(
    y_true,
    scores,
    out_path,
    title,
    subtitle="",
    xlabel="Predicted risk",
    pos_label="Incident CeD",
    meta_lines=None,
    category_col=None,
    dca_threshold=None,
    spec95_threshold=None,
    youden_threshold=None,
    alpha_threshold=None,
    metrics_at_thresholds=None,
    x_limits=None,
    target_spec=0.95,
):
    """
    Plot risk score distribution with optional thresholds and case-type subplots.

    Args:
        y_true: Binary outcome labels (0/1)
        scores: Risk scores
        out_path: Path to save figure
        title: Plot title
        subtitle: Optional subtitle
        xlabel: X-axis label
        pos_label: Label for positive class
        meta_lines: Metadata lines for bottom of figure
        category_col: Array of category labels ("Controls", "Incident", "Prevalent")
        dca_threshold: DCA zero-crossing threshold (optional)
        spec95_threshold: Specificity threshold (optional)
        youden_threshold: Youden's J statistic threshold (optional)
        alpha_threshold: Alpha/target specificity threshold (optional)
        metrics_at_thresholds: Dict with metrics at thresholds (optional)
            Format: {
                'spec95': {'sensitivity': float, 'precision': float, 'fp': int, 'n_celiac': int},
                'dca': {'sensitivity': float, 'precision': float, 'fp': int, 'n_celiac': int},
                'youden': {'sensitivity': float, 'precision': float, 'fp': int, 'n_celiac': int},
                'alpha': {'sensitivity': float, 'precision': float, 'fp': int, 'n_celiac': int}
            }
        x_limits: Optional tuple (xmin, xmax) for x-axis range (default: auto)
        target_spec: Target specificity value for annotation label (default: 0.95)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[PLOT] Risk distribution plot failed to import dependencies: {e}")
        return

    s = np.asarray(scores).astype(float)

    # Determine if we have incident/prevalent subplots to show
    has_incident = False
    has_prevalent = False
    if category_col is not None:
        cat = np.asarray(category_col)
        mask = np.isfinite(s)
        s_clean = s[mask]
        cat_clean = cat[mask]
        has_incident = np.any(cat_clean == "Incident")
        has_prevalent = np.any(cat_clean == "Prevalent")

    # Calculate number of subplots needed
    n_subplots = 1
    if has_incident:
        n_subplots += 1
    if has_prevalent:
        n_subplots += 1

    # Create figure with appropriate number of subplots
    # Determine figure size based on plot type (histogram vs KDE)
    if n_subplots == 1:
        # Single plot: use different aspect ratios for KDE vs histogram
        height_ratios = [1]
        if category_col is not None:
            # KDE plot with categories: 16:9 aspect ratio
            figsize = (12, 6.75)
        else:
            # Histogram or controls distribution: 3:2 aspect ratio
            figsize = (9, 6)
    elif n_subplots == 2:
        # Main + 1 strip plot: maintain 3:2 overall aspect
        height_ratios = [9, 1]
        figsize = (9, 10)
    else:
        # Main + 2 strip plots: maintain 3:2 overall aspect
        height_ratios = [9, 1, 1]
        figsize = (9, 11)

    fig, axes = plt.subplots(n_subplots, 1, figsize=figsize, height_ratios=height_ratios)
    if n_subplots == 1:
        axes = [axes]

    ax_main = axes[0]

    # === MAIN HISTOGRAM (ax_main) ===
    if y_true is None and category_col is None:
        mask = np.isfinite(s)
        s = s[mask]
        if len(s) == 0:
            plt.close()
            return
        bins = min(60, max(10, int(np.sqrt(len(s)))))
        ax_main.hist(s, bins=bins, density=True, alpha=0.7, color="steelblue", edgecolor="white")
    elif category_col is not None:
        # Use category column for three-way split (Controls, Incident, Prevalent)
        cat = np.asarray(category_col)
        mask = np.isfinite(s)
        s = s[mask]
        cat = cat[mask]
        if len(s) == 0:
            plt.close()
            return
        bins = min(60, max(10, int(np.sqrt(len(s)))))

        # Define three categories with distinct colors
        categories = [
            ("Controls", "steelblue", "Controls"),
            ("Incident", "firebrick", "Incident"),
            ("Prevalent", "darkorange", "Prevalent"),
        ]

        for label, color, cat_name in categories:
            vals = s[cat == cat_name]
            if len(vals) == 0:
                continue
            ax_main.hist(vals, bins=bins, density=True, alpha=0.45, color=color, edgecolor="white", label=label)

        if ax_main.get_legend_handles_labels()[0]:
            ax_main.legend(loc="upper right", fontsize=10)
    else:
        y = np.asarray(y_true).astype(int)
        mask = np.isfinite(s) & np.isfinite(y)
        s = s[mask]
        y = y[mask]
        if len(s) == 0:
            plt.close()
            return
        bins = min(60, max(10, int(np.sqrt(len(s)))))
        for label, color, target in [("Control", "steelblue", 0), (pos_label, "firebrick", 1)]:
            vals = s[y == target]
            if len(vals) == 0:
                continue
            ax_main.hist(vals, bins=bins, density=True, alpha=0.45, color=color, edgecolor="white", label=label)
        if ax_main.get_legend_handles_labels()[0]:
            ax_main.legend(loc="upper right", fontsize=10)

    # Add threshold lines and annotations
    if spec95_threshold is not None and 0 <= spec95_threshold <= 1:
        spec_label = f"{target_spec*100:.0f}% Spec"
        ax_main.axvline(spec95_threshold, color='red', linestyle='--', linewidth=2, alpha=0.7, label=spec_label)

    if youden_threshold is not None and 0 <= youden_threshold <= 1:
        ax_main.axvline(youden_threshold, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Youden')


    if dca_threshold is not None and 0 <= dca_threshold <= 1:
        ax_main.axvline(dca_threshold, color='purple', linestyle='--', linewidth=2, alpha=0.7, label='DCA Threshold')

    # Create comprehensive legend with threshold metrics
    handles, labels = ax_main.get_legend_handles_labels()
    threshold_handles = []
    threshold_labels = []

    if spec95_threshold is not None and metrics_at_thresholds and 'spec95' in metrics_at_thresholds:
        from matplotlib.lines import Line2D
        m = metrics_at_thresholds['spec95']
        sens = m.get('sensitivity', np.nan)
        ppv = m.get('precision', np.nan)
        fp = m.get('fp', np.nan)

        line_handle = Line2D([0], [0], color='red', linestyle='--', linewidth=2, alpha=0.7)
        threshold_handles.append(line_handle)

        # Multi-line label format
        label_text = f"{target_spec*100:.0f}% Spec\n"
        if not np.isnan(sens) and not np.isnan(ppv) and not np.isnan(fp):
            label_text += f"Sens {sens*100:.1f}%, PPV {ppv*100:.1f}%, FP: {int(fp)}"
        threshold_labels.append(label_text)

    if youden_threshold is not None and metrics_at_thresholds and 'youden' in metrics_at_thresholds:
        from matplotlib.lines import Line2D
        m = metrics_at_thresholds['youden']
        sens = m.get('sensitivity', np.nan)
        ppv = m.get('precision', np.nan)
        fp = m.get('fp', np.nan)

        line_handle = Line2D([0], [0], color='green', linestyle='--', linewidth=2, alpha=0.7)
        threshold_handles.append(line_handle)

        # Multi-line label format
        label_text = f"Youden\n"
        if not np.isnan(sens) and not np.isnan(ppv) and not np.isnan(fp):
            label_text += f"Sens {sens*100:.1f}%, PPV {ppv*100:.1f}%, FP: {int(fp)}"
        threshold_labels.append(label_text)

    if dca_threshold is not None and metrics_at_thresholds and 'dca' in metrics_at_thresholds:
        from matplotlib.lines import Line2D
        m = metrics_at_thresholds['dca']
        sens = m.get('sensitivity', np.nan)
        ppv = m.get('precision', np.nan)
        fp = m.get('fp', np.nan)

        line_handle = Line2D([0], [0], color='purple', linestyle='--', linewidth=2, alpha=0.7)
        threshold_handles.append(line_handle)

        # Multi-line label format
        label_text = f"DCA\n"
        if not np.isnan(sens) and not np.isnan(ppv) and not np.isnan(fp):
            label_text += f"Sens {sens*100:.1f}%, PPV {ppv*100:.1f}%, FP: {int(fp)}"
        threshold_labels.append(label_text)

    # Combine all handles and labels
    all_handles = handles + threshold_handles
    all_labels = labels + threshold_labels

    # Create legend outside plot area
    if all_handles:
        ax_main.legend(all_handles, all_labels,
                       loc='upper left',
                       bbox_to_anchor=(1.05, 1),
                       fontsize=9,
                       framealpha=0.9)

    if subtitle:
        ax_main.set_title(f"{title}\n{subtitle}", fontsize=12)
    else:
        ax_main.set_title(title, fontsize=12)
    ax_main.set_ylabel("Density")
    ax_main.grid(True, alpha=0.2)

    # Apply x-axis limits if provided
    if x_limits is not None:
        ax_main.set_xlim(x_limits)

    # === INCIDENT DENSITY PLOT (if applicable) ===
    subplot_idx = 1
    if has_incident:
        ax_incident = axes[subplot_idx]
        subplot_idx += 1

        incident_scores = s[cat == "Incident"]
        stats = compute_distribution_stats(incident_scores)

        # Create KDE density plot
        if len(incident_scores) > 0:
            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(incident_scores, bw_method='scott')
                x_range = np.linspace(0, 1, 200)
                density = kde(x_range)
                ax_incident.plot(x_range, density, color='firebrick', linewidth=2, alpha=0.8)
                ax_incident.fill_between(x_range, density, alpha=0.3, color='firebrick')
            except Exception:
                # Fallback to histogram if KDE fails (e.g., too few points)
                ax_incident.hist(incident_scores, bins=20, density=True, alpha=0.7, color='firebrick', edgecolor='white')

        # Add threshold lines (no labels)
        if spec95_threshold is not None and 0 <= spec95_threshold <= 1:
            ax_incident.axvline(spec95_threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        if youden_threshold is not None and 0 <= youden_threshold <= 1:
            ax_incident.axvline(youden_threshold, color='green', linestyle='--', linewidth=1.5, alpha=0.5)
        if dca_threshold is not None and 0 <= dca_threshold <= 1:
            ax_incident.axvline(dca_threshold, color='purple', linestyle='--', linewidth=1.5, alpha=0.5)

        ax_incident.set_xlim(0, 1)
        ax_incident.set_ylabel("Incident\nDensity", fontsize=9)
        ax_incident.grid(True, alpha=0.2, axis='x')
        ax_incident.set_yticks([])

        # Add statistics text
        stats_text = (f"Mean: {stats['mean']:.3f} | Median: {stats['median']:.3f} | "
                     f"IQR: {stats['iqr']:.3f} | SD: {stats['sd']:.3f}")
        ax_incident.text(0.02, 0.95, stats_text, transform=ax_incident.transAxes,
                        fontsize=8, va='top', ha='left',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # === PREVALENT DENSITY PLOT (if applicable) ===
    if has_prevalent:
        ax_prevalent = axes[subplot_idx]

        prevalent_scores = s[cat == "Prevalent"]
        stats = compute_distribution_stats(prevalent_scores)

        # Create KDE density plot
        if len(prevalent_scores) > 0:
            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(prevalent_scores, bw_method='scott')
                x_range = np.linspace(0, 1, 200)
                density = kde(x_range)
                ax_prevalent.plot(x_range, density, color='darkorange', linewidth=2, alpha=0.8)
                ax_prevalent.fill_between(x_range, density, alpha=0.3, color='darkorange')
            except Exception:
                # Fallback to histogram if KDE fails
                ax_prevalent.hist(prevalent_scores, bins=20, density=True, alpha=0.7, color='darkorange', edgecolor='white')

        # Add threshold lines (no labels)
        if spec95_threshold is not None and 0 <= spec95_threshold <= 1:
            ax_prevalent.axvline(spec95_threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        if youden_threshold is not None and 0 <= youden_threshold <= 1:
            ax_prevalent.axvline(youden_threshold, color='green', linestyle='--', linewidth=1.5, alpha=0.5)
        if dca_threshold is not None and 0 <= dca_threshold <= 1:
            ax_prevalent.axvline(dca_threshold, color='purple', linestyle='--', linewidth=1.5, alpha=0.5)

        ax_prevalent.set_xlim(0, 1)
        ax_prevalent.set_ylabel("Prevalent\nDensity", fontsize=9)
        ax_prevalent.set_xlabel(xlabel)
        ax_prevalent.grid(True, alpha=0.2, axis='x')
        ax_prevalent.set_yticks([])

        # Add statistics text
        stats_text = (f"Mean: {stats['mean']:.3f} | Median: {stats['median']:.3f} | "
                     f"IQR: {stats['iqr']:.3f} | SD: {stats['sd']:.3f}")
        ax_prevalent.text(0.02, 0.95, stats_text, transform=ax_prevalent.transAxes,
                         fontsize=8, va='top', ha='left',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    else:
        # If no prevalent subplot, add xlabel to last subplot
        if has_incident:
            axes[-1].set_xlabel(xlabel)
        else:
            ax_main.set_xlabel(xlabel)

    # Apply metadata and adjust layout
    bottom_margin = _apply_plot_metadata(fig, meta_lines) if meta_lines else 0.1
    plt.subplots_adjust(left=0.12, right=0.70, top=0.92, bottom=bottom_margin, hspace=0.3)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()


