"""
ROC and Precision-Recall curve plotting.

Provides functions to generate ROC and PR curves with confidence intervals
and threshold annotations.
"""

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    _HAS_PLOTTING = True
except ImportError:
    _HAS_PLOTTING = False


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
        return 0.10

    fig.text(
        0.5, 0.005, "\n".join(lines), ha="center", va="bottom", fontsize=8, wrap=True
    )
    required_bottom = 0.10 + (0.018 * len(lines))
    return min(required_bottom, 0.30)


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    title: str,
    subtitle: str = "",
    split_ids: Optional[np.ndarray] = None,
    meta_lines: Optional[Sequence[str]] = None,
    youden_threshold: Optional[float] = None,
    alpha_threshold: Optional[float] = None,
    metrics_at_thresholds: Optional[dict] = None,
) -> None:
    """
    Plot ROC curve with optional split-wise confidence bands and threshold markers.

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        out_path: Output file path
        title: Plot title
        subtitle: Optional subtitle
        split_ids: Array indicating split membership for each sample
        meta_lines: Optional metadata lines to display at bottom
        youden_threshold: Youden threshold value (for marker)
        alpha_threshold: Alpha threshold value (for marker)
        metrics_at_thresholds: Dict with 'youden' and 'alpha' keys containing
            dicts with 'fpr' and 'tpr' values

    Returns:
        None. Saves plot to out_path.
    """
    if not _HAS_PLOTTING:
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

            ax.fill_between(
                base_fpr, tpr_lo, tpr_hi, color="steelblue", alpha=0.15, label="95% CI"
            )
            ax.fill_between(
                base_fpr,
                np.maximum(0, tpr_mean - tpr_sd),
                np.minimum(1, tpr_mean + tpr_sd),
                color="steelblue",
                alpha=0.30,
                label="±1 SD",
            )
            ax.plot(
                base_fpr,
                tpr_mean,
                color="steelblue",
                linewidth=2,
                label=f"AUC = {auc_mean:.3f} ± {auc_sd:.3f}",
            )
        else:
            fpr, tpr, _ = roc_curve(y, p)
            auc = roc_auc_score(y, p)
            ax.plot(fpr, tpr, color="steelblue", linewidth=2, label=f"AUC = {auc:.3f}")
    else:
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ax.plot(fpr, tpr, color="steelblue", linewidth=2, label=f"AUC = {auc:.3f}")

    if metrics_at_thresholds is not None:
        if youden_threshold is not None and "youden" in metrics_at_thresholds:
            m = metrics_at_thresholds["youden"]
            fpr_youden = m.get("fpr", None)
            tpr_youden = m.get("tpr", None)
            if (
                fpr_youden is not None
                and tpr_youden is not None
                and 0 <= fpr_youden <= 1
                and 0 <= tpr_youden <= 1
            ):
                ax.scatter(
                    [fpr_youden],
                    [tpr_youden],
                    s=100,
                    color="green",
                    marker="o",
                    edgecolors="darkgreen",
                    linewidths=2,
                    label="Youden",
                    zorder=5,
                )

        if alpha_threshold is not None and "alpha" in metrics_at_thresholds:
            m = metrics_at_thresholds["alpha"]
            fpr_alpha = m.get("fpr", None)
            tpr_alpha = m.get("tpr", None)
            if (
                fpr_alpha is not None
                and tpr_alpha is not None
                and 0 <= fpr_alpha <= 1
                and 0 <= tpr_alpha <= 1
            ):
                ax.scatter(
                    [fpr_alpha],
                    [tpr_alpha],
                    s=100,
                    color="orange",
                    marker="D",
                    edgecolors="darkorange",
                    linewidths=2,
                    label="Alpha threshold",
                    zorder=5,
                )

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


def plot_pr_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    title: str,
    subtitle: str = "",
    split_ids: Optional[np.ndarray] = None,
    meta_lines: Optional[Sequence[str]] = None,
) -> None:
    """
    Plot Precision-Recall curve with optional split-wise confidence bands.

    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        out_path: Output file path
        title: Plot title
        subtitle: Optional subtitle
        split_ids: Array indicating split membership for each sample
        meta_lines: Optional metadata lines to display at bottom

    Returns:
        None. Saves plot to out_path.
    """
    if not _HAS_PLOTTING:
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
    ax.axhline(
        y=baseline,
        color="k",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
        label=f"Prevalence = {baseline:.4f}",
    )

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

            ax.fill_between(
                base_recall,
                np.clip(prec_lo, 0, 1),
                np.clip(prec_hi, 0, 1),
                color="steelblue",
                alpha=0.15,
                label="95% CI",
            )
            ax.fill_between(
                base_recall,
                np.clip(prec_mean - prec_sd, 0, 1),
                np.clip(prec_mean + prec_sd, 0, 1),
                color="steelblue",
                alpha=0.30,
                label="±1 SD",
            )
            ax.plot(
                base_recall,
                prec_mean,
                color="steelblue",
                linewidth=2,
                label=f"AP = {ap_mean:.3f} ± {ap_sd:.3f}",
            )
        else:
            precision, recall, _ = precision_recall_curve(y, p)
            ap = average_precision_score(y, p)
            ax.plot(
                recall,
                precision,
                color="steelblue",
                linewidth=2,
                label=f"AP = {ap:.3f}",
            )
    else:
        precision, recall, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ax.plot(
            recall, precision, color="steelblue", linewidth=2, label=f"AP = {ap:.3f}"
        )

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
