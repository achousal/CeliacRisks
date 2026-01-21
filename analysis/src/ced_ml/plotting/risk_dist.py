"""Risk distribution plotting for CeliacRiskML.

Creates publication-quality risk score distribution plots with:
- Multi-panel layouts for incident/prevalent/control cases
- Clinical threshold overlays (DCA, Youden, specificity)
- Performance metrics at thresholds
- Density estimation and summary statistics
"""

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from .dca import apply_plot_metadata


def compute_distribution_stats(scores: np.ndarray) -> Dict[str, float]:
    """Compute summary statistics for a distribution of scores.

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
        "sd": float(np.std(scores)),
    }


def plot_risk_distribution(
    y_true: Optional[np.ndarray],
    scores: np.ndarray,
    out_path: Path,
    title: str,
    subtitle: str = "",
    xlabel: str = "Predicted risk",
    pos_label: str = "Incident CeD",
    meta_lines: Optional[Sequence[str]] = None,
    category_col: Optional[np.ndarray] = None,
    dca_threshold: Optional[float] = None,
    spec95_threshold: Optional[float] = None,
    youden_threshold: Optional[float] = None,
    alpha_threshold: Optional[float] = None,
    metrics_at_thresholds: Optional[Dict[str, Dict[str, float]]] = None,
    x_limits: Optional[Tuple[float, float]] = None,
    target_spec: float = 0.95,
) -> None:
    """Plot risk score distribution with optional thresholds and case-type subplots.

    Creates a multi-panel plot with:
    - Main histogram/KDE showing overall distribution
    - Optional incident-only density subplot
    - Optional prevalent-only density subplot
    - Threshold lines with performance metrics

    Args:
        y_true: Binary outcome labels (0/1), optional if category_col provided
        scores: Risk scores (0-1 range)
        out_path: Path to save figure
        title: Plot title
        subtitle: Optional subtitle
        xlabel: X-axis label
        pos_label: Label for positive class (e.g., "Incident CeD")
        meta_lines: Metadata lines for bottom of figure
        category_col: Array of category labels ("Controls", "Incident", "Prevalent")
        dca_threshold: DCA zero-crossing threshold (0-1)
        spec95_threshold: Specificity threshold (0-1)
        youden_threshold: Youden's J statistic threshold (0-1)
        alpha_threshold: Alpha/target specificity threshold (0-1)
        metrics_at_thresholds: Performance metrics at each threshold
            Format: {
                'spec95': {'sensitivity': float, 'precision': float, 'fp': int, 'n_celiac': int},
                'dca': {...},
                'youden': {...},
                'alpha': {...}
            }
        x_limits: Optional tuple (xmin, xmax) for x-axis range
        target_spec: Target specificity value for annotation label (default: 0.95)

    Notes:
        - If category_col is provided, creates three-category KDE plot
        - If y_true is provided without category_col, creates binary histogram
        - If neither is provided, creates single-category histogram
        - Incident/prevalent subplots only shown if category_col includes those categories
    """
    matplotlib.use("Agg")

    s = np.asarray(scores).astype(float)

    # Determine if we have incident/prevalent subplots to show
    has_incident = False
    has_prevalent = False
    if category_col is not None:
        cat = np.asarray(category_col)
        mask = np.isfinite(s)
        s[mask]
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
            # KDE plot with categories: 4:1 aspect ratio
            figsize = (12, 3)
        else:
            # Histogram or controls distribution: 3:2 aspect ratio
            figsize = (9, 6)
    elif n_subplots == 2:
        # Main + 1 KDE subplot: each KDE subplot is 3:2
        height_ratios = [3, 2]
        figsize = (9, 15)
    else:
        # Main + 2 KDE subplots: each KDE subplot is 3:2
        height_ratios = [3, 2, 2]
        figsize = (9, 21)

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
        ax_main.hist(
            s,
            bins=bins,
            density=True,
            alpha=0.7,
            color="steelblue",
            edgecolor="white",
        )
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
            ax_main.hist(
                vals,
                bins=bins,
                density=True,
                alpha=0.45,
                color=color,
                edgecolor="white",
                label=label,
            )

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
        for label, color, target in [
            ("Control", "steelblue", 0),
            (pos_label, "firebrick", 1),
        ]:
            vals = s[y == target]
            if len(vals) == 0:
                continue
            ax_main.hist(
                vals,
                bins=bins,
                density=True,
                alpha=0.45,
                color=color,
                edgecolor="white",
                label=label,
            )
        if ax_main.get_legend_handles_labels()[0]:
            ax_main.legend(loc="upper right", fontsize=10)

    # Add threshold lines (without labels - will be added to legend separately)
    if spec95_threshold is not None and 0 <= spec95_threshold <= 1:
        ax_main.axvline(
            spec95_threshold,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
        )

    if youden_threshold is not None and 0 <= youden_threshold <= 1:
        ax_main.axvline(
            youden_threshold,
            color="green",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
        )

    if dca_threshold is not None and 0 <= dca_threshold <= 1:
        ax_main.axvline(
            dca_threshold,
            color="purple",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
        )

    # Create comprehensive legend with threshold metrics
    handles, labels = ax_main.get_legend_handles_labels()
    threshold_handles = []
    threshold_labels = []

    if spec95_threshold is not None and metrics_at_thresholds and "spec95" in metrics_at_thresholds:
        m = metrics_at_thresholds["spec95"]
        sens = m.get("sensitivity", np.nan)
        ppv = m.get("precision", np.nan)
        fp = m.get("fp", np.nan)

        line_handle = Line2D([0], [0], color="red", linestyle="--", linewidth=2, alpha=0.7)
        threshold_handles.append(line_handle)

        # Multi-line label format with each metric on separate line
        label_text = f"{target_spec*100:.0f}% Spec"
        if not np.isnan(sens) and not np.isnan(ppv) and not np.isnan(fp):
            label_text += f"\nSens: {sens*100:.1f}%\nPPV: {ppv*100:.1f}%\nFP: {int(fp)}"
        threshold_labels.append(label_text)

    if youden_threshold is not None and metrics_at_thresholds and "youden" in metrics_at_thresholds:
        m = metrics_at_thresholds["youden"]
        sens = m.get("sensitivity", np.nan)
        ppv = m.get("precision", np.nan)
        fp = m.get("fp", np.nan)

        line_handle = Line2D([0], [0], color="green", linestyle="--", linewidth=2, alpha=0.7)
        threshold_handles.append(line_handle)

        # Multi-line label format with each metric on separate line
        label_text = "Youden"
        if not np.isnan(sens) and not np.isnan(ppv) and not np.isnan(fp):
            label_text += f"\nSens: {sens*100:.1f}%\nPPV: {ppv*100:.1f}%\nFP: {int(fp)}"
        threshold_labels.append(label_text)

    if dca_threshold is not None and metrics_at_thresholds and "dca" in metrics_at_thresholds:
        m = metrics_at_thresholds["dca"]
        sens = m.get("sensitivity", np.nan)
        ppv = m.get("precision", np.nan)
        fp = m.get("fp", np.nan)

        line_handle = Line2D([0], [0], color="purple", linestyle="--", linewidth=2, alpha=0.7)
        threshold_handles.append(line_handle)

        # Multi-line label format with each metric on separate line
        label_text = "DCA"
        if not np.isnan(sens) and not np.isnan(ppv) and not np.isnan(fp):
            label_text += f"\nSens: {sens*100:.1f}%\nPPV: {ppv*100:.1f}%\nFP: {int(fp)}"
        threshold_labels.append(label_text)

    # Combine all handles and labels
    all_handles = handles + threshold_handles
    all_labels = labels + threshold_labels

    # Create legend outside plot area
    if all_handles:
        ax_main.legend(
            all_handles,
            all_labels,
            loc="upper left",
            bbox_to_anchor=(1.05, 1),
            fontsize=9,
            framealpha=0.9,
        )

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
            try:
                kde = gaussian_kde(incident_scores, bw_method="scott")
                x_range = np.linspace(0, 1, 200)
                density = kde(x_range)
                ax_incident.plot(x_range, density, color="firebrick", linewidth=2, alpha=0.8)
                ax_incident.fill_between(x_range, density, alpha=0.3, color="firebrick")
            except Exception:
                # Fallback to histogram if KDE fails (e.g., too few points)
                ax_incident.hist(
                    incident_scores,
                    bins=20,
                    density=True,
                    alpha=0.7,
                    color="firebrick",
                    edgecolor="white",
                )

        # Add threshold lines (no labels)
        if spec95_threshold is not None and 0 <= spec95_threshold <= 1:
            ax_incident.axvline(
                spec95_threshold, color="red", linestyle="--", linewidth=1.5, alpha=0.5
            )
        if youden_threshold is not None and 0 <= youden_threshold <= 1:
            ax_incident.axvline(
                youden_threshold,
                color="green",
                linestyle="--",
                linewidth=1.5,
                alpha=0.5,
            )
        if dca_threshold is not None and 0 <= dca_threshold <= 1:
            ax_incident.axvline(
                dca_threshold,
                color="purple",
                linestyle="--",
                linewidth=1.5,
                alpha=0.5,
            )

        ax_incident.set_xlim(0, 1)
        ax_incident.set_ylabel("Incident\nDensity", fontsize=9)
        ax_incident.grid(True, alpha=0.2, axis="x")
        ax_incident.set_yticks([])

        # Add statistics text
        stats_text = (
            f"Mean: {stats['mean']:.3f} | Median: {stats['median']:.3f} | "
            f"IQR: {stats['iqr']:.3f} | SD: {stats['sd']:.3f}"
        )
        ax_incident.text(
            0.02,
            0.95,
            stats_text,
            transform=ax_incident.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7},
        )

    # === PREVALENT DENSITY PLOT (if applicable) ===
    if has_prevalent:
        ax_prevalent = axes[subplot_idx]

        prevalent_scores = s[cat == "Prevalent"]
        stats = compute_distribution_stats(prevalent_scores)

        # Create KDE density plot
        if len(prevalent_scores) > 0:
            try:
                kde = gaussian_kde(prevalent_scores, bw_method="scott")
                x_range = np.linspace(0, 1, 200)
                density = kde(x_range)
                ax_prevalent.plot(x_range, density, color="darkorange", linewidth=2, alpha=0.8)
                ax_prevalent.fill_between(x_range, density, alpha=0.3, color="darkorange")
            except Exception:
                # Fallback to histogram if KDE fails
                ax_prevalent.hist(
                    prevalent_scores,
                    bins=20,
                    density=True,
                    alpha=0.7,
                    color="darkorange",
                    edgecolor="white",
                )

        # Add threshold lines (no labels)
        if spec95_threshold is not None and 0 <= spec95_threshold <= 1:
            ax_prevalent.axvline(
                spec95_threshold, color="red", linestyle="--", linewidth=1.5, alpha=0.5
            )
        if youden_threshold is not None and 0 <= youden_threshold <= 1:
            ax_prevalent.axvline(
                youden_threshold,
                color="green",
                linestyle="--",
                linewidth=1.5,
                alpha=0.5,
            )
        if dca_threshold is not None and 0 <= dca_threshold <= 1:
            ax_prevalent.axvline(
                dca_threshold,
                color="purple",
                linestyle="--",
                linewidth=1.5,
                alpha=0.5,
            )

        ax_prevalent.set_xlim(0, 1)
        ax_prevalent.set_ylabel("Prevalent\nDensity", fontsize=9)
        ax_prevalent.set_xlabel(xlabel)
        ax_prevalent.grid(True, alpha=0.2, axis="x")
        ax_prevalent.set_yticks([])

        # Add statistics text
        stats_text = (
            f"Mean: {stats['mean']:.3f} | Median: {stats['median']:.3f} | "
            f"IQR: {stats['iqr']:.3f} | SD: {stats['sd']:.3f}"
        )
        ax_prevalent.text(
            0.02,
            0.95,
            stats_text,
            transform=ax_prevalent.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7},
        )
    else:
        # If no prevalent subplot, add xlabel to last subplot
        if has_incident:
            axes[-1].set_xlabel(xlabel)
        else:
            ax_main.set_xlabel(xlabel)

    # Apply metadata and adjust layout
    bottom_margin = apply_plot_metadata(fig, meta_lines) if meta_lines else 0.1
    plt.subplots_adjust(left=0.12, right=0.70, top=0.92, bottom=bottom_margin, hspace=0.3)
    plt.savefig(out_path, dpi=150)
    plt.close()
