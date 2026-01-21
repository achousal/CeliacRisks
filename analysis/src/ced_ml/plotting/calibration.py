"""
Calibration plotting utilities.

This module provides calibration curve plotting in both probability and logit space:
- Probability-space calibration (observed vs predicted frequencies)
- Logit-space calibration (log-odds)
- Multi-split aggregation with confidence bands
- LOESS smoothing for logit calibration
- Binomial confidence intervals

References:
    Van Calster et al. (2016). Calibration: the Achilles heel of predictive analytics.
    BMC Medicine.

    Austin & Steyerberg (2019). The Integrated Calibration Index (ICI).
    Statistics in Medicine.
"""

import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _get_legend_reference_sizes(actual_sizes: np.ndarray) -> list:
    """
    Compute appropriate legend reference sizes based on actual bin sizes.

    Args:
        actual_sizes: Array of actual bin sizes from the data

    Returns:
        List of 3-4 representative sample counts for legend
    """
    if len(actual_sizes) == 0 or actual_sizes.max() == 0:
        return [10, 50, 100, 200]

    min_size = int(actual_sizes.min())
    max_size = int(actual_sizes.max())

    # If range is small, use actual min/max and interpolate
    if max_size - min_size < 50:
        return [min_size, max_size]

    # Generate 3-4 evenly spaced reference points
    # Round to nice numbers (multiples of 10, 50, or 100)
    def round_to_nice(x):
        if x < 50:
            return int(np.round(x / 10) * 10)
        elif x < 200:
            return int(np.round(x / 25) * 25)
        else:
            return int(np.round(x / 50) * 50)

    # Create quartile-based reference points
    q25 = round_to_nice(np.percentile(actual_sizes, 25))
    q50 = round_to_nice(np.percentile(actual_sizes, 50))
    q75 = round_to_nice(np.percentile(actual_sizes, 75))
    q_max = round_to_nice(max_size)

    # Filter duplicates and sort
    sizes = sorted({q25, q50, q75, q_max})

    # Ensure we have at least 2 reference points
    if len(sizes) < 2:
        sizes = [min_size, max_size]

    return sizes


def _plot_prob_calibration_panel(
    ax,
    y: np.ndarray,
    p: np.ndarray,
    bins: np.ndarray,
    bin_centers: np.ndarray,
    actual_n_bins: int,
    bin_strategy: str,
    split_ids: Optional[np.ndarray] = None,
    unique_splits: Optional[list] = None,
    panel_title: str = "",
    variable_sizes: bool = True,
) -> None:
    """
    Plot a single probability-space calibration panel.

    Args:
        ax: Matplotlib axis to plot on
        y: True labels (0/1)
        p: Predicted probabilities
        bins: Bin edges
        bin_centers: Center points of bins
        actual_n_bins: Number of bins
        bin_strategy: 'uniform' or 'quantile'
        split_ids: Optional split identifiers
        unique_splits: List of unique split IDs
        panel_title: Title for this panel
        variable_sizes: If True, circle sizes vary with bin sample counts
    """
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration", alpha=0.7)

    if unique_splits is not None and len(unique_splits) > 1:
        curves = []
        counts_all = []
        for sid in unique_splits:
            m_split = (split_ids == sid) if sid is not None else np.isnan(split_ids)
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
        np.nanmean(counts_all, axis=0)
        sum_counts = np.nansum(counts_all, axis=0)

        ax.fill_between(
            bin_centers,
            np.clip(obs_lo, 0, 1),
            np.clip(obs_hi, 0, 1),
            color="steelblue",
            alpha=0.15,
            label="95% CI",
        )
        ax.fill_between(
            bin_centers,
            np.clip(obs_mean - obs_sd, 0, 1),
            np.clip(obs_mean + obs_sd, 0, 1),
            color="steelblue",
            alpha=0.30,
            label="±1 SD",
        )

        valid = ~np.isnan(obs_mean) & (sum_counts > 0)
        # Only use variable marker sizes for uniform binning; quantile binning gets fixed sizes
        if bin_strategy == "quantile":
            scatter_sizes = 50  # Fixed size for quantile binning
        elif variable_sizes:
            # Use variable marker sizes based on aggregate counts
            scatter_sizes = np.clip(sum_counts[valid] * 1, 5, 300)
        else:
            # Fixed marker size for all points
            scatter_sizes = 30
        ax.scatter(
            bin_centers[valid],
            obs_mean[valid],
            s=scatter_sizes,
            color="steelblue",
            alpha=0.7,
            edgecolors="darkblue",
            linewidths=0.5,
        )
        ax.plot(
            bin_centers,
            obs_mean,
            color="steelblue",
            linewidth=2,
            alpha=0.6,
            label=f"Mean (n={len(curves)} splits)",
        )
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
        ax.scatter(
            pred_means[valid],
            obs[valid],
            s=scatter_sizes,
            color="steelblue",
            alpha=0.7,
            edgecolors="darkblue",
            linewidths=0.5,
        )
        ax.plot(pred_means[valid], obs[valid], color="steelblue", linewidth=1.5, alpha=0.6)

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

    # Add size legend for uniform binning
    if bin_strategy == "uniform":
        from matplotlib.lines import Line2D

        # Determine the actual sizing formula used in the scatter plot
        if unique_splits is not None and len(unique_splits) > 1:
            # Multi-split case: uses sum_counts * 1
            size_multiplier = 1
            actual_bin_sizes = sum_counts[sum_counts > 0]
        else:
            # Single-split case: uses sizes * 3
            size_multiplier = 3
            actual_bin_sizes = sizes[sizes > 0]

        # Get legend reference sizes based on actual data
        reference_sizes = _get_legend_reference_sizes(actual_bin_sizes)

        size_handles = []
        size_labels = []
        for sample_count in reference_sizes:
            # Match the actual scatter plot sizing formula
            # scatter() 's' parameter is area in points^2
            scatter_area = np.clip(sample_count * size_multiplier, 5, 300)
            # Line2D markersize is the marker width/diameter in points
            # Convert: diameter = sqrt(area) because area = pi*r^2 and diameter ≈ 2*r
            # But matplotlib scatter uses a simpler area calculation, so: diameter = sqrt(area)
            markersize = np.sqrt(scatter_area)
            handle = Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="steelblue",
                markersize=markersize,
                markeredgecolor="darkblue",
                markeredgewidth=0.5,
                linestyle="None",
                alpha=0.6,
            )
            size_handles.append(handle)
            size_labels.append(f"{sample_count}")

        # Position legend to the right with adequate spacing
        size_legend = ax.legend(
            size_handles,
            size_labels,
            title="Bin size (n)",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            fontsize=8,
            title_fontsize=9,
            framealpha=0.9,
            labelspacing=1.2,  # Increase vertical spacing between legend entries
        )
        ax.add_artist(size_legend)
        # Re-add main legend (was overwritten by size legend)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9, labelspacing=1.0)
    else:
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9, labelspacing=1.0)


def _binned_logits(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
    bin_strategy: str = "quantile",
    min_bin_size: int = 30,
    merge_tail: bool = True,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """
    Compute binned logits for calibration plot in logit space with binomial CIs.

    Logits are log-odds of probabilities: log(p/(1-p)). Creates calibration curve
    with predicted logits on x-axis and observed logits on y-axis.

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

    # Assign samples to bins
    bin_idx = np.digitize(p, bins) - 1
    bin_idx = np.clip(bin_idx, 0, len(bins) - 2)

    # Compute per-bin statistics
    xs_list, ys_list, ys_lo_list, ys_hi_list, sizes_list = [], [], [], [], []
    eps = 1e-7

    for i in range(len(bins) - 1):
        mask_bin = bin_idx == i
        if mask_bin.sum() < min_bin_size and merge_tail:
            continue

        y_bin = y[mask_bin]
        p_bin = p[mask_bin]

        if len(y_bin) == 0:
            continue

        # Predicted logit (mean of bin)
        p_mean = np.mean(p_bin)
        p_mean_clipped = np.clip(p_mean, eps, 1 - eps)
        logit_pred = np.log(p_mean_clipped / (1 - p_mean_clipped))

        # Observed proportion and Wilson CI
        n = len(y_bin)
        k = int(y_bin.sum())
        ci_lo, ci_hi = proportion_confint(k, n, alpha=0.05, method="wilson")

        # Convert to logits (with clipping for numerical stability)
        obs_prop = k / n
        obs_prop_clipped = np.clip(obs_prop, eps, 1 - eps)
        ci_lo_clipped = np.clip(ci_lo, eps, 1 - eps)
        ci_hi_clipped = np.clip(ci_hi, eps, 1 - eps)

        logit_obs = np.log(obs_prop_clipped / (1 - obs_prop_clipped))
        logit_obs_lo = np.log(ci_lo_clipped / (1 - ci_lo_clipped))
        logit_obs_hi = np.log(ci_hi_clipped / (1 - ci_hi_clipped))

        xs_list.append(logit_pred)
        ys_list.append(logit_obs)
        ys_lo_list.append(logit_obs_lo)
        ys_hi_list.append(logit_obs_hi)
        sizes_list.append(n)

    if len(xs_list) == 0:
        return None, None, None, None, None

    return (
        np.array(xs_list),
        np.array(ys_list),
        np.array(ys_lo_list),
        np.array(ys_hi_list),
        np.array(sizes_list),
    )


def _plot_logit_calibration_panel(
    ax,
    y: np.ndarray,
    p: np.ndarray,
    n_bins: int,
    bin_strategy: str,
    split_ids: Optional[np.ndarray],
    unique_splits: Optional[list],
    panel_title: str,
    lowess,
    calib_intercept: Optional[float],
    calib_slope: Optional[float],
    eps: float = 1e-7,
) -> None:
    """
    Plot a single logit-space calibration panel.

    Args:
        ax: Matplotlib axis to plot on
        y: True labels (0/1)
        p: Predicted probabilities
        n_bins: Number of bins for binning
        bin_strategy: 'uniform' or 'quantile'
        split_ids: Optional split identifiers
        unique_splits: List of unique split IDs
        panel_title: Title for this panel
        lowess: LOESS function from statsmodels (or None)
        calib_intercept: Calibration intercept (alpha) from logistic recalibration
        calib_slope: Calibration slope (beta) from logistic recalibration
        eps: Small epsilon for clipping probabilities
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
        # Define fixed probability bins
        prob_x_bins = []
        prob_y_bins = []
        bin_sizes_per_split = []

        for sid in unique_splits:
            m_split = (split_ids == sid) if sid is not None else np.isnan(split_ids)
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

                    # Apply Jeffreys smoothing to avoid 0/1
                    n_in_bin = m_bin.sum()
                    k_in_bin = np.sum(y_s[m_bin])
                    obs_freq_smoothed = (k_in_bin + 0.5) / (n_in_bin + 1)
                    prob_y_per_split.append(obs_freq_smoothed)
                    bin_sizes_per_bin.append(n_in_bin)

            prob_x_bins.append(prob_x_per_split)
            prob_y_bins.append(prob_y_per_split)
            bin_sizes_per_split.append(bin_sizes_per_bin)

        # Aggregate across splits in PROBABILITY SPACE
        prob_x_bins = np.array(prob_x_bins, dtype=float)
        prob_y_bins = np.array(prob_y_bins, dtype=float)
        bin_sizes_per_split = np.array(bin_sizes_per_split, dtype=int)

        # Aggregate predicted and observed probabilities across splits
        prob_x_mean = np.nanmean(prob_x_bins, axis=0)
        prob_y_mean = np.nanmean(prob_y_bins, axis=0)
        prob_y_lo = np.nanpercentile(prob_y_bins, 2.5, axis=0)
        prob_y_hi = np.nanpercentile(prob_y_bins, 97.5, axis=0)
        np.nanstd(prob_y_bins, axis=0)

        # Aggregate bin sizes across splits
        bin_sizes_mean = np.nanmean(bin_sizes_per_split, axis=0)

        # NOW convert aggregated probabilities to logit space
        logit_x_mean = np.log(
            np.clip(prob_x_mean, eps, 1 - eps) / (1 - np.clip(prob_x_mean, eps, 1 - eps))
        )
        logit_y_mean = np.log(
            np.clip(prob_y_mean, eps, 1 - eps) / (1 - np.clip(prob_y_mean, eps, 1 - eps))
        )
        logit_y_lo = np.log(
            np.clip(prob_y_lo, eps, 1 - eps) / (1 - np.clip(prob_y_lo, eps, 1 - eps))
        )
        logit_y_hi = np.log(
            np.clip(prob_y_hi, eps, 1 - eps) / (1 - np.clip(prob_y_hi, eps, 1 - eps))
        )

        # For SD: compute logit of each smoothed split value, then take SD in logit space
        logit_curves_smooth = np.log(
            np.clip(prob_y_bins, eps, 1 - eps) / (1 - np.clip(prob_y_bins, eps, 1 - eps))
        )
        logit_y_sd = np.nanstd(logit_curves_smooth, axis=0)

        # Plot aggregated logit calibration bands
        valid_logit = ~np.isnan(logit_x_mean) & ~np.isnan(logit_y_mean)
        if valid_logit.sum() > 0:
            ax.fill_between(
                logit_x_mean[valid_logit],
                logit_y_lo[valid_logit],
                logit_y_hi[valid_logit],
                color="steelblue",
                alpha=0.15,
                label="95% CI",
            )
            ax.fill_between(
                logit_x_mean[valid_logit],
                np.clip(logit_y_mean[valid_logit] - logit_y_sd[valid_logit], -20, 20),
                np.clip(logit_y_mean[valid_logit] + logit_y_sd[valid_logit], -20, 20),
                color="steelblue",
                alpha=0.30,
                label="±1 SD",
            )

            # Plot line connecting bin centers
            ax.plot(
                logit_x_mean[valid_logit],
                logit_y_mean[valid_logit],
                "-",
                color="steelblue",
                linewidth=2,
                alpha=0.6,
                zorder=4,
            )

            # Compute marker sizes: only variable for uniform binning
            if bin_strategy == "quantile":
                marker_sizes = 6
            else:
                valid_sizes = bin_sizes_mean[valid_logit]
                if len(valid_sizes) > 0 and valid_sizes.max() > 0:
                    min_size, max_size = 4, 16
                    norm_sizes = (valid_sizes - valid_sizes.min()) / (
                        valid_sizes.max() - valid_sizes.min() + 1e-7
                    )
                    marker_sizes = min_size + norm_sizes * (max_size - min_size)
                else:
                    marker_sizes = 6

            # Plot markers
            ax.scatter(
                logit_x_mean[valid_logit],
                logit_y_mean[valid_logit],
                s=marker_sizes**2,
                marker="o",
                color="steelblue",
                alpha=0.7,
                edgecolors="darkblue",
                linewidth=0.5,
                label=f"Mean logit calib (n={len(unique_splits)} splits)",
                zorder=5,
            )

            loess_ok = True  # Skip LOESS when multi-split aggregation is used

        # Determine axis ranges from aggregated data
        if valid_logit.sum() > 0:
            logit_range_x = [
                np.nanpercentile(logit_x_mean[valid_logit], 1) - 0.5,
                np.nanpercentile(logit_x_mean[valid_logit], 99) + 0.5,
            ]
            logit_range_y = [
                np.nanpercentile(logit_y_lo[valid_logit], 1) - 0.5,
                np.nanpercentile(logit_y_hi[valid_logit], 99) + 0.5,
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
        y_sorted = y[sort_idx].astype(np.float64)

        # Apply LOESS
        loess_result = lowess(y_sorted, p_sorted, frac=0.3, return_sorted=True, it=0)

        loess_p = loess_result[:, 0]
        loess_prob = loess_result[:, 1]

        # LOESS can overshoot [0,1] bounds at boundaries
        # Truncate the curve where LOESS overshoots
        clip_lower = 0.001
        clip_upper = 0.999

        valid_loess_mask = (loess_prob >= clip_lower) & (loess_prob <= clip_upper)

        if valid_loess_mask.sum() < 10:
            # Fallback to clipping
            loess_prob = np.clip(loess_prob, clip_lower, clip_upper)
            loess_p_clipped = np.clip(loess_p, clip_lower, clip_upper)
            loess_logit_y = np.log(loess_prob / (1 - loess_prob))
            loess_x = np.log(loess_p_clipped / (1 - loess_p_clipped))
        else:
            # Truncate to valid region
            loess_prob = loess_prob[valid_loess_mask]
            loess_p = loess_p[valid_loess_mask]
            loess_p = np.clip(loess_p, clip_lower, clip_upper)

            # Convert to log-odds
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
    except Exception:
        loess_ok = False

    # Determine axis ranges based on actual data
    if not (unique_splits is not None and len(unique_splits) > 1):
        logit_min = np.percentile(logit_pred, 1)
        logit_max = np.percentile(logit_pred, 99)
        logit_range_x = [logit_min - 0.5, logit_max + 0.5]

        # Y-axis range should accommodate both LOESS and recalibration line
        logit_range_y = list(logit_range_x)

        if loess_ok and loess_logit_y is not None:
            loess_min = np.percentile(loess_logit_y, 1)
            loess_max = np.percentile(loess_logit_y, 99)
            logit_range_y = [
                min(logit_range_y[0], loess_min - 0.5),
                max(logit_range_y[1], loess_max + 0.5),
            ]

    # Plot ideal calibration line
    ax.plot(
        logit_range_x,
        logit_range_x,
        "k--",
        linewidth=1.5,
        alpha=0.7,
        label="Ideal (α=0, β=1)",
    )

    # Plot recalibration line if available
    if (
        calib_intercept is not None
        and calib_slope is not None
        and np.isfinite(calib_intercept)
        and np.isfinite(calib_slope)
    ):
        recal_x = np.array(logit_range_x)
        recal_y = calib_intercept + calib_slope * recal_x
        ax.plot(
            recal_x,
            recal_y,
            "r-",
            linewidth=2,
            alpha=0.8,
            label=f"Recalibration (α={calib_intercept:.2f}, β={calib_slope:.2f})",
        )
        # Extend y-range if recalibration line goes outside
        logit_range_y = [
            min(logit_range_y[0], recal_y.min() - 0.5),
            max(logit_range_y[1], recal_y.max() + 0.5),
        ]

    # Compute binned observations (skip for multi-split, already computed above)
    if not (unique_splits is not None and len(unique_splits) > 1):
        binned_result = _binned_logits(
            y,
            p,
            n_bins=n_bins,
            bin_strategy=bin_strategy,
            min_bin_size=30,
            merge_tail=True,
        )
        bx, by, by_lo, by_hi, bin_sizes = binned_result

        # Plot observed calibration (LOESS or binned)
        method_label = ""
        if loess_ok and loess_x is not None and loess_logit_y is not None:
            # Plot LOESS curve on top of CI band
            label = "LOESS (smoothed)"
            ax.plot(
                loess_x,
                loess_logit_y,
                "steelblue",
                linewidth=2.5,
                alpha=0.6,
                label=label,
                zorder=5,
            )
            method_label = "LOESS"

            # Overlay binned observations with binomial CIs
            if bx is not None and by is not None and by_lo is not None and by_hi is not None:
                yerr_lo = by - by_lo
                yerr_hi = by_hi - by
                yerr = np.vstack([yerr_lo, yerr_hi])

                # Compute marker sizes scaled by bin sample numbers
                if bin_sizes is not None and len(bin_sizes) > 0:
                    min_size, max_size = 4, 16
                    norm_sizes = (bin_sizes - bin_sizes.min()) / (
                        bin_sizes.max() - bin_sizes.min() + 1e-7
                    )
                    marker_sizes = min_size + norm_sizes * (max_size - min_size)
                else:
                    marker_sizes = 7

                # Plot error bars
                ax.errorbar(
                    bx,
                    by,
                    yerr=yerr,
                    fmt="none",
                    capsize=4,
                    capthick=1.5,
                    color="darkorange",
                    ecolor="darkorange",
                    alpha=0.8,
                    zorder=9,
                )

                # Plot markers
                ax.scatter(
                    bx,
                    by,
                    s=marker_sizes**2,
                    marker="o",
                    color="darkorange",
                    alpha=0.7,
                    edgecolors="darkred",
                    linewidth=0.5,
                    label=f"Binned observations (n={len(bx)} bins, Wilson CI)",
                    zorder=10,
                )

                # Extend y-range for binned CIs
                logit_range_y = [
                    min(logit_range_y[0], by_lo.min() - 0.3),
                    max(logit_range_y[1], by_hi.max() + 0.3),
                ]
        else:
            # Fallback: only binned data available
            if bx is not None and by is not None and by_lo is not None and by_hi is not None:
                yerr_lo = by - by_lo
                yerr_hi = by_hi - by
                yerr = np.vstack([yerr_lo, yerr_hi])

                # Compute marker sizes
                if bin_sizes is not None and len(bin_sizes) > 0:
                    min_size, max_size = 4, 16
                    norm_sizes = (bin_sizes - bin_sizes.min()) / (
                        bin_sizes.max() - bin_sizes.min() + 1e-7
                    )
                    marker_sizes = min_size + norm_sizes * (max_size - min_size)
                else:
                    marker_sizes = 7

                # Plot error bars
                ax.errorbar(
                    bx,
                    by,
                    yerr=yerr,
                    fmt="none",
                    capsize=4,
                    capthick=1.5,
                    color="steelblue",
                    ecolor="steelblue",
                    alpha=0.8,
                    linewidth=2,
                    zorder=4,
                )

                # Plot line connecting bin centers
                ax.plot(bx, by, "-", color="steelblue", linewidth=2, alpha=0.6, zorder=4)

                # Plot markers
                ax.scatter(
                    bx,
                    by,
                    s=marker_sizes**2,
                    marker="o",
                    color="steelblue",
                    alpha=0.7,
                    edgecolors="darkblue",
                    linewidth=0.5,
                    label=f"Binned logits (n={len(bx)} bins, Wilson CI)",
                    zorder=5,
                )

                method_label = "Binned"
                # Extend y-range for binned data
                logit_range_y = [
                    min(logit_range_y[0], by_lo.min() - 0.5),
                    max(logit_range_y[1], by_hi.max() + 0.5),
                ]
    else:
        # Multi-split mode: aggregated bands already plotted
        method_label = "Multi-split aggregated"

    ax.set_title(panel_title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted logit: logit(p̂)", fontsize=11)
    ylabel = f"Empirical logit ({method_label})" if method_label else "Empirical logit"
    ax.set_ylabel(ylabel, fontsize=11)

    # Add size legend for uniform binning (match probability-space legend style)
    if bin_strategy == "uniform":
        from matplotlib.lines import Line2D

        # Determine actual bin sizes from the data
        # Multi-split case: bin_sizes_mean already computed
        # Single-split case: bin_sizes from _binned_logits
        if unique_splits is not None and len(unique_splits) > 1:
            # Multi-split: use bin_sizes_mean from earlier computation
            actual_bin_sizes = bin_sizes_mean[bin_sizes_mean > 0]
        else:
            # Single-split: use bin_sizes from binned_result
            if bin_sizes is not None and len(bin_sizes) > 0:
                actual_bin_sizes = bin_sizes[bin_sizes > 0]
            else:
                actual_bin_sizes = np.array([])

        # Get legend reference sizes based on actual data
        reference_sizes = _get_legend_reference_sizes(actual_bin_sizes)

        size_handles = []
        size_labels = []

        # Determine min/max for normalization (matching actual plot logic)
        if len(actual_bin_sizes) > 0:
            data_min = actual_bin_sizes.min()
            data_max = actual_bin_sizes.max()
        else:
            data_min = reference_sizes[0] if reference_sizes else 10
            data_max = reference_sizes[-1] if reference_sizes else 200

        for sample_count in reference_sizes:
            # Compute normalized marker size matching actual plot logic
            # Logit uses: norm_sizes = (bin_sizes - min) / (max - min + eps)
            # marker_sizes = min_size + norm_sizes * (max_size - min_size)
            if data_max > data_min:
                norm_size = (sample_count - data_min) / (data_max - data_min + 1e-7)
            else:
                norm_size = 0.5
            min_marker, max_marker = 4, 16
            marker_size = min_marker + norm_size * (max_marker - min_marker)

            handle = Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="steelblue",
                markersize=marker_size,
                markeredgecolor="darkblue",
                markeredgewidth=0.5,
                linestyle="None",
                alpha=0.8,
            )
            size_handles.append(handle)
            size_labels.append(f"{sample_count}")

        # Position legend to the right with adequate spacing
        size_legend = ax.legend(
            size_handles,
            size_labels,
            title="Bin size (n)",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            fontsize=8,
            title_fontsize=9,
            framealpha=0.9,
            labelspacing=1.2,  # Increase vertical spacing between legend entries
        )
        ax.add_artist(size_legend)
        # Re-add main legend
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9, labelspacing=1.0)
    else:
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9, labelspacing=1.0)

    ax.grid(True, alpha=0.3)
    ax.set_xlim(logit_range_x)
    ax.set_ylim(logit_range_y)


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
        return 0.12  # Default minimum bottom margin

    # Position metadata at very bottom with fixed offset from edge
    fig.text(0.5, 0.005, "\n".join(lines), ha="center", va="bottom", fontsize=8, wrap=True)

    # Calculate required bottom margin: base + space per line
    # Increased spacing for better separation between metadata and figures
    required_bottom = 0.12 + (0.022 * len(lines))
    return min(required_bottom, 0.30)  # Cap at 30%


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    title: str,
    subtitle: str = "",
    n_bins: int = 10,
    split_ids: Optional[np.ndarray] = None,
    meta_lines: Optional[Sequence[str]] = None,
    bin_strategy: str = "uniform",
    calib_intercept: Optional[float] = None,
    calib_slope: Optional[float] = None,
    four_panel: bool = False,
) -> None:
    """
    Generate 4-panel calibration plot.

    Always generates a 2x2 layout:
        Panel 1 (top-left): Calibration curve with quantile binning
        Panel 2 (top-right): Calibration curve with uniform binning
        Panel 3 (bottom-left): Logit calibration curve with quantile binning
        Panel 4 (bottom-right): Logit calibration curve with uniform binning

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
        four_panel: Deprecated parameter (always True, kept for backward compatibility)
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.error(f"Calibration plot failed to import matplotlib: {e}")
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
    np.clip(p, eps, 1 - eps)

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

    # Compute actual bin sizes from the quantile bins
    bin_idx_q = np.digitize(p, bins_quantile) - 1
    bin_idx_q = np.clip(bin_idx_q, 0, actual_n_bins_q - 1)
    bin_sizes_q = np.array([int((bin_idx_q == i).sum()) for i in range(actual_n_bins_q)])

    # Compute per-bin sample counts
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
        panel_title_1 = (
            f"{subtitle} – Calibration (quantile bins)\nk={actual_n_bins_q}, {bin_size_str}"
        )

    _plot_prob_calibration_panel(
        ax1,
        y,
        p,
        bins_quantile,
        bin_centers_q,
        actual_n_bins_q,
        "quantile",
        split_ids=split_ids,
        unique_splits=unique_splits,
        panel_title=panel_title_1,
        variable_sizes=False,
    )

    # ========== Panel 2 (top-right): Probability-space calibration curve with uniform binning ==========
    bins_uniform = np.linspace(0, 1, n_bins + 1)
    actual_n_bins_u = len(bins_uniform) - 1
    bin_centers_u = (bins_uniform[:-1] + bins_uniform[1:]) / 2

    panel_title_2 = f"Calibration (uniform bins)\nk={actual_n_bins_u}"
    if subtitle:
        panel_title_2 = f"{subtitle} – Calibration (uniform bins)\nk={actual_n_bins_u}"

    _plot_prob_calibration_panel(
        ax2,
        y,
        p,
        bins_uniform,
        bin_centers_u,
        actual_n_bins_u,
        "uniform",
        split_ids=split_ids,
        unique_splits=unique_splits,
        panel_title=panel_title_2,
        variable_sizes=True,
    )

    # ========== Panel 3 (bottom-left): Log-odds calibration with quantile binning ==========
    logit_title_q = "Logit calibration (quantile bins)"
    _plot_logit_calibration_panel(
        ax3,
        y,
        p,
        n_bins,
        "quantile",
        split_ids,
        unique_splits,
        logit_title_q,
        lowess,
        calib_intercept,
        calib_slope,
        eps=eps,
    )

    # ========== Panel 4 (bottom-right): Log-odds calibration with uniform binning ==========
    logit_title_u = "Logit calibration (uniform bins)"
    _plot_logit_calibration_panel(
        ax4,
        y,
        p,
        n_bins,
        "uniform",
        split_ids,
        unique_splits,
        logit_title_u,
        lowess,
        calib_intercept,
        calib_slope,
        eps=eps,
    )

    # Add title at the top
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    # Apply metadata and adjust layout
    bottom_margin = _apply_plot_metadata(fig, meta_lines)
    # Increase right margin to accommodate size legend in uniform binning panels
    plt.subplots_adjust(left=0.10, right=0.88, top=0.92, bottom=bottom_margin)

    # Save figure
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, pad_inches=0.1)
    plt.close()

    logger.info(f"Calibration plot saved to {out_path}")
