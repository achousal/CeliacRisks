"""
Out-of-fold (OOF) prediction plotting.

Provides functions to generate combined plots across CV repeats
with confidence bands showing variability across splits.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ced_ml.metrics.thresholds import compute_threshold_bundle
from ced_ml.plotting.calibration import plot_calibration_curve
from ced_ml.plotting.roc_pr import plot_pr_curve, plot_roc_curve


def plot_oof_combined(
    y_true: np.ndarray,
    oof_preds: np.ndarray,
    out_dir: Path,
    model_name: str,
    plot_format: str = "png",
    calib_bins: int = 10,
    meta_lines: Sequence[str] = None,
    target_spec: float = 0.95,
) -> None:
    """
    Generate combined OOF plots across CV repeats.

    Creates ROC, PR, and calibration plots with confidence bands
    showing variability across CV repeats.

    Args:
        y_true: True labels for training set (n_samples,)
        oof_preds: OOF predictions array (n_repeats, n_samples)
        out_dir: Output directory for plots
        model_name: Model identifier (e.g., "LR_EN", "XGBoost")
        plot_format: Output format ("png" or "pdf")
        calib_bins: Number of bins for calibration plots
        meta_lines: Metadata lines for plot annotation (required).
                   Use build_oof_metadata() or build_aggregated_metadata() to construct.
        target_spec: Target specificity for threshold computation (default 0.95)

    Returns:
        None. Saves plots to out_dir.

    Note:
        Requires n_repeats > 1 to show meaningful confidence bands.
        If n_repeats == 1, plots will still be generated but without bands.
    """
    n_repeats = oof_preds.shape[0]
    n_samples = len(y_true)

    if oof_preds.shape[1] != n_samples:
        raise ValueError(
            f"oof_preds shape {oof_preds.shape} incompatible with y_true length {n_samples}"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reshape for combined plotting:
    # - Stack predictions from all repeats
    # - Create split_ids to identify which repeat each prediction came from
    y_stacked = np.tile(y_true, n_repeats)
    p_stacked = oof_preds.ravel()  # Row-major: repeat0 samples, then repeat1, etc.
    split_ids = np.repeat(np.arange(n_repeats), n_samples)

    # Filter out NaN values (can occur if pooled data has incomplete repeat columns)
    valid_mask = ~np.isnan(p_stacked)
    if not valid_mask.all():
        y_stacked = y_stacked[valid_mask]
        p_stacked = p_stacked[valid_mask]
        split_ids = split_ids[valid_mask]

    # Validate metadata is provided
    if meta_lines is None:
        raise ValueError(
            "meta_lines is required. Use build_oof_metadata() or build_aggregated_metadata() "
            "to construct metadata before calling this function."
        )

    # Build plot titles
    oof_title = f"{model_name} - OOF (Train)"
    subtitle_suffix = " (mean +/- SD across repeats)" if n_repeats > 1 else ""

    # Compute thresholds on the stacked OOF predictions using standardized bundle
    threshold_bundle = compute_threshold_bundle(y_stacked, p_stacked, target_spec=target_spec)

    # ROC curve
    plot_roc_curve(
        y_true=y_stacked,
        y_pred=p_stacked,
        out_path=out_dir / f"{model_name}__oof_roc.{plot_format}",
        title=oof_title,
        subtitle=f"ROC Curve{subtitle_suffix}",
        split_ids=split_ids if n_repeats > 1 else None,
        meta_lines=meta_lines,
        threshold_bundle=threshold_bundle,
    )

    # PR curve
    plot_pr_curve(
        y_true=y_stacked,
        y_pred=p_stacked,
        out_path=out_dir / f"{model_name}__oof_pr.{plot_format}",
        title=oof_title,
        subtitle=f"PR Curve{subtitle_suffix}",
        split_ids=split_ids if n_repeats > 1 else None,
        meta_lines=meta_lines,
    )

    # Calibration plot
    calib_subtitle = "Calibration (quantile)" if n_repeats > 1 else "Calibration"
    plot_calibration_curve(
        y_true=y_stacked,
        y_pred=p_stacked,
        out_path=out_dir / f"{model_name}__oof_calibration.{plot_format}",
        title=oof_title,
        subtitle=calib_subtitle,
        n_bins=calib_bins,
        split_ids=split_ids if n_repeats > 1 else None,
        meta_lines=meta_lines,
    )
