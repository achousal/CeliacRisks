"""
CLI implementation for aggregate-splits command.

Aggregates results across multiple split seeds into summary statistics,
pooled predictions, aggregated plots, and consensus panels.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ced_ml.cli.aggregation.collection import (
    collect_best_hyperparams,
    collect_ensemble_hyperparams,
    collect_ensemble_predictions,
    collect_feature_reports,
    collect_metrics,
    collect_predictions,
)
from ced_ml.cli.aggregation.discovery import (
    discover_ensemble_dirs,
    discover_split_dirs,
)
from ced_ml.config.loader import load_aggregate_config
from ced_ml.evaluation.scoring import compute_selection_scores_for_models
from ced_ml.metrics.dca import threshold_dca_zero_crossing
from ced_ml.metrics.discrimination import (
    compute_brier_score,
    compute_discrimination_metrics,
)
from ced_ml.metrics.thresholds import (
    binary_metrics_at_threshold,
    compute_multi_target_specificity_metrics,
    compute_threshold_bundle,
    threshold_for_specificity,
    threshold_youden,
)
from ced_ml.utils.logging import log_section, setup_logger
from ced_ml.utils.metadata import build_aggregated_metadata


def run_aggregate_splits_with_config(
    config_file: str | None = None,
    overrides: list[str] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Wrapper for run_aggregate_splits that loads config from YAML.

    Args:
        config_file: Path to aggregate_config.yaml (optional)
        overrides: List of CLI overrides in "key=value" format
        **kwargs: Additional keyword arguments override config values

    Returns:
        Dictionary with aggregation results summary
    """
    config = load_aggregate_config(config_file=config_file, overrides=overrides)

    params = {
        "results_dir": str(config.results_dir),
        "stability_threshold": config.min_stability,
        "plot_formats": [config.plot_format] if hasattr(config, "plot_format") else ["png"],
        "target_specificity": 0.95,
        "n_boot": 500,
        "verbose": 0,
        "save_plots": config.save_plots,
        "plot_roc": config.plot_roc,
        "plot_pr": config.plot_pr,
        "plot_calibration": config.plot_calibration,
        "plot_risk_distribution": config.plot_risk_distribution,
        "plot_dca": config.plot_dca,
        "plot_oof_combined": config.plot_oof_combined,
        "plot_learning_curve": config.plot_learning_curve,
    }

    params.update(kwargs)

    return run_aggregate_splits(**params)


def compute_summary_stats(
    metrics_df: pd.DataFrame,
    group_cols: list[str] | None = None,
    metric_cols: list[str] | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Compute summary statistics (mean, std, min, max, count) across splits.

    Args:
        metrics_df: DataFrame with metrics from all splits
        group_cols: Columns to group by (e.g., ["scenario", "model"])
        metric_cols: Numeric columns to summarize (auto-detected if None)
        logger: Optional logger instance

    Returns:
        DataFrame with summary statistics
    """
    if metrics_df.empty:
        if logger:
            logger.debug("No metrics to summarize (empty DataFrame)")
        return pd.DataFrame()

    if group_cols is None:
        group_cols = []
        for col in ["scenario", "model"]:
            if col in metrics_df.columns:
                group_cols.append(col)

    if metric_cols is None:
        exclude = {"split_seed", "scenario", "model", "seed", "random_state"}
        metric_cols = [
            col
            for col in metrics_df.columns
            if metrics_df[col].dtype in [np.float64, np.int64, float, int] and col not in exclude
        ]

    if not group_cols:
        summary_rows = []
        for col in metric_cols:
            values = metrics_df[col].dropna()
            if len(values) > 0:
                summary_rows.append(
                    {
                        "metric": col,
                        "mean": values.mean(),
                        "std": values.std(),
                        "min": values.min(),
                        "max": values.max(),
                        "count": len(values),
                    }
                )
        return pd.DataFrame(summary_rows)

    summary_rows = []
    for group_vals, group_df in metrics_df.groupby(group_cols):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)

        row = dict(zip(group_cols, group_vals, strict=False))
        row["n_splits"] = len(group_df)

        for col in metric_cols:
            values = group_df[col].dropna()
            if len(values) > 0:
                row[f"{col}_mean"] = values.mean()
                row[f"{col}_std"] = values.std()
                row[f"{col}_ci95_lo"] = values.mean() - 1.96 * values.std() / np.sqrt(len(values))
                row[f"{col}_ci95_hi"] = values.mean() + 1.96 * values.std() / np.sqrt(len(values))

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    if logger:
        logger.info(f"Computed summary stats: {len(summary_df)} groups, {len(metric_cols)} metrics")

    return summary_df


def aggregate_hyperparams_summary(
    params_df: pd.DataFrame,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Compute summary statistics for hyperparameters across splits.

    For each model and hyperparameter:
    - Numeric params: mean, std, min, max
    - Categorical params: mode, unique values

    Args:
        params_df: DataFrame with hyperparameters from all splits
        logger: Optional logger instance

    Returns:
        DataFrame with aggregated hyperparameter statistics per model
    """
    if params_df.empty:
        if logger:
            logger.debug("No hyperparameters to aggregate (empty DataFrame)")
        return pd.DataFrame()

    # Identify parameter columns (exclude metadata)
    metadata_cols = {
        "split_seed",
        "model",
        "repeat",
        "outer_split",
        "best_score_inner",
        "optuna_n_trials",
        "optuna_sampler",
        "optuna_pruner",
    }
    param_cols = [col for col in params_df.columns if col not in metadata_cols]

    if not param_cols:
        if logger:
            logger.warning("No hyperparameter columns found to aggregate")
        return pd.DataFrame()

    summary_rows = []

    # Group by model
    for model_name, model_df in params_df.groupby("model"):
        row = {"model": model_name, "n_cv_folds": len(model_df)}

        for param in param_cols:
            if param not in model_df.columns:
                continue

            values = model_df[param].dropna()
            if len(values) == 0:
                continue

            # Check if numeric
            if pd.api.types.is_numeric_dtype(values):
                row[f"{param}_mean"] = values.mean()
                row[f"{param}_std"] = values.std()
                row[f"{param}_min"] = values.min()
                row[f"{param}_max"] = values.max()
            else:
                # Categorical: get mode and unique count
                mode_val = values.mode()
                row[f"{param}_mode"] = mode_val[0] if len(mode_val) > 0 else None
                row[f"{param}_n_unique"] = values.nunique()
                # Include all unique values as comma-separated string
                unique_vals = sorted(values.unique())
                row[f"{param}_values"] = ", ".join(str(v) for v in unique_vals)

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    if logger:
        logger.info(
            f"Aggregated hyperparams for {len(summary_df)} models, " f"{len(param_cols)} parameters"
        )

    return summary_df


def compute_pooled_metrics(
    pooled_df: pd.DataFrame,
    y_col: str = "y_true",
    pred_col: str = "y_prob",
    spec_targets: list[float] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, float]:
    """
    Compute metrics on pooled predictions.

    Args:
        pooled_df: DataFrame with pooled predictions
        y_col: Column name for true labels
        pred_col: Column name for predicted probabilities
        spec_targets: List of specificity targets for threshold computation
        logger: Optional logger instance

    Returns:
        Dictionary of computed metrics
    """
    if pooled_df.empty:
        if logger:
            logger.debug("Empty pooled DataFrame, no metrics computed")
        return {}

    # Find the prediction column (might be y_prob, y_pred, risk_score, etc.)
    pred_cols = [
        c
        for c in pooled_df.columns
        if c in ["y_prob", pred_col, "risk_score", "prob", "prediction"]
    ]
    if not pred_cols:
        return {}
    actual_pred_col = pred_cols[0]

    y_true = pooled_df[y_col].values
    y_pred = pooled_df[actual_pred_col].values

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask].astype(int)
    y_pred = y_pred[mask].astype(float)

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {}

    metrics = compute_discrimination_metrics(y_true, y_pred)
    metrics["Brier"] = compute_brier_score(y_true, y_pred)
    metrics["n_samples"] = len(y_true)
    metrics["n_positive"] = int(y_true.sum())
    metrics["prevalence"] = float(y_true.mean())

    # Multi-target specificity metrics
    if spec_targets:
        multi_target_metrics = compute_multi_target_specificity_metrics(
            y_true=y_true, y_pred=y_pred, spec_targets=spec_targets
        )
        metrics.update(multi_target_metrics)

    if logger:
        logger.debug(
            f"Computed pooled metrics: n={len(y_true)}, "
            f"AUROC={metrics.get('AUROC', -1):.3f}, "
            f"Brier={metrics.get('Brier', -1):.4f}"
        )

    return metrics


def compute_pooled_metrics_by_model(
    pooled_df: pd.DataFrame,
    y_col: str = "y_true",
    pred_col: str = "y_prob",
    spec_targets: list[float] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, dict[str, float]]:
    """
    Compute metrics on pooled predictions, grouped by model.

    Args:
        pooled_df: DataFrame with pooled predictions (must have 'model' column)
        y_col: Column name for true labels
        pred_col: Column name for predicted probabilities
        spec_targets: List of specificity targets for threshold computation
        logger: Optional logger instance

    Returns:
        Dictionary mapping model name to metrics dict
    """
    if pooled_df.empty:
        if logger:
            logger.debug("Empty pooled DataFrame, no model metrics computed")
        return {}

    if "model" not in pooled_df.columns:
        # Fall back to single-model behavior
        if logger:
            logger.debug("No 'model' column found, computing single-model metrics")
        metrics = compute_pooled_metrics(pooled_df, y_col, pred_col, spec_targets, logger=logger)
        return {"unknown": metrics} if metrics else {}

    results = {}
    for model_name, model_df in pooled_df.groupby("model"):
        if logger:
            logger.debug(f"Computing pooled metrics for model: {model_name}")
        metrics = compute_pooled_metrics(model_df, y_col, pred_col, spec_targets, logger=logger)
        if metrics:
            metrics["model"] = model_name
            results[model_name] = metrics

    if logger:
        logger.info(f"Computed pooled metrics for {len(results)} models")

    return results


def compute_pooled_threshold_metrics(
    pooled_df: pd.DataFrame,
    y_col: str = "y_true",
    pred_col: str = "y_prob",
    target_spec: float = 0.95,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """
    Compute threshold-based metrics from pooled predictions.

    Uses Youden threshold from pooled data.

    Args:
        pooled_df: DataFrame with pooled predictions
        y_col: Column name for true labels
        pred_col: Column name for predicted probabilities
        target_spec: Target specificity for alpha threshold

    Returns:
        Dictionary with thresholds and metrics at each threshold
    """
    if pooled_df.empty:
        return {}

    pred_cols = [
        c
        for c in pooled_df.columns
        if c in ["y_prob", pred_col, "risk_score", "prob", "prediction"]
    ]
    if not pred_cols:
        return {}
    actual_pred_col = pred_cols[0]

    y_true = pooled_df[y_col].values
    y_pred = pooled_df[actual_pred_col].values

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask].astype(int)
    y_pred = y_pred[mask].astype(float)

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {}

    youden_thr = threshold_youden(y_true, y_pred)
    alpha_thr = threshold_for_specificity(y_true, y_pred, target_spec=target_spec)
    dca_thr = threshold_dca_zero_crossing(y_true, y_pred)

    youden_metrics = binary_metrics_at_threshold(y_true, y_pred, youden_thr)
    alpha_metrics = binary_metrics_at_threshold(y_true, y_pred, alpha_thr)

    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_pred)

    def get_fpr_tpr_at_threshold(threshold: float) -> tuple[float, float]:
        y_hat = (y_pred >= threshold).astype(int)
        tp = np.sum((y_hat == 1) & (y_true == 1))
        fp = np.sum((y_hat == 1) & (y_true == 0))
        fn = np.sum((y_hat == 0) & (y_true == 1))
        tn = np.sum((y_hat == 0) & (y_true == 0))
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return fpr_val, tpr_val

    youden_fpr, youden_tpr = get_fpr_tpr_at_threshold(youden_thr)
    alpha_fpr, alpha_tpr = get_fpr_tpr_at_threshold(alpha_thr)

    result = {
        "youden_threshold": youden_thr,
        "alpha_threshold": alpha_thr,
        "target_specificity": target_spec,
        "youden_metrics": {
            **youden_metrics,
            "fpr": youden_fpr,
            "tpr": youden_tpr,
        },
        "alpha_metrics": {
            **alpha_metrics,
            "fpr": alpha_fpr,
            "tpr": alpha_tpr,
        },
    }

    if dca_thr is not None:
        dca_metrics = binary_metrics_at_threshold(y_true, y_pred, dca_thr)
        dca_fpr, dca_tpr = get_fpr_tpr_at_threshold(dca_thr)
        result["dca_threshold"] = dca_thr
        result["dca_metrics"] = {
            **dca_metrics,
            "fpr": dca_fpr,
            "tpr": dca_tpr,
        }

    if logger:
        logger.debug(
            f"Computed thresholds: Youden={youden_thr:.4f}, "
            f"Alpha(spec={target_spec})={alpha_thr:.4f}, "
            f"DCA={dca_thr if dca_thr is not None else 'N/A'}"
        )

    return result


def save_threshold_data(
    threshold_info: dict[str, dict[str, Any]],
    out_dir: Path,
    logger: logging.Logger | None = None,
) -> None:
    """
    Save threshold information to CSV files (one per model).

    Args:
        threshold_info: Dictionary mapping model name to threshold data
        out_dir: Output directory for threshold files
        logger: Optional logger instance
    """
    if not threshold_info:
        if logger:
            logger.info("No threshold data to save")
        return

    thresholds_dir = out_dir / "core" / "thresholds"
    thresholds_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model_thresholds in threshold_info.items():
        rows = []

        # Youden threshold
        if "youden_threshold" in model_thresholds:
            youden_metrics = model_thresholds.get("youden_metrics", {})
            rows.append(
                {
                    "threshold_type": "youden",
                    "threshold_value": model_thresholds["youden_threshold"],
                    "sensitivity": youden_metrics.get("sensitivity"),
                    "specificity": youden_metrics.get("specificity"),
                    "precision": youden_metrics.get("precision"),
                    "f1": youden_metrics.get("f1"),
                    "fpr": youden_metrics.get("fpr"),
                    "tpr": youden_metrics.get("tpr"),
                    "tp": youden_metrics.get("tp"),
                    "fp": youden_metrics.get("fp"),
                    "tn": youden_metrics.get("tn"),
                    "fn": youden_metrics.get("fn"),
                }
            )

        # Alpha threshold (spec95)
        if "alpha_threshold" in model_thresholds:
            alpha_metrics = model_thresholds.get("alpha_metrics", {})
            rows.append(
                {
                    "threshold_type": "alpha",
                    "threshold_value": model_thresholds["alpha_threshold"],
                    "target_specificity": model_thresholds.get("target_specificity"),
                    "sensitivity": alpha_metrics.get("sensitivity"),
                    "specificity": alpha_metrics.get("specificity"),
                    "precision": alpha_metrics.get("precision"),
                    "f1": alpha_metrics.get("f1"),
                    "fpr": alpha_metrics.get("fpr"),
                    "tpr": alpha_metrics.get("tpr"),
                    "tp": alpha_metrics.get("tp"),
                    "fp": alpha_metrics.get("fp"),
                    "tn": alpha_metrics.get("tn"),
                    "fn": alpha_metrics.get("fn"),
                }
            )

        # DCA threshold
        if "dca_threshold" in model_thresholds:
            dca_metrics = model_thresholds.get("dca_metrics", {})
            rows.append(
                {
                    "threshold_type": "dca",
                    "threshold_value": model_thresholds["dca_threshold"],
                    "sensitivity": dca_metrics.get("sensitivity"),
                    "specificity": dca_metrics.get("specificity"),
                    "precision": dca_metrics.get("precision"),
                    "f1": dca_metrics.get("f1"),
                    "fpr": dca_metrics.get("fpr"),
                    "tpr": dca_metrics.get("tpr"),
                    "tp": dca_metrics.get("tp"),
                    "fp": dca_metrics.get("fp"),
                    "tn": dca_metrics.get("tn"),
                    "fn": dca_metrics.get("fn"),
                }
            )

        if rows:
            df = pd.DataFrame(rows)
            csv_path = thresholds_dir / f"thresholds__{model_name}.csv"
            df.to_csv(csv_path, index=False)
            if logger:
                logger.info(f"Thresholds saved for {model_name}: {csv_path}")

    # Also save a combined file with all models
    all_rows = []
    for model_name, model_thresholds in threshold_info.items():
        # Youden
        if "youden_threshold" in model_thresholds:
            youden_metrics = model_thresholds.get("youden_metrics", {})
            all_rows.append(
                {
                    "model": model_name,
                    "threshold_type": "youden",
                    "threshold_value": model_thresholds["youden_threshold"],
                    "sensitivity": youden_metrics.get("sensitivity"),
                    "specificity": youden_metrics.get("specificity"),
                    "precision": youden_metrics.get("precision"),
                    "f1": youden_metrics.get("f1"),
                    "fpr": youden_metrics.get("fpr"),
                    "tpr": youden_metrics.get("tpr"),
                    "tp": youden_metrics.get("tp"),
                    "fp": youden_metrics.get("fp"),
                    "tn": youden_metrics.get("tn"),
                    "fn": youden_metrics.get("fn"),
                }
            )

        # Alpha
        if "alpha_threshold" in model_thresholds:
            alpha_metrics = model_thresholds.get("alpha_metrics", {})
            all_rows.append(
                {
                    "model": model_name,
                    "threshold_type": "alpha",
                    "threshold_value": model_thresholds["alpha_threshold"],
                    "target_specificity": model_thresholds.get("target_specificity"),
                    "sensitivity": alpha_metrics.get("sensitivity"),
                    "specificity": alpha_metrics.get("specificity"),
                    "precision": alpha_metrics.get("precision"),
                    "f1": alpha_metrics.get("f1"),
                    "fpr": alpha_metrics.get("fpr"),
                    "tpr": alpha_metrics.get("tpr"),
                    "tp": alpha_metrics.get("tp"),
                    "fp": alpha_metrics.get("fp"),
                    "tn": alpha_metrics.get("tn"),
                    "fn": alpha_metrics.get("fn"),
                }
            )

        # DCA
        if "dca_threshold" in model_thresholds:
            dca_metrics = model_thresholds.get("dca_metrics", {})
            all_rows.append(
                {
                    "model": model_name,
                    "threshold_type": "dca",
                    "threshold_value": model_thresholds["dca_threshold"],
                    "sensitivity": dca_metrics.get("sensitivity"),
                    "specificity": dca_metrics.get("specificity"),
                    "precision": dca_metrics.get("precision"),
                    "f1": dca_metrics.get("f1"),
                    "fpr": dca_metrics.get("fpr"),
                    "tpr": dca_metrics.get("tpr"),
                    "tp": dca_metrics.get("tp"),
                    "fp": dca_metrics.get("fp"),
                    "tn": dca_metrics.get("tn"),
                    "fn": dca_metrics.get("fn"),
                }
            )

    if all_rows:
        combined_df = pd.DataFrame(all_rows)
        combined_path = thresholds_dir / "thresholds_all_models.csv"
        combined_df.to_csv(combined_path, index=False)
        if logger:
            logger.info(f"Combined thresholds saved: {combined_path}")


def aggregate_feature_stability(
    split_dirs: list[Path],
    stability_threshold: float = 0.75,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate feature selection across splits.

    Args:
        split_dirs: List of split subdirectory paths
        stability_threshold: Fraction of splits a feature must appear in to be "stable"
        logger: Optional logger instance

    Returns:
        Tuple of (feature_stability_df, stable_features_df)
        - feature_stability_df: All features with selection counts
        - stable_features_df: Features meeting stability threshold
    """
    all_selections = []

    for split_dir in split_dirs:
        seed = int(split_dir.name.replace("split_seed", ""))

        cv_path = split_dir / "cv" / "selected_proteins_per_split.csv"
        if not cv_path.exists():
            if logger:
                logger.debug(f"No selected proteins file in {split_dir.name}")
            continue

        try:
            df = pd.read_csv(cv_path)
            proteins_col = None
            for col in ["selected_proteins_split", "selected_proteins", "proteins"]:
                if col in df.columns:
                    proteins_col = col
                    break

            if proteins_col is None:
                continue

            for _, row in df.iterrows():
                proteins_str = row[proteins_col]
                if pd.isna(proteins_str) or not proteins_str:
                    continue

                if isinstance(proteins_str, str):
                    proteins = [p.strip() for p in proteins_str.split(",") if p.strip()]
                else:
                    proteins = []

                for protein in proteins:
                    all_selections.append(
                        {
                            "split_seed": seed,
                            "protein": protein,
                        }
                    )
        except Exception as e:
            if logger:
                logger.warning(f"Failed to read {cv_path}: {e}")

    if not all_selections:
        return pd.DataFrame(), pd.DataFrame()

    selection_df = pd.DataFrame(all_selections)

    n_splits = len(split_dirs)

    protein_counts = (
        selection_df.groupby("protein")["split_seed"]
        .nunique()
        .reset_index()
        .rename(columns={"split_seed": "n_splits_selected"})
    )
    protein_counts["selection_fraction"] = protein_counts["n_splits_selected"] / n_splits
    protein_counts = protein_counts.sort_values("selection_fraction", ascending=False)

    stable_features = protein_counts[
        protein_counts["selection_fraction"] >= stability_threshold
    ].copy()

    return protein_counts, stable_features


def aggregate_feature_reports(
    feature_reports_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate feature reports across splits.

    Computes mean, std, min, max, and count for selection_freq, effect_size, p_value.

    Args:
        feature_reports_df: DataFrame with feature reports from all splits

    Returns:
        DataFrame with aggregated feature statistics
    """
    if feature_reports_df.empty:
        return pd.DataFrame()

    agg_funcs = {
        "selection_freq": ["mean", "std", "min", "max", "count"],
    }

    if "effect_size" in feature_reports_df.columns:
        agg_funcs["effect_size"] = ["mean", "std"]
    if "p_value" in feature_reports_df.columns:
        agg_funcs["p_value"] = ["mean", "std"]

    agg_df = feature_reports_df.groupby("protein").agg(agg_funcs).reset_index()

    agg_df.columns = [
        "_".join(col).strip("_") if col[1] else col[0] for col in agg_df.columns.values
    ]

    if "selection_freq_count" in agg_df.columns:
        agg_df.rename(columns={"selection_freq_count": "n_splits"}, inplace=True)

    agg_df = agg_df.sort_values("selection_freq_mean", ascending=False).reset_index(drop=True)
    agg_df["rank"] = range(1, len(agg_df) + 1)

    col_order = ["rank", "protein", "selection_freq_mean", "selection_freq_std", "n_splits"]
    if "effect_size_mean" in agg_df.columns:
        col_order.extend(["effect_size_mean", "effect_size_std"])
    if "p_value_mean" in agg_df.columns:
        col_order.extend(["p_value_mean", "p_value_std"])

    remaining_cols = [c for c in agg_df.columns if c not in col_order]
    col_order.extend(remaining_cols)

    agg_df = agg_df[[c for c in col_order if c in agg_df.columns]]

    return agg_df


def build_consensus_panels(
    split_dirs: list[Path],
    panel_sizes: list[int] | None = None,
    threshold: float = 0.75,
    logger: logging.Logger | None = None,
) -> dict[int, dict[str, Any]]:
    """
    Build consensus panels from per-split panels.

    Args:
        split_dirs: List of split subdirectory paths
        panel_sizes: List of panel sizes to build (e.g., [10, 25, 50])
        threshold: Fraction of splits a protein must appear in
        logger: Optional logger instance

    Returns:
        Dictionary mapping panel_size -> panel manifest dict
    """
    if panel_sizes is None:
        panel_sizes = [10, 25, 50]

    results = {}

    for panel_size in panel_sizes:
        panel_proteins_per_split = []

        for split_dir in split_dirs:
            panel_dir = split_dir / "reports" / "panels"
            if not panel_dir.exists():
                continue

            manifest_files = list(panel_dir.glob(f"*__N{panel_size}__panel_manifest.json"))
            if not manifest_files:
                continue

            try:
                with open(manifest_files[0]) as f:
                    manifest = json.load(f)
                    proteins = manifest.get("panel_proteins", [])
                    if proteins:
                        panel_proteins_per_split.append(set(proteins))
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to read panel manifest: {e}")

        if not panel_proteins_per_split:
            continue

        protein_counts: dict[str, int] = {}
        for protein_set in panel_proteins_per_split:
            for protein in protein_set:
                protein_counts[protein] = protein_counts.get(protein, 0) + 1

        n_splits = len(panel_proteins_per_split)
        min_count = int(np.ceil(threshold * n_splits))

        consensus_proteins = [
            protein
            for protein, count in sorted(protein_counts.items(), key=lambda x: (-x[1], x[0]))
            if count >= min_count
        ]

        results[panel_size] = {
            "panel_size": panel_size,
            "n_splits_with_panel": n_splits,
            "consensus_threshold": threshold,
            "min_splits_required": min_count,
            "n_consensus_proteins": len(consensus_proteins),
            "consensus_proteins": consensus_proteins,
            "protein_counts": {p: c for p, c in protein_counts.items() if c >= min_count},
        }

    return results


def generate_aggregated_plots(
    pooled_test_df: pd.DataFrame,
    pooled_val_df: pd.DataFrame,
    pooled_train_oof_df: pd.DataFrame,
    out_dir: Path,
    threshold_info: dict[str, Any],
    plot_formats: list[str],
    meta_lines: list[str] | None = None,
    logger: logging.Logger | None = None,
    plot_roc: bool = True,
    plot_pr: bool = True,
    plot_calibration: bool = True,
    plot_risk_distribution: bool = True,
    plot_dca: bool = True,
    plot_oof_combined: bool = True,
    target_specificity: float = 0.95,
) -> None:
    """
    Generate all aggregated diagnostic plots, separated by model.

    Args:
        pooled_test_df: DataFrame with pooled test predictions
        pooled_val_df: DataFrame with pooled validation predictions
        pooled_train_oof_df: DataFrame with pooled train OOF predictions
        out_dir: Output directory for plots
        threshold_info: Dictionary with threshold information (keyed by model)
        plot_formats: List of plot formats (e.g., ["png", "pdf"])
        meta_lines: Metadata lines to add to plots
        logger: Optional logger instance
        plot_roc: Whether to generate ROC plots
        plot_pr: Whether to generate PR plots
        plot_calibration: Whether to generate calibration plots
        plot_risk_distribution: Whether to generate risk distribution plots
        plot_dca: Whether to generate DCA plots
        plot_oof_combined: Whether to generate OOF combined plots
    """
    try:
        from ced_ml.plotting.calibration import plot_calibration_curve
        from ced_ml.plotting.dca import plot_dca_curve
        from ced_ml.plotting.risk_dist import plot_risk_distribution
        from ced_ml.plotting.roc_pr import plot_pr_curve, plot_roc_curve
    except ImportError as e:
        if logger:
            logger.warning(f"Plotting not available: {e}")
        return

    pred_col_names = ["y_prob", "y_pred", "risk_score", "prob", "prediction"]

    def get_arrays(
        df: pd.DataFrame,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        if df.empty:
            return None, None, None, None

        pred_col = None
        for col in pred_col_names:
            if col in df.columns:
                pred_col = col
                break

        if pred_col is None or "y_true" not in df.columns:
            return None, None, None, None

        y_true = df["y_true"].values
        y_pred = df[pred_col].values
        split_ids = df["split_seed"].values if "split_seed" in df.columns else None
        category = df["category"].values if "category" in df.columns else None

        return y_true, y_pred, split_ids, category

    # Detect models present in the data
    test_models = (
        pooled_test_df["model"].unique().tolist()
        if not pooled_test_df.empty and "model" in pooled_test_df.columns
        else []
    )
    val_models = (
        pooled_val_df["model"].unique().tolist()
        if not pooled_val_df.empty and "model" in pooled_val_df.columns
        else []
    )
    train_models = (
        pooled_train_oof_df["model"].unique().tolist()
        if not pooled_train_oof_df.empty and "model" in pooled_train_oof_df.columns
        else []
    )
    all_models = sorted(set(test_models + val_models + train_models))

    if not all_models:
        all_models = ["unknown"]

    if logger:
        logger.info(f"Generating plots for {len(all_models)} model(s): {', '.join(all_models)}")

    # Generate plots for each model separately
    for model_name in all_models:
        if logger:
            logger.info(f"Generating plots for model: {model_name}")

        # Create output directories (model name not needed since parent folder already specifies it)
        model_plots_dir = out_dir / "diagnostics" / "plots"
        model_plots_dir.mkdir(parents=True, exist_ok=True)

        model_preds_plots_dir = out_dir / "preds" / "plots"
        model_preds_plots_dir.mkdir(parents=True, exist_ok=True)

        # Filter data for this model
        model_test_df = (
            pooled_test_df[pooled_test_df["model"] == model_name]
            if not pooled_test_df.empty and "model" in pooled_test_df.columns
            else pooled_test_df
        )
        model_val_df = (
            pooled_val_df[pooled_val_df["model"] == model_name]
            if not pooled_val_df.empty and "model" in pooled_val_df.columns
            else pooled_val_df
        )
        model_train_oof_df = (
            pooled_train_oof_df[pooled_train_oof_df["model"] == model_name]
            if not pooled_train_oof_df.empty and "model" in pooled_train_oof_df.columns
            else pooled_train_oof_df
        )

        # Get model-specific threshold info
        model_threshold_info = threshold_info.get(model_name, {})
        dca_thr = model_threshold_info.get("dca_threshold")
        youden_metrics = model_threshold_info.get("youden_metrics", {})
        alpha_metrics = model_threshold_info.get("alpha_metrics", {})
        dca_metrics = model_threshold_info.get("dca_metrics", {})

        metrics_at_thresholds = {}
        if youden_metrics:
            metrics_at_thresholds["youden"] = {
                "fpr": youden_metrics.get("fpr"),
                "tpr": youden_metrics.get("tpr"),
            }
        if alpha_metrics:
            metrics_at_thresholds["alpha"] = {
                "fpr": alpha_metrics.get("fpr"),
                "tpr": alpha_metrics.get("tpr"),
            }
        if dca_metrics:
            metrics_at_thresholds["dca"] = {
                "fpr": dca_metrics.get("fpr"),
                "tpr": dca_metrics.get("tpr"),
            }

        # Add model name to metadata lines
        model_meta_lines = (meta_lines or []) + [f"Model: {model_name}"]

        # Generate test/val plots
        for data_name, df in [("test", model_test_df), ("val", model_val_df)]:
            y_true, y_pred, split_ids, category = get_arrays(df)
            if y_true is None:
                if logger:
                    logger.debug(f"Skipping {data_name} plots for {model_name}: no valid data")
                continue

            if logger:
                logger.info(f"Generating aggregated {data_name} plots for {model_name}")

            # Compute threshold bundle (standardized interface) - always compute fresh
            # for aggregated data to ensure consistency
            local_bundle = compute_threshold_bundle(
                y_true,
                y_pred,
                target_spec=target_specificity,
                dca_threshold=dca_thr,
            )

            # Ensemble models: skip 95% CI band (only use SD) since CI and SD are redundant
            # when there are no CV repeats per split
            skip_ci_band = model_name == "ENSEMBLE"

            for fmt in plot_formats:
                if plot_roc:
                    plot_roc_curve(
                        y_true=y_true,
                        y_pred=y_pred,
                        out_path=model_plots_dir / f"{data_name}_roc.{fmt}",
                        title=f"Aggregated {data_name.capitalize()} Set ROC - {model_name}",
                        split_ids=split_ids,
                        meta_lines=model_meta_lines,
                        threshold_bundle=local_bundle,
                        skip_ci_band=skip_ci_band,
                    )

                if plot_pr:
                    plot_pr_curve(
                        y_true=y_true,
                        y_pred=y_pred,
                        out_path=model_plots_dir / f"{data_name}_pr.{fmt}",
                        title=f"Aggregated {data_name.capitalize()} Set PR Curve - {model_name}",
                        split_ids=split_ids,
                        meta_lines=model_meta_lines,
                        skip_ci_band=skip_ci_band,
                    )

                if plot_calibration:
                    plot_calibration_curve(
                        y_true=y_true,
                        y_pred=y_pred,
                        out_path=model_plots_dir / f"{data_name}_calibration.{fmt}",
                        title=f"Aggregated {data_name.capitalize()} Set Calibration - {model_name}",
                        split_ids=split_ids,
                        meta_lines=model_meta_lines,
                        skip_ci_band=skip_ci_band,
                    )

                if plot_dca:
                    plot_dca_curve(
                        y_true=y_true,
                        y_pred=y_pred,
                        out_path=str(model_plots_dir / f"{data_name}_dca.{fmt}"),
                        title=f"Aggregated {data_name.capitalize()} Set DCA - {model_name}",
                        split_ids=split_ids,
                        meta_lines=model_meta_lines,
                        skip_ci_band=skip_ci_band,
                    )

                if plot_risk_distribution:
                    plot_risk_distribution(
                        y_true=y_true,
                        scores=y_pred,
                        out_path=model_preds_plots_dir / f"{data_name}_risk_dist.{fmt}",
                        title=f"Aggregated {data_name.capitalize()} Set Risk Distribution - {model_name}",
                        category_col=category,
                        threshold_bundle=local_bundle,
                        meta_lines=model_meta_lines,
                    )

        # Generate train OOF plots if available
        if not model_train_oof_df.empty:
            y_true_train, y_pred_train, split_ids_train, category_train = get_arrays(
                model_train_oof_df
            )
            if y_true_train is not None:
                if logger:
                    logger.info(f"Generating aggregated train OOF plots for {model_name}")

                # Compute threshold bundle for OOF data
                oof_bundle = compute_threshold_bundle(
                    y_true_train,
                    y_pred_train,
                    target_spec=target_specificity,
                    dca_threshold=dca_thr,
                )

                for fmt in plot_formats:
                    if plot_risk_distribution:
                        plot_risk_distribution(
                            y_true=y_true_train,
                            scores=y_pred_train,
                            out_path=model_preds_plots_dir / f"train_oof_risk_dist.{fmt}",
                            title=f"Aggregated Train OOF Risk Distribution - {model_name}",
                            category_col=category_train,
                            threshold_bundle=oof_bundle,
                            meta_lines=model_meta_lines,
                        )

                    # Generate OOF combined plots (ROC, PR, Calibration)
                    if plot_oof_combined:
                        try:
                            from ced_ml.plotting.oof import plot_oof_combined

                            # For OOF plots, we need to use the per-repeat predictions
                            repeat_cols = [
                                c
                                for c in model_train_oof_df.columns
                                if c.startswith("y_prob_repeat")
                            ]

                            if repeat_cols:
                                # Get unique sample indices
                                unique_idx = model_train_oof_df["idx"].unique()
                                n_samples = len(unique_idx)
                                n_repeats = len(repeat_cols)

                                # Create oof_preds array (n_repeats x n_samples)
                                oof_preds = np.full((n_repeats, n_samples), np.nan)
                                y_true_oof = np.zeros(n_samples)

                                # Map idx to position
                                idx_to_pos = {idx: pos for pos, idx in enumerate(unique_idx)}

                                # Fill in predictions from first split_seed (they should all have same idx mapping)
                                first_seed = model_train_oof_df["split_seed"].iloc[0]
                                seed_df = model_train_oof_df[
                                    model_train_oof_df["split_seed"] == first_seed
                                ]

                                for _, row in seed_df.iterrows():
                                    pos = idx_to_pos[row["idx"]]
                                    y_true_oof[pos] = row["y_true"]
                                    for repeat_idx, col in enumerate(repeat_cols):
                                        oof_preds[repeat_idx, pos] = row[col]

                                plot_oof_combined(
                                    y_true=y_true_oof,
                                    oof_preds=oof_preds,
                                    out_dir=model_plots_dir,
                                    model_name=model_name,
                                    scenario="pooled",
                                    seed=None,
                                    plot_format=fmt,
                                    calib_bins=10,
                                    meta_lines=model_meta_lines,
                                )
                        except Exception as e:
                            if logger:
                                logger.warning(
                                    f"Failed to generate OOF combined plots for {model_name}: {e}"
                                )

                # Generate controls-only risk distribution if available


def generate_model_comparison_report(
    pooled_test_metrics: dict[str, dict[str, float]],
    pooled_val_metrics: dict[str, dict[str, float]],
    threshold_info: dict[str, dict[str, Any]],
    out_dir: Path,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Generate a model comparison report comparing all models including ENSEMBLE.

    Args:
        pooled_test_metrics: Dict mapping model name to test metrics
        pooled_val_metrics: Dict mapping model name to val metrics
        threshold_info: Dict mapping model name to threshold information
        out_dir: Output directory for the report
        logger: Optional logger instance

    Returns:
        DataFrame with model comparison data
    """
    if not pooled_test_metrics:
        return pd.DataFrame()

    rows = []
    for model_name in sorted(pooled_test_metrics.keys()):
        test_metrics = pooled_test_metrics.get(model_name, {})
        val_metrics = pooled_val_metrics.get(model_name, {})
        model_threshold = threshold_info.get(model_name, {})

        row = {
            "model": model_name,
            "is_ensemble": model_name == "ENSEMBLE",
            # Test metrics
            "test_AUROC": test_metrics.get("AUROC"),
            "test_PR_AUC": test_metrics.get("PR_AUC"),
            "test_Brier": test_metrics.get("Brier"),
            "test_n_samples": test_metrics.get("n_samples"),
            "test_n_positive": test_metrics.get("n_positive"),
            "test_prevalence": test_metrics.get("prevalence"),
            # Calibration
            "test_calib_slope": test_metrics.get("calib_slope"),
            # Val metrics
            "val_AUROC": val_metrics.get("AUROC"),
            "val_PR_AUC": val_metrics.get("PR_AUC"),
            "val_Brier": val_metrics.get("Brier"),
            # Thresholds
            "youden_threshold": model_threshold.get("youden_threshold"),
            "alpha_threshold": model_threshold.get("alpha_threshold"),
            "dca_threshold": model_threshold.get("dca_threshold"),
        }

        # Add multi-target specificity metrics if present
        for key in test_metrics:
            if key.startswith("sens_ctrl_") or key.startswith("prec_ctrl_"):
                row[f"test_{key}"] = test_metrics[key]

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by test AUROC descending
    if "test_AUROC" in df.columns:
        df = df.sort_values("test_AUROC", ascending=False)

    # Save to reports directory
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / "model_comparison.csv"
    df.to_csv(report_path, index=False)

    if logger:
        logger.info(f"Model comparison report saved: {report_path}")

        # Log summary of comparison
        if "ENSEMBLE" in df["model"].values:
            ensemble_row = df[df["model"] == "ENSEMBLE"].iloc[0]
            best_base = df[df["model"] != "ENSEMBLE"].iloc[0] if len(df) > 1 else None

            if best_base is not None:
                ensemble_auroc = ensemble_row.get("test_AUROC")
                best_base_auroc = best_base.get("test_AUROC")
                if ensemble_auroc is not None and best_base_auroc is not None:
                    improvement = (ensemble_auroc - best_base_auroc) * 100
                    logger.info(
                        f"ENSEMBLE vs best base ({best_base['model']}): "
                        f"AUROC {ensemble_auroc:.4f} vs {best_base_auroc:.4f} "
                        f"({improvement:+.2f} pp)"
                    )

    return df


def run_aggregate_splits(
    results_dir: str,
    stability_threshold: float = 0.75,
    plot_formats: list[str] | None = None,
    target_specificity: float = 0.95,
    n_boot: int = 500,
    verbose: int = 0,
    save_plots: bool = True,
    plot_roc: bool = True,
    plot_pr: bool = True,
    plot_calibration: bool = True,
    plot_risk_distribution: bool = True,
    plot_dca: bool = True,
    plot_oof_combined: bool = True,
    plot_learning_curve: bool = True,
    control_spec_targets: list[float] | None = None,
) -> dict[str, Any]:
    """
    Aggregate results across multiple split seeds.

    Args:
        results_dir: Directory containing split_seedX subdirectories
        stability_threshold: Fraction of splits for feature stability (default 0.75)
        plot_formats: List of plot formats (default ["png"])
        target_specificity: Target specificity for alpha threshold (default 0.95)
        n_boot: Number of bootstrap iterations (for future CI computation)
        verbose: Verbosity level (0=INFO, 1=DEBUG)
        save_plots: Whether to save plots at all (default True)
        plot_roc: Whether to generate ROC plots (default True)
        plot_pr: Whether to generate PR plots (default True)
        plot_calibration: Whether to generate calibration plots (default True)
        plot_risk_distribution: Whether to generate risk distribution plots (default True)
        plot_dca: Whether to generate DCA plots (default True)
        plot_oof_combined: Whether to generate OOF combined plots (default True)
        plot_learning_curve: Whether to generate learning curve plots (default True)

    Returns:
        Dictionary with aggregation results summary
    """
    if plot_formats is None:
        plot_formats = ["png"]

    if control_spec_targets is None:
        control_spec_targets = [0.90, 0.95, 0.99]

    log_level = 20 - (verbose * 10)
    logger = setup_logger("ced_ml.aggregate", level=log_level)

    log_section(logger, "CeD-ML Split Aggregation")

    results_path = Path(results_dir)
    if not results_path.exists():
        logger.error(f"Results directory not found: {results_dir}")
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    split_dirs = discover_split_dirs(results_path, logger=logger)
    logger.info(f"Found {len(split_dirs)} split directories")

    # Also discover ENSEMBLE model directories (stored separately)
    ensemble_dirs = discover_ensemble_dirs(results_path, logger=logger)
    if ensemble_dirs:
        logger.info(f"Found {len(ensemble_dirs)} ENSEMBLE split directories")

    if not split_dirs and not ensemble_dirs:
        logger.warning("No split_seedX directories found. Nothing to aggregate.")
        return {"status": "no_splits_found"}

    for sd in split_dirs:
        logger.info(f"  {sd.name}")
    for ed in ensemble_dirs:
        logger.info(f"  ENSEMBLE/{ed.name}")

    agg_dir = results_path / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)

    core_dir = agg_dir / "core"
    core_dir.mkdir(parents=True, exist_ok=True)

    cv_dir = agg_dir / "cv"
    cv_dir.mkdir(parents=True, exist_ok=True)

    preds_dir = agg_dir / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    reports_dir = agg_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output: {agg_dir}")

    log_section(logger, "Collecting Pooled Predictions")

    pooled_test_df = collect_predictions(split_dirs, "test", logger)
    pooled_val_df = collect_predictions(split_dirs, "val", logger)
    pooled_train_oof_df = collect_predictions(split_dirs, "train_oof", logger)

    # Collect ENSEMBLE predictions if available and merge with other model predictions
    if ensemble_dirs:
        ensemble_test_df = collect_ensemble_predictions(ensemble_dirs, "test", logger)
        ensemble_val_df = collect_ensemble_predictions(ensemble_dirs, "val", logger)
        ensemble_oof_df = collect_ensemble_predictions(ensemble_dirs, "train_oof", logger)

        if not ensemble_test_df.empty:
            pooled_test_df = pd.concat([pooled_test_df, ensemble_test_df], ignore_index=True)
            logger.info(f"Merged ENSEMBLE test predictions: {len(ensemble_test_df)} samples")

        if not ensemble_val_df.empty:
            pooled_val_df = pd.concat([pooled_val_df, ensemble_val_df], ignore_index=True)
            logger.info(f"Merged ENSEMBLE val predictions: {len(ensemble_val_df)} samples")

        if not ensemble_oof_df.empty:
            pooled_train_oof_df = pd.concat(
                [pooled_train_oof_df, ensemble_oof_df], ignore_index=True
            )
            logger.info(f"Merged ENSEMBLE OOF predictions: {len(ensemble_oof_df)} samples")

    if not pooled_test_df.empty:
        test_preds_dir = preds_dir / "test_preds"
        test_preds_dir.mkdir(parents=True, exist_ok=True)
        # Save combined file (all models)
        pooled_test_df.to_csv(test_preds_dir / "pooled_test_preds.csv", index=False)
        # Save per-model files
        if "model" in pooled_test_df.columns:
            for model_name, model_df in pooled_test_df.groupby("model"):
                model_df.to_csv(
                    test_preds_dir / f"pooled_test_preds__{model_name}.csv", index=False
                )
        logger.info(
            f"Pooled test predictions: {len(pooled_test_df)} samples from "
            f"{pooled_test_df['split_seed'].nunique()} splits, "
            f"{pooled_test_df['model'].nunique() if 'model' in pooled_test_df.columns else 1} model(s)"
        )

    if not pooled_val_df.empty:
        val_preds_dir = preds_dir / "val_preds"
        val_preds_dir.mkdir(parents=True, exist_ok=True)
        # Save combined file (all models)
        pooled_val_df.to_csv(val_preds_dir / "pooled_val_preds.csv", index=False)
        # Save per-model files
        if "model" in pooled_val_df.columns:
            for model_name, model_df in pooled_val_df.groupby("model"):
                model_df.to_csv(val_preds_dir / f"pooled_val_preds__{model_name}.csv", index=False)
        logger.info(
            f"Pooled val predictions: {len(pooled_val_df)} samples from "
            f"{pooled_val_df['split_seed'].nunique()} splits, "
            f"{pooled_val_df['model'].nunique() if 'model' in pooled_val_df.columns else 1} model(s)"
        )

    if not pooled_train_oof_df.empty:
        train_oof_dir = preds_dir / "train_oof"
        train_oof_dir.mkdir(parents=True, exist_ok=True)

        # Compute mean across CV repeats for each split
        repeat_cols = [c for c in pooled_train_oof_df.columns if c.startswith("y_prob_repeat")]
        if repeat_cols:
            pooled_train_oof_df["y_prob"] = pooled_train_oof_df[repeat_cols].mean(axis=1)

        # Save combined file (all models)
        pooled_train_oof_df.to_csv(train_oof_dir / "pooled_train_oof.csv", index=False)
        # Save per-model files
        if "model" in pooled_train_oof_df.columns:
            for model_name, model_df in pooled_train_oof_df.groupby("model"):
                model_df.to_csv(train_oof_dir / f"pooled_train_oof__{model_name}.csv", index=False)
        logger.info(
            f"Pooled train OOF predictions: {len(pooled_train_oof_df)} samples from "
            f"{pooled_train_oof_df['split_seed'].nunique()} splits, "
            f"{pooled_train_oof_df['model'].nunique() if 'model' in pooled_train_oof_df.columns else 1} model(s)"
        )

    log_section(logger, "Computing Pooled Metrics")

    pooled_test_metrics: dict[str, dict[str, float]] = {}
    pooled_val_metrics: dict[str, dict[str, float]] = {}
    threshold_info: dict[str, Any] = {}

    # Detect models present in predictions
    test_models = (
        pooled_test_df["model"].unique().tolist()
        if not pooled_test_df.empty and "model" in pooled_test_df.columns
        else []
    )
    val_models = (
        pooled_val_df["model"].unique().tolist()
        if not pooled_val_df.empty and "model" in pooled_val_df.columns
        else []
    )
    all_models = sorted(set(test_models + val_models))

    if len(all_models) > 1:
        logger.info(f"Multiple models detected: {', '.join(all_models)}")
    elif all_models:
        logger.info(f"Single model: {all_models[0]}")

    if not pooled_test_df.empty:
        # Compute per-model pooled metrics
        pooled_test_metrics = compute_pooled_metrics_by_model(
            pooled_test_df, spec_targets=control_spec_targets
        )

        if pooled_test_metrics:
            # Save per-model metrics
            metrics_rows = list(pooled_test_metrics.values())
            pd.DataFrame(metrics_rows).to_csv(core_dir / "pooled_test_metrics.csv", index=False)

            # Log per-model results
            for model_name, metrics in pooled_test_metrics.items():
                logger.info(f"Pooled test [{model_name}] AUROC: {metrics.get('AUROC', 'N/A'):.4f}")
                logger.info(
                    f"Pooled test [{model_name}] PR-AUC: {metrics.get('PR_AUC', 'N/A'):.4f}"
                )
                logger.info(f"Pooled test [{model_name}] Brier: {metrics.get('Brier', 'N/A'):.4f}")

            # Compute and save selection scores
            selection_scores = compute_selection_scores_for_models(pooled_test_metrics)
            if selection_scores:
                selection_df = pd.DataFrame(
                    [
                        {"model": model_name, "selection_score": score}
                        for model_name, score in sorted(
                            selection_scores.items(), key=lambda x: x[1], reverse=True
                        )
                    ]
                )
                selection_df.to_csv(core_dir / "selection_scores.csv", index=False)
                logger.info("Selection scores computed and saved")
                for model_name, score in selection_scores.items():
                    logger.info(f"Selection score [{model_name}]: {score:.4f}")

        # Compute threshold info per model
        threshold_info = {}
        for model_name in test_models:
            model_df = pooled_test_df[pooled_test_df["model"] == model_name]
            model_threshold = compute_pooled_threshold_metrics(
                model_df, target_spec=target_specificity, logger=logger
            )
            if model_threshold:
                threshold_info[model_name] = model_threshold
                logger.info(
                    f"Youden threshold [{model_name}]: {model_threshold.get('youden_threshold', 'N/A'):.4f}"
                )
                logger.info(
                    f"Alpha threshold [{model_name}] (spec={target_specificity}): "
                    f"{model_threshold.get('alpha_threshold', 'N/A'):.4f}"
                )

        # Save threshold data to CSV
        if threshold_info:
            save_threshold_data(threshold_info, agg_dir, logger)

    if not pooled_val_df.empty:
        pooled_val_metrics = compute_pooled_metrics_by_model(
            pooled_val_df, spec_targets=control_spec_targets
        )
        if pooled_val_metrics:
            metrics_rows = list(pooled_val_metrics.values())
            pd.DataFrame(metrics_rows).to_csv(core_dir / "pooled_val_metrics.csv", index=False)
            for model_name, metrics in pooled_val_metrics.items():
                logger.info(f"Pooled val [{model_name}] AUROC: {metrics.get('AUROC', 'N/A'):.4f}")

    # Generate model comparison report (includes ENSEMBLE if available)
    log_section(logger, "Generating Model Comparison Report")
    _ = generate_model_comparison_report(
        pooled_test_metrics=pooled_test_metrics,
        pooled_val_metrics=pooled_val_metrics,
        threshold_info=threshold_info,
        out_dir=agg_dir,
        logger=logger,
    )

    # Generate ensemble-specific aggregate plots and metadata
    if ensemble_dirs and pooled_test_metrics:
        try:
            from ced_ml.plotting.ensemble import (
                plot_aggregated_weights,
                plot_model_comparison,
                save_ensemble_aggregation_metadata,
            )

            agg_plots_dir = agg_dir / "diagnostics" / "plots"
            agg_plots_dir.mkdir(parents=True, exist_ok=True)

            # Collect meta-learner coefficients from each ensemble split
            coefs_per_split: dict[int, dict[str, float]] = {}
            base_models_list = []
            meta_penalty = "l2"
            meta_C = 1.0

            for ed in ensemble_dirs:
                settings_path = ed / "core" / "run_settings.json"
                config_path = ed / "config.yaml"

                if settings_path.exists():
                    try:
                        with open(settings_path) as f:
                            settings = json.load(f)
                        meta_coef = settings.get("meta_coef", {})
                        if meta_coef:
                            seed = settings.get("split_seed", 0)
                            coefs_per_split[seed] = meta_coef
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.debug(f"Could not read ensemble settings from {ed}: {e}")

                # Extract ensemble config
                if config_path.exists() and not base_models_list:
                    try:
                        import yaml

                        with open(config_path) as f:
                            config = yaml.safe_load(f)
                        if "ensemble" in config:
                            ensemble_cfg = config["ensemble"]
                            base_models_list = ensemble_cfg.get("base_models", [])
                            meta_learner_cfg = ensemble_cfg.get("meta_model", {})
                            meta_penalty = meta_learner_cfg.get("penalty", "l2")
                            meta_C = meta_learner_cfg.get("C", 1.0)
                    except Exception as e:
                        logger.debug(f"Could not read ensemble config from {config_path}: {e}")

            if coefs_per_split:
                # Generate aggregated weights plot
                plot_aggregated_weights(
                    coefs_per_split=coefs_per_split,
                    out_path=agg_plots_dir / "ensemble_weights_aggregated.png",
                    title="Aggregated Meta-Learner Coefficients",
                    meta_lines=[f"n_splits={len(coefs_per_split)}"],
                )
                logger.info("Aggregated ensemble weights plot saved")

                # Generate and save ensemble metadata JSON
                save_ensemble_aggregation_metadata(
                    coefs_per_split=coefs_per_split,
                    pooled_test_metrics=pooled_test_metrics,
                    base_models=base_models_list,
                    meta_penalty=meta_penalty,
                    meta_C=meta_C,
                    out_dir=agg_plots_dir,
                )
                logger.info("Ensemble aggregation metadata saved")

            # Model comparison chart across all discovered models
            if len(pooled_test_metrics) >= 2:
                plot_model_comparison(
                    metrics=pooled_test_metrics,
                    out_path=agg_plots_dir / "model_comparison.png",
                    title="Model Comparison (Pooled Test Set)",
                    highlight_model="ENSEMBLE",
                    meta_lines=[f"n_models={len(pooled_test_metrics)}"],
                )
                logger.info("Model comparison plot saved")

        except Exception as e:
            logger.warning(f"Ensemble aggregate plot generation failed (non-fatal): {e}")

    log_section(logger, "Aggregating Per-Split Metrics")

    test_metrics = collect_metrics(split_dirs, "core/test_metrics.csv", logger=logger)
    if not test_metrics.empty:
        all_test_path = agg_dir / "all_test_metrics.csv"
        test_metrics.to_csv(all_test_path, index=False)
        logger.info(f"All test metrics saved: {all_test_path}")
        logger.info(
            f"  {len(test_metrics)} rows from {test_metrics['split_seed'].nunique()} splits"
        )

        summary = compute_summary_stats(test_metrics, logger=logger)
        if not summary.empty:
            summary_path = core_dir / "test_metrics_summary.csv"
            summary.to_csv(summary_path, index=False)
            logger.info(f"Summary stats saved: {summary_path}")

    val_metrics = collect_metrics(split_dirs, "core/val_metrics.csv", logger=logger)
    if not val_metrics.empty:
        all_val_path = agg_dir / "all_val_metrics.csv"
        val_metrics.to_csv(all_val_path, index=False)
        logger.info(f"All val metrics saved: {all_val_path}")

        val_summary = compute_summary_stats(val_metrics, logger=logger)
        if not val_summary.empty:
            val_summary_path = core_dir / "val_metrics_summary.csv"
            val_summary.to_csv(val_summary_path, index=False)
            logger.info(f"Val summary saved: {val_summary_path}")

    cv_metrics = collect_metrics(split_dirs, "cv/cv_repeat_metrics.csv", logger=logger)
    if not cv_metrics.empty:
        all_cv_path = cv_dir / "all_cv_repeat_metrics.csv"
        cv_metrics.to_csv(all_cv_path, index=False)
        logger.info(f"All CV metrics saved: {all_cv_path}")

        cv_summary = compute_summary_stats(cv_metrics, logger=logger)
        if not cv_summary.empty:
            cv_summary_path = cv_dir / "cv_metrics_summary.csv"
            cv_summary.to_csv(cv_summary_path, index=False)
            logger.info(f"CV summary saved: {cv_summary_path}")
    else:
        logger.info("No CV metrics found (optional)")

    # Aggregate best hyperparameters
    log_section(logger, "Aggregating Hyperparameters")

    best_params = collect_best_hyperparams(split_dirs, logger=logger)
    if not best_params.empty:
        all_params_path = cv_dir / "all_best_params_per_split.csv"
        best_params.to_csv(all_params_path, index=False)
        logger.info(f"All best hyperparameters saved: {all_params_path}")
        logger.info(
            f"  {len(best_params)} hyperparameter sets from {best_params['split_seed'].nunique()} splits"
        )

        params_summary = aggregate_hyperparams_summary(best_params, logger=logger)
        if not params_summary.empty:
            params_summary_path = cv_dir / "hyperparams_summary.csv"
            params_summary.to_csv(params_summary_path, index=False)
            logger.info(f"Hyperparameters summary saved: {params_summary_path}")
    else:
        logger.info("No hyperparameters found (Optuna may not be enabled)")

    # Aggregate ensemble hyperparameters if available
    if ensemble_dirs:
        ensemble_params = collect_ensemble_hyperparams(ensemble_dirs, logger=logger)
        if not ensemble_params.empty:
            ensemble_params_path = cv_dir / "ensemble_config_per_split.csv"
            ensemble_params.to_csv(ensemble_params_path, index=False)
            logger.info(f"Ensemble configurations saved: {ensemble_params_path}")

    log_section(logger, "Feature Stability Analysis")

    feature_stability_df, stable_features_df = aggregate_feature_stability(
        split_dirs, stability_threshold=stability_threshold, logger=logger
    )

    feature_reports_dir = reports_dir / "feature_reports"
    feature_reports_dir.mkdir(parents=True, exist_ok=True)

    stable_panel_dir = reports_dir / "stable_panel"
    stable_panel_dir.mkdir(parents=True, exist_ok=True)

    if not feature_stability_df.empty:
        feature_stability_df.to_csv(
            feature_reports_dir / "feature_stability_summary.csv", index=False
        )
        logger.info(f"Feature stability: {len(feature_stability_df)} features analyzed")

    if not stable_features_df.empty:
        stable_features_df.to_csv(stable_panel_dir / "consensus_stable_features.csv", index=False)
        logger.info(
            f"Stable features (>={stability_threshold*100:.0f}% splits): "
            f"{len(stable_features_df)} features"
        )
    else:
        logger.info("No stable features found (or no feature selection data)")

    log_section(logger, "Building Consensus Panels")

    consensus_panels = build_consensus_panels(
        split_dirs, threshold=stability_threshold, logger=logger
    )

    panels_dir = reports_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    for panel_size, manifest in consensus_panels.items():
        manifest_path = panels_dir / f"consensus_panel_N{panel_size}.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(
            f"Consensus panel N={panel_size}: {manifest['n_consensus_proteins']} proteins "
            f"(from {manifest['n_splits_with_panel']} splits)"
        )

    log_section(logger, "Aggregating Feature Reports")

    all_feature_reports = collect_feature_reports(split_dirs, logger=logger)
    agg_feature_report = pd.DataFrame()

    if not all_feature_reports.empty:
        all_feature_reports_path = feature_reports_dir / "all_feature_reports.csv"
        all_feature_reports.to_csv(all_feature_reports_path, index=False)
        logger.info(
            f"All feature reports: {len(all_feature_reports)} entries from "
            f"{all_feature_reports['split_seed'].nunique()} splits"
        )

        agg_feature_report = aggregate_feature_reports(all_feature_reports, logger=logger)
        if not agg_feature_report.empty:
            agg_feature_report_path = feature_reports_dir / "feature_report.csv"
            agg_feature_report.to_csv(agg_feature_report_path, index=False)
            logger.info(f"Aggregated feature report: {len(agg_feature_report)} proteins analyzed")
            logger.info(
                f"Top 5 proteins by selection frequency: "
                f"{', '.join(agg_feature_report.head(5)['protein'].tolist())}"
            )
    else:
        logger.info("No feature reports found (optional - depends on feature selection)")

    log_section(logger, "Saving Aggregation Metadata")

    # Aggregate sample category breakdowns from pooled predictions
    sample_categories_metadata = {}

    for split_name, df in [
        ("test", pooled_test_df),
        ("val", pooled_val_df),
        ("train_oof", pooled_train_oof_df),
    ]:
        if df.empty:
            continue

        if "category" in df.columns:
            cat_counts = df["category"].value_counts().to_dict()
            sample_categories_metadata[split_name] = {
                "controls": int(cat_counts.get("Controls", 0)),
                "incident": int(cat_counts.get("Incident", 0)),
                "prevalent": int(cat_counts.get("Prevalent", 0)),
                "total": len(df),
            }
        else:
            # Fallback: just total count
            sample_categories_metadata[split_name] = {
                "total": len(df),
                "controls": None,
                "incident": None,
                "prevalent": None,
            }

    log_section(logger, "Generating Aggregated Plots")

    n_splits = len(split_dirs)
    split_seeds = [int(sd.name.replace("split_seed", "")) for sd in split_dirs]
    meta_lines = build_aggregated_metadata(
        n_splits=n_splits,
        split_seeds=split_seeds,
        sample_categories=sample_categories_metadata,
        timestamp=True,
    )

    if save_plots:
        generate_aggregated_plots(
            pooled_test_df=pooled_test_df,
            pooled_val_df=pooled_val_df,
            pooled_train_oof_df=pooled_train_oof_df,
            out_dir=agg_dir,
            threshold_info=threshold_info,
            plot_formats=plot_formats,
            meta_lines=meta_lines,
            logger=logger,
            plot_roc=plot_roc,
            plot_pr=plot_pr,
            plot_calibration=plot_calibration,
            plot_risk_distribution=plot_risk_distribution,
            plot_dca=plot_dca,
            plot_oof_combined=plot_oof_combined,
            target_specificity=target_specificity,
        )

    log_section(logger, "Aggregating Optuna Trials")

    # Aggregate Optuna hyperparameter tuning trials across splits
    try:
        from ced_ml.plotting.optuna_plots import aggregate_optuna_trials

        optuna_trials = []
        for split_dir in split_dirs:
            optuna_csv = split_dir / "cv" / "optuna" / "optuna_trials.csv"
            if optuna_csv.exists():
                try:
                    df = pd.read_csv(optuna_csv)
                    optuna_trials.append(df)
                except Exception as e:
                    logger.warning(f"Failed to load optuna trials from {optuna_csv}: {e}")

        if optuna_trials:
            optuna_dir = agg_dir / "cv" / "optuna"
            optuna_dir.mkdir(parents=True, exist_ok=True)

            aggregate_optuna_trials(
                trials_dfs=optuna_trials,
                out_dir=optuna_dir,
                prefix="",
            )
            logger.info(f"Aggregated {len(optuna_trials)} Optuna trial sets: {optuna_dir}")
        else:
            logger.info("No Optuna trials found (optional - depends on config.optuna.enabled)")

    except Exception as e:
        logger.warning(f"Failed to aggregate Optuna trials: {e}")

    log_section(logger, "Generating Additional Artifacts")

    # --- Calibration CSV export ---
    try:
        from sklearn.calibration import calibration_curve

        diag_calibration_dir = agg_dir / "diagnostics" / "calibration"
        diag_calibration_dir.mkdir(parents=True, exist_ok=True)

        calib_bins = 10  # Match train.py default
        calib_rows = []

        for split_name, df in [("test", pooled_test_df), ("val", pooled_val_df)]:
            if df.empty:
                continue

            pred_col = None
            for col in ["y_prob", "y_pred", "risk_score", "prob", "prediction"]:
                if col in df.columns:
                    pred_col = col
                    break

            if pred_col is None or "y_true" not in df.columns:
                continue

            y_true = df["y_true"].values
            y_pred = df[pred_col].values

            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            y_true = y_true[mask].astype(int)
            y_pred = y_pred[mask].astype(float)

            if len(y_true) == 0 or len(np.unique(y_true)) < 2:
                continue

            prob_true, prob_pred = calibration_curve(
                y_true, y_pred, n_bins=calib_bins, strategy="uniform"
            )

            for bin_center, obs_freq in zip(prob_pred, prob_true, strict=False):
                calib_rows.append(
                    {
                        "split": split_name,
                        "bin_center": bin_center,
                        "observed_freq": obs_freq,
                        "scenario": "aggregated",
                        "model": "pooled",
                    }
                )

        if calib_rows:
            calib_df = pd.DataFrame(calib_rows)
            calib_csv_path = diag_calibration_dir / "calibration.csv"
            calib_df.to_csv(calib_csv_path, index=False)
            logger.info(f"Calibration CSV saved: {calib_csv_path}")
    except Exception as e:
        if logger:
            logger.warning(f"Failed to save calibration CSV: {e}")

    # --- DCA CSV export ---
    try:
        from ced_ml.metrics.dca import save_dca_results

        diag_dca_dir = agg_dir / "diagnostics" / "dca"
        diag_dca_dir.mkdir(parents=True, exist_ok=True)

        for split_name, df in [("test", pooled_test_df), ("val", pooled_val_df)]:
            if df.empty:
                continue

            pred_col = None
            for col in ["y_prob", "y_pred", "risk_score", "prob", "prediction"]:
                if col in df.columns:
                    pred_col = col
                    break

            if pred_col is None or "y_true" not in df.columns:
                continue

            y_true = df["y_true"].values
            y_pred = df[pred_col].values

            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            y_true = y_true[mask].astype(int)
            y_pred = y_pred[mask].astype(float)

            if len(y_true) == 0 or len(np.unique(y_true)) < 2:
                continue

            dca_result = save_dca_results(
                y_true=y_true,
                y_pred_prob=y_pred,
                out_dir=str(diag_dca_dir),
                prefix=f"{split_name}__",
                thresholds=None,
                report_points=None,
                prevalence_adjustment=None,
            )
            if logger:
                logger.info(f"DCA CSV ({split_name}): {dca_result.get('dca_csv_path', 'N/A')}")
    except Exception as e:
        if logger:
            logger.warning(f"Failed to save DCA CSV: {e}")

    # --- Screening results aggregation ---
    try:
        diag_screening_dir = agg_dir / "diagnostics" / "screening"
        diag_screening_dir.mkdir(parents=True, exist_ok=True)

        all_screening = []
        for split_dir in split_dirs:
            seed = int(split_dir.name.replace("split_seed", ""))
            screening_path = split_dir / "diagnostics" / "screening"

            if not screening_path.exists():
                continue

            for csv_file in screening_path.glob("*_screening_results.csv"):
                try:
                    df = pd.read_csv(csv_file)
                    df["split_seed"] = seed
                    all_screening.append(df)
                except Exception as e:
                    if logger:
                        logger.debug(f"Failed to read {csv_file}: {e}")

        if all_screening:
            combined_screening = pd.concat(all_screening, ignore_index=True)
            screening_csv_path = diag_screening_dir / "all_screening_results.csv"
            combined_screening.to_csv(screening_csv_path, index=False)
            logger.info(f"Screening results aggregated: {screening_csv_path}")

            # Compute summary statistics
            if "protein" in combined_screening.columns:
                protein_cols = [
                    c
                    for c in combined_screening.columns
                    if c not in ["split_seed", "scenario", "model", "protein"]
                ]
                if protein_cols:
                    screening_summary = (
                        combined_screening.groupby("protein")[protein_cols]
                        .agg(["mean", "std"])
                        .reset_index()
                    )
                    screening_summary.columns = [
                        "_".join(col).strip("_") for col in screening_summary.columns
                    ]
                    screening_summary_path = diag_screening_dir / "screening_summary.csv"
                    screening_summary.to_csv(screening_summary_path, index=False)
                    logger.info(f"Screening summary saved: {screening_summary_path}")
    except Exception as e:
        if logger:
            logger.warning(f"Failed to aggregate screening results: {e}")

    # --- Learning curve aggregation ---
    try:
        # CSVs go to diagnostics/learning_curve/, plots go to diagnostics/plots/
        diag_learning_dir = agg_dir / "diagnostics" / "learning_curve"
        diag_learning_dir.mkdir(parents=True, exist_ok=True)
        diag_plots_dir = agg_dir / "diagnostics" / "plots"
        diag_plots_dir.mkdir(parents=True, exist_ok=True)

        all_learning_curves = []
        for split_dir in split_dirs:
            seed = int(split_dir.name.replace("split_seed", ""))
            # Individual splits store CSVs in diagnostics/learning_curve/ (singular)
            lc_path = split_dir / "diagnostics" / "learning_curve"

            if not lc_path.exists():
                continue

            for csv_file in lc_path.glob("*_learning_curve.csv"):
                try:
                    df = pd.read_csv(csv_file)
                    df["split_seed"] = seed
                    df["run_dir"] = split_dir.name
                    all_learning_curves.append(df)
                except Exception as e:
                    if logger:
                        logger.debug(f"Failed to read {csv_file}: {e}")

        if all_learning_curves:
            combined_lc = pd.concat(all_learning_curves, ignore_index=True)
            lc_csv_path = diag_learning_dir / "all_learning_curves.csv"
            combined_lc.to_csv(lc_csv_path, index=False)
            logger.info(f"Learning curves aggregated: {lc_csv_path}")

            # Generate learning curve summary plot
            if save_plots and plot_learning_curve:
                try:
                    from ced_ml.plotting.learning_curve import (
                        aggregate_learning_curve_runs,
                        plot_learning_curve_summary,
                    )

                    if "train_size" in combined_lc.columns:
                        # aggregate_learning_curve_runs expects list[pd.DataFrame]
                        agg_lc = aggregate_learning_curve_runs(all_learning_curves)
                        if not agg_lc.empty:
                            # Save aggregated summary CSV to learning_curve dir
                            agg_lc_path = diag_learning_dir / "learning_curve_summary.csv"
                            agg_lc.to_csv(agg_lc_path, index=False)
                            logger.info(f"Learning curve summary: {agg_lc_path}")

                            # Save plots to diagnostics/plots/
                            for fmt in plot_formats:
                                plot_learning_curve_summary(
                                    df=agg_lc,
                                    out_path=diag_plots_dir / f"learning_curve.{fmt}",
                                    title="Aggregated Learning Curve",
                                    meta_lines=meta_lines,
                                )
                            logger.info("Learning curve summary plot saved")
                except Exception as e:
                    if logger:
                        logger.debug(f"Failed to generate learning curve plot: {e}")
    except Exception as e:
        if logger:
            logger.warning(f"Failed to aggregate learning curves: {e}")

    # Collect ensemble-specific metadata if ENSEMBLE model present
    ensemble_metadata = {}
    if "ENSEMBLE" in all_models and ensemble_dirs:
        ensemble_coefs_agg = {}
        ensemble_configs = {}

        for ed in ensemble_dirs:
            settings_path = ed / "core" / "run_settings.json"
            config_path = ed / "config.yaml"

            # Extract coefficients
            if settings_path.exists():
                try:
                    with open(settings_path) as f:
                        settings = json.load(f)
                    meta_coef = settings.get("meta_coef", {})
                    split_seed = settings.get("split_seed", 0)
                    if meta_coef:
                        ensemble_coefs_agg[split_seed] = meta_coef
                except Exception as e:
                    logger.debug(f"Could not read ensemble settings from {ed}: {e}")

            # Extract base model list and meta-learner config
            if config_path.exists():
                try:
                    import yaml

                    with open(config_path) as f:
                        config = yaml.safe_load(f)
                    if "ensemble" in config:
                        ensemble_cfg = config["ensemble"]
                        split_seed = int(ed.name.replace("split_seed", "").replace("split_", ""))
                        ensemble_configs[split_seed] = {
                            "base_models": ensemble_cfg.get("base_models", []),
                            "meta_penalty": ensemble_cfg.get("meta_model", {}).get("penalty", "l2"),
                            "meta_C": ensemble_cfg.get("meta_model", {}).get("C", 1.0),
                        }
                except Exception as e:
                    logger.debug(f"Could not read ensemble config from {config_path}: {e}")

        # Aggregate coefficients across splits
        if ensemble_coefs_agg:
            all_coef_names = set()
            for coef_dict in ensemble_coefs_agg.values():
                all_coef_names.update(coef_dict.keys())

            coef_stats = {}
            for name in all_coef_names:
                vals = [cd.get(name) for cd in ensemble_coefs_agg.values() if name in cd]
                if vals:
                    coef_stats[name] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "min": float(np.min(vals)),
                        "max": float(np.max(vals)),
                        "n_splits": len(vals),
                    }

            ensemble_metadata["meta_learner_coefficients"] = {
                "aggregated_stats": coef_stats,
                "per_split": ensemble_coefs_agg,
                "n_splits_with_coefs": len(ensemble_coefs_agg),
            }

        # Add ensemble configuration metadata
        if ensemble_configs:
            # Get base models from first available config
            first_config = next(iter(ensemble_configs.values()), {})
            ensemble_metadata["ensemble_config"] = {
                "base_models": first_config.get("base_models", []),
                "n_base_models": len(first_config.get("base_models", [])),
                "meta_penalty": first_config.get("meta_penalty", "l2"),
                "meta_C": first_config.get("meta_C", 1.0),
                "n_splits_with_config": len(ensemble_configs),
            }

        # Add ENSEMBLE model performance vs best single model
        if "ENSEMBLE" in pooled_test_metrics and len(all_models) > 1:
            ensemble_test = pooled_test_metrics["ENSEMBLE"]
            base_models_test = {m: pooled_test_metrics[m] for m in all_models if m != "ENSEMBLE"}

            if base_models_test:
                best_base_auroc = max(
                    (m.get("AUROC", 0) for m in base_models_test.values()), default=0
                )
                ensemble_auroc = ensemble_test.get("AUROC", 0)

                if best_base_auroc > 0:
                    improvement = ((ensemble_auroc - best_base_auroc) / best_base_auroc) * 100
                    ensemble_metadata["performance"] = {
                        "test_AUROC": ensemble_auroc,
                        "best_base_model_AUROC": best_base_auroc,
                        "AUROC_improvement_percent": improvement,
                        "test_PR_AUC": ensemble_test.get("PR_AUC"),
                        "test_Brier": ensemble_test.get("Brier"),
                    }

    agg_metadata: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "n_splits": n_splits,
        "split_seeds": split_seeds,
        "models": all_models,
        "n_models": len(all_models),
        "n_boot": n_boot,
        "stability_threshold": stability_threshold,
        "target_specificity": target_specificity,
        "sample_categories": sample_categories_metadata,
        "pooled_metrics": {
            "test": pooled_test_metrics,  # Now keyed by model
            "val": pooled_val_metrics,  # Now keyed by model
        },
        "thresholds": threshold_info,  # Now keyed by model
        "ensemble": ensemble_metadata if ensemble_metadata else None,
        "feature_consensus": {
            "n_features_analyzed": (
                len(feature_stability_df) if not feature_stability_df.empty else 0
            ),
            "n_stable_features": len(stable_features_df) if not stable_features_df.empty else 0,
            "top_10_features": (
                stable_features_df["protein"].head(10).tolist()
                if not stable_features_df.empty
                else []
            ),
        },
        "feature_reports": {
            "n_proteins_in_reports": len(agg_feature_report) if not agg_feature_report.empty else 0,
            "n_splits_with_reports": (
                all_feature_reports["split_seed"].nunique() if not all_feature_reports.empty else 0
            ),
            "top_10_by_selection_freq": (
                agg_feature_report.head(10)["protein"].tolist()
                if not agg_feature_report.empty
                else []
            ),
        },
        "consensus_panels": {
            str(k): {
                "n_proteins": v["n_consensus_proteins"],
                "n_splits_with_panel": v["n_splits_with_panel"],
            }
            for k, v in consensus_panels.items()
        },
        "files_generated": [f.name for f in agg_dir.rglob("*") if f.is_file()],
    }

    agg_metadata_path = agg_dir / "aggregation_metadata.json"
    with open(agg_metadata_path, "w") as f:
        json.dump(agg_metadata, f, indent=2)
    logger.info(f"Metadata saved: {agg_metadata_path}")

    log_section(logger, "Aggregation Complete")
    logger.info(f"Results saved to: {agg_dir}")

    # Build per-model summary for return value
    per_model_summary = {}
    for model_name in all_models:
        model_test = pooled_test_metrics.get(model_name, {})
        model_threshold = threshold_info.get(model_name, {})
        per_model_summary[model_name] = {
            "pooled_test_auroc": model_test.get("AUROC"),
            "pooled_test_prauc": model_test.get("PR_AUC"),
            "youden_threshold": model_threshold.get("youden_threshold"),
        }

    return {
        "status": "success",
        "output_dir": str(agg_dir),
        "n_splits": n_splits,
        "models": all_models,
        "per_model": per_model_summary,
        "n_stable_features": len(stable_features_df) if not stable_features_df.empty else 0,
    }
