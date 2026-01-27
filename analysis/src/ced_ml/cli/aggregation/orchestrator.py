"""
Orchestration helper functions for aggregate_splits.

This module contains the stage-level orchestration functions extracted from
the original mega-function run_aggregate_splits(). Each function represents
a logical stage in the aggregation pipeline.

Progress: PR 1.6 - Reduce run_aggregate_splits() complexity from C901=130
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from logging import Logger


def setup_aggregation_directories(results_path: Path) -> dict[str, Path]:
    """
    Create output directory structure for aggregation results.

    Args:
        results_path: Root results directory

    Returns:
        Dictionary mapping directory names to Path objects
    """
    agg_dir = results_path / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)

    dirs = {
        "agg": agg_dir,
        "core": agg_dir / "core",
        "cv": agg_dir / "cv",
        "preds": agg_dir / "preds",
        "reports": agg_dir / "reports",
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def save_pooled_predictions(
    pooled_test_df: pd.DataFrame,
    pooled_val_df: pd.DataFrame,
    pooled_train_oof_df: pd.DataFrame,
    preds_dir: Path,
    logger: Logger,
) -> None:
    """
    Save pooled predictions to CSV files.

    Saves both combined (all models) and per-model prediction files.

    Args:
        pooled_test_df: Pooled test predictions
        pooled_val_df: Pooled validation predictions
        pooled_train_oof_df: Pooled train OOF predictions
        preds_dir: Directory to save predictions
        logger: Logger instance
    """
    # Test predictions
    if not pooled_test_df.empty:
        test_preds_dir = preds_dir / "test_preds"
        test_preds_dir.mkdir(parents=True, exist_ok=True)
        pooled_test_df.to_csv(test_preds_dir / "pooled_test_preds.csv", index=False)

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

    # Validation predictions
    if not pooled_val_df.empty:
        val_preds_dir = preds_dir / "val_preds"
        val_preds_dir.mkdir(parents=True, exist_ok=True)
        pooled_val_df.to_csv(val_preds_dir / "pooled_val_preds.csv", index=False)

        if "model" in pooled_val_df.columns:
            for model_name, model_df in pooled_val_df.groupby("model"):
                model_df.to_csv(val_preds_dir / f"pooled_val_preds__{model_name}.csv", index=False)

        logger.info(
            f"Pooled val predictions: {len(pooled_val_df)} samples from "
            f"{pooled_val_df['split_seed'].nunique()} splits, "
            f"{pooled_val_df['model'].nunique() if 'model' in pooled_val_df.columns else 1} model(s)"
        )

    # Train OOF predictions
    if not pooled_train_oof_df.empty:
        train_oof_dir = preds_dir / "train_oof"
        train_oof_dir.mkdir(parents=True, exist_ok=True)

        # Compute mean across CV repeats for each split
        repeat_cols = [c for c in pooled_train_oof_df.columns if c.startswith("y_prob_repeat")]
        if repeat_cols:
            pooled_train_oof_df["y_prob"] = pooled_train_oof_df[repeat_cols].mean(axis=1)

        pooled_train_oof_df.to_csv(train_oof_dir / "pooled_train_oof.csv", index=False)

        if "model" in pooled_train_oof_df.columns:
            for model_name, model_df in pooled_train_oof_df.groupby("model"):
                model_df.to_csv(train_oof_dir / f"pooled_train_oof__{model_name}.csv", index=False)

        logger.info(
            f"Pooled train OOF predictions: {len(pooled_train_oof_df)} samples from "
            f"{pooled_train_oof_df['split_seed'].nunique()} splits, "
            f"{pooled_train_oof_df['model'].nunique() if 'model' in pooled_train_oof_df.columns else 1} model(s)"
        )


def compute_and_save_pooled_metrics(
    pooled_test_df: pd.DataFrame,
    pooled_val_df: pd.DataFrame,
    target_specificity: float,
    control_spec_targets: list[float],
    core_dir: Path,
    agg_dir: Path,
    logger: Logger,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, Any]]:
    """
    Compute and save pooled metrics for test and validation sets.

    Args:
        pooled_test_df: Pooled test predictions
        pooled_val_df: Pooled validation predictions
        target_specificity: Target specificity for threshold
        control_spec_targets: List of specificity targets for metrics
        core_dir: Directory for core metrics
        agg_dir: Aggregation root directory
        logger: Logger instance

    Returns:
        Tuple of (pooled_test_metrics, pooled_val_metrics, threshold_info)
    """
    from ced_ml.cli.aggregation.aggregation import (
        compute_pooled_metrics_by_model,
        compute_pooled_threshold_metrics,
        compute_selection_scores_for_models,
        save_threshold_data,
    )

    pooled_test_metrics: dict[str, dict[str, float]] = {}
    pooled_val_metrics: dict[str, dict[str, float]] = {}
    threshold_info: dict[str, Any] = {}

    # Detect models
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

    # Test metrics
    if not pooled_test_df.empty:
        pooled_test_metrics = compute_pooled_metrics_by_model(
            pooled_test_df, spec_targets=control_spec_targets
        )

        if pooled_test_metrics:
            metrics_rows = list(pooled_test_metrics.values())
            pd.DataFrame(metrics_rows).to_csv(core_dir / "pooled_test_metrics.csv", index=False)

            for model_name, metrics in pooled_test_metrics.items():
                logger.info(f"Pooled test [{model_name}] AUROC: {metrics.get('AUROC', 'N/A'):.4f}")
                logger.info(
                    f"Pooled test [{model_name}] PR-AUC: {metrics.get('PR_AUC', 'N/A'):.4f}"
                )
                logger.info(f"Pooled test [{model_name}] Brier: {metrics.get('Brier', 'N/A'):.4f}")

            # Selection scores
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

        # Threshold info
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

        if threshold_info:
            save_threshold_data(threshold_info, agg_dir, logger)

    # Validation metrics
    if not pooled_val_df.empty:
        pooled_val_metrics = compute_pooled_metrics_by_model(
            pooled_val_df, spec_targets=control_spec_targets
        )
        if pooled_val_metrics:
            metrics_rows = list(pooled_val_metrics.values())
            pd.DataFrame(metrics_rows).to_csv(core_dir / "pooled_val_metrics.csv", index=False)
            for model_name, metrics in pooled_val_metrics.items():
                logger.info(f"Pooled val [{model_name}] AUROC: {metrics.get('AUROC', 'N/A'):.4f}")

    return pooled_test_metrics, pooled_val_metrics, threshold_info


def build_aggregation_metadata(
    n_splits: int,
    split_seeds: list[int],
    all_models: list[str],
    n_boot: int,
    stability_threshold: float,
    target_specificity: float,
    sample_categories_metadata: dict[str, Any],
    pooled_test_metrics: dict[str, dict[str, float]],
    pooled_val_metrics: dict[str, dict[str, float]],
    threshold_info: dict[str, Any],
    feature_stability_df: pd.DataFrame,
    stable_features_df: pd.DataFrame,
    agg_feature_report: pd.DataFrame,
    all_feature_reports: pd.DataFrame,
    consensus_panels: dict[int, dict[str, Any]],
    ensemble_metadata: dict[str, Any],
    agg_dir: Path,
) -> dict[str, Any]:
    """
    Build and save aggregation metadata JSON.

    Args:
        n_splits: Number of splits
        split_seeds: List of split seeds
        all_models: List of all models
        n_boot: Number of bootstrap iterations
        stability_threshold: Feature stability threshold
        target_specificity: Target specificity
        sample_categories_metadata: Sample category counts per split
        pooled_test_metrics: Test metrics by model
        pooled_val_metrics: Validation metrics by model
        threshold_info: Threshold info by model
        feature_stability_df: Feature stability dataframe
        stable_features_df: Stable features dataframe
        agg_feature_report: Aggregated feature report
        all_feature_reports: All feature reports
        consensus_panels: Consensus panel manifests
        ensemble_metadata: Ensemble-specific metadata
        agg_dir: Aggregation directory

    Returns:
        Metadata dictionary
    """
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
            "test": pooled_test_metrics,
            "val": pooled_val_metrics,
        },
        "thresholds": threshold_info,
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

    return agg_metadata


def build_return_summary(
    all_models: list[str],
    pooled_test_metrics: dict[str, dict[str, float]],
    threshold_info: dict[str, Any],
    n_splits: int,
    stable_features_df: pd.DataFrame,
    agg_dir: Path,
) -> dict[str, Any]:
    """
    Build summary dictionary for return value.

    Args:
        all_models: List of all models
        pooled_test_metrics: Test metrics by model
        threshold_info: Threshold info by model
        n_splits: Number of splits
        stable_features_df: Stable features dataframe
        agg_dir: Aggregation directory

    Returns:
        Summary dictionary
    """
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
