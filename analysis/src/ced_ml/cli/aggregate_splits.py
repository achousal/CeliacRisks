"""
CLI implementation for aggregate-splits command.

Aggregates results across multiple split seeds into summary statistics,
pooled predictions, aggregated plots, and consensus panels.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ced_ml.cli.aggregation.aggregation import (
    compute_summary_stats,
)
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
from ced_ml.cli.aggregation.orchestrator import (
    build_aggregation_metadata as build_agg_metadata,
)
from ced_ml.cli.aggregation.orchestrator import (
    build_return_summary,
    compute_and_save_pooled_metrics,
    save_pooled_predictions,
    setup_aggregation_directories,
)
from ced_ml.cli.aggregation.plot_generator import (
    generate_aggregated_plots,
    generate_model_comparison_report,
)
from ced_ml.cli.aggregation.reporting import (
    aggregate_feature_reports,
    aggregate_feature_stability,
    build_consensus_panels,
)
from ced_ml.config.loader import load_aggregate_config
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

    # Setup directory structure
    dirs = setup_aggregation_directories(results_path)
    agg_dir = dirs["agg"]
    core_dir = dirs["core"]
    cv_dir = dirs["cv"]
    preds_dir = dirs["preds"]
    reports_dir = dirs["reports"]

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

    # Save pooled predictions
    save_pooled_predictions(pooled_test_df, pooled_val_df, pooled_train_oof_df, preds_dir, logger)

    log_section(logger, "Computing Pooled Metrics")

    # Compute and save pooled metrics
    pooled_test_metrics, pooled_val_metrics, threshold_info = compute_and_save_pooled_metrics(
        pooled_test_df=pooled_test_df,
        pooled_val_df=pooled_val_df,
        target_specificity=target_specificity,
        control_spec_targets=control_spec_targets,
        core_dir=core_dir,
        agg_dir=agg_dir,
        logger=logger,
    )

    # Detect all models
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

        # Concat-as-you-go to avoid accumulating all DataFrames in memory
        optuna_trials_combined = None
        n_optuna_trials = 0
        for split_dir in split_dirs:
            optuna_csv = split_dir / "cv" / "optuna" / "optuna_trials.csv"
            if optuna_csv.exists():
                try:
                    df = pd.read_csv(optuna_csv)
                    if optuna_trials_combined is None:
                        optuna_trials_combined = df
                    else:
                        optuna_trials_combined = pd.concat(
                            [optuna_trials_combined, df], ignore_index=True
                        )
                    n_optuna_trials += 1
                except Exception as e:
                    logger.warning(f"Failed to load optuna trials from {optuna_csv}: {e}")

        if optuna_trials_combined is not None:
            optuna_dir = agg_dir / "cv" / "optuna"
            optuna_dir.mkdir(parents=True, exist_ok=True)

            # Save combined trials directly (already concatenated)
            combined_csv = optuna_dir / "optuna_trials.csv"
            optuna_trials_combined.to_csv(combined_csv, index=False)
            logger.info(f"Aggregated {n_optuna_trials} Optuna trial sets: {optuna_dir}")
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

        # Concat-as-you-go to avoid accumulating all DataFrames in memory (saves ~500MB)
        combined_screening = None
        for split_dir in split_dirs:
            seed = int(split_dir.name.replace("split_seed", ""))
            screening_path = split_dir / "diagnostics" / "screening"

            if not screening_path.exists():
                continue

            for csv_file in screening_path.glob("*_screening_results.csv"):
                try:
                    df = pd.read_csv(csv_file)
                    df["split_seed"] = seed
                    if combined_screening is None:
                        combined_screening = df
                    else:
                        combined_screening = pd.concat([combined_screening, df], ignore_index=True)
                except Exception as e:
                    if logger:
                        logger.debug(f"Failed to read {csv_file}: {e}")

        if combined_screening is not None:
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

        # Note: Keep list for aggregate_learning_curve_runs (needs individual DFs)
        # Memory impact is lower than screening (smaller DataFrames)
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

    # Build and save aggregation metadata
    _ = build_agg_metadata(
        n_splits=n_splits,
        split_seeds=split_seeds,
        all_models=all_models,
        n_boot=n_boot,
        stability_threshold=stability_threshold,
        target_specificity=target_specificity,
        sample_categories_metadata=sample_categories_metadata,
        pooled_test_metrics=pooled_test_metrics,
        pooled_val_metrics=pooled_val_metrics,
        threshold_info=threshold_info,
        feature_stability_df=feature_stability_df,
        stable_features_df=stable_features_df,
        agg_feature_report=agg_feature_report,
        all_feature_reports=all_feature_reports,
        consensus_panels=consensus_panels,
        ensemble_metadata=ensemble_metadata,
        agg_dir=agg_dir,
    )
    logger.info(f"Metadata saved: {agg_dir / 'aggregation_metadata.json'}")

    log_section(logger, "Aggregation Complete")
    logger.info(f"Results saved to: {agg_dir}")

    # Build and return summary
    return build_return_summary(
        all_models=all_models,
        pooled_test_metrics=pooled_test_metrics,
        threshold_info=threshold_info,
        n_splits=n_splits,
        stable_features_df=stable_features_df,
        agg_dir=agg_dir,
    )
