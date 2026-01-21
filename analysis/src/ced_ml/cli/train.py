"""
CLI implementation for train command.

Thin wrapper around existing celiacML_faith.py logic with new config system.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ced_ml.config.loader import load_training_config, save_config
from ced_ml.config.validation import validate_training_config
from ced_ml.data.columns import get_available_columns_from_file, resolve_columns
from ced_ml.data.filters import apply_row_filters
from ced_ml.data.io import read_proteomics_file, usecols_for_proteomics
from ced_ml.data.schema import (
    CONTROL_LABEL,
    SCENARIO_DEFINITIONS,
    TARGET_COL,
)
from ced_ml.evaluation.reports import OutputDirectories, ResultsWriter
from ced_ml.features.kbest import (
    build_kbest_pipeline_step,
)
from ced_ml.features.screening import screen_proteins
from ced_ml.features.stability import (
    compute_selection_frequencies,
    extract_stable_panel,
)
from ced_ml.features.panels import build_multi_size_panels
from ced_ml.metrics.dca import save_dca_results
from ced_ml.metrics.bootstrap import stratified_bootstrap_ci
from ced_ml.plotting.learning_curve import save_learning_curve_csv

# Feature selection modules
# Metrics modules
from ced_ml.metrics.discrimination import (
    compute_discrimination_metrics,
)
from ced_ml.metrics.thresholds import (
    binary_metrics_at_threshold,
    choose_threshold_objective,
    threshold_for_specificity,
    threshold_youden,
)
from ced_ml.models.prevalence import (
    PrevalenceAdjustedModel,
    adjust_probabilities_for_prevalence,
)

# Model modules
from ced_ml.models.registry import (
    build_models,
)
from ced_ml.models.training import (
    oof_predictions_with_nested_cv,
)

# Plotting modules
from ced_ml.plotting import (
    plot_calibration_curve,
    plot_oof_combined,
    plot_pr_curve,
    plot_risk_distribution,
    plot_roc_curve,
)
from ced_ml.utils.logging import log_section, setup_logger


def build_preprocessor(
    protein_cols: List[str], cat_cols: List[str], meta_num_cols: List[str]
) -> ColumnTransformer:
    """
    Build preprocessing pipeline for model training.

    Args:
        protein_cols: List of protein column names
        cat_cols: List of categorical column names
        meta_num_cols: List of numeric metadata column names

    Returns:
        ColumnTransformer with StandardScaler for numeric and OneHotEncoder for categorical
    """
    numeric_cols = protein_cols + meta_num_cols

    transformers = [
        ("num", StandardScaler(), numeric_cols),
    ]

    if cat_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                cat_cols,
            )
        )

    return ColumnTransformer(transformers=transformers, verbose_feature_names_out=False)


def build_training_pipeline(
    config: Any,
    classifier: Any,
    protein_cols: List[str],
    cat_cols: List[str],
    meta_num_cols: List[str],
) -> Pipeline:
    """
    Build complete training pipeline with preprocessing, feature selection, and classifier.

    Args:
        config: TrainingConfig object
        classifier: Unfitted sklearn classifier
        protein_cols: List of protein column names
        cat_cols: List of categorical column names
        meta_num_cols: List of numeric metadata column names

    Returns:
        Pipeline with named steps: pre (preprocessing), sel (feature selection), clf (classifier)
    """
    preprocessor = build_preprocessor(protein_cols, cat_cols, meta_num_cols)

    steps = [("pre", preprocessor)]

    if config.features.feature_select and config.features.feature_select != "none":
        k_val = config.features.kbest_max if hasattr(config.features, "kbest_max") else 500
        kbest = build_kbest_pipeline_step(k=k_val)
        steps.append(("sel", kbest))

    steps.append(("clf", classifier))

    return Pipeline(steps=steps)


def load_split_indices(
    split_dir: str, scenario: str, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load train/val/test split indices from CSVs.

    Args:
        split_dir: Directory containing split CSV files
        scenario: Scenario name (e.g., IncidentOnly)
        seed: Random seed used for splits

    Returns:
        (train_idx, val_idx, test_idx) as numpy arrays

    Raises:
        FileNotFoundError: If any split file is missing
    """
    split_dir = Path(split_dir)

    train_file = split_dir / f"{scenario}_train_idx_seed{seed}.csv"
    val_file = split_dir / f"{scenario}_val_idx_seed{seed}.csv"
    test_file = split_dir / f"{scenario}_test_idx_seed{seed}.csv"

    if not train_file.exists():
        raise FileNotFoundError(f"Train split file not found: {train_file}")
    if not val_file.exists():
        raise FileNotFoundError(f"Val split file not found: {val_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test split file not found: {test_file}")

    train_idx = pd.read_csv(train_file)["idx"].values
    val_idx = pd.read_csv(val_file)["idx"].values
    test_idx = pd.read_csv(test_file)["idx"].values

    return train_idx, val_idx, test_idx


def evaluate_on_split(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    train_prev: float,
    target_prev: float,
    config: Any,
    logger: Any,
) -> Dict[str, float]:
    """
    Evaluate model on a data split.

    Args:
        model: Fitted sklearn model with predict_proba method
        X: Features
        y: True labels
        train_prev: Training prevalence
        target_prev: Target prevalence for adjustment
        config: TrainingConfig object
        logger: Logger instance

    Returns:
        Dictionary of metrics
    """
    y_probs = model.predict_proba(X)[:, 1]

    y_probs_adj = adjust_probabilities_for_prevalence(
        y_probs, sample_prev=train_prev, target_prev=target_prev
    )

    metrics = compute_discrimination_metrics(y, y_probs_adj)

    threshold_obj = config.thresholds.objective if hasattr(config, "thresholds") else "youden"
    threshold_name, threshold = choose_threshold_objective(y, y_probs_adj, objective=threshold_obj)

    binary_metrics = binary_metrics_at_threshold(y, y_probs_adj, threshold)

    metrics.update({"threshold": threshold, **binary_metrics})

    return metrics


def run_train(
    config_file: Optional[str] = None,
    cli_args: Optional[Dict[str, Any]] = None,
    overrides: Optional[List[str]] = None,
    verbose: int = 0,
):
    """
    Run model training with new config system.

    Args:
        config_file: Path to YAML config file (optional)
        cli_args: Dictionary of CLI arguments (optional)
        overrides: List of config overrides (optional)
        verbose: Verbosity level (0=INFO, 1=DEBUG)
    """
    # Setup logger
    log_level = 20 - (verbose * 10)
    logger = setup_logger("ced_ml.train", level=log_level)

    log_section(logger, "CeD-ML Model Training")

    # Build overrides list from CLI args
    all_overrides = list(overrides) if overrides else []
    if cli_args:
        for key, value in cli_args.items():
            if value is not None:
                all_overrides.append(f"{key}={value}")

    # Load and validate config
    logger.info("Loading configuration...")
    config = load_training_config(config_file=config_file, overrides=all_overrides)

    logger.info("Validating configuration...")
    validate_training_config(config)

    # Save resolved config
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    config_path = outdir / "training_config.yaml"
    save_config(config, config_path)
    logger.info(f"Saved resolved config to: {config_path}")

    # Add file handler for run.log
    log_file = outdir / "run.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )
    logger.addHandler(file_handler)
    logger.info(f"Logging to: {log_file}")

    # Log config summary
    logger.info(f"Model: {config.model}")
    logger.info(f"Scenario: {config.scenario}")
    logger.info(f"CV: {config.cv.folds} folds × {config.cv.repeats} repeats")
    logger.info(f"Scoring: {config.cv.scoring}")

    # Step 1: Resolve columns (auto-detect or explicit)
    log_section(logger, "Resolving Columns")
    logger.info(f"Column mode: {config.columns.mode}")
    available_columns = get_available_columns_from_file(str(config.infile))
    resolved = resolve_columns(available_columns, config.columns)

    logger.info("Resolved columns:")
    logger.info(f"  Proteins: {len(resolved.protein_cols)}")
    logger.info(f"  Numeric metadata: {resolved.numeric_metadata}")
    logger.info(f"  Categorical metadata: {resolved.categorical_metadata}")

    # Step 2: Load data with resolved columns
    log_section(logger, "Loading Data")
    logger.info(f"Reading: {config.infile}")
    usecols_fn = usecols_for_proteomics(
        numeric_metadata=resolved.numeric_metadata,
        categorical_metadata=resolved.categorical_metadata,
    )
    df_raw = read_proteomics_file(config.infile, usecols=usecols_fn)

    # Step 3: Apply row filters (defaults: drop_uncertain_controls=True, dropna_meta_num=True)
    logger.info("Applying row filters...")
    df_filtered, filter_stats = apply_row_filters(df_raw, meta_num_cols=resolved.numeric_metadata)
    logger.info(f"Filtered: {filter_stats['n_in']:,} → {filter_stats['n_out']:,} rows")
    logger.info(f"  Removed {filter_stats['n_removed_uncertain_controls']} uncertain controls")
    logger.info(f"  Removed {filter_stats['n_removed_dropna_meta_num']} rows with missing metadata")

    # Step 4: Use resolved columns
    protein_cols = resolved.protein_cols
    logger.info(f"Using {len(protein_cols)} protein columns")

    # Step 4: Create output directories
    log_section(logger, "Setting Up Output Structure")
    outdirs = OutputDirectories.create(config.outdir, exist_ok=True)
    logger.info(f"Output root: {outdirs.root}")

    # Step 5: Load split indices
    log_section(logger, "Loading Splits")
    seed = getattr(config, "seed", getattr(config, "split_seed", 0))
    try:
        train_idx, val_idx, test_idx = load_split_indices(
            str(config.split_dir), config.scenario, seed
        )
        logger.info(f"Loaded splits for seed {seed}:")
        logger.info(f"  Train: {len(train_idx):,} samples")
        logger.info(f"  Val:   {len(val_idx):,} samples")
        logger.info(f"  Test:  {len(test_idx):,} samples")

        # Save split trace
        split_trace_df = pd.DataFrame(
            {
                "idx": np.concatenate([train_idx, val_idx, test_idx]),
                "split": (
                    ["train"] * len(train_idx)
                    + ["val"] * len(val_idx)
                    + ["test"] * len(test_idx)
                ),
                "scenario": config.scenario,
                "seed": seed,
            }
        )
        split_trace_path = (
            Path(outdirs.diag_splits)
            / f"{config.scenario}__train_test_split_trace.csv"
        )
        split_trace_df.to_csv(split_trace_path, index=False)
        logger.info(f"Split trace saved: {split_trace_path}")
    except FileNotFoundError as e:
        logger.error(f"Split files not found: {e}")
        logger.error("Please run 'ced save-splits' first to generate splits")
        raise

    # Step 6: Prepare X, y for each split
    scenario_def = SCENARIO_DEFINITIONS[config.scenario]
    target_labels = scenario_def["labels"]
    scenario_def["positive_label"]

    df_scenario = df_filtered[df_filtered[TARGET_COL].isin(target_labels)].copy()
    df_scenario["y"] = (df_scenario[TARGET_COL] != CONTROL_LABEL).astype(int)

    feature_cols = resolved.all_feature_cols

    X_train = df_scenario.iloc[train_idx][feature_cols]
    y_train = df_scenario.iloc[train_idx]["y"].values

    X_val = df_scenario.iloc[val_idx][feature_cols]
    y_val = df_scenario.iloc[val_idx]["y"].values

    X_test = df_scenario.iloc[test_idx][feature_cols]
    y_test = df_scenario.iloc[test_idx]["y"].values

    # Extract original category labels for 3-panel KDE plots
    cat_train = df_scenario.iloc[train_idx][TARGET_COL].values
    cat_val = df_scenario.iloc[val_idx][TARGET_COL].values
    cat_test = df_scenario.iloc[test_idx][TARGET_COL].values

    train_prev = float(y_train.mean())
    logger.info(f"Training prevalence: {train_prev:.3f}")

    # Step 7: Build classifier
    log_section(logger, "Building Model")
    logger.info(f"Model type: {config.model}")

    classifier = build_models(
        model_name=config.model,
        config=config,
        random_state=seed,
        n_jobs=config.n_jobs,
    )

    # Step 8: Build full pipeline
    pipeline = build_training_pipeline(
        config,
        classifier,
        protein_cols,
        resolved.categorical_metadata,
        resolved.numeric_metadata,
    )
    logger.info(f"Pipeline steps: {[name for name, _ in pipeline.steps]}")

    # Step 9: Run nested CV for OOF predictions
    log_section(logger, "Nested Cross-Validation")
    logger.info(f"Running {config.cv.folds}-fold CV × {config.cv.repeats} repeats...")

    # Create grid RNG if grid randomization is enabled
    grid_rng = np.random.default_rng(seed) if config.cv.grid_randomize else None

    oof_preds, elapsed_sec, best_params_df, selected_proteins_df = oof_predictions_with_nested_cv(
        pipeline=pipeline,
        model_name=config.model,
        X=X_train,
        y=y_train,
        protein_cols=protein_cols,
        config=config,
        random_state=seed,
        grid_rng=grid_rng,
    )

    logger.info(f"CV completed in {elapsed_sec:.1f}s")

    # Step 10: Fit final model on full train set
    log_section(logger, "Training Final Model")
    logger.info("Fitting on full training set...")

    final_pipeline = build_training_pipeline(
        config,
        classifier,
        protein_cols,
        resolved.categorical_metadata,
        resolved.numeric_metadata,
    )
    final_pipeline.fit(X_train, y_train)
    logger.info("Final model fitted")

    # Step 11: Evaluate on validation set (threshold selection)
    log_section(logger, "Validation Set Evaluation")
    # Determine target prevalence based on config
    if config.thresholds.target_prevalence_source == "fixed":
        target_prev = config.thresholds.target_prevalence_fixed
    elif config.thresholds.target_prevalence_source == "train":
        target_prev = train_prev
    else:
        # For "val" or "test" source, use train_prev as default (will be refined later)
        target_prev = train_prev

    val_metrics = evaluate_on_split(
        final_pipeline, X_val, y_val, train_prev, target_prev, config, logger
    )

    logger.info(f"Val AUROC: {val_metrics['AUROC']:.3f}")
    logger.info(f"Val PRAUC: {val_metrics['PR_AUC']:.3f}")
    logger.info(f"Selected threshold: {val_metrics['threshold']:.3f}")

    # Step 12: Evaluate on test set
    log_section(logger, "Test Set Evaluation")
    test_metrics = evaluate_on_split(
        final_pipeline, X_test, y_test, train_prev, target_prev, config, logger
    )

    logger.info(f"Test AUROC: {test_metrics['AUROC']:.3f}")
    logger.info(f"Test PRAUC: {test_metrics['PR_AUC']:.3f}")

    # Step 13: Wrap in prevalence-adjusted model
    prevalence_model = PrevalenceAdjustedModel(
        base_model=final_pipeline,
        sample_prevalence=train_prev,
        target_prevalence=target_prev,
    )

    # Step 14: Save outputs
    log_section(logger, "Saving Results")

    ResultsWriter(outdirs)

    # Save model
    model_filename = f"{config.scenario}__{config.model}__final_model.joblib"
    model_path = Path(outdirs.core) / model_filename
    import joblib

    joblib.dump(prevalence_model, model_path)
    logger.info(f"Model saved: {model_path}")

    # Save metrics
    val_metrics_df = pd.DataFrame([val_metrics])
    val_metrics_path = Path(outdirs.core) / "val_metrics.csv"
    val_metrics_df.to_csv(val_metrics_path, index=False)
    logger.info(f"Val metrics saved: {val_metrics_path}")

    test_metrics_df = pd.DataFrame([test_metrics])
    test_metrics_path = Path(outdirs.core) / "test_metrics.csv"
    test_metrics_df.to_csv(test_metrics_path, index=False)
    logger.info(f"Test metrics saved: {test_metrics_path}")

    # Save CV artifacts
    best_params_path = Path(outdirs.cv) / "best_params_per_split.csv"
    best_params_df.to_csv(best_params_path, index=False)
    logger.info(f"Best params saved: {best_params_path}")

    selected_proteins_path = Path(outdirs.cv) / "selected_proteins_per_split.csv"
    selected_proteins_df.to_csv(selected_proteins_path, index=False)
    logger.info(f"Selected proteins saved: {selected_proteins_path}")

    # Save cv_repeat_metrics.csv (per-repeat OOF metrics)
    cv_repeat_rows = []
    for repeat in range(oof_preds.shape[0]):
        repeat_preds = oof_preds[repeat, :]
        valid_mask = ~np.isnan(repeat_preds)
        if valid_mask.sum() > 0:
            y_repeat = y_train[valid_mask]
            p_repeat = repeat_preds[valid_mask]
            auroc = (
                roc_auc_score(y_repeat, p_repeat)
                if len(np.unique(y_repeat)) > 1
                else np.nan
            )
            prauc = (
                average_precision_score(y_repeat, p_repeat)
                if len(np.unique(y_repeat)) > 1
                else np.nan
            )
            brier = float(np.mean((y_repeat - p_repeat) ** 2))
            cv_repeat_rows.append(
                {
                    "scenario": config.scenario,
                    "model": config.model,
                    "repeat": repeat,
                    "folds": config.cv.folds,
                    "repeats": config.cv.repeats,
                    "n_train": len(y_train),
                    "n_train_pos": int(y_train.sum()),
                    "AUROC_oof": auroc,
                    "PR_AUC_oof": prauc,
                    "Brier_oof": brier,
                    "cv_seconds": elapsed_sec,
                    "feature_select": config.features.feature_select,
                    "random_state": seed,
                }
            )
    if cv_repeat_rows:
        cv_repeat_df = pd.DataFrame(cv_repeat_rows)
        cv_repeat_path = Path(outdirs.cv) / "cv_repeat_metrics.csv"
        cv_repeat_df.to_csv(cv_repeat_path, index=False)
        logger.info(f"CV repeat metrics saved: {cv_repeat_path}")

    # Save run settings
    run_settings = {
        "model": config.model,
        "scenario": config.scenario,
        "seed": seed,
        "train_prevalence": float(train_prev),
        "target_prevalence": float(target_prev),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "cv_elapsed_sec": elapsed_sec,
        "columns": {
            "mode": config.columns.mode,
            "n_proteins": len(resolved.protein_cols),
            "numeric_metadata": resolved.numeric_metadata,
            "categorical_metadata": resolved.categorical_metadata,
        },
    }
    run_settings_path = Path(outdirs.core) / "run_settings.json"
    with open(run_settings_path, "w") as f:
        json.dump(run_settings, f, indent=2)
    logger.info(f"Run settings saved: {run_settings_path}")

    # Save config_metadata.json at root (comprehensive run configuration)
    config_metadata = {
        "pipeline_version": "ced_ml_v2",
        "scenario": config.scenario,
        "model": config.model,
        "folds": config.cv.folds,
        "repeats": config.cv.repeats,
        "val_size": getattr(config, "val_size", 0.25),
        "test_size": getattr(config, "test_size", 0.25),
        "random_state": seed,
        "scoring": config.cv.scoring,
        "inner_folds": getattr(config.cv, "inner_folds", 5),
        "n_iter": getattr(config.cv, "n_iter", 50),
        "feature_select": config.features.feature_select,
        "kbest_scope": getattr(config.features, "kbest_scope", "inner"),
        "kbest_max": getattr(config.features, "kbest_max", 500),
        "screen_method": getattr(config.features, "screen_method", "none"),
        "screen_top_n": getattr(config.features, "screen_top_n", 1000),
        "calibrate_final_models": int(getattr(config.calibration, "enabled", False)),
        "threshold_source": getattr(config.thresholds, "threshold_source", "val"),
        "target_prevalence_source": getattr(config.thresholds, "target_prevalence_source", "train"),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "train_prevalence": float(train_prev),
        "target_prevalence": float(target_prev),
        "cv_elapsed_sec": elapsed_sec,
        "n_proteins": len(resolved.protein_cols),
        "timestamp": datetime.now().isoformat(),
    }
    config_metadata_path = Path(outdirs.root) / "config_metadata.json"
    with open(config_metadata_path, "w") as f:
        json.dump(config_metadata, f, indent=2, sort_keys=True)
    logger.info(f"Config metadata saved: {config_metadata_path}")

    # Save predictions
    test_preds_df = pd.DataFrame(
        {
            "idx": test_idx,
            "y_true": y_test,
            "y_prob": prevalence_model.predict_proba(X_test)[:, 1],
        }
    )
    test_preds_path = (
        Path(outdirs.preds_test) / f"{config.scenario}__test_preds__{config.model}.csv"
    )
    test_preds_df.to_csv(test_preds_path, index=False)
    logger.info(f"Test predictions saved: {test_preds_path}")

    val_preds_df = pd.DataFrame(
        {
            "idx": val_idx,
            "y_true": y_val,
            "y_prob": prevalence_model.predict_proba(X_val)[:, 1],
        }
    )
    val_preds_path = Path(outdirs.preds_val) / f"{config.scenario}__val_preds__{config.model}.csv"
    val_preds_df.to_csv(val_preds_path, index=False)
    logger.info(f"Val predictions saved: {val_preds_path}")

    # Save OOF predictions
    oof_preds_df = pd.DataFrame(
        {
            "idx": train_idx,
            "y_true": y_train,
        }
    )
    for repeat in range(oof_preds.shape[0]):
        oof_preds_df[f"y_prob_repeat{repeat}"] = oof_preds[repeat, :]
    oof_preds_path = (
        Path(outdirs.preds_train_oof) / f"{config.scenario}__train_oof__{config.model}.csv"
    )
    oof_preds_df.to_csv(oof_preds_path, index=False)
    logger.info(f"OOF predictions saved: {oof_preds_path}")

    # Save controls OOF predictions (mean across repeats)
    controls_mask = y_train == 0
    if controls_mask.sum() > 0:
        controls_idx = train_idx[controls_mask]
        controls_oof_mean = oof_preds[:, controls_mask].mean(axis=0)
        controls_oof_df = pd.DataFrame(
            {
                "idx": controls_idx,
                "y_true": y_train[controls_mask],
                "y_prob_oof_mean": controls_oof_mean,
            }
        )
        controls_oof_path = (
            Path(outdirs.preds_controls)
            / f"{config.scenario}__controls_risk__{config.model}__oof_mean.csv"
        )
        controls_oof_df.to_csv(controls_oof_path, index=False)
        logger.info(f"Controls OOF predictions saved: {controls_oof_path}")

    # Step 15: Generate plots (if enabled)
    if config.output.save_plots:
        log_section(logger, "Generating Plots")
        plots_dir = Path(outdirs.diag_plots)
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Common metadata for plots
        meta_lines = [
            f"Model: {config.model} | Scenario: {config.scenario} | Seed: {seed}",
            f"Train prev: {train_prev:.3f} | Target prev: {target_prev:.3f}",
        ]

        # Validation set plots
        val_y_prob = val_preds_df["y_prob"].values
        val_title = f"{config.model} - Validation Set"

        plot_roc_curve(
            y_true=y_val,
            y_pred=val_y_prob,
            out_path=plots_dir / f"{config.scenario}__{config.model}__val_roc.{config.output.plot_format}",
            title=val_title,
            subtitle="ROC Curve",
            meta_lines=meta_lines,
        )
        logger.info("Val ROC curve saved")

        plot_pr_curve(
            y_true=y_val,
            y_pred=val_y_prob,
            out_path=plots_dir / f"{config.scenario}__{config.model}__val_pr.{config.output.plot_format}",
            title=val_title,
            subtitle="Precision-Recall Curve",
            meta_lines=meta_lines,
        )
        logger.info("Val PR curve saved")

        plot_calibration_curve(
            y_true=y_val,
            y_pred=val_y_prob,
            out_path=plots_dir / f"{config.scenario}__{config.model}__val_calibration.{config.output.plot_format}",
            title=val_title,
            subtitle="Calibration",
            n_bins=config.output.calib_bins,
            meta_lines=meta_lines,
        )
        logger.info("Val calibration plot saved")

        # Test set plots
        test_y_prob = test_preds_df["y_prob"].values
        test_title = f"{config.model} - Test Set"

        # Compute thresholds for plotting
        youden_thr = threshold_youden(y_test, test_y_prob)
        spec95_thr = threshold_for_specificity(y_test, test_y_prob, target_spec=0.95)
        logger.info(f"Thresholds: Youden={youden_thr:.4f}, Spec95={spec95_thr:.4f}")

        # Compute metrics at each threshold
        metrics_at_thresholds = {
            "youden": binary_metrics_at_threshold(y_test, test_y_prob, youden_thr),
            "spec95": binary_metrics_at_threshold(y_test, test_y_prob, spec95_thr),
        }

        plot_roc_curve(
            y_true=y_test,
            y_pred=test_y_prob,
            out_path=plots_dir / f"{config.scenario}__{config.model}__test_roc.{config.output.plot_format}",
            title=test_title,
            subtitle="ROC Curve",
            meta_lines=meta_lines,
            youden_threshold=youden_thr,
            alpha_threshold=spec95_thr,
            metrics_at_thresholds=metrics_at_thresholds,
        )
        logger.info("Test ROC curve saved")

        plot_pr_curve(
            y_true=y_test,
            y_pred=test_y_prob,
            out_path=plots_dir / f"{config.scenario}__{config.model}__test_pr.{config.output.plot_format}",
            title=test_title,
            subtitle="Precision-Recall Curve",
            meta_lines=meta_lines,
        )
        logger.info("Test PR curve saved")

        plot_calibration_curve(
            y_true=y_test,
            y_pred=test_y_prob,
            out_path=plots_dir / f"{config.scenario}__{config.model}__test_calibration.{config.output.plot_format}",
            title=test_title,
            subtitle="Calibration",
            n_bins=config.output.calib_bins,
            meta_lines=meta_lines,
        )
        logger.info("Test calibration plot saved")

        # Combined OOF plots across CV repeats
        plot_oof_combined(
            y_true=y_train,
            oof_preds=oof_preds,
            out_dir=plots_dir,
            model_name=config.model,
            scenario=config.scenario,
            seed=seed,
            cv_folds=config.cv.folds,
            train_prev=train_prev,
            plot_format=config.output.plot_format,
            calib_bins=config.output.calib_bins,
        )
        logger.info("OOF combined plots saved")

        # Generate risk distribution plots (go to preds/plots/)
        preds_plots_dir = Path(outdirs.preds_plots)
        preds_plots_dir.mkdir(parents=True, exist_ok=True)

        # Test set risk distribution
        plot_risk_distribution(
            y_true=y_test,
            scores=test_preds_df["y_prob"].values,
            out_path=preds_plots_dir
            / f"{config.scenario}__{config.model}__TEST_risk_distribution.{config.output.plot_format}",
            title=f"{config.model} - Test Set",
            subtitle="Risk Score Distribution",
            meta_lines=meta_lines,
            category_col=cat_test,
            youden_threshold=youden_thr,
            spec95_threshold=spec95_thr,
            metrics_at_thresholds=metrics_at_thresholds,
        )
        logger.info("Test risk distribution plot saved")

        # Val set risk distribution
        plot_risk_distribution(
            y_true=y_val,
            scores=val_preds_df["y_prob"].values,
            out_path=preds_plots_dir
            / f"{config.scenario}__{config.model}__VAL_risk_distribution.{config.output.plot_format}",
            title=f"{config.model} - Validation Set",
            subtitle="Risk Score Distribution",
            meta_lines=meta_lines,
            category_col=cat_val,
            youden_threshold=youden_thr,
            spec95_threshold=spec95_thr,
            metrics_at_thresholds=metrics_at_thresholds,
        )
        logger.info("Val risk distribution plot saved")

        # Train OOF risk distribution (mean across repeats)
        oof_mean = oof_preds.mean(axis=0)
        plot_risk_distribution(
            y_true=y_train,
            scores=oof_mean,
            out_path=preds_plots_dir
            / f"{config.scenario}__{config.model}__TRAIN_OOF_risk_distribution.{config.output.plot_format}",
            title=f"{config.model} - Train OOF",
            subtitle="Risk Score Distribution (mean across repeats)",
            meta_lines=meta_lines,
            category_col=cat_train,
            youden_threshold=youden_thr,
            spec95_threshold=spec95_thr,
            metrics_at_thresholds=metrics_at_thresholds,
        )
        logger.info("Train OOF risk distribution plot saved")

        # Controls-only risk distribution (if any controls in training set)
        if controls_mask.sum() > 0:
            plot_risk_distribution(
                y_true=None,
                scores=controls_oof_mean,
                out_path=preds_plots_dir
                / f"{config.scenario}__{config.model}__TRAIN_OOF_controls_risk_distribution.{config.output.plot_format}",
                title=f"{config.model} - Controls (Train OOF)",
                subtitle="Risk Score Distribution",
                meta_lines=meta_lines,
                youden_threshold=youden_thr,
                spec95_threshold=spec95_thr,
            )
            logger.info("Controls risk distribution plot saved")

        logger.info(f"Diagnostic plots saved to: {plots_dir}")
        logger.info(f"Risk distribution plots saved to: {preds_plots_dir}")

    # Step 16: Generate additional artifacts
    log_section(logger, "Generating Additional Artifacts")

    # --- Calibration CSV export (raw + adjusted) ---
    try:
        from sklearn.calibration import calibration_curve

        # Test set calibration data
        test_y_prob = test_preds_df["y_prob"].values
        prob_true_test, prob_pred_test = calibration_curve(
            y_test, test_y_prob, n_bins=config.output.calib_bins, strategy="uniform"
        )
        calib_df_test = pd.DataFrame({
            "bin_center": prob_pred_test,
            "observed_freq": prob_true_test,
            "split": "test",
            "scenario": config.scenario,
            "model": config.model,
        })

        # Val set calibration data
        val_y_prob = val_preds_df["y_prob"].values
        prob_true_val, prob_pred_val = calibration_curve(
            y_val, val_y_prob, n_bins=config.output.calib_bins, strategy="uniform"
        )
        calib_df_val = pd.DataFrame({
            "bin_center": prob_pred_val,
            "observed_freq": prob_true_val,
            "split": "val",
            "scenario": config.scenario,
            "model": config.model,
        })

        # Combine and save
        calib_df = pd.concat([calib_df_test, calib_df_val], ignore_index=True)
        calib_csv_path = (
            Path(outdirs.diag_calibration)
            / f"{config.scenario}__{config.model}__calibration.csv"
        )
        calib_df.to_csv(calib_csv_path, index=False)
        logger.info(f"Calibration data saved: {calib_csv_path}")
    except Exception as e:
        logger.warning(f"Failed to save calibration CSV: {e}")

    # --- DCA data export ---
    try:
        dca_summary = save_dca_results(
            y_true=y_test,
            y_pred_prob=test_preds_df["y_prob"].values,
            out_dir=str(outdirs.diag_dca),
            prefix=f"{config.scenario}__{config.model}__test__",
            thresholds=None,  # Use default thresholds
            report_points=None,
            prevalence_adjustment=target_prev,
        )
        logger.info(f"DCA results saved: {dca_summary.get('dca_csv_path', 'N/A')}")

        # Also compute DCA for validation set
        dca_summary_val = save_dca_results(
            y_true=y_val,
            y_pred_prob=val_preds_df["y_prob"].values,
            out_dir=str(outdirs.diag_dca),
            prefix=f"{config.scenario}__{config.model}__val__",
            thresholds=None,
            report_points=None,
            prevalence_adjustment=target_prev,
        )
        logger.info(f"DCA (val) results saved: {dca_summary_val.get('dca_csv_path', 'N/A')}")
    except Exception as e:
        logger.warning(f"Failed to save DCA results: {e}")

    # --- Learning curve CSV export ---
    try:
        lc_enabled = getattr(config.evaluation, "learning_curve", False)
        if lc_enabled:
            lc_csv_path = (
                Path(outdirs.diag_learning)
                / f"{config.scenario}__{config.model}__learning_curve.csv"
            )
            lc_plot_path = (
                Path(outdirs.diag_learning)
                / f"{config.scenario}__{config.model}__learning_curve.{config.output.plot_format}"
            )
            lc_meta = [
                f"Model: {config.model} | Scenario: {config.scenario}",
                f"CV: {config.cv.folds} folds | Scoring: {config.cv.scoring}",
            ]
            # Build a fresh pipeline for learning curve (don't use fitted one)
            lc_pipeline = build_training_pipeline(
                config,
                build_models(config.model, config, seed, config.n_jobs),
                protein_cols,
                resolved.categorical_metadata,
                resolved.numeric_metadata,
            )
            save_learning_curve_csv(
                estimator=lc_pipeline,
                X=X_train,
                y=y_train,
                out_csv=lc_csv_path,
                scoring=config.cv.scoring,
                cv=min(config.cv.folds, 5),  # Use at most 5 folds for speed
                min_frac=0.3,
                n_points=5,
                seed=seed,
                out_plot=lc_plot_path,
                meta_lines=lc_meta,
            )
            logger.info(f"Learning curve saved: {lc_csv_path}")
    except Exception as e:
        logger.warning(f"Failed to save learning curve: {e}")

    # --- Screening results export ---
    try:
        screen_method = getattr(config.features, "screen_method", "none")
        if screen_method and screen_method != "none":
            screen_top_n = getattr(config.features, "screen_top_n", 1000)
            _, screening_stats = screen_proteins(
                X_train=X_train,
                y_train=y_train,
                protein_cols=protein_cols,
                method=screen_method,
                top_n=screen_top_n,
            )
            if not screening_stats.empty:
                screening_stats["scenario"] = config.scenario
                screening_stats["model"] = config.model
                screening_path = (
                    Path(outdirs.diag_screening)
                    / f"{config.scenario}__{config.model}__screening_results.csv"
                )
                screening_stats.to_csv(screening_path, index=False)
                logger.info(f"Screening results saved: {screening_path}")
    except Exception as e:
        logger.warning(f"Failed to save screening results: {e}")

    # --- Feature reports and stable panel export ---
    try:
        # Compute selection frequencies from CV results
        selection_freq = compute_selection_frequencies(
            selected_proteins_df,
            selection_col="selected_proteins",
        )

        if selection_freq:
            # Feature report: frequencies and rankings
            feature_report = pd.DataFrame([
                {"protein": p, "selection_freq": f}
                for p, f in selection_freq.items()
            ])
            feature_report = feature_report.sort_values(
                "selection_freq", ascending=False
            ).reset_index(drop=True)
            feature_report["rank"] = range(1, len(feature_report) + 1)
            feature_report["scenario"] = config.scenario
            feature_report["model"] = config.model

            feature_report_path = (
                Path(outdirs.reports_features)
                / f"{config.scenario}__{config.model}__feature_report_train.csv"
            )
            feature_report.to_csv(feature_report_path, index=False)
            logger.info(f"Feature report saved: {feature_report_path}")

            # Stable panel extraction
            stable_panel_df, stable_proteins, _ = extract_stable_panel(
                selection_log=selected_proteins_df,
                n_repeats=config.cv.repeats,
                stability_threshold=0.75,
                selection_col="selected_proteins",
                fallback_top_n=20,
            )
            if not stable_panel_df.empty:
                stable_panel_df["scenario"] = config.scenario
                stable_panel_path = (
                    Path(outdirs.reports_stable)
                    / f"{config.scenario}__stable_panel__KBest.csv"
                )
                stable_panel_df.to_csv(stable_panel_path, index=False)
                logger.info(f"Stable panel saved: {stable_panel_path} ({len(stable_proteins)} proteins)")

            # Panel manifests (multiple sizes)
            panel_sizes = getattr(config.panels, "panel_sizes", [10, 25, 50])
            if panel_sizes and len(selection_freq) >= min(panel_sizes):
                corr_threshold = getattr(config.panels, "panel_corr_thresh", 0.80)
                panels = build_multi_size_panels(
                    df=X_train,
                    y=y_train,
                    selection_freq=selection_freq,
                    panel_sizes=panel_sizes,
                    corr_threshold=corr_threshold,
                    pool_limit=1000,
                )
                for size, (_comp_map, panel_proteins) in panels.items():
                    manifest = {
                        "scenario": config.scenario,
                        "model": config.model,
                        "panel_size": size,
                        "actual_size": len(panel_proteins),
                        "corr_threshold": corr_threshold,
                        "proteins": panel_proteins,
                    }
                    manifest_path = (
                        Path(outdirs.reports_panels)
                        / f"{config.scenario}__{config.model}__N{size}__panel_manifest.json"
                    )
                    with open(manifest_path, "w") as f:
                        json.dump(manifest, f, indent=2)
                    logger.info(f"Panel manifest saved: {manifest_path}")
    except Exception as e:
        logger.warning(f"Failed to save feature reports/panels: {e}")

    # --- Bootstrap CI for small test sets ---
    try:
        # Only run bootstrap if test set is small (< threshold samples)
        min_bootstrap_threshold = getattr(config.evaluation, "bootstrap_min_samples", 100)
        if len(y_test) < min_bootstrap_threshold:
            logger.info(f"Test set small ({len(y_test)} samples) - computing bootstrap CI")

            # Bootstrap CI for AUROC
            auroc_lo, auroc_hi = stratified_bootstrap_ci(
                y_true=y_test,
                y_pred=test_preds_df["y_prob"].values,
                metric_fn=roc_auc_score,
                n_boot=1000,
                seed=seed,
            )

            # Bootstrap CI for PR-AUC
            prauc_lo, prauc_hi = stratified_bootstrap_ci(
                y_true=y_test,
                y_pred=test_preds_df["y_prob"].values,
                metric_fn=average_precision_score,
                n_boot=1000,
                seed=seed,
            )

            bootstrap_ci_df = pd.DataFrame([{
                "scenario": config.scenario,
                "model": config.model,
                "n_test": len(y_test),
                "n_boot": 1000,
                "AUROC": test_metrics["AUROC"],
                "AUROC_ci_lo": auroc_lo,
                "AUROC_ci_hi": auroc_hi,
                "PR_AUC": test_metrics["PR_AUC"],
                "PR_AUC_ci_lo": prauc_lo,
                "PR_AUC_ci_hi": prauc_hi,
            }])
            bootstrap_ci_path = (
                Path(outdirs.diag_test_ci)
                / f"{config.scenario}__{config.model}__test_bootstrap_ci.csv"
            )
            bootstrap_ci_df.to_csv(bootstrap_ci_path, index=False)
            logger.info(
                f"Bootstrap CI saved: {bootstrap_ci_path} "
                f"(AUROC: {auroc_lo:.3f}-{auroc_hi:.3f}, PR-AUC: {prauc_lo:.3f}-{prauc_hi:.3f})"
            )
    except Exception as e:
        logger.warning(f"Failed to compute bootstrap CI: {e}")

    log_section(logger, "Training Complete")
    logger.info(f"All results saved to: {config.outdir}")
