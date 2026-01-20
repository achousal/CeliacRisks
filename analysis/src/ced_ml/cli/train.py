"""
CLI implementation for train command.

Thin wrapper around existing celiacML_faith.py logic with new config system.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ced_ml.config.loader import load_training_config, save_config
from ced_ml.config.validation import validate_training_config
from ced_ml.data.columns import get_available_columns_from_file, resolve_columns
from ced_ml.data.filters import apply_row_filters
from ced_ml.data.io import read_proteomics_csv, usecols_for_proteomics
from ced_ml.data.schema import (
    CAT_COLS,
    CONTROL_LABEL,
    META_NUM_COLS,
    PROTEIN_SUFFIX,
    SCENARIO_DEFINITIONS,
    TARGET_COL,
)
from ced_ml.evaluation.reports import OutputDirectories, ResultsWriter
from ced_ml.features.kbest import (
    build_kbest_pipeline_step,
)

# Feature selection modules
# Metrics modules
from ced_ml.metrics.discrimination import (
    compute_discrimination_metrics,
)
from ced_ml.metrics.thresholds import (
    binary_metrics_at_threshold,
    choose_threshold_objective,
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

    logger.info(f"Resolved columns:")
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
    df_raw = read_proteomics_csv(config.infile, usecols=usecols_fn)

    # Step 3: Apply row filters (defaults: drop_uncertain_controls=True, dropna_meta_num=True)
    logger.info("Applying row filters...")
    df_filtered, filter_stats = apply_row_filters(
        df_raw, meta_num_cols=resolved.numeric_metadata
    )
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

    train_prev = float(y_train.mean())
    logger.info(f"Training prevalence: {train_prev:.3f}")

    # Step 7: Build classifier
    log_section(logger, "Building Model")
    logger.info(f"Model type: {config.model}")

    class_weight = None
    if hasattr(config, "lr") and hasattr(config.lr, "class_weight"):
        class_weight = config.lr.class_weight
    elif hasattr(config, "rf") and hasattr(config.rf, "class_weight"):
        class_weight = config.rf.class_weight

    classifiers = build_models(
        class_weight_option=class_weight if class_weight else "balanced",
        y_for_scale_pos_weight=y_train if config.model == "XGBoost" else None,
    )

    if config.model not in classifiers:
        raise ValueError(f"Unknown model: {config.model}. Available: {list(classifiers.keys())}")

    classifier = classifiers[config.model]

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

    oof_preds, elapsed_sec, best_params_df, selected_proteins_df = oof_predictions_with_nested_cv(
        pipeline=pipeline,
        model_name=config.model,
        X=X_train,
        y=y_train,
        protein_cols=protein_cols,
        config=config,
        random_state=seed,
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
    target_prev = (
        config.thresholds.target_prevalence
        if hasattr(config.thresholds, "target_prevalence")
        else train_prev
    )

    val_metrics = evaluate_on_split(
        final_pipeline, X_val, y_val, train_prev, target_prev, config, logger
    )

    logger.info(f"Val AUROC: {val_metrics['auroc']:.3f}")
    logger.info(f"Val PRAUC: {val_metrics['prauc']:.3f}")
    logger.info(f"Selected threshold: {val_metrics['threshold']:.3f}")

    # Step 12: Evaluate on test set
    log_section(logger, "Test Set Evaluation")
    test_metrics = evaluate_on_split(
        final_pipeline, X_test, y_test, train_prev, target_prev, config, logger
    )

    logger.info(f"Test AUROC: {test_metrics['auroc']:.3f}")
    logger.info(f"Test PRAUC: {test_metrics['prauc']:.3f}")

    # Step 13: Wrap in prevalence-adjusted model
    prevalence_model = PrevalenceAdjustedModel(
        base_model=final_pipeline, sample_prevalence=train_prev, target_prevalence=target_prev
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

    # Save predictions
    test_preds_df = pd.DataFrame(
        {"idx": test_idx, "y_true": y_test, "y_prob": prevalence_model.predict_proba(X_test)[:, 1]}
    )
    test_preds_path = (
        Path(outdirs.preds_test) / f"{config.scenario}__test_preds__{config.model}.csv"
    )
    test_preds_df.to_csv(test_preds_path, index=False)
    logger.info(f"Test predictions saved: {test_preds_path}")

    val_preds_df = pd.DataFrame(
        {"idx": val_idx, "y_true": y_val, "y_prob": prevalence_model.predict_proba(X_val)[:, 1]}
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

    log_section(logger, "Training Complete")
    logger.info(f"All results saved to: {config.outdir}")
