"""CLI for training stacking ensemble from base model outputs.

This module provides the entry point for ensemble training. It collects
OOF predictions from previously trained base models, trains a meta-learner,
and generates ensemble predictions.

Usage:
    ced train-ensemble --results-dir results/ --base-models LR_EN,RF,XGBoost
    ced train-ensemble --config configs/training_config.yaml --split-seed 0
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from ced_ml.config.loader import load_training_config
from ced_ml.models.stacking import (
    StackingEnsemble,
    _find_model_split_dir,
    collect_oof_predictions,
    collect_split_predictions,
    load_calibration_info_for_models,
)
from ced_ml.utils.logging import log_section, setup_logger

logger = logging.getLogger(__name__)


def compute_ensemble_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    split_name: str = "test",
) -> dict[str, float]:
    """Compute standard metrics for ensemble predictions.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        split_name: Name of the split (for logging)

    Returns:
        Dict with AUROC, PR_AUC, Brier score
    """
    metrics = {}

    # Handle edge cases
    if len(np.unique(y_true)) < 2:
        logger.warning(f"Only one class present in {split_name} set, metrics may be undefined")
        return metrics

    metrics["AUROC"] = float(roc_auc_score(y_true, y_prob))
    metrics["PR_AUC"] = float(average_precision_score(y_true, y_prob))
    metrics["Brier"] = float(brier_score_loss(y_true, y_prob))
    metrics["n_samples"] = int(len(y_true))
    metrics["n_pos"] = int(y_true.sum())
    metrics["prevalence"] = float(y_true.mean())

    return metrics


def validate_probabilities(
    y_prob: np.ndarray,
    split_name: str,
    logger: logging.Logger,
) -> None:
    """Validate predicted probabilities are in [0, 1] and contain no NaN/Inf.

    Args:
        y_prob: Array of predicted probabilities
        split_name: Name of the split (for error messages)
        logger: Logger instance

    Raises:
        ValueError: If probabilities are invalid (NaN, Inf, or out of bounds)
    """
    if not isinstance(y_prob, np.ndarray):
        y_prob = np.asarray(y_prob)

    # Check for NaN
    n_nan = np.isnan(y_prob).sum()
    if n_nan > 0:
        raise ValueError(
            f"Ensemble predictions on {split_name} contain {n_nan} NaN values. "
            "This indicates a meta-learner or calibration error."
        )

    # Check for Inf
    n_inf = np.isinf(y_prob).sum()
    if n_inf > 0:
        raise ValueError(
            f"Ensemble predictions on {split_name} contain {n_inf} Inf values. "
            "This indicates a meta-learner or calibration error."
        )

    # Check bounds [0, 1]
    if y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError(
            f"Ensemble predictions on {split_name} are out of bounds [0, 1]. "
            f"min={y_prob.min():.6f}, max={y_prob.max():.6f}. "
            "This indicates a calibration error."
        )

    logger.debug(
        f"Ensemble {split_name} probabilities validated: n={len(y_prob)}, "
        f"min={y_prob.min():.4f}, max={y_prob.max():.4f}, mean={y_prob.mean():.4f}"
    )


def run_train_ensemble(
    config_file: str | None = None,
    results_dir: str | None = None,
    base_models: list[str] | None = None,
    split_seed: int = 0,
    outdir: str | None = None,
    meta_penalty: str | None = None,
    meta_C: float | None = None,
    verbose: int = 0,
) -> dict[str, Any]:
    """Run ensemble training from base model outputs.

    This function:
    1. Loads configuration (from file or defaults)
    2. Collects OOF predictions from base models
    3. Trains the meta-learner
    4. Generates and saves ensemble predictions
    5. Computes and reports metrics

    Args:
        config_file: Path to YAML config file (optional)
        results_dir: Directory containing base model results
        base_models: List of base model names (overrides config)
        split_seed: Split seed for identifying model outputs
        outdir: Output directory (overrides results_dir/ENSEMBLE/split_{seed})
        meta_penalty: Meta-learner regularization (overrides config)
        meta_C: Meta-learner regularization strength (overrides config)
        verbose: Verbosity level

    Returns:
        Dict with ensemble results and metrics
    """
    # Setup logger
    log_level = 20 - (verbose * 10)
    logger = setup_logger("ced_ml.train_ensemble", level=log_level)

    log_section(logger, "CeD-ML Ensemble Training")

    # Load config if provided
    config = None
    if config_file:
        logger.info(f"Loading config from: {config_file}")
        config = load_training_config(config_file=config_file)

    # Determine results directory
    if results_dir is None and config is not None:
        results_dir = str(config.outdir)
    if results_dir is None:
        raise ValueError("Must provide --results-dir or config with outdir")

    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_path}")

    # Determine base models
    if base_models is None and config is not None:
        base_models = config.ensemble.base_models
    if base_models is None:
        base_models = ["LR_EN", "RF", "XGBoost", "LinSVM_cal"]

    logger.info(f"Results directory: {results_path}")
    logger.info(f"Base models: {base_models}")
    logger.info(f"Split seed: {split_seed}")

    # Determine meta-learner hyperparameters
    if meta_penalty is None:
        meta_penalty = config.ensemble.meta_model.penalty if config else "l2"
    if meta_C is None:
        meta_C = config.ensemble.meta_model.C if config else 1.0

    logger.info(f"Meta-learner: LogisticRegression(penalty={meta_penalty}, C={meta_C})")

    # Check which base models have results (using flexible path discovery)
    available_models = []
    missing_models = []
    for model in base_models:
        try:
            # Use _find_model_split_dir for flexible path resolution (H1 fix)
            # This handles both legacy (split_{seed}) and new (run_{id}/split_seed{seed}) layouts
            model_dir = _find_model_split_dir(results_path, model, split_seed)
            oof_path = model_dir / "preds" / "train_oof" / f"train_oof__{model}.csv"
            if oof_path.exists():
                available_models.append(model)
            else:
                missing_models.append(model)
        except FileNotFoundError:
            missing_models.append(model)

    if missing_models:
        logger.warning(f"Missing OOF predictions for: {missing_models}")

    if len(available_models) < 2:
        raise ValueError(
            f"Need at least 2 base models with OOF predictions. "
            f"Available: {available_models}, missing: {missing_models}"
        )

    logger.info(f"Using {len(available_models)} base models: {available_models}")

    # Load calibration info for base models
    log_section(logger, "Loading Calibration Info")
    calibration_info = load_calibration_info_for_models(results_path, available_models, split_seed)

    # Log calibration strategies
    for model_name, calib_info in calibration_info.items():
        strategy = calib_info.strategy
        has_oof_calib = calib_info.oof_calibrator is not None
        logger.info(
            f"  {model_name}: strategy={strategy}, "
            f"needs_posthoc={calib_info.needs_posthoc_calibration}, "
            f"has_oof_calibrator={has_oof_calib}"
        )

    # Collect OOF predictions
    log_section(logger, "Collecting OOF Predictions")
    oof_dict, y_train, train_idx = collect_oof_predictions(
        results_path, available_models, split_seed
    )

    logger.info(f"Training samples: {len(y_train)}")
    logger.info(f"Training prevalence: {y_train.mean():.4f}")

    # Train ensemble
    log_section(logger, "Training Meta-Learner")

    random_state = config.cv.random_state if config else 42
    calibrate_meta = True  # Always calibrate for probability estimates

    ensemble = StackingEnsemble(
        base_model_names=available_models,
        meta_penalty=meta_penalty,
        meta_C=meta_C,
        calibrate_meta=calibrate_meta,
        random_state=random_state,
    )
    ensemble.fit_from_oof(oof_dict, y_train)

    # Log meta-model coefficients
    coef = ensemble.get_meta_model_coef()
    if coef:
        logger.info("Meta-learner coefficients:")
        for name, value in coef.items():
            logger.info(f"  {name}: {value:.4f}")

    # Generate predictions on val/test sets
    log_section(logger, "Generating Ensemble Predictions")

    results = {
        "base_models": available_models,
        "split_seed": split_seed,
        "meta_penalty": meta_penalty,
        "meta_C": meta_C,
        "meta_coef": coef,
        "random_state": random_state,
        "calibration_strategies": {name: info.strategy for name, info in calibration_info.items()},
    }

    # Validation set (apply calibration to base model predictions)
    try:
        val_preds_dict, y_val, val_idx = collect_split_predictions(
            results_path,
            available_models,
            split_seed,
            "val",
            calibration_info=calibration_info,
        )
        val_proba = ensemble.predict_proba_from_base_preds(val_preds_dict)[:, 1]
        # Validate ensemble val predictions (H4 fix)
        validate_probabilities(val_proba, "val", logger)
        results["val_proba"] = val_proba
        results["y_val"] = y_val
        results["val_idx"] = val_idx

        val_metrics = compute_ensemble_metrics(y_val, val_proba, "val")
        results["val_metrics"] = val_metrics
        logger.info(f"Validation AUROC: {val_metrics.get('AUROC', 'N/A'):.4f}")
        logger.info(f"Validation PR-AUC: {val_metrics.get('PR_AUC', 'N/A'):.4f}")
    except FileNotFoundError as e:
        logger.warning(f"Could not load validation predictions: {e}")

    # Test set (apply calibration to base model predictions)
    try:
        test_preds_dict, y_test, test_idx = collect_split_predictions(
            results_path,
            available_models,
            split_seed,
            "test",
            calibration_info=calibration_info,
        )
        test_proba = ensemble.predict_proba_from_base_preds(test_preds_dict)[:, 1]
        # Validate ensemble test predictions (H4 fix)
        validate_probabilities(test_proba, "test", logger)
        results["test_proba"] = test_proba
        results["y_test"] = y_test
        results["test_idx"] = test_idx

        test_metrics = compute_ensemble_metrics(y_test, test_proba, "test")
        results["test_metrics"] = test_metrics
        logger.info(f"Test AUROC: {test_metrics.get('AUROC', 'N/A'):.4f}")
        logger.info(f"Test PR-AUC: {test_metrics.get('PR_AUC', 'N/A'):.4f}")
    except FileNotFoundError as e:
        logger.warning(f"Could not load test predictions: {e}")

    # Save results
    log_section(logger, "Saving Ensemble Results")

    if outdir is None:
        outdir = results_path / "ENSEMBLE" / f"split_{split_seed}"
    else:
        outdir = Path(outdir)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Create output subdirectories matching standard structure
    core_dir = outdir / "core"
    preds_dir = outdir / "preds"
    preds_test_dir = preds_dir / "test_preds"
    preds_val_dir = preds_dir / "val_preds"
    preds_train_oof_dir = preds_dir / "train_oof"
    core_dir.mkdir(exist_ok=True)
    preds_dir.mkdir(exist_ok=True)
    preds_test_dir.mkdir(parents=True, exist_ok=True)
    preds_val_dir.mkdir(parents=True, exist_ok=True)
    preds_train_oof_dir.mkdir(parents=True, exist_ok=True)

    # Save ensemble model bundle
    import joblib
    import sklearn

    model_bundle = {
        "model": ensemble,
        "model_name": "ENSEMBLE",
        "base_models": available_models,
        "meta_penalty": meta_penalty,
        "meta_C": meta_C,
        "meta_coef": coef,
        "split_seed": split_seed,
        "random_state": random_state,
        "base_calibration_strategies": results.get("calibration_strategies", {}),
        "versions": {
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    model_path = core_dir / "ENSEMBLE__final_model.joblib"
    joblib.dump(model_bundle, model_path)
    logger.info(f"Ensemble model saved: {model_path}")

    # Save predictions
    if "val_proba" in results:
        val_df = pd.DataFrame(
            {
                "idx": results["val_idx"],
                "y_true": results["y_val"],
                "y_prob": results["val_proba"],
            }
        )
        val_path = preds_val_dir / "val_preds__ENSEMBLE.csv"
        val_df.to_csv(val_path, index=False)
        logger.info(f"Validation predictions saved: {val_path}")

    if "test_proba" in results:
        test_df = pd.DataFrame(
            {
                "idx": results["test_idx"],
                "y_true": results["y_test"],
                "y_prob": results["test_proba"],
            }
        )
        test_path = preds_test_dir / "test_preds__ENSEMBLE.csv"
        test_df.to_csv(test_path, index=False)
        logger.info(f"Test predictions saved: {test_path}")

    # Save OOF predictions (aggregated meta-features used for training)
    oof_meta = ensemble._build_meta_features(oof_dict, aggregate_repeats=True)
    oof_df = pd.DataFrame(oof_meta, columns=[f"oof_{m}" for m in available_models])
    oof_df["idx"] = train_idx
    oof_df["y_true"] = y_train
    oof_path = preds_train_oof_dir / "train_oof__ENSEMBLE.csv"
    oof_df.to_csv(oof_path, index=False)
    logger.info(f"OOF predictions saved: {oof_path}")

    # Save metrics
    metrics_summary = {
        "model": "ENSEMBLE",
        "base_models": available_models,
        "split_seed": split_seed,
        "meta_penalty": meta_penalty,
        "meta_C": meta_C,
        "timestamp": datetime.now().isoformat(),
    }
    if "val_metrics" in results:
        metrics_summary["val"] = results["val_metrics"]
    if "test_metrics" in results:
        metrics_summary["test"] = results["test_metrics"]

    metrics_path = core_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info(f"Metrics saved: {metrics_path}")

    # Save run settings
    run_settings = {
        "model": "ENSEMBLE",
        "base_models": available_models,
        "split_seed": split_seed,
        "meta_penalty": meta_penalty,
        "meta_C": meta_C,
        "meta_coef": coef,
        "n_train": len(y_train),
        "train_prevalence": float(y_train.mean()),
        "random_state": random_state,
    }
    settings_path = core_dir / "run_settings.json"
    with open(settings_path, "w") as f:
        json.dump(run_settings, f, indent=2)
    logger.info(f"Run settings saved: {settings_path}")

    log_section(logger, "Ensemble Training Complete")
    logger.info(f"All results saved to: {outdir}")

    return results
