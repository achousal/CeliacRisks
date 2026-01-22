#!/usr/bin/env python3
"""
Validate artifact isolation across multi-model training runs.

This script checks that:
1. Each model has its own directory structure
2. best_params_per_split.csv files contain only relevant hyperparameters
3. No cross-model contamination in metrics files
4. Row counts match expected values

Usage:
    python scripts/validate_artifact_isolation.py --results-dir ../results
    python scripts/validate_artifact_isolation.py --results-dir ../results --verbose
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

# Model-specific hyperparameters (used to detect cross-contamination)
MODEL_HYPERPARAMS: Dict[str, Set[str]] = {
    "LR_EN": {"C", "l1_ratio", "penalty", "solver", "max_iter"},
    "LR_L1": {"C", "penalty", "solver", "max_iter"},
    "RF": {"n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features"},
    "XGBoost": {"n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "gamma", "reg_alpha", "reg_lambda"},
    "LinSVM_cal": {"C", "class_weight", "kernel"},
}

# Expected row counts for standard 5-fold x 3-repeat CV
EXPECTED_ROWS = {
    "best_params_per_split.csv": 15,  # 5 folds x 3 repeats
    "cv_repeat_metrics.csv": 3,       # 3 repeats
    "selected_proteins_per_split.csv": 15,  # 5 folds x 3 repeats
}


def setup_logging(verbose: bool) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )
    return logging.getLogger(__name__)


def discover_model_dirs(results_dir: Path) -> Dict[str, List[Path]]:
    """
    Discover model directories and their split subdirectories.

    Returns:
        Dict mapping model name to list of split directories
    """
    model_dirs: Dict[str, List[Path]] = {}

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        if model_dir.name in ("aggregated", "logs", "__pycache__"):
            continue

        split_dirs = sorted(model_dir.glob("split_seed*"))
        if split_dirs:
            model_dirs[model_dir.name] = split_dirs

    return model_dirs


def check_best_params_isolation(
    model_name: str,
    split_dir: Path,
    logger: logging.Logger,
) -> Tuple[bool, List[str]]:
    """
    Check that best_params_per_split.csv contains only relevant hyperparameters.

    Returns:
        (is_valid, list of issues)
    """
    issues: List[str] = []
    best_params_path = split_dir / "cv" / "best_params_per_split.csv"

    if not best_params_path.exists():
        issues.append(f"Missing: {best_params_path}")
        return False, issues

    try:
        df = pd.read_csv(best_params_path)
    except Exception as e:
        issues.append(f"Failed to read {best_params_path}: {e}")
        return False, issues

    # Check row count
    expected_rows = EXPECTED_ROWS.get("best_params_per_split.csv", 15)
    if len(df) != expected_rows:
        issues.append(
            f"Row count mismatch in {best_params_path}: "
            f"expected {expected_rows}, got {len(df)}"
        )

    # Check for foreign hyperparameters
    file_cols = set(df.columns)
    expected_params = MODEL_HYPERPARAMS.get(model_name, set())

    for other_model, other_params in MODEL_HYPERPARAMS.items():
        if other_model == model_name:
            continue

        # Find params that are unique to other models (not shared)
        unique_to_other = other_params - expected_params
        foreign_found = file_cols & unique_to_other

        if foreign_found:
            issues.append(
                f"Foreign hyperparameters from {other_model} found in {model_name}: "
                f"{sorted(foreign_found)}"
            )

    return len(issues) == 0, issues


def check_metrics_isolation(
    model_name: str,
    split_dir: Path,
    logger: logging.Logger,
) -> Tuple[bool, List[str]]:
    """
    Check that metrics files contain only this model's data.

    Returns:
        (is_valid, list of issues)
    """
    issues: List[str] = []

    metrics_files = [
        ("core/test_metrics.csv", 1),
        ("core/val_metrics.csv", 1),
        ("cv/cv_repeat_metrics.csv", 3),
    ]

    for rel_path, expected_rows in metrics_files:
        metrics_path = split_dir / rel_path

        if not metrics_path.exists():
            logger.debug(f"Optional file missing: {metrics_path}")
            continue

        try:
            df = pd.read_csv(metrics_path)
        except Exception as e:
            issues.append(f"Failed to read {metrics_path}: {e}")
            continue

        # Check model column
        if "model" in df.columns:
            models_in_file = df["model"].unique().tolist()
            if len(models_in_file) > 1:
                issues.append(
                    f"Multiple models in {metrics_path}: {models_in_file}"
                )
            elif models_in_file and models_in_file[0] != model_name:
                issues.append(
                    f"Wrong model in {metrics_path}: "
                    f"expected {model_name}, got {models_in_file[0]}"
                )

        # Check row count (with tolerance for append mode)
        if len(df) < expected_rows:
            issues.append(
                f"Too few rows in {metrics_path}: "
                f"expected >={expected_rows}, got {len(df)}"
            )
        elif len(df) > expected_rows * 2:
            issues.append(
                f"Suspiciously many rows in {metrics_path}: "
                f"expected ~{expected_rows}, got {len(df)} (possible re-runs)"
            )

    return len(issues) == 0, issues


def check_predictions_isolation(
    model_name: str,
    split_dir: Path,
    logger: logging.Logger,
) -> Tuple[bool, List[str]]:
    """
    Check that prediction files have correct model prefix.

    Returns:
        (is_valid, list of issues)
    """
    issues: List[str] = []

    pred_dirs = [
        "preds/test_preds",
        "preds/val_preds",
        "preds/train_oof",
    ]

    for pred_rel in pred_dirs:
        pred_dir = split_dir / pred_rel

        if not pred_dir.exists():
            continue

        for csv_file in pred_dir.glob("*.csv"):
            # Check filename contains model name
            if model_name not in csv_file.name:
                issues.append(
                    f"Prediction file missing model prefix: {csv_file.name} "
                    f"(expected to contain '{model_name}')"
                )

    return len(issues) == 0, issues


def validate_results_dir(
    results_dir: Path,
    logger: logging.Logger,
) -> Tuple[int, int, List[str]]:
    """
    Validate all model directories for artifact isolation.

    Returns:
        (n_valid, n_invalid, all_issues)
    """
    model_dirs = discover_model_dirs(results_dir)

    if not model_dirs:
        logger.warning(f"No model directories found in {results_dir}")
        return 0, 0, ["No model directories found"]

    logger.info(f"Found {len(model_dirs)} model directories")

    all_issues: List[str] = []
    n_valid = 0
    n_invalid = 0

    for model_name, split_dirs in model_dirs.items():
        logger.info(f"\nValidating {model_name} ({len(split_dirs)} splits)...")

        model_issues: List[str] = []

        for split_dir in split_dirs:
            seed = split_dir.name

            # Check best_params isolation
            valid, issues = check_best_params_isolation(model_name, split_dir, logger)
            if not valid:
                for issue in issues:
                    model_issues.append(f"[{seed}] {issue}")

            # Check metrics isolation
            valid, issues = check_metrics_isolation(model_name, split_dir, logger)
            if not valid:
                for issue in issues:
                    model_issues.append(f"[{seed}] {issue}")

            # Check predictions isolation
            valid, issues = check_predictions_isolation(model_name, split_dir, logger)
            if not valid:
                for issue in issues:
                    model_issues.append(f"[{seed}] {issue}")

        if model_issues:
            logger.warning(f"  {len(model_issues)} issue(s) found")
            for issue in model_issues:
                logger.warning(f"    - {issue}")
            all_issues.extend([f"[{model_name}] {i}" for i in model_issues])
            n_invalid += 1
        else:
            logger.info(f"  OK - All checks passed")
            n_valid += 1

    return n_valid, n_invalid, all_issues


def main():
    parser = argparse.ArgumentParser(
        description="Validate artifact isolation in multi-model training results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="../results",
        help="Path to results directory (default: ../results)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()
    logger = setup_logging(args.verbose)

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        sys.exit(1)

    logger.info(f"Validating artifact isolation in: {results_dir}")

    n_valid, n_invalid, all_issues = validate_results_dir(results_dir, logger)

    if args.json:
        result = {
            "results_dir": str(results_dir),
            "n_valid_models": n_valid,
            "n_invalid_models": n_invalid,
            "total_issues": len(all_issues),
            "issues": all_issues,
            "status": "pass" if n_invalid == 0 else "fail",
        }
        print(json.dumps(result, indent=2))
    else:
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Valid models:   {n_valid}")
        logger.info(f"Invalid models: {n_invalid}")
        logger.info(f"Total issues:   {len(all_issues)}")

        if n_invalid == 0:
            logger.info("\nAll artifact isolation checks PASSED")
            sys.exit(0)
        else:
            logger.error("\nArtifact isolation checks FAILED")
            sys.exit(1)


if __name__ == "__main__":
    main()
