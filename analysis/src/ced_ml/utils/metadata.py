"""
Metadata building utilities for plot annotations.

Provides functions to create rich, reproducible metadata lines
for plot annotations with run configuration details.

Usage Guide:
    - build_plot_metadata: Main builder for test/val plots during training
    - build_oof_metadata: Specialized builder for out-of-fold plots during training
    - build_aggregated_metadata: Builder for plots aggregating multiple splits
    - build_holdout_metadata: Builder for holdout evaluation plots (currently unused)

"""

from datetime import datetime
from typing import List, Optional


def build_plot_metadata(
    model: str,
    scenario: str,
    seed: int,
    train_prev: float,
    target_prev: Optional[float] = None,
    cv_folds: Optional[int] = None,
    cv_repeats: Optional[int] = None,
    cv_scoring: Optional[str] = None,
    n_features: Optional[int] = None,
    feature_method: Optional[str] = None,
    n_train: Optional[int] = None,
    n_val: Optional[int] = None,
    n_test: Optional[int] = None,
    n_train_pos: Optional[int] = None,
    n_val_pos: Optional[int] = None,
    n_test_pos: Optional[int] = None,
    split_mode: Optional[str] = None,
    optuna_enabled: bool = False,
    n_trials: Optional[int] = None,
    n_iter: Optional[int] = None,
    threshold_objective: Optional[str] = None,
    prevalence_adjusted: bool = False,
    timestamp: bool = True,
    extra_lines: Optional[List[str]] = None,
) -> List[str]:
    """
    Build enriched metadata lines for plot annotations.

    Creates a structured set of metadata lines that capture key aspects
    of the model training run for reproducibility and context.

    Args:
        model: Model identifier (e.g., "LR_EN", "XGBoost")
        scenario: Scenario name (e.g., "IncidentOnly")
        seed: Random seed used
        train_prev: Training set prevalence
        target_prev: Target prevalence for calibration (optional)
        cv_folds: Number of CV folds (optional)
        cv_repeats: Number of CV repeats (optional)
        cv_scoring: CV scoring metric (optional)
        n_features: Number of features selected (optional)
        feature_method: Feature selection method (optional)
        n_train: Training set size (optional)
        n_val: Validation set size (optional)
        n_test: Test set size (optional)
        n_train_pos: Number of positive cases in training set (optional)
        n_val_pos: Number of positive cases in validation set (optional)
        n_test_pos: Number of positive cases in test set (optional)
        split_mode: Split mode ("development" or "holdout") (optional)
        optuna_enabled: Whether Optuna was used (default: False)
        n_trials: Number of Optuna trials (optional)
        n_iter: Number of RandomizedSearchCV iterations (optional)
        threshold_objective: Threshold selection objective (optional)
        prevalence_adjusted: Whether prevalence adjustment was applied (default: False)
        timestamp: Include generation timestamp (default: True)
        extra_lines: Additional custom metadata lines (optional)

    Returns:
        List of metadata strings suitable for plot annotation

    Example:
        >>> meta = build_plot_metadata(
        ...     model="LR_EN",
        ...     scenario="IncidentOnly",
        ...     seed=0,
        ...     train_prev=0.167,
        ...     target_prev=0.0034,
        ...     cv_folds=5,
        ...     cv_repeats=10,
        ...     n_train=1000,
        ...     n_train_pos=150,
        ...     n_features=200,
        ...     feature_method="hybrid"
        ... )
    """
    lines = []

    # Line 1: Core identifiers and split mode
    line1_parts = [f"Model: {model}", f"Scenario: {scenario}"]
    if split_mode:
        line1_parts.append(f"Split: {split_mode}")
    line1_parts.append(f"Seed: {seed}")
    lines.append(" | ".join(line1_parts))

    # Line 2: Sample sizes with positive counts
    size_parts = []
    if n_train is not None:
        if n_train_pos is not None:
            size_parts.append(f"Train: n={n_train} (pos={n_train_pos})")
        else:
            size_parts.append(f"Train: n={n_train}")

    if n_val is not None:
        if n_val_pos is not None:
            size_parts.append(f"Val: n={n_val} (pos={n_val_pos})")
        else:
            size_parts.append(f"Val: n={n_val}")

    if n_test is not None:
        if n_test_pos is not None:
            size_parts.append(f"Test: n={n_test} (pos={n_test_pos})")
        else:
            size_parts.append(f"Test: n={n_test}")

    if size_parts:
        lines.append(" | ".join(size_parts))

    # Line 3: CV configuration and scoring
    line3_parts = []
    if cv_folds and cv_repeats:
        cv_str = f"CV: {cv_folds}-fold x {cv_repeats} repeats"
        line3_parts.append(cv_str)

    if optuna_enabled and n_trials:
        line3_parts.append(f"Optuna: {n_trials} trials")
    elif cv_scoring:
        line3_parts.append(f"Scoring: {cv_scoring}")
        if n_iter:
            line3_parts.append(f"n_iter={n_iter}")

    if line3_parts:
        lines.append(" | ".join(line3_parts))

    # Line 4: Features and prevalence
    line4_parts = []
    if n_features is not None:
        feat_str = f"Features: {n_features}"
        if feature_method:
            feat_str += f" ({feature_method})"
        line4_parts.append(feat_str)

    # Prevalence info
    prev_parts = [f"train={train_prev:.3f}"]
    if target_prev is not None:
        prev_parts.append(f"target={target_prev:.3f}")
    line4_parts.append(f"Prev: {', '.join(prev_parts)}")

    if line4_parts:
        lines.append(" | ".join(line4_parts))

    # Line 5: Advanced settings (if present)
    line5_parts = []
    if threshold_objective:
        line5_parts.append(f"Threshold: {threshold_objective}")

    if prevalence_adjusted:
        line5_parts.append("Prevalence-adjusted")

    if line5_parts:
        lines.append(" | ".join(line5_parts))

    # Timestamp
    if timestamp:
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines.append(f"Generated: {timestamp_str}")

    # Extra custom lines
    if extra_lines:
        lines.extend(extra_lines)

    return lines


def build_oof_metadata(
    model: str,
    scenario: str,
    seed: int,
    cv_folds: int,
    cv_repeats: int,
    train_prev: float,
    n_train: Optional[int] = None,
    n_train_pos: Optional[int] = None,
    n_features: Optional[int] = None,
    feature_method: Optional[str] = None,
    cv_scoring: Optional[str] = None,
    extra_lines: Optional[List[str]] = None,
) -> List[str]:
    """
    Build metadata for out-of-fold (OOF) plots.

    Specialized metadata builder for OOF predictions across CV repeats.

    Args:
        model: Model identifier
        scenario: Scenario name
        seed: Random seed
        cv_folds: Number of CV folds
        cv_repeats: Number of CV repeats
        train_prev: Training set prevalence
        n_train: Training set size (optional)
        n_train_pos: Number of positive cases in training set (optional)
        n_features: Number of features selected (optional)
        feature_method: Feature selection method (optional)
        cv_scoring: CV scoring metric (optional)
        extra_lines: Additional metadata lines (optional)

    Returns:
        List of metadata strings
    """
    return build_plot_metadata(
        model=model,
        scenario=scenario,
        seed=seed,
        train_prev=train_prev,
        cv_folds=cv_folds,
        cv_repeats=cv_repeats,
        cv_scoring=cv_scoring,
        n_train=n_train,
        n_train_pos=n_train_pos,
        n_features=n_features,
        feature_method=feature_method,
        timestamp=True,
        extra_lines=extra_lines,
    )


def build_aggregated_metadata(
    n_splits: int,
    split_seeds: List[int],
    timestamp: bool = True,
) -> List[str]:
    """
    Build metadata for aggregated plots across multiple splits.

    Args:
        n_splits: Number of splits aggregated
        split_seeds: List of seed values used
        timestamp: Include generation timestamp (default: True)

    Returns:
        List of metadata strings
    """
    lines = [f"Pooled from {n_splits} splits (seeds: {split_seeds})"]

    if timestamp:
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines.append(f"Generated: {timestamp_str}")

    return lines


def build_holdout_metadata(
    model_name: str,
    n_holdout: int,
    holdout_prev: float,
    n_holdout_pos: Optional[int] = None,
    timestamp: bool = True,
) -> List[str]:
    """
    Build metadata for holdout evaluation plots.

    NOTE: Currently unused. The holdout evaluation module (evaluation.holdout)
    does not generate plots. This function is provided for future use if holdout
    plotting is implemented.

    Args:
        model_name: Model identifier
        n_holdout: Holdout set size
        holdout_prev: Holdout set prevalence
        n_holdout_pos: Number of positive cases in holdout set (optional)
        timestamp: Include generation timestamp (default: True)

    Returns:
        List of metadata strings
    """
    lines = [f"Model: {model_name} | Holdout Evaluation"]

    if n_holdout_pos is not None:
        lines.append(f"Holdout: n={n_holdout} (pos={n_holdout_pos}) | Prev: {holdout_prev:.3f}")
    else:
        lines.append(f"Holdout: n={n_holdout} | Prev: {holdout_prev:.3f}")

    if timestamp:
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines.append(f"Generated: {timestamp_str}")

    return lines
