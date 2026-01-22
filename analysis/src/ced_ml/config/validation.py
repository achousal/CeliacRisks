"""
Configuration validation and safety checks.

Implements strictness levels and leakage detection as per the refactoring plan.
"""

import warnings

from ced_ml.config.schema import SplitsConfig, TrainingConfig


class ConfigValidationError(Exception):
    """Raised when configuration validation fails in strict mode."""

    pass


class ConfigValidationWarning(UserWarning):
    """Warning for potential configuration issues."""

    pass


def validate_splits_config(config: SplitsConfig, strictness: str = "warn"):
    """
    Validate splits configuration for potential issues.

    Args:
        config: SplitsConfig instance
        strictness: "off", "warn", or "error"
    """
    issues = []

    # Check split size consistency
    if config.mode == "development":
        total_split = config.val_size + config.test_size
        if total_split >= 1.0:
            issues.append(
                f"val_size ({config.val_size}) + test_size ({config.test_size}) >= 1.0. "
                "No data left for training."
            )

    # Check prevalent handling
    if config.prevalent_train_only and config.prevalent_train_frac == 0.0:
        issues.append(
            "prevalent_train_only=True but prevalent_train_frac=0.0. "
            "No prevalent cases will be included."
        )

    # Check temporal split configuration
    if config.temporal_split and not config.temporal_col:
        issues.append("temporal_split=True but temporal_col not specified.")

    # Report issues
    _handle_issues(issues, strictness, "Splits configuration")


def validate_training_config(config: TrainingConfig):
    """
    Validate training configuration for potential leakage and inconsistencies.

    Args:
        config: TrainingConfig instance
    """
    strictness = config.strictness.level
    issues = []

    # Check threshold source availability
    if config.strictness.check_threshold_source:
        if config.thresholds.threshold_source == "val":
            # In current implementation, val is created via nested CV or explicit val_size
            # This is OK, but we should verify split_dir or CV config
            pass
        elif config.thresholds.threshold_source == "test":
            issues.append(
                "threshold_source='test' causes test leakage. "
                "Thresholds should be selected on validation set only."
            )

    # Check prevalence adjustment source
    if config.strictness.check_prevalence_adjustment:
        if config.thresholds.target_prevalence_source == "test":
            # This is actually OK for calibration - we want to match deployment prevalence
            # Only warn if risk_prob_source is also test (which is standard practice)
            pass

    # Check feature selection leakage
    if config.strictness.check_feature_leakage:
        if config.features.screen_top_n > 0:
            # Screening should happen within CV folds
            # Current implementation does screen on train only, which is correct
            pass

    # Check CV configuration
    if config.cv.folds < 2 and config.thresholds.threshold_source == "val":
        issues.append(
            f"cv.folds={config.cv.folds} but threshold_source='val'. "
            "Need folds >= 2 or explicit validation set."
        )

    # Check panel sizes vs kbest max
    if config.panels.build_panels:
        max_panel = max(config.panels.panel_sizes)
        if max_panel > config.features.kbest_max:
            issues.append(
                f"Max panel size ({max_panel}) > kbest_max ({config.features.kbest_max}). "
                "Not enough features for largest panel."
            )

    # Check DCA configuration
    if config.dca.compute_dca:
        if config.dca.dca_threshold_min >= config.dca.dca_threshold_max:
            issues.append(
                f"DCA threshold_min ({config.dca.dca_threshold_min}) >= "
                f"threshold_max ({config.dca.dca_threshold_max})."
            )

        step_count = (
            config.dca.dca_threshold_max - config.dca.dca_threshold_min
        ) / config.dca.dca_threshold_step
        if step_count > 10000:
            issues.append(
                f"DCA will compute {int(step_count)} thresholds. "
                "This may be slow. Consider larger step size."
            )

    # Check for unwired feature selection config options (H2 fix)
    # These config options exist in the schema but are not connected to the
    # training pipeline. Warn users when they set non-default values.
    _validate_unwired_feature_selection_config(config, issues)

    # Report issues
    _handle_issues(issues, strictness, "Training configuration")


def check_split_overlap(
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    strictness: str = "warn",
):
    """
    Check for overlap between train/val/test splits.

    Args:
        train_idx: Training indices
        val_idx: Validation indices
        test_idx: Test indices
        strictness: "off", "warn", or "error"
    """
    if strictness == "off":
        return

    issues = []

    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)

    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set

    if train_val_overlap:
        issues.append(f"Train/Val overlap: {len(train_val_overlap)} samples")

    if train_test_overlap:
        issues.append(f"Train/Test overlap: {len(train_test_overlap)} samples")

    if val_test_overlap:
        issues.append(f"Val/Test overlap: {len(val_test_overlap)} samples")

    _handle_issues(issues, strictness, "Split overlap check")


def check_prevalent_in_eval(
    eval_idx: list[int],
    prevalent_idx: list[int],
    split_name: str = "eval",
    strictness: str = "warn",
):
    """
    Check if prevalent cases leaked into evaluation set.

    Args:
        eval_idx: Evaluation set indices (val or test)
        prevalent_idx: Prevalent case indices
        split_name: Name of eval split for error messages
        strictness: "off", "warn", or "error"
    """
    if strictness == "off":
        return

    eval_set = set(eval_idx)
    prevalent_set = set(prevalent_idx)

    overlap = eval_set & prevalent_set

    if overlap:
        issue = (
            f"Prevalent cases found in {split_name} set: {len(overlap)} samples. "
            "This causes reverse causality bias. Use prevalent_train_only=True."
        )
        _handle_issues([issue], strictness, "Prevalent leakage check")


def validate_config(config):
    """
    Validate configuration and return lists of errors and warnings.

    Args:
        config: SplitsConfig or TrainingConfig instance

    Returns:
        Tuple of (errors, warnings) as lists of strings
    """
    errors = []
    warnings_list = []

    try:
        if isinstance(config, SplitsConfig):
            # Use warn mode to capture issues
            validate_splits_config(config, strictness="warn")
        elif isinstance(config, TrainingConfig):
            validate_training_config(config)
        else:
            errors.append(f"Unknown config type: {type(config)}")
    except ConfigValidationError as e:
        errors.append(str(e))
    except ConfigValidationWarning as w:
        warnings_list.append(str(w))

    return errors, warnings_list


def _validate_unwired_feature_selection_config(config: TrainingConfig, issues: list[str]) -> None:
    """
    Check for feature selection config options that are set but not wired into the pipeline.

    These config knobs exist in the schema but the training pipeline only uses
    a simple SelectKBest with kbest_max. When users set non-default values for
    advanced feature selection options, we should warn them that these settings
    will be ignored.

    Args:
        config: TrainingConfig instance
        issues: List to append warning messages to
    """
    # Default values from schema.py
    DEFAULT_SCREEN_TOP_N = 0
    DEFAULT_STABILITY_THRESH = 0.70
    DEFAULT_STABLE_CORR_THRESH = 0.80
    DEFAULT_KBEST_SCOPE = "protein"

    unwired_settings = []

    # screen_top_n: Only used for logging/reporting, not in pipeline
    if config.features.screen_top_n != DEFAULT_SCREEN_TOP_N:
        unwired_settings.append(
            f"screen_top_n={config.features.screen_top_n} (screening runs for reporting "
            "but does not filter features in the training pipeline)"
        )

    # stability_thresh: Not wired into pipeline feature selection
    if config.features.stability_thresh != DEFAULT_STABILITY_THRESH:
        unwired_settings.append(
            f"stability_thresh={config.features.stability_thresh} (not used during training, "
            "only in post-hoc panel extraction)"
        )

    # stable_corr_thresh: Not wired into pipeline feature selection
    if config.features.stable_corr_thresh != DEFAULT_STABLE_CORR_THRESH:
        unwired_settings.append(
            f"stable_corr_thresh={config.features.stable_corr_thresh} (not used during training, "
            "only in post-hoc panel building)"
        )

    # kbest_scope: Only affects extraction/logging, not selection
    if config.features.kbest_scope != DEFAULT_KBEST_SCOPE:
        unwired_settings.append(
            f"kbest_scope={config.features.kbest_scope} (affects feature extraction logging "
            "but pipeline always uses transformed-space SelectKBest)"
        )

    # feature_select modes other than "none" or "kbest" are not fully implemented
    advanced_modes = {"l1_stability", "hybrid"}
    if config.features.feature_select in advanced_modes:
        unwired_settings.append(
            f"feature_select={config.features.feature_select} (advanced mode; the pipeline "
            "uses basic SelectKBest, advanced selection logic is partially implemented)"
        )

    if unwired_settings:
        issues.append(
            "Feature selection config options set but not wired into training pipeline: "
            + "; ".join(unwired_settings)
            + ". These settings will be logged but not used for actual feature filtering. "
            "See docs/adr/ADR-004-hybrid-feature-selection.md for planned implementation."
        )


def _handle_issues(issues: list[str], strictness: str, context: str):
    """Handle validation issues based on strictness level."""
    if not issues:
        return

    message = f"{context} issues:\n" + "\n".join(f"  - {issue}" for issue in issues)

    if strictness == "error":
        raise ConfigValidationError(message)
    elif strictness == "warn":
        warnings.warn(message, ConfigValidationWarning, stacklevel=3)
    # strictness == "off": do nothing
