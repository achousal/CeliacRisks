"""
Configuration management tools.

Commands:
- ced config migrate: Convert legacy CLI args to YAML config
- ced config validate: Validate and report on config files
- ced config diff: Compare two config files
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import yaml

from ced_ml.config.loader import load_yaml, load_splits_config, load_training_config, save_config
from ced_ml.config.schema import SplitsConfig, TrainingConfig
from ced_ml.config.validation import validate_config
from ced_ml.utils.logging import setup_logger


def _verbose_to_level(verbose: int) -> int:
    """Convert verbose count to logging level."""
    if verbose == 0:
        return logging.WARNING
    elif verbose == 1:
        return logging.INFO
    else:
        return logging.DEBUG


def migrate_legacy_args_to_yaml(
    legacy_args: List[str],
    command: str,
    output_file: Optional[Path] = None,
    verbose: int = 0,
) -> Path:
    """
    Convert legacy CLI arguments to YAML config file.

    Args:
        legacy_args: List of legacy CLI arguments (e.g., ['--val-size', '0.25', '--test-size', '0.25'])
        command: Command type ('save-splits' or 'train')
        output_file: Output YAML file path (optional)
        verbose: Verbosity level

    Returns:
        Path to generated YAML config file
    """
    logger = setup_logger("ced.config.migrate", level=_verbose_to_level(verbose))

    # Parse legacy args into dict
    config_dict = _parse_legacy_args(legacy_args, command)

    logger.info(f"Parsed {len(legacy_args)} legacy arguments into config")

    # Load and validate config
    if command == "save-splits":
        config = SplitsConfig(**config_dict)
    elif command == "train":
        config = TrainingConfig(**config_dict)
    else:
        raise ValueError(f"Unknown command: {command}. Expected 'save-splits' or 'train'")

    # Determine output file
    if output_file is None:
        output_file = Path(f"{command.replace('-', '_')}_config.yaml")

    # Save config
    save_config(config, output_file)

    logger.info(f"Saved config to: {output_file}")

    return output_file


def _parse_legacy_args(args: List[str], command: str) -> Dict[str, Any]:
    """
    Parse legacy CLI arguments into config dictionary.

    Handles both --flag and --key value patterns.
    """
    config_dict = {}
    i = 0

    while i < len(args):
        arg = args[i]

        # Skip if not a flag
        if not arg.startswith("--"):
            i += 1
            continue

        # Remove '--' prefix and convert to underscore notation
        key = arg[2:].replace("-", "_")

        # Check if this is a boolean flag or has a value
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            # Has a value
            value_str = args[i + 1]
            value = _parse_value_smart(value_str)
            i += 2
        else:
            # Boolean flag
            value = True
            i += 1

        # Map to nested structure for training configs
        if command == "train":
            key = _map_to_nested_key(key)

        # Set value (handle dot-notation for nested keys)
        _set_nested_value(config_dict, key, value)

    return config_dict


def _parse_value_smart(value_str: str) -> Any:
    """Parse value string to appropriate type."""
    # Boolean
    if value_str.lower() in ("true", "yes"):
        return True
    if value_str.lower() in ("false", "no"):
        return False

    # None
    if value_str.lower() in ("none", "null"):
        return None

    # List (comma-separated)
    if "," in value_str:
        return [_parse_value_smart(v.strip()) for v in value_str.split(",")]

    # Integer
    try:
        return int(value_str)
    except ValueError:
        pass

    # Float
    try:
        return float(value_str)
    except ValueError:
        pass

    # String
    return value_str


def _map_to_nested_key(key: str) -> str:
    """
    Map flat legacy keys to nested config keys for training.

    Examples:
        folds -> cv.folds
        scoring -> cv.scoring
        feature_select -> features.feature_select
        screen_top_n -> features.screen_top_n
    """
    # CV keys
    cv_keys = {
        "folds", "repeats", "inner_folds", "scoring", "n_iter", "random_state",
        "tune_n_jobs", "error_score", "grid_randomize"
    }

    # Feature keys
    feature_keys = {
        "feature_select", "screen_method", "screen_top_n", "kbest_scope",
        "kbest_max", "k_grid", "stability_thresh", "stable_corr_thresh",
        "stable_corr_method", "stable_summary", "kbest_n_jobs", "screening_n_jobs",
        "screen_equal_var", "screen_filter_mode"
    }

    # Panel keys
    panel_keys = {
        "build_panels", "panel_sizes", "panel_corr_thresh", "panel_corr_method",
        "panel_rep_tiebreak", "panel_refit", "panel_stability_mode",
        "panel_min_stability", "panel_use_all_sizes"
    }

    # Threshold keys
    threshold_keys = {
        "threshold_objective", "fbeta", "fixed_spec", "fixed_ppv",
        "threshold_source", "target_prevalence_source", "risk_prob_source",
        "prevalence_fixed"
    }

    # Evaluation keys
    evaluation_keys = {
        "test_ci_bootstrap", "n_boot", "learning_curve", "feature_reports",
        "control_spec_targets", "toprisk_fracs", "n_boot_lc", "lc_train_sizes",
        "lc_cv", "lc_scoring", "lc_n_jobs", "feature_report_max"
    }

    # DCA keys
    dca_keys = {
        "compute_dca", "dca_threshold_min", "dca_threshold_max",
        "dca_threshold_step", "dca_report_points"
    }

    # Output keys
    output_keys = {
        "save_train_preds", "save_val_preds", "save_test_preds",
        "save_calibration", "calib_bins", "save_oof_splits",
        "save_inner_model", "save_search_results"
    }

    # Strictness keys
    strictness_keys = {
        "strictness_level", "check_split_overlap", "check_prevalent_in_eval",
        "check_threshold_source", "check_feature_leakage", "check_missing_handling"
    }

    # Map to nested structure
    if key in cv_keys:
        return f"cv.{key}"
    elif key in feature_keys:
        return f"features.{key}"
    elif key in panel_keys:
        return f"panels.{key}"
    elif key in threshold_keys:
        # Handle special renaming
        if key == "threshold_objective":
            return "thresholds.objective"
        return f"thresholds.{key}"
    elif key in evaluation_keys:
        return f"evaluation.{key}"
    elif key in dca_keys:
        return f"dca.{key}"
    elif key in output_keys:
        return f"output.{key}"
    elif key in strictness_keys:
        # Handle special renaming
        if key == "strictness_level":
            return "strictness.level"
        return f"strictness.{key}"
    else:
        # Top-level key (model, scenario, infile, etc.)
        return key


def _set_nested_value(d: Dict[str, Any], key_path: str, value: Any):
    """Set value in nested dictionary using dot-notation key path."""
    keys = key_path.split(".")

    # Navigate to nested dict
    target = d
    for key in keys[:-1]:
        if key not in target:
            target[key] = {}
        target = target[key]

    # Set value
    target[keys[-1]] = value


def validate_config_file(
    config_file: Path,
    command: str,
    strict: bool = False,
    verbose: int = 0,
) -> Tuple[bool, List[str], List[str]]:
    """
    Validate configuration file and return diagnostic report.

    Args:
        config_file: Path to YAML config file
        command: Command type ('save-splits' or 'train')
        strict: If True, treat warnings as errors
        verbose: Verbosity level

    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    logger = setup_logger("ced.config.validate", level=_verbose_to_level(verbose))

    errors = []
    warnings = []

    # Load config
    try:
        config_dict = load_yaml(config_file)
    except Exception as e:
        errors.append(f"Failed to load YAML: {e}")
        return False, errors, warnings

    # Validate schema
    try:
        if command == "save-splits":
            config = load_splits_config(config_file)
        elif command == "train":
            config = load_training_config(config_file)
        else:
            errors.append(f"Unknown command: {command}")
            return False, errors, warnings
    except Exception as e:
        errors.append(f"Schema validation failed: {e}")
        return False, errors, warnings

    # Run validation checks
    validation_errors, validation_warnings = validate_config(config)
    errors.extend(validation_errors)
    warnings.extend(validation_warnings)

    # Determine validity
    is_valid = len(errors) == 0 and (not strict or len(warnings) == 0)

    logger.info(f"Validation complete: {len(errors)} errors, {len(warnings)} warnings")

    return is_valid, errors, warnings


def diff_configs(
    config_file1: Path,
    config_file2: Path,
    output_file: Optional[Path] = None,
    verbose: int = 0,
) -> Dict[str, Any]:
    """
    Compare two config files and report differences.

    Args:
        config_file1: First config file
        config_file2: Second config file
        output_file: Optional output file for diff report
        verbose: Verbosity level

    Returns:
        Dictionary with diff results
    """
    logger = setup_logger("ced.config.diff", level=_verbose_to_level(verbose))

    # Load both configs
    config1 = load_yaml(config_file1)
    config2 = load_yaml(config_file2)

    # Compute differences
    diff_result = {
        "only_in_first": {},
        "only_in_second": {},
        "different_values": {},
        "same": {},
    }

    _diff_dicts(config1, config2, diff_result)

    # Generate report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(f"Config Diff: {config_file1.name} vs {config_file2.name}")
    report_lines.append("=" * 80)

    if diff_result["only_in_first"]:
        report_lines.append(f"\nOnly in {config_file1.name}:")
        for key, value in diff_result["only_in_first"].items():
            report_lines.append(f"  {key}: {value}")

    if diff_result["only_in_second"]:
        report_lines.append(f"\nOnly in {config_file2.name}:")
        for key, value in diff_result["only_in_second"].items():
            report_lines.append(f"  {key}: {value}")

    if diff_result["different_values"]:
        report_lines.append("\nDifferent values:")
        for key, (val1, val2) in diff_result["different_values"].items():
            report_lines.append(f"  {key}:")
            report_lines.append(f"    {config_file1.name}: {val1}")
            report_lines.append(f"    {config_file2.name}: {val2}")

    report_lines.append(f"\nTotal differences: {len(diff_result['only_in_first']) + len(diff_result['only_in_second']) + len(diff_result['different_values'])}")
    report_lines.append("=" * 80)

    report = "\n".join(report_lines)

    # Print report
    logger.info(report)

    # Save report if requested
    if output_file:
        output_file.write_text(report)
        logger.info(f"Diff report saved to: {output_file}")

    return diff_result


def _diff_dicts(
    d1: Dict[str, Any],
    d2: Dict[str, Any],
    result: Dict[str, Any],
    prefix: str = "",
):
    """Recursively compare two dictionaries."""
    all_keys = set(d1.keys()) | set(d2.keys())

    for key in all_keys:
        full_key = f"{prefix}{key}" if prefix else key

        if key not in d1:
            result["only_in_second"][full_key] = d2[key]
        elif key not in d2:
            result["only_in_first"][full_key] = d1[key]
        elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
            # Recurse into nested dicts
            _diff_dicts(d1[key], d2[key], result, prefix=f"{full_key}.")
        elif d1[key] != d2[key]:
            result["different_values"][full_key] = (d1[key], d2[key])
        else:
            result["same"][full_key] = d1[key]


def run_config_migrate(
    input_file: Optional[Path] = None,
    args: Optional[List[str]] = None,
    command: str = "save-splits",
    output_file: Optional[Path] = None,
    verbose: int = 0,
):
    """
    Run config migration command.

    Args:
        input_file: File containing legacy CLI args (one per line)
        args: List of legacy args directly
        command: Command type
        output_file: Output YAML file
        verbose: Verbosity level
    """
    logger = setup_logger("ced.config.migrate", level=_verbose_to_level(verbose))

    # Get args from file or direct input
    if input_file:
        with open(input_file) as f:
            legacy_args = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    elif args:
        legacy_args = args
    else:
        raise ValueError("Must provide either input_file or args")

    logger.info(f"Migrating {len(legacy_args)} legacy arguments to YAML")

    # Migrate
    output_path = migrate_legacy_args_to_yaml(
        legacy_args=legacy_args,
        command=command,
        output_file=output_file,
        verbose=verbose,
    )

    logger.info(f"Migration complete: {output_path}")


def run_config_validate(
    config_file: Path,
    command: str = "train",
    strict: bool = False,
    verbose: int = 0,
):
    """
    Run config validation command.

    Args:
        config_file: Path to config file
        command: Command type
        strict: Treat warnings as errors
        verbose: Verbosity level
    """
    logger = setup_logger("ced.config.validate", level=_verbose_to_level(verbose))

    logger.info(f"Validating config: {config_file}")

    is_valid, errors, warnings = validate_config_file(
        config_file=config_file,
        command=command,
        strict=strict,
        verbose=verbose,
    )

    # Print results
    print("\n" + "=" * 80)
    print(f"Validation Report: {config_file.name}")
    print("=" * 80)

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for err in errors:
            print(f"  - {err}")

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for warn in warnings:
            print(f"  - {warn}")

    if is_valid:
        print(f"\n[OK] Config is valid")
    else:
        print(f"\n[FAIL] Config is invalid")
        if strict:
            print("  (strict mode: warnings treated as errors)")

    print("=" * 80)

    sys.exit(0 if is_valid else 1)


def run_config_diff(
    config_file1: Path,
    config_file2: Path,
    output_file: Optional[Path] = None,
    verbose: int = 0,
):
    """
    Run config diff command.

    Args:
        config_file1: First config file
        config_file2: Second config file
        output_file: Output file for diff report
        verbose: Verbosity level
    """
    logger = setup_logger("ced.config.diff", level=_verbose_to_level(verbose))

    diff_result = diff_configs(
        config_file1=config_file1,
        config_file2=config_file2,
        output_file=output_file,
        verbose=verbose,
    )

    # Exit with non-zero if differences found
    total_diffs = (
        len(diff_result["only_in_first"]) +
        len(diff_result["only_in_second"]) +
        len(diff_result["different_values"])
    )

    sys.exit(0 if total_diffs == 0 else 1)
