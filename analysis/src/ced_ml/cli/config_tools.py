"""
Configuration management tools.

Commands:
- ced config validate: Validate and report on config files
- ced config diff: Compare two config files
"""

import logging
import sys
from pathlib import Path
from typing import Any

from ced_ml.config.loader import (
    load_splits_config,
    load_training_config,
    load_yaml,
)
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


def validate_config_file(
    config_file: Path,
    command: str,
    strict: bool = False,
    verbose: int = 0,
) -> tuple[bool, list[str], list[str]]:
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
        load_yaml(config_file)
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
    output_file: Path | None = None,
    verbose: int = 0,
) -> dict[str, Any]:
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

    report_lines.append(
        f"\nTotal differences: {len(diff_result['only_in_first']) + len(diff_result['only_in_second']) + len(diff_result['different_values'])}"
    )
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
    d1: dict[str, Any],
    d2: dict[str, Any],
    result: dict[str, Any],
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
        print("\n[OK] Config is valid")
    else:
        print("\n[FAIL] Config is invalid")
        if strict:
            print("  (strict mode: warnings treated as errors)")

    print("=" * 80)

    sys.exit(0 if is_valid else 1)


def run_config_diff(
    config_file1: Path,
    config_file2: Path,
    output_file: Path | None = None,
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
    setup_logger("ced.config.diff", level=_verbose_to_level(verbose))

    diff_result = diff_configs(
        config_file1=config_file1,
        config_file2=config_file2,
        output_file=output_file,
        verbose=verbose,
    )

    # Exit with non-zero if differences found
    total_diffs = (
        len(diff_result["only_in_first"])
        + len(diff_result["only_in_second"])
        + len(diff_result["different_values"])
    )

    sys.exit(0 if total_diffs == 0 else 1)
