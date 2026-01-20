"""
Configuration loading and merging logic.

Supports:
1. Loading from YAML files
2. CLI argument overrides (dot-notation: e.g., cv.folds=10)
3. Environment variable overrides
4. Validation and resolution
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import ValidationError

from ced_ml.config.defaults import (
    DEFAULT_CV_CONFIG,
    DEFAULT_DCA_CONFIG,
    DEFAULT_EVALUATION_CONFIG,
    DEFAULT_FEATURE_CONFIG,
    DEFAULT_OUTPUT_CONFIG,
    DEFAULT_PANEL_CONFIG,
    DEFAULT_SPLITS_CONFIG,
    DEFAULT_STRICTNESS_CONFIG,
    DEFAULT_THRESHOLD_CONFIG,
)
from ced_ml.config.schema import SplitsConfig, TrainingConfig


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    with open(file_path) as f:
        config_dict = yaml.safe_load(f)

    return config_dict or {}


def apply_overrides(config_dict: Dict[str, Any], overrides: list[str]) -> Dict[str, Any]:
    """
    Apply CLI overrides to config dictionary.

    Supports dot-notation for nested keys:
        cv.folds=10 -> config_dict['cv']['folds'] = 10
        features.screen_top_n=1000 -> config_dict['features']['screen_top_n'] = 1000

    Args:
        config_dict: Base configuration dictionary
        overrides: List of "key=value" or "nested.key=value" strings

    Returns:
        Updated config dictionary
    """
    # Keys that should always be lists
    LIST_KEYS = {"scenarios", "k_grid", "panel_sizes", "control_spec_targets", "toprisk_fracs"}

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format: {override}. Expected 'key=value'")

        key_path, value_str = override.split("=", 1)
        keys = key_path.split(".")

        # Navigate to the nested dict
        target = config_dict
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]

        # Parse value (try int, float, bool, then string)
        final_key = keys[-1]
        force_list = final_key in LIST_KEYS
        value = _parse_value(value_str, force_list=force_list)
        target[final_key] = value

    return config_dict


def _parse_value(value_str: str, force_list: bool = False) -> Any:
    """
    Parse string value to appropriate Python type.

    Args:
        value_str: String to parse
        force_list: If True, always return a list (for comma-separated or single values)
    """
    # Boolean
    if value_str.lower() in ("true", "yes", "1"):
        return [True] if force_list else True
    if value_str.lower() in ("false", "no", "0"):
        return [False] if force_list else False

    # None
    if value_str.lower() in ("none", "null"):
        return [None] if force_list else None

    # List (comma-separated) or forced list
    if "," in value_str or force_list:
        values = [v.strip() for v in value_str.split(",")]
        parsed = []
        for v in values:
            # Try int
            try:
                parsed.append(int(v))
                continue
            except ValueError:
                pass
            # Try float
            try:
                parsed.append(float(v))
                continue
            except ValueError:
                pass
            # String
            parsed.append(v)
        return parsed

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


def load_splits_config(
    config_file: Optional[Union[str, Path]] = None,
    overrides: Optional[list[str]] = None,
) -> SplitsConfig:
    """
    Load splits configuration from file and CLI overrides.

    Args:
        config_file: Path to YAML config file (optional)
        overrides: List of CLI overrides in "key=value" format (optional)

    Returns:
        Validated SplitsConfig instance
    """
    # Start with defaults
    config_dict = DEFAULT_SPLITS_CONFIG.copy()

    # Load from file if provided
    if config_file is not None:
        file_config = load_yaml(config_file)
        config_dict.update(file_config)

    # Apply CLI overrides
    if overrides:
        config_dict = apply_overrides(config_dict, overrides)

    # Validate and return
    try:
        return SplitsConfig(**config_dict)
    except ValidationError as e:
        raise ValueError(f"Invalid splits configuration:\n{e}") from e


def load_training_config(
    config_file: Optional[Union[str, Path]] = None,
    overrides: Optional[list[str]] = None,
) -> TrainingConfig:
    """
    Load training configuration from file and CLI overrides.

    Args:
        config_file: Path to YAML config file (optional)
        overrides: List of CLI overrides in "key=value" format (optional)

    Returns:
        Validated TrainingConfig instance
    """
    # Start with defaults
    config_dict = {
        "cv": DEFAULT_CV_CONFIG.copy(),
        "features": DEFAULT_FEATURE_CONFIG.copy(),
        "panels": DEFAULT_PANEL_CONFIG.copy(),
        "thresholds": DEFAULT_THRESHOLD_CONFIG.copy(),
        "evaluation": DEFAULT_EVALUATION_CONFIG.copy(),
        "dca": DEFAULT_DCA_CONFIG.copy(),
        "output": DEFAULT_OUTPUT_CONFIG.copy(),
        "strictness": DEFAULT_STRICTNESS_CONFIG.copy(),
    }

    # Load from file if provided
    if config_file is not None:
        file_config = load_yaml(config_file)

        # Deep merge nested dicts
        for key, value in file_config.items():
            if key in config_dict and isinstance(value, dict):
                config_dict[key].update(value)
            else:
                config_dict[key] = value

    # Apply CLI overrides
    if overrides:
        config_dict = apply_overrides(config_dict, overrides)

    # Validate and return
    try:
        return TrainingConfig(**config_dict)
    except ValidationError as e:
        raise ValueError(f"Invalid training configuration:\n{e}") from e


def save_config(config: Union[SplitsConfig, TrainingConfig], output_path: Union[str, Path]):
    """Save resolved configuration to YAML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict
    config_dict = config.model_dump()

    # Convert Path objects to strings for YAML serialization
    def convert_paths(d):
        for key, value in d.items():
            if isinstance(value, Path):
                d[key] = str(value)
            elif isinstance(value, dict):
                convert_paths(value)
        return d

    config_dict = convert_paths(config_dict)

    # Write YAML
    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def print_config_summary(config: Union[SplitsConfig, TrainingConfig], logger=None):
    """Print human-readable configuration summary."""
    lines = []
    lines.append("=" * 80)
    lines.append("Configuration Summary")
    lines.append("=" * 80)

    config_dict = config.model_dump()

    def format_dict(d, indent=0):
        result = []
        for key, value in d.items():
            if isinstance(value, dict):
                result.append(f"{'  ' * indent}{key}:")
                result.extend(format_dict(value, indent + 1))
            else:
                result.append(f"{'  ' * indent}{key}: {value}")
        return result

    lines.extend(format_dict(config_dict))
    lines.append("=" * 80)

    summary = "\n".join(lines)

    if logger:
        logger.info(summary)
    else:
        print(summary)
