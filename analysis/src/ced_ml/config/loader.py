"""
Configuration loading and merging logic.

Supports:
1. Loading from YAML files
2. CLI argument overrides (dot-notation: e.g., cv.folds=10)
3. Environment variable overrides
4. Validation and resolution
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from ced_ml.config.defaults import (
    DEFAULT_CV_CONFIG,
    DEFAULT_DCA_CONFIG,
    DEFAULT_EVALUATION_CONFIG,
    DEFAULT_FEATURE_CONFIG,
    DEFAULT_OPTUNA_CONFIG,
    DEFAULT_OUTPUT_CONFIG,
    DEFAULT_PANEL_CONFIG,
    DEFAULT_SPLITS_CONFIG,
    DEFAULT_STRICTNESS_CONFIG,
    DEFAULT_THRESHOLD_CONFIG,
)
from ced_ml.config.schema import (
    AggregateConfig,
    HoldoutEvalConfig,
    PanelOptimizeConfig,
    SplitsConfig,
    TrainingConfig,
)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *overlay* into *base* (overlay wins on leaf conflicts).

    Returns a new dict; neither input is mutated.
    """
    merged = base.copy()
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(file_path: str | Path) -> dict[str, Any]:
    """Load configuration from YAML file.

    Supports a ``_base`` key: if present, the referenced YAML file is loaded
    first and the current file's values are deep-merged on top.  The ``_base``
    path is resolved relative to the directory containing *file_path*.
    Bases can be chained (a base may itself declare ``_base``).
    """
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    with open(file_path) as f:
        config_dict = yaml.safe_load(f) or {}

    base_ref = config_dict.pop("_base", None)
    if base_ref is not None:
        base_path = (file_path.parent / base_ref).resolve()
        base_dict = load_yaml(base_path)
        config_dict = _deep_merge(base_dict, config_dict)

    return config_dict


def resolve_paths_relative_to_config(
    config_dict: dict[str, Any], config_file: Path
) -> dict[str, Any]:
    """
    Resolve relative paths in config dict relative to config file directory.

    Args:
        config_dict: Configuration dictionary (may contain Path-like strings)
        config_file: Path to the config file

    Returns:
        Config dict with relative paths resolved

    Note:
        Only resolves paths that:
        - Are relative (not absolute)
        - Exist as keys containing "file", "dir", or "path" (case-insensitive)
        - Point to files/dirs that can be resolved relative to config dir
    """
    config_dir = config_file.resolve().parent
    resolved_dict = config_dict.copy()

    def resolve_value(value: Any) -> Any:
        """Recursively resolve Path-like values."""
        if isinstance(value, dict):
            return {k: resolve_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [resolve_value(v) for v in value]
        elif isinstance(value, str):
            # Check if this looks like a path (not absolute)
            path = Path(value)
            if not path.is_absolute() and len(path.parts) > 0:
                # Try resolving relative to config dir
                resolved = config_dir / path
                # Only replace if resolved path exists or value is clearly a path
                if resolved.exists() or "/" in value or "\\" in value:
                    return str(resolved)
        return value

    # Only resolve keys that look like paths
    path_like_keys = [
        "infile",
        "outdir",
        "split_dir",
        "results_dir",
        "model_artifact",
        "model_path",
        "holdout_idx",
    ]

    for key, val in resolved_dict.items():
        if key in path_like_keys:
            resolved_dict[key] = resolve_value(val)
        elif isinstance(val, dict):
            # Recursively resolve nested dicts
            for nested_key, nested_val in val.items():
                if nested_key in path_like_keys:
                    val[nested_key] = resolve_value(nested_val)

    return resolved_dict


def apply_overrides(config_dict: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
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
    LIST_KEYS = {
        "scenarios",
        "k_grid",
        "panel_sizes",
        "control_spec_targets",
        "toprisk_fracs",
    }

    # Keys that should always be strings (not parsed as int/float)
    STRING_KEYS = {
        "run_id",
        "run_name",
    }

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
        force_string = final_key in STRING_KEYS
        value = _parse_value(value_str, force_list=force_list, force_string=force_string)
        target[final_key] = value

    return config_dict


def _parse_value(value_str: str, force_list: bool = False, force_string: bool = False) -> Any:
    """
    Parse string value to appropriate Python type.

    Args:
        value_str: String to parse
        force_list: If True, always return a list (for comma-separated or single values)
        force_string: If True, always return a string (skip int/float parsing)
    """
    # Force string if requested
    if force_string:
        return value_str

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
    config_file: str | Path | None = None,
    overrides: list[str] | None = None,
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
        config_file_path = Path(config_file)
        file_config = load_yaml(config_file_path)

        # Resolve relative paths relative to config file directory
        file_config = resolve_paths_relative_to_config(file_config, config_file_path)

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
    config_file: str | Path | None = None,
    overrides: list[str] | None = None,
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
        "optuna": DEFAULT_OPTUNA_CONFIG.copy(),
    }

    # Load from file if provided
    if config_file is not None:
        config_file_path = Path(config_file)
        file_config = load_yaml(config_file_path)

        # Resolve relative paths relative to config file directory
        file_config = resolve_paths_relative_to_config(file_config, config_file_path)

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


def load_aggregate_config(
    config_file: str | Path | None = None,
    overrides: list[str] | None = None,
) -> AggregateConfig:
    """
    Load aggregate configuration from file and CLI overrides.

    Args:
        config_file: Path to YAML config file (optional)
        overrides: List of CLI overrides in "key=value" format (optional)

    Returns:
        Validated AggregateConfig instance
    """
    config_dict = {}

    if config_file is not None:
        config_file_path = Path(config_file)
        config_dict = load_yaml(config_file_path)

        # Resolve relative paths relative to config file directory
        config_dict = resolve_paths_relative_to_config(config_dict, config_file_path)

    if overrides:
        config_dict = apply_overrides(config_dict, overrides)

    try:
        return AggregateConfig(**config_dict)
    except ValidationError as e:
        raise ValueError(f"Invalid aggregate configuration:\n{e}") from e


def load_holdout_config(
    config_file: str | Path | None = None,
    overrides: list[str] | None = None,
) -> HoldoutEvalConfig:
    """
    Load holdout evaluation configuration from file and CLI overrides.

    Args:
        config_file: Path to YAML config file (optional)
        overrides: List of CLI overrides in "key=value" format (optional)

    Returns:
        Validated HoldoutEvalConfig instance
    """
    config_dict = {}

    if config_file is not None:
        config_file_path = Path(config_file)
        config_dict = load_yaml(config_file_path)

        # Resolve relative paths relative to config file directory
        config_dict = resolve_paths_relative_to_config(config_dict, config_file_path)

    if overrides:
        config_dict = apply_overrides(config_dict, overrides)

    try:
        return HoldoutEvalConfig(**config_dict)
    except ValidationError as e:
        raise ValueError(f"Invalid holdout configuration:\n{e}") from e


def load_panel_optimize_config(
    config_file: str | Path | None = None,
    cli_args: dict | None = None,
    overrides: list[str] | None = None,
) -> "PanelOptimizeConfig":
    """
    Load panel optimization configuration from file and CLI overrides.

    Auto-detection: if config_file is None, searches for:
    1. configs/optimize_panel.yaml
    2. configs/panel_optimize.yaml

    Args:
        config_file: Path to YAML config file (optional, auto-detects if None)
        cli_args: Dict of CLI arguments (infile, split_dir, model_path, etc.)
        overrides: List of CLI overrides in "key=value" format (optional)

    Returns:
        Validated PanelOptimizeConfig instance

    Raises:
        FileNotFoundError: If required config not found
        ValueError: If configuration is invalid
    """
    from ced_ml.config.schema import PanelOptimizeConfig

    config_dict = {}

    # Auto-detect config file if not provided
    if config_file is None:
        for candidate in ["configs/optimize_panel.yaml", "configs/panel_optimize.yaml"]:
            candidate_path = Path(candidate)
            if candidate_path.exists():
                config_file = candidate_path
                break

    # Load from file if found
    if config_file is not None:
        config_file_path = Path(config_file)
        if not config_file_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file_path}")
        file_config = load_yaml(config_file_path)
        config_dict = resolve_paths_relative_to_config(file_config, config_file_path)

    # Merge CLI arguments (they override config file)
    if cli_args:
        for key, value in cli_args.items():
            if value is not None:
                config_dict[key] = value

    # Apply CLI overrides
    if overrides:
        config_dict = apply_overrides(config_dict, overrides)

    # Validate and return
    try:
        return PanelOptimizeConfig(**config_dict)
    except Exception as e:
        raise ValueError(f"Invalid panel optimize configuration: {e}") from e


def save_config(config: SplitsConfig | TrainingConfig, output_path: str | Path):
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


def print_config_summary(config: SplitsConfig | TrainingConfig, logger=None):
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
