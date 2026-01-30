"""
Path utilities for standardized file/directory operations.

This module provides two sets of utilities:

1. Directory management (ensure_dir, get_run_dir, etc.)
2. Path resolution for CLI commands

**IMPORTANT**: All CLI commands must be run from the project root (CeliacRisks/).

Project structure:
    CeliacRisks/                    <- Project root (run ced here)
    ├── data/                       <- Input data
    ├── splits/                     <- Split indices
    ├── results/                    <- Model outputs
    ├── analysis/                   <- Analysis package
    │   ├── configs/               <- Configuration files
    │   ├── src/ced_ml/            <- Package source
    │   └── tests/                  <- Test suite

Path resolution:
    - Run from CeliacRisks/: paths like "data/" resolve correctly
    - Paths in configs are relative to analysis/ directory
"""

from pathlib import Path

# ============================================================================
# Directory Management Utilities (legacy compatibility)
# ============================================================================


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist, return Path object."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_run_dir(
    base_dir: str | Path,
    scenario: str,
    model: str,
    cv_config: str,
    val_config: str = "",
    test_config: str = "",
    suffix: str = "",
) -> Path:
    """
    Generate standardized run directory name matching current implementation.

    Format: {scenario}__{model}__{cv_config}__{val_config}__{test_config}{suffix}

    Example:
        RF__5x10__val0.25__test0.25__hybrid
    """
    parts = [scenario, model, cv_config]

    if val_config:
        parts.append(val_config)

    if test_config:
        parts.append(test_config)

    run_name = "__".join(parts)

    if suffix:
        run_name = f"{run_name}__{suffix}"

    run_dir = Path(base_dir) / run_name
    return run_dir


def get_core_dir(run_dir: str | Path) -> Path:
    """Get core results directory."""
    return ensure_dir(Path(run_dir) / "core")


def get_preds_dir(run_dir: str | Path) -> Path:
    """Get predictions directory."""
    return ensure_dir(Path(run_dir) / "preds")


def get_diagnostics_dir(run_dir: str | Path) -> Path:
    """Get diagnostics directory."""
    return ensure_dir(Path(run_dir) / "diagnostics")


def get_reports_dir(run_dir: str | Path) -> Path:
    """Get reports directory."""
    return ensure_dir(Path(run_dir) / "reports")


# ============================================================================
# Path Resolution (must run from project root)
# ============================================================================


def get_project_root() -> Path:
    """
    Get the project root directory (CeliacRisks/).

    **IMPORTANT**: This assumes you are running from the project root.
    If cwd is not the project root, raises an error.

    Returns:
        Path to project root (current working directory)

    Raises:
        RuntimeError: If current directory doesn't look like project root
    """
    cwd = Path.cwd()

    # Check for expected project structure
    has_data = (cwd / "data").is_dir()
    has_analysis = (cwd / "analysis").is_dir()

    if not (has_data and has_analysis):
        raise RuntimeError(
            f"Must run 'ced' commands from project root (CeliacRisks/).\n"
            f"Current directory: {cwd}\n"
            f"Expected structure: data/, analysis/, splits/, results/"
        )

    return cwd


def get_analysis_dir() -> Path:
    """Get the analysis/ directory path."""
    return get_project_root() / "analysis"


def get_default_paths() -> dict:
    """
    Get default paths for common directories.

    Returns:
        Dict with keys: project_root, analysis, data, splits, results, configs
    """
    root = get_project_root()
    return {
        "project_root": root,
        "analysis": root / "analysis",
        "data": root / "data",
        "splits": root / "splits",
        "results": root / "results",
        "configs": root / "analysis" / "configs",
        "logs": root / "logs",
    }
