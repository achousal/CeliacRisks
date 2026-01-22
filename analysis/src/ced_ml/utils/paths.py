"""
Path utilities for standardized file/directory operations.
"""

from pathlib import Path


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
