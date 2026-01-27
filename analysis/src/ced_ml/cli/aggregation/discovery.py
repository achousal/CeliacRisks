"""Directory discovery utilities for aggregating split results."""

import logging
from pathlib import Path


def discover_split_dirs(
    results_dir: Path,
    logger: logging.Logger | None = None,
) -> list[Path]:
    """
    Discover all split_seedX subdirectories in results_dir.

    Args:
        results_dir: Base results directory
        logger: Optional logger instance

    Returns:
        List of split subdirectory paths, sorted by seed number
    """
    # Filter directories first, then sort
    split_dirs = [d for d in results_dir.glob("split_seed*") if d.is_dir()]
    split_dirs = sorted(
        split_dirs,
        key=lambda p: int(p.name.replace("split_seed", "")),
    )
    if logger:
        logger.debug(f"Discovered {len(split_dirs)} split directories in {results_dir}")
    return split_dirs


def discover_ensemble_dirs(
    results_dir: Path,
    logger: logging.Logger | None = None,
) -> list[Path]:
    """
    Discover ENSEMBLE model split directories.

    The ensemble model outputs to a model-specific directory structure:
    results/ENSEMBLE/split_{seed}/ or results/ENSEMBLE/split_seed{seed}/

    Args:
        results_dir: Base results directory
        logger: Optional logger instance

    Returns:
        List of ensemble split subdirectory paths, sorted by seed number
    """
    ensemble_base = results_dir / "ENSEMBLE"
    if not ensemble_base.exists():
        if logger:
            logger.debug(f"No ENSEMBLE directory found at {ensemble_base}")
        return []

    # Try both naming conventions: split_{seed} and split_seed{seed}
    split_dirs = []

    for d in ensemble_base.glob("split_*"):
        if d.is_dir():
            name = d.name
            if name.startswith("split_seed"):
                # split_seed{X} format
                try:
                    seed = int(name.replace("split_seed", ""))
                    split_dirs.append((seed, d))
                except ValueError:
                    pass
            elif name.startswith("split_"):
                # split_{X} format
                try:
                    seed = int(name.replace("split_", ""))
                    split_dirs.append((seed, d))
                except ValueError:
                    pass

    # Sort by seed number
    split_dirs.sort(key=lambda x: x[0])
    dirs = [d for _, d in split_dirs]

    if logger:
        logger.debug(f"Discovered {len(dirs)} ENSEMBLE split directories")

    return dirs
