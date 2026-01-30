"""Directory discovery utilities for aggregating split results."""

import logging
from pathlib import Path


def discover_split_dirs(
    results_dir: Path,
    logger: logging.Logger | None = None,
) -> list[Path]:
    """
    Discover all split_seedX subdirectories in results_dir.

    Directory structure:
    - results_dir/splits/split_seed*/

    Args:
        results_dir: Base results directory (typically MODEL/run_ID/)
        logger: Optional logger instance

    Returns:
        List of split subdirectory paths, sorted by seed number
    """
    splits_subdir = results_dir / "splits"
    if not splits_subdir.exists() or not splits_subdir.is_dir():
        if logger:
            logger.warning(f"Splits directory not found: {splits_subdir}")
        return []

    split_dirs = [
        split_dir for split_dir in splits_subdir.glob("split_seed*") if split_dir.is_dir()
    ]

    if logger:
        logger.debug(f"Found {len(split_dirs)} splits in {splits_subdir}")

    split_dirs = sorted(
        split_dirs,
        key=lambda p: int(p.name.replace("split_seed", "")),
    )

    return split_dirs


def discover_ensemble_dirs(
    results_dir: Path,
    logger: logging.Logger | None = None,
) -> list[Path]:
    """
    Discover ENSEMBLE model split directories.

    Directory layout: results_dir/ENSEMBLE/splits/split_seed*/
    where results_dir is the run-level directory (e.g., results/run_{RUN_ID}/).

    Args:
        results_dir: Run-level directory containing model subdirectories
        logger: Optional logger instance

    Returns:
        List of ensemble split subdirectory paths, sorted by seed number
    """
    ensemble_base = results_dir / "ENSEMBLE"
    if not ensemble_base.exists():
        if logger:
            logger.debug(f"No ENSEMBLE directory found at {ensemble_base}")
        return []

    splits_subdir = ensemble_base / "splits"
    if not splits_subdir.exists() or not splits_subdir.is_dir():
        if logger:
            logger.debug(f"No splits directory in ENSEMBLE: {splits_subdir}")
        return []

    split_dirs = []
    for d in splits_subdir.glob("split_seed*"):
        if d.is_dir():
            try:
                seed = int(d.name.replace("split_seed", ""))
                split_dirs.append((seed, d))
            except ValueError:
                pass

    split_dirs.sort(key=lambda x: x[0])
    dirs = [d for _, d in split_dirs]

    if logger:
        logger.debug(f"Discovered {len(dirs)} ENSEMBLE split directories")

    return dirs
