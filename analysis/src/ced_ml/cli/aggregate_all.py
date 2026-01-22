"""
CLI implementation for aggregate-all command.

Scans a results directory tree for completed runs and aggregates them.
"""

import json
import logging
from pathlib import Path

from ced_ml.cli.aggregate_splits import discover_split_dirs, run_aggregate_splits
from ced_ml.utils.logging import setup_logger


def _verbose_to_level(verbose: int) -> int:
    """Convert verbose count to logging level."""
    if verbose == 0:
        return logging.WARNING
    elif verbose == 1:
        return logging.INFO
    else:
        return logging.DEBUG


def discover_runs(results_root: Path) -> list[dict]:
    """
    Discover all model/run directories in the results tree.

    Expected structure:
        results_root/
            ModelA/
                run_YYYYMMDD_HHMMSS/
                    split_seed0/
                    split_seed1/
                    ...
            ModelB/
                run_YYYYMMDD_HHMMSS/
                    ...

    Args:
        results_root: Root results directory

    Returns:
        List of dicts with keys: model, run_id, run_dir, split_dirs
    """
    runs = []

    for model_dir in sorted(results_root.iterdir()):
        if not model_dir.is_dir():
            continue
        # Skip aggregated directories and other non-model dirs
        if model_dir.name in ("aggregated", "logs", ".git"):
            continue

        for run_dir in sorted(model_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            # Match run_* pattern
            if not run_dir.name.startswith("run_"):
                continue

            split_dirs = discover_split_dirs(run_dir)
            if split_dirs:
                runs.append(
                    {
                        "model": model_dir.name,
                        "run_id": run_dir.name,
                        "run_dir": run_dir,
                        "split_dirs": split_dirs,
                    }
                )

    return runs


def check_run_completion(run_info: dict) -> dict:
    """
    Check if a run is complete (all splits have test_metrics.csv).

    Args:
        run_info: Dict with run_dir and split_dirs

    Returns:
        Updated dict with completion status
    """
    run_dir = run_info["run_dir"]
    split_dirs = run_info["split_dirs"]

    # Check each split for completion marker
    completed_splits = []
    for split_dir in split_dirs:
        metrics_file = split_dir / "core" / "test_metrics.csv"
        if metrics_file.exists():
            completed_splits.append(split_dir)

    # Check if already aggregated
    aggregated_dir = run_dir / "aggregated"
    already_aggregated = (
        aggregated_dir.exists() and (aggregated_dir / "core" / "pooled_metrics.csv").exists()
    )

    # Try to get expected split count from metadata
    expected_splits = None
    metadata_file = run_dir / "run_metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file) as f:
                metadata = json.load(f)
                expected_splits = metadata.get("n_splits")
        except (json.JSONDecodeError, KeyError):
            pass

    # If no metadata, check if any split_seed dirs exist
    if expected_splits is None:
        expected_splits = len(split_dirs) if split_dirs else 0

    return {
        **run_info,
        "completed_splits": len(completed_splits),
        "expected_splits": expected_splits,
        "is_complete": len(completed_splits) == expected_splits and expected_splits > 0,
        "already_aggregated": already_aggregated,
    }


def run_aggregate_all(
    results_root: str,
    force: bool = False,
    dry_run: bool = False,
    stability_threshold: float = 0.75,
    target_specificity: float = 0.95,
    n_boot: int = 500,
    plot_formats: list[str] | None = None,
    verbose: int = 0,
) -> dict:
    """
    Scan results directory and aggregate all completed runs.

    Args:
        results_root: Root results directory to scan
        force: Re-aggregate even if already aggregated
        dry_run: Show what would be aggregated without doing it
        stability_threshold: Feature stability threshold
        target_specificity: Target specificity for thresholds
        n_boot: Bootstrap iterations
        plot_formats: Output plot formats
        verbose: Verbosity level

    Returns:
        Summary dict with counts and details
    """
    logger = setup_logger("ced_ml.aggregate_all", level=_verbose_to_level(verbose))
    results_root = Path(results_root)

    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    if plot_formats is None:
        plot_formats = ["png"]

    logger.info("=" * 60)
    logger.info("Scanning for completed runs")
    logger.info("=" * 60)
    logger.info(f"Results root: {results_root}")

    # Discover all runs
    runs = discover_runs(results_root)
    logger.info(f"Found {len(runs)} run(s)")

    # Check completion status
    run_statuses = [check_run_completion(r) for r in runs]

    # Categorize
    complete_not_aggregated = [
        r for r in run_statuses if r["is_complete"] and not r["already_aggregated"]
    ]
    complete_already_aggregated = [
        r for r in run_statuses if r["is_complete"] and r["already_aggregated"]
    ]
    incomplete = [r for r in run_statuses if not r["is_complete"]]

    logger.info("")
    logger.info(f"Complete, needs aggregation: {len(complete_not_aggregated)}")
    logger.info(f"Complete, already aggregated: {len(complete_already_aggregated)}")
    logger.info(f"Incomplete: {len(incomplete)}")

    # Show incomplete runs
    if incomplete:
        logger.info("")
        logger.info("Incomplete runs:")
        for r in incomplete:
            logger.info(
                f"  {r['model']}/{r['run_id']}: "
                f"{r['completed_splits']}/{r['expected_splits']} splits"
            )

    # Determine what to aggregate
    to_aggregate = complete_not_aggregated.copy()
    if force:
        to_aggregate.extend(complete_already_aggregated)

    if not to_aggregate:
        logger.info("")
        logger.info("Nothing to aggregate.")
        if complete_already_aggregated and not force:
            logger.info("Use --force to re-aggregate already aggregated runs.")
        return {
            "scanned": len(runs),
            "aggregated": 0,
            "skipped_already_done": len(complete_already_aggregated),
            "skipped_incomplete": len(incomplete),
        }

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Aggregating {len(to_aggregate)} run(s)")
    logger.info("=" * 60)

    aggregated = []
    failed = []

    for run_info in to_aggregate:
        run_dir = run_info["run_dir"]
        model = run_info["model"]
        run_id = run_info["run_id"]

        logger.info("")
        logger.info(f"[{model}/{run_id}] {run_info['completed_splits']} splits")

        if dry_run:
            logger.info("  [DRY RUN] Would aggregate")
            aggregated.append(run_info)
            continue

        try:
            run_aggregate_splits(
                results_dir=str(run_dir),
                stability_threshold=stability_threshold,
                target_specificity=target_specificity,
                n_boot=n_boot,
                plot_formats=plot_formats,
                verbose=verbose,
            )
            logger.info("  [OK] Aggregated")
            aggregated.append(run_info)
        except Exception as e:
            logger.error(f"  [FAIL] {e}")
            failed.append({"run_info": run_info, "error": str(e)})

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"Scanned: {len(runs)} runs")
    logger.info(f"Aggregated: {len(aggregated)}")
    if failed:
        logger.info(f"Failed: {len(failed)}")
        for f in failed:
            logger.info(f"  {f['run_info']['model']}/{f['run_info']['run_id']}: {f['error']}")
    logger.info(f"Skipped (already done): {len(complete_already_aggregated) if not force else 0}")
    logger.info(f"Skipped (incomplete): {len(incomplete)}")

    return {
        "scanned": len(runs),
        "aggregated": len(aggregated),
        "failed": len(failed),
        "skipped_already_done": len(complete_already_aggregated) if not force else 0,
        "skipped_incomplete": len(incomplete),
        "details": {
            "aggregated": [{"model": r["model"], "run_id": r["run_id"]} for r in aggregated],
            "failed": [
                {
                    "model": f["run_info"]["model"],
                    "run_id": f["run_info"]["run_id"],
                    "error": f["error"],
                }
                for f in failed
            ],
            "incomplete": [
                {
                    "model": r["model"],
                    "run_id": r["run_id"],
                    "completed": r["completed_splits"],
                    "expected": r["expected_splits"],
                }
                for r in incomplete
            ],
        },
    }
