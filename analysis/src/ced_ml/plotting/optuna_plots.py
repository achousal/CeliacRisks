"""
Optuna hyperparameter optimization visualization utilities.

Provides functions to visualize Optuna study results, including:
- Optimization history (value over trials)
- Parameter importances
- Parallel coordinate plots
- Hyperparameter relationships
"""

import logging
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Attempt optuna import
try:
    import optuna

    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False
    optuna = None  # type: ignore[assignment]


def save_optuna_plots(
    study: Any,
    out_dir: Path,
    prefix: str = "",
    plot_format: str = "png",
) -> None:
    """
    Generate and save Optuna study visualization plots.

    Creates:
    - Optimization history plot (objective value over trials)
    - Parameter importance plot (if study has completed trials)
    - Parallel coordinate plot (parameter relationships)
    - Slice plot (parameter vs objective)

    Args:
        study: Optuna Study object
        out_dir: Output directory for plots
        prefix: Filename prefix (e.g., "RF__")
        plot_format: Output format ("png" or "pdf")

    Returns:
        None. Plots saved to out_dir.

    Note:
        Requires optuna[plotly] for interactive plots.
        Falls back gracefully if optuna or plotly not available.
    """
    if not _OPTUNA_AVAILABLE or study is None:
        logger.warning("Optuna not available or study is None, skipping optuna plots")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_trials = len(study.trials)
    if n_trials == 0:
        logger.warning("No trials in study, skipping optuna plots")
        return

    logger.info(f"Generating Optuna plots for {n_trials} trials")

    # Try plotly-based plots first (more interactive)
    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_parallel_coordinate,
            plot_param_importances,
            plot_slice,
        )

        # 1. Optimization history
        try:
            fig = plot_optimization_history(study)
            fig.update_layout(title=f"Optimization History ({n_trials} trials)")
            out_path = out_dir / f"{prefix}optuna_history.html"
            fig.write_html(str(out_path))
            logger.info(f"Saved optimization history plot: {out_path}")
        except Exception as e:
            logger.warning(f"Failed to create optimization history plot: {e}")

        # 2. Parameter importances (requires multiple trials)
        if n_trials >= 2:
            try:
                fig = plot_param_importances(study)
                fig.update_layout(title="Parameter Importances")
                out_path = out_dir / f"{prefix}optuna_importances.html"
                fig.write_html(str(out_path))
                logger.info(f"Saved parameter importances plot: {out_path}")
            except Exception as e:
                logger.warning(f"Failed to create parameter importances plot: {e}")

        # 3. Parallel coordinate plot
        if n_trials >= 2:
            try:
                fig = plot_parallel_coordinate(study)
                fig.update_layout(title="Parallel Coordinate Plot")
                out_path = out_dir / f"{prefix}optuna_parallel.html"
                fig.write_html(str(out_path))
                logger.info(f"Saved parallel coordinate plot: {out_path}")
            except Exception as e:
                logger.warning(f"Failed to create parallel coordinate plot: {e}")

        # 4. Slice plot (parameter vs objective)
        if n_trials >= 2:
            try:
                fig = plot_slice(study)
                fig.update_layout(title="Slice Plot (Parameter vs Objective)")
                out_path = out_dir / f"{prefix}optuna_slice.html"
                fig.write_html(str(out_path))
                logger.info(f"Saved slice plot: {out_path}")
            except Exception as e:
                logger.warning(f"Failed to create slice plot: {e}")

    except ImportError:
        logger.warning(
            "Optuna visualization not available (install optuna[plotly]). "
            "Saving trials dataframe only."
        )

    # Always save trials dataframe as CSV
    try:
        trials_df = study.trials_dataframe()
        csv_path = out_dir / f"{prefix}optuna_trials.csv"
        trials_df.to_csv(csv_path, index=False)
        logger.info(f"Saved trials dataframe: {csv_path}")
    except Exception as e:
        logger.warning(f"Failed to save trials dataframe: {e}")


def aggregate_optuna_trials(
    trials_dfs: List[pd.DataFrame],
    out_dir: Path,
    prefix: str = "aggregated_",
) -> Optional[pd.DataFrame]:
    """
    Aggregate Optuna trials across multiple splits.

    Combines trials from multiple runs into a single dataframe
    and computes summary statistics.

    Args:
        trials_dfs: List of trials DataFrames from different splits
        out_dir: Output directory
        prefix: Filename prefix

    Returns:
        Combined trials DataFrame with split_id column added

    Note:
        Adds 'split_id' column to track which split each trial came from.
    """
    if not trials_dfs:
        logger.warning("No trials dataframes to aggregate")
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Add split_id to each dataframe
    combined_dfs = []
    for split_id, df in enumerate(trials_dfs):
        df = df.copy()
        df["split_id"] = split_id
        combined_dfs.append(df)

    combined_df = pd.concat(combined_dfs, ignore_index=True)

    # Save combined trials
    csv_path = out_dir / f"{prefix}optuna_trials.csv"
    combined_df.to_csv(csv_path, index=False)
    logger.info(f"Saved aggregated trials ({len(combined_df)} trials): {csv_path}")

    # Compute summary statistics
    try:
        summary_stats = {
            "n_splits": len(trials_dfs),
            "total_trials": len(combined_df),
            "mean_best_value": combined_df.groupby("split_id")["value"].min().mean(),
            "std_best_value": combined_df.groupby("split_id")["value"].min().std(),
            "median_n_trials_per_split": len(combined_df) / len(trials_dfs),
        }

        # Save summary
        summary_path = out_dir / f"{prefix}optuna_summary.json"
        import json

        with open(summary_path, "w") as f:
            json.dump(summary_stats, f, indent=2)
        logger.info(f"Saved optuna summary: {summary_path}")

    except Exception as e:
        logger.warning(f"Failed to compute optuna summary stats: {e}")

    return combined_df
