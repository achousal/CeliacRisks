"""CLI implementation for panel size optimization via RFE.

This module provides the `ced optimize-panel` command for finding minimum
viable protein panels through Recursive Feature Elimination.
"""

import logging
import os
from pathlib import Path

import joblib
import pandas as pd

from ced_ml.data.io import read_data_file
from ced_ml.data.persistence import load_split
from ced_ml.features.rfe import (
    RFEResult,
    recursive_feature_elimination,
    save_rfe_results,
)
from ced_ml.features.stability import compute_selection_frequencies, rank_proteins_by_frequency
from ced_ml.plotting.panel_curve import plot_feature_ranking, plot_pareto_curve


def run_optimize_panel(
    model_path: str,
    infile: str,
    split_dir: str,
    split_seed: int = 0,
    start_size: int = 100,
    min_size: int = 5,
    min_auroc_frac: float = 0.90,
    cv_folds: int = 5,
    step_strategy: str = "adaptive",
    outdir: str | None = None,
    use_stability_panel: bool = True,
    verbose: int = 0,
) -> RFEResult:
    """Run panel optimization via Recursive Feature Elimination.

    Loads a trained model, extracts the stability panel (or uses all features),
    and performs RFE to find the minimum viable panel maintaining acceptable AUROC.

    Args:
        model_path: Path to trained model bundle (.joblib).
        infile: Path to input data file (Parquet/CSV).
        split_dir: Directory containing split indices.
        split_seed: Split seed to use.
        start_size: Starting panel size (top N from stability ranking).
        min_size: Minimum panel size to evaluate.
        min_auroc_frac: Early stop if AUROC drops below this fraction of max.
        cv_folds: CV folds for OOF AUROC estimation.
        step_strategy: Elimination strategy ("adaptive", "linear", "geometric").
        outdir: Output directory (default: alongside model in optimize_panel/).
        use_stability_panel: If True, start from stability ranking; else all proteins.
        verbose: Verbosity level.

    Returns:
        RFEResult with curve, feature_ranking, and recommendations.

    Raises:
        FileNotFoundError: If model or data files not found.
        ValueError: If required data is missing from model bundle.
    """
    # Setup logging
    log_level = logging.WARNING
    if verbose >= 2:
        log_level = logging.DEBUG
    elif verbose >= 1:
        log_level = logging.INFO

    logging.basicConfig(level=log_level, format="%(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    logger.info(f"Loading model from {model_path}")

    # Load model bundle
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    bundle = joblib.load(model_path)

    if not isinstance(bundle, dict):
        raise ValueError("Model bundle must be a dictionary (not bare model)")

    model_name = bundle.get("model_name", "unknown")
    pipeline = bundle.get("model")
    resolved_cols = bundle.get("resolved_columns", {})
    config_dict = bundle.get("config", {})

    if pipeline is None:
        raise ValueError("Model bundle missing 'model' key")

    protein_cols = resolved_cols.get("protein_cols", [])
    cat_cols = resolved_cols.get("categorical_metadata", [])
    meta_num_cols = resolved_cols.get("numeric_metadata", [])

    if not protein_cols:
        raise ValueError("Model bundle missing protein_cols in resolved_columns")

    logger.info(f"Model: {model_name}, {len(protein_cols)} proteins available")

    # Load data
    logger.info(f"Loading data from {infile}")
    df = read_data_file(infile)

    # Load split
    split_path = Path(split_dir)
    scenario = bundle.get("scenario", "IncidentOnly")

    # Try to find split file
    split_patterns = [
        split_path / f"split_{split_seed}.pkl",
        split_path / f"splits_{split_seed}.pkl",
        split_path / scenario / f"split_{split_seed}.pkl",
    ]

    split_file = None
    for pattern in split_patterns:
        if pattern.exists():
            split_file = pattern
            break

    if split_file is None:
        raise FileNotFoundError(f"Split file not found. Tried: {[str(p) for p in split_patterns]}")

    logger.info(f"Loading split from {split_file}")
    split_data = load_split(str(split_file))

    train_idx = split_data.get("train_idx", split_data.get("train", []))
    val_idx = split_data.get("val_idx", split_data.get("val", []))

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Split file missing train_idx or val_idx")

    # Prepare data
    feature_cols = protein_cols + cat_cols + meta_num_cols

    # Check for missing columns
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning(f"Missing columns in data: {missing[:10]}...")
        feature_cols = [c for c in feature_cols if c in df.columns]
        protein_cols = [c for c in protein_cols if c in df.columns]

    # Get target column
    target_col = config_dict.get("columns", {}).get("target", "celiac_category")
    if target_col not in df.columns:
        # Try common alternatives
        for alt in ["celiac_category", "target", "label", "y"]:
            if alt in df.columns:
                target_col = alt
                break

    # Create binary target
    scenario_labels = config_dict.get("scenarios", {}).get(scenario, {}).get("labels", [1])
    if not scenario_labels:
        scenario_labels = [1]  # Default: 1 = case

    y_all = df[target_col].isin(scenario_labels).astype(int).values

    # Subset to train/val
    X_train = df.loc[train_idx, feature_cols].copy()
    y_train = y_all[train_idx]

    X_val = df.loc[val_idx, feature_cols].copy()
    y_val = y_all[val_idx]

    logger.info(f"Train: {len(X_train)} samples, {y_train.sum()} cases")
    logger.info(f"Val: {len(X_val)} samples, {y_val.sum()} cases")

    # Determine initial proteins
    if use_stability_panel:
        # Try to load stability panel from model directory
        model_dir = Path(model_path).parent.parent  # Go up from core/
        selected_proteins_path = model_dir / "cv" / "selected_proteins_per_split.csv"

        initial_proteins = None

        if selected_proteins_path.exists():
            logger.info(f"Loading stability panel from {selected_proteins_path}")
            sel_df = pd.read_csv(selected_proteins_path)

            # Compute selection frequencies
            freq = compute_selection_frequencies(sel_df, selection_col="selected_proteins_split")
            if not freq:
                freq = compute_selection_frequencies(sel_df, selection_col="selected_proteins")
            if not freq:
                freq = compute_selection_frequencies(
                    sel_df, selection_col="selected_proteins_final"
                )

            if freq:
                ranked = rank_proteins_by_frequency(freq)
                initial_proteins = ranked[:start_size]
                logger.info(f"Using top {len(initial_proteins)} proteins from stability ranking")

        if initial_proteins is None:
            logger.warning("Could not load stability panel, using all proteins")
            initial_proteins = protein_cols[:start_size]
    else:
        initial_proteins = protein_cols[:start_size]

    # Ensure initial proteins are in the data
    initial_proteins = [p for p in initial_proteins if p in X_train.columns]

    if len(initial_proteins) < min_size:
        raise ValueError(
            f"Only {len(initial_proteins)} valid proteins, less than min_size={min_size}"
        )

    logger.info(f"Starting RFE with {len(initial_proteins)} proteins")

    # Run RFE
    result = recursive_feature_elimination(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        base_pipeline=pipeline,
        model_name=model_name,
        initial_proteins=initial_proteins,
        cat_cols=[c for c in cat_cols if c in X_train.columns],
        meta_num_cols=[c for c in meta_num_cols if c in X_train.columns],
        min_size=min_size,
        cv_folds=cv_folds,
        step_strategy=step_strategy,
        min_auroc_frac=min_auroc_frac,
        random_state=split_seed,
    )

    # Determine output directory
    if outdir is None:
        model_dir = Path(model_path).parent.parent
        outdir = str(model_dir / "optimize_panel")

    os.makedirs(outdir, exist_ok=True)

    # Save results
    paths = save_rfe_results(result, outdir, model_name, split_seed)

    # Generate plots
    try:
        plot_path = Path(outdir) / "panel_curve.png"
        plot_pareto_curve(
            curve=result.curve,
            recommended=result.recommended_panels,
            out_path=plot_path,
            title="Panel Size vs AUROC (RFE)",
            model_name=model_name,
        )
        paths["panel_curve_plot"] = str(plot_path)
        logger.info(f"Saved panel curve plot to {plot_path}")

        ranking_plot_path = Path(outdir) / "feature_ranking.png"
        plot_feature_ranking(
            feature_ranking=result.feature_ranking,
            out_path=ranking_plot_path,
            top_n=30,
            title=f"Feature Importance Ranking ({model_name})",
        )
        paths["feature_ranking_plot"] = str(ranking_plot_path)
        logger.info(f"Saved feature ranking plot to {ranking_plot_path}")
    except Exception as e:
        logger.warning(f"Could not generate plots: {e}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Panel Optimization Complete: {model_name}")
    print(f"{'='*60}")
    print(f"Starting panel size: {len(initial_proteins)}")
    print(f"Max AUROC: {result.max_auroc:.4f}")
    print("\nRecommended panel sizes:")
    for key, size in result.recommended_panels.items():
        print(f"  {key}: {size} proteins")
    print(f"\nResults saved to: {outdir}")
    print(f"{'='*60}\n")

    return result
