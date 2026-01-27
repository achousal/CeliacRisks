"""CLI implementation for panel size optimization via RFE.

This module provides the `ced optimize-panel` command for finding minimum
viable protein panels through Recursive Feature Elimination.
"""

import logging
import os
from pathlib import Path

import joblib
import pandas as pd

from ced_ml.data.filters import apply_row_filters
from ced_ml.data.io import read_proteomics_file
from ced_ml.data.schema import TARGET_COL, get_positive_label
from ced_ml.features.rfe import (
    RFEResult,
    recursive_feature_elimination,
    save_rfe_results,
)
from ced_ml.features.stability import compute_selection_frequencies, rank_proteins_by_frequency
from ced_ml.plotting.panel_curve import plot_feature_ranking, plot_pareto_curve


def find_model_path_for_run(
    run_id: str | None = None,
    model: str | None = None,
    split_seed: int = 0,
) -> str:
    """Auto-detect model path from run_id and model name.

    If run_id is None, auto-detects the latest run.
    If model is None, uses the first available model.

    Args:
        run_id: Run ID (e.g., "20260127_104409"). If None, auto-detects latest.
        model: Model name (e.g., "LR_EN"). If None, uses first available.
        split_seed: Split seed (default: 0)

    Returns:
        Path to the model file

    Raises:
        FileNotFoundError: If model not found
        ValueError: If configuration is invalid
    """
    from pathlib import Path

    # Determine results directory (project root / results)
    # __file__ is: .../analysis/src/ced_ml/cli/optimize_panel.py
    # So we need to go up: .. (cli) .. (ced_ml) .. (src) .. (analysis) .. (project_root) / results
    results_dir = Path(__file__).parent.parent.parent.parent.parent / "results"

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Auto-detect run_id if not provided
    if not run_id:
        run_ids = []
        for model_dir in results_dir.glob("*/"):
            if model_dir.name.startswith(".") or model_dir.name == "investigations":
                continue
            for run_dir in model_dir.glob("run_*"):
                if run_dir.is_dir():
                    rid = run_dir.name.replace("run_", "")
                    run_ids.append(rid)

        if not run_ids:
            raise FileNotFoundError("No runs found in results directory")

        # Sort by timestamp (format: YYYYMMDD_HHMMSS)
        run_ids.sort(reverse=True)
        run_id = run_ids[0]

    # Auto-detect model if not provided
    if not model:
        models = []
        for model_dir in sorted(results_dir.glob("*/")):
            if model_dir.name.startswith(".") or model_dir.name == "investigations":
                continue
            run_path = model_dir / f"run_{run_id}"
            if run_path.exists():
                models.append(model_dir.name)

        if not models:
            raise FileNotFoundError(f"No models found for run {run_id}")

        model = sorted(models)[0]

    # Find the model file
    model_path = (
        results_dir
        / model
        / f"run_{run_id}"
        / f"split_seed{split_seed}"
        / "core"
        / f"{model}__final_model.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n" f"Run: {run_id}, Model: {model}, Split: {split_seed}"
        )

    return str(model_path)


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
    # Setup logging to file and console
    log_level = logging.WARNING
    if verbose >= 2:
        log_level = logging.DEBUG
    elif verbose >= 1:
        log_level = logging.INFO

    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # Create logs/features directory at root level (parallel to results/)
    # __file__ is: .../analysis/src/ced_ml/cli/optimize_panel.py
    # Go up to project root: .. (cli) .. (ced_ml) .. (src) .. (analysis) .. (root) / logs / features
    log_dir = Path(__file__).parent.parent.parent.parent.parent / "logs" / "features"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Timestamped log file
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"optimize_panel_{timestamp}.log"

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    # Add handlers (avoid duplicates)
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    logger.info(f"Panel optimization started at {log_file}")
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
    df_raw = read_proteomics_file(infile, validate=True)

    # Apply row filters (must match split generation)
    logger.info("Applying row filters...")
    df, filter_stats = apply_row_filters(df_raw, meta_num_cols=meta_num_cols)
    logger.info(f"Filtered: {filter_stats['n_in']:,} → {filter_stats['n_out']:,} rows")
    logger.info(f"  Removed {filter_stats['n_removed_uncertain_controls']} uncertain controls")
    logger.info(f"  Removed {filter_stats['n_removed_dropna_meta_num']} rows with missing metadata")

    # Load split indices (CSV format)
    split_path = Path(split_dir)
    scenario = bundle.get("scenario", "IncidentOnly")

    # Try new format first (with scenario)
    train_file = split_path / f"train_idx_{scenario}_seed{split_seed}.csv"
    val_file = split_path / f"val_idx_{scenario}_seed{split_seed}.csv"

    # Fallback to old format (without scenario)
    if not train_file.exists():
        train_file = split_path / f"train_idx_seed{split_seed}.csv"
        val_file = split_path / f"val_idx_seed{split_seed}.csv"

    if not train_file.exists() or not val_file.exists():
        raise FileNotFoundError(f"Split files not found. Tried: {train_file}, {val_file}")

    logger.info(f"Loading splits from {split_path}")
    train_idx = pd.read_csv(train_file).squeeze().values
    val_idx = pd.read_csv(val_file).squeeze().values

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

    # Create binary target using schema-defined target column and positive label
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found in data. "
            f"Available columns: {list(df.columns[:10])}..."
        )

    positive_label = get_positive_label(scenario)
    y_all = (df[TARGET_COL] == positive_label).astype(int).values

    # Subset to train/val (use iloc for position-based indexing)
    X_train = df.iloc[train_idx][feature_cols].copy()
    y_train = y_all[train_idx]

    X_val = df.iloc[val_idx][feature_cols].copy()
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
    logger.info(
        f"RFE parameters: step_strategy={step_strategy}, cv_folds={cv_folds}, "
        f"min_auroc_frac={min_auroc_frac}, min_size={min_size}"
    )

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

    logger.info(f"RFE complete. Max AUROC: {result.max_auroc:.4f}")

    # Determine output directory
    if outdir is None:
        model_dir = Path(model_path).parent.parent
        outdir = str(model_dir / "optimize_panel")

    os.makedirs(outdir, exist_ok=True)

    # Save results
    paths = save_rfe_results(result, outdir, model_name, split_seed)
    logger.info(f"Saved RFE results to {outdir}")

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
    print(f"Log file: {log_file}")
    print(f"{'='*60}\n")

    logger.info("Panel optimization completed successfully")

    return result
