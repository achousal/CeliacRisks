"""
Main CLI entry point for CeD-ML pipeline.

Provides subcommands:
  - ced save-splits: Generate train/val/test splits
  - ced train: Train ML models
  - ced aggregate-splits: Aggregate results across split seeds
  - ced eval-holdout: Evaluate on holdout set
"""

import click

from ced_ml import __version__


@click.group()
@click.version_option(version=__version__, prog_name="ced")
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Increase verbosity (can be repeated: -v, -vv, -vvv)",
)
@click.option(
    "--log-level",
    type=click.Choice(["debug", "info", "warning", "error"], case_sensitive=False),
    default="info",
    help="Logging level (default: info). Use 'debug' for detailed algorithm insights.",
)
@click.pass_context
def cli(ctx, verbose, log_level):
    """
    CeD-ML: Machine Learning Pipeline for Celiac Disease Risk Prediction

    A modular, reproducible ML pipeline for predicting incident Celiac Disease
    risk from proteomics biomarkers.
    """
    import logging

    from ced_ml.utils.random import apply_seed_global

    # Store global options in context
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    # Convert log_level string to logging constant
    log_level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    ctx.obj["log_level"] = log_level_map[log_level.lower()]

    # Apply SEED_GLOBAL if set (for single-threaded reproducibility debugging)
    seed_applied = apply_seed_global()
    if seed_applied is not None:
        ctx.obj["seed_global"] = seed_applied


@cli.command("save-splits")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to YAML configuration file",
)
@click.option(
    "--infile",
    type=click.Path(exists=True),
    required=True,
    help="Input CSV file with proteomics data",
)
@click.option(
    "--outdir",
    type=click.Path(),
    default="splits",
    help="Output directory for splits (default: splits/)",
)
@click.option(
    "--mode",
    type=click.Choice(["development", "holdout"]),
    default=None,
    help="Split mode: development (TRAIN/VAL/TEST) or holdout (TRAIN/VAL/TEST + HOLDOUT)",
)
@click.option(
    "--scenarios",
    multiple=True,
    default=None,
    help="Scenarios to generate (can be repeated)",
)
@click.option(
    "--n-splits",
    type=int,
    default=None,
    help="Number of repeated splits with different seeds",
)
@click.option(
    "--val-size",
    type=float,
    default=None,
    help="Validation set proportion (0-1)",
)
@click.option(
    "--test-size",
    type=float,
    default=None,
    help="Test set proportion (0-1)",
)
@click.option(
    "--holdout-size",
    type=float,
    default=None,
    help="Holdout set proportion (only if mode=holdout)",
)
@click.option(
    "--seed-start",
    type=int,
    default=None,
    help="Starting random seed",
)
@click.option(
    "--prevalent-train-only",
    is_flag=True,
    default=None,
    help="Restrict prevalent cases to TRAIN set only (prevents reverse causality)",
)
@click.option(
    "--prevalent-train-frac",
    type=float,
    default=None,
    help="Fraction of prevalent cases to include in TRAIN (0-1)",
)
@click.option(
    "--train-control-per-case",
    type=float,
    default=None,
    help="Downsample TRAIN controls to N per case (e.g., 5 for 1:5 ratio)",
)
@click.option(
    "--eval-control-per-case",
    type=float,
    default=None,
    help="Downsample VAL/TEST controls to N per case",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing split files",
)
@click.option(
    "--override",
    multiple=True,
    help="Override config values (format: key=value or nested.key=value)",
)
@click.pass_context
def save_splits(ctx, config, **kwargs):
    """Generate train/val/test splits with stratification and optional downsampling."""
    from ced_ml.cli.save_splits import run_save_splits

    # Collect CLI args
    cli_args = {k: v for k, v in kwargs.items() if k != "override"}
    overrides = list(kwargs.get("override", []))

    # Run split generation
    run_save_splits(
        config_file=config,
        cli_args=cli_args,
        overrides=overrides,
        verbose=ctx.obj.get("verbose", 0),
        log_level=ctx.obj.get("log_level"),
    )


@cli.command("train")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to YAML configuration file",
)
@click.option(
    "--infile",
    type=click.Path(exists=True),
    required=True,
    help="Input CSV file with proteomics data",
)
@click.option(
    "--split-dir",
    type=click.Path(exists=True),
    help="Directory containing split indices",
)
@click.option(
    "--scenario",
    default=None,
    help="Scenario name (must match split scenario)",
)
@click.option(
    "--model",
    default="LR_EN",
    help="Model to train (LR, LR_EN, RF, XGBoost, LinSVM_cal, etc.)",
)
@click.option(
    "--split-seed",
    type=int,
    default=0,
    help="Split seed to use (if multiple splits generated)",
)
@click.option(
    "--outdir",
    type=click.Path(),
    default="results",
    help="Output directory for results",
)
@click.option(
    "--fixed-panel",
    type=click.Path(exists=True),
    default=None,
    help="Path to CSV with fixed feature panel (bypasses feature selection)",
)
@click.option(
    "--override",
    multiple=True,
    help="Override config values (format: key=value or nested.key=value)",
)
@click.pass_context
def train(ctx, config, **kwargs):
    """Train machine learning models with nested cross-validation."""
    from ced_ml.cli.train import run_train

    # Collect CLI args
    cli_args = {k: v for k, v in kwargs.items() if k != "override"}
    overrides = list(kwargs.get("override", []))

    # Run training
    run_train(
        config_file=config,
        cli_args=cli_args,
        overrides=overrides,
        verbose=ctx.obj.get("verbose", 0),
        log_level=ctx.obj.get("log_level"),
    )


@cli.command("aggregate-splits")
@click.option(
    "--results-dir",
    type=click.Path(exists=True),
    required=False,
    help="Directory containing split_seedX subdirectories (mutually exclusive with --run-id)",
)
@click.option(
    "--run-id",
    type=str,
    required=False,
    help="Run ID for auto-detection (e.g., 20260127_115115, mutually exclusive with --results-dir)",
)
@click.option(
    "--model",
    type=str,
    required=False,
    help="Model name for --run-id mode (e.g., LR_EN). If not specified, uses all models for run.",
)
@click.option(
    "--stability-threshold",
    type=float,
    default=0.75,
    help="Fraction of splits a feature must appear in to be 'stable' (default: 0.75)",
)
@click.option(
    "--target-specificity",
    type=float,
    default=0.95,
    help="Target specificity for alpha threshold (default: 0.95)",
)
@click.option(
    "--plot-formats",
    multiple=True,
    default=["png"],
    help="Plot output formats (can be repeated, e.g., --plot-formats png --plot-formats pdf)",
)
@click.option(
    "--n-boot",
    type=int,
    default=500,
    help="Number of bootstrap iterations for CIs (reserved for future use)",
)
@click.pass_context
def aggregate_splits(ctx, **kwargs):
    """
    Aggregate results across multiple split seeds.

    Discovers split_seedX subdirectories, collects metrics, computes pooled
    metrics, generates aggregated plots with CI bands, and builds consensus
    feature panels. Results are saved to an aggregated/ subdirectory.

    Output structure:
        aggregated/
          core/                 # Pooled and summary metrics
          cv/                   # CV metrics summary
          preds/                # Pooled predictions
          reports/              # Feature stability and consensus panels
          diagnostics/plots/    # Aggregated ROC, PR, calibration, DCA plots

    Usage:
        # Explicit path (original)
        ced aggregate-splits --results-dir results/LR_EN/run_20260127_115115/

        # Auto-detection (new)
        ced aggregate-splits --run-id 20260127_115115 --model LR_EN
        ced aggregate-splits --run-id 20260127_115115  # All models for run

    Example:
        ced aggregate-splits --results-dir results_local/
        ced aggregate-splits --results-dir results_local/ --stability-threshold 0.80
        ced aggregate-splits --results-dir results_local/ --plot-formats png --plot-formats pdf
        ced aggregate-splits --run-id 20260127_115115 --model LR_EN
    """
    from ced_ml.cli.aggregate_splits import resolve_results_dir_from_run_id, run_aggregate_splits

    # Validate mutually exclusive options
    results_dir = kwargs.get("results_dir")
    run_id = kwargs.get("run_id")
    model = kwargs.get("model")

    if not results_dir and not run_id:
        raise click.UsageError(
            "Either --results-dir or --run-id must be provided.\n"
            "Examples:\n"
            "  ced aggregate-splits --results-dir results/LR_EN/run_20260127_115115/\n"
            "  ced aggregate-splits --run-id 20260127_115115 --model LR_EN"
        )

    if results_dir and run_id:
        raise click.UsageError(
            "--results-dir and --run-id are mutually exclusive.\n"
            "Use --results-dir for explicit path OR --run-id for auto-detection."
        )

    # Auto-detect results_dir from run_id
    if run_id:
        results_dir = resolve_results_dir_from_run_id(run_id=run_id, model=model)
        kwargs["results_dir"] = results_dir
        click.echo(f"Auto-detected results directory: {results_dir}")

    # Convert tuple to list for plot_formats
    kwargs["plot_formats"] = list(kwargs["plot_formats"]) if kwargs["plot_formats"] else ["png"]

    # Remove run_id and model from kwargs (not needed by run_aggregate_splits)
    kwargs.pop("run_id", None)
    kwargs.pop("model", None)

    run_aggregate_splits(
        **kwargs, verbose=ctx.obj.get("verbose", 0), log_level=ctx.obj.get("log_level")
    )


@cli.command("eval-holdout")
@click.option(
    "--infile",
    type=click.Path(exists=True),
    required=True,
    help="Input CSV file with proteomics data",
)
@click.option(
    "--model-artifact",
    type=click.Path(exists=True),
    required=True,
    help="Path to trained model (.joblib file)",
)
@click.option(
    "--holdout-idx",
    type=click.Path(exists=True),
    required=True,
    help="Path to holdout indices CSV",
)
@click.option(
    "--outdir",
    type=click.Path(),
    required=True,
    help="Output directory for holdout evaluation results",
)
@click.option(
    "--compute-dca",
    is_flag=True,
    help="Compute decision curve analysis",
)
@click.pass_context
def eval_holdout(ctx, **kwargs):
    """Evaluate trained model on holdout set (run ONCE only)."""
    from ced_ml.cli.eval_holdout import run_eval_holdout

    run_eval_holdout(**kwargs, verbose=ctx.obj.get("verbose", 0))


@cli.command("train-ensemble")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to YAML configuration file (uses ensemble section)",
)
@click.option(
    "--results-dir",
    type=click.Path(exists=True),
    required=False,
    help="Directory containing base model results (auto-detected if --run-id provided)",
)
@click.option(
    "--base-models",
    type=str,
    default=None,
    help="Comma-separated list of base models (auto-detected if --run-id provided)",
)
@click.option(
    "--run-id",
    type=str,
    default=None,
    help="Run ID for auto-detection (e.g., 20260127_115115). Auto-discovers results-dir and base-models.",
)
@click.option(
    "--split-seed",
    type=int,
    default=0,
    help="Split seed for identifying model outputs",
)
@click.option(
    "--outdir",
    type=click.Path(),
    default=None,
    help="Output directory (default: results_dir/ENSEMBLE/split_{seed})",
)
@click.option(
    "--meta-penalty",
    type=click.Choice(["l2", "l1", "elasticnet", "none"]),
    default=None,
    help="Meta-learner regularization penalty",
)
@click.option(
    "--meta-C",
    type=float,
    default=None,
    help="Meta-learner regularization strength (inverse)",
)
@click.pass_context
def train_ensemble(ctx, config, base_models, **kwargs):
    """Train stacking ensemble from base model OOF predictions.

    This command collects out-of-fold (OOF) predictions from previously trained
    base models and trains a meta-learner (Logistic Regression) to combine them.

    AUTO-DETECTION MODE (recommended):
        Use --run-id to automatically discover results directory and base models:
            ced train-ensemble --run-id 20260127_115115 --split-seed 0

    MANUAL MODE:
        Explicitly specify results directory and base models:
            ced train-ensemble --results-dir results/ --base-models LR_EN,RF,XGBoost

    Requirements:
        - Base models must be trained first using 'ced train'
        - OOF predictions must exist in results_dir/{model}/split_{seed}/preds/train_oof/

    Examples:
        # Auto-detect from run-id (simplest)
        ced train-ensemble --run-id 20260127_115115 --split-seed 0

        # Manual specification
        ced train-ensemble --results-dir results/ --base-models LR_EN,RF,XGBoost

        # With config file
        ced train-ensemble --config configs/training_config.yaml --run-id 20260127_115115
    """
    from ced_ml.cli.train_ensemble import run_train_ensemble

    # Parse base models from comma-separated string
    base_model_list = None
    if base_models:
        base_model_list = [m.strip() for m in base_models.split(",")]

    run_train_ensemble(
        config_file=config,
        base_models=base_model_list,
        **kwargs,
        verbose=ctx.obj.get("verbose", 0),
        log_level=ctx.obj.get("log_level"),
    )


@cli.command("optimize-panel")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to YAML configuration file",
)
@click.option(
    "--results-dir",
    "-d",
    type=click.Path(exists=True),
    required=False,
    help="Path to model results directory (e.g., results/LR_EN/run_20260127_115115/). Mutually exclusive with --run-id.",
)
@click.option(
    "--run-id",
    type=str,
    required=False,
    help="Run ID to auto-discover all models (e.g., 20260127_115115). Mutually exclusive with --results-dir.",
)
@click.option(
    "--infile",
    type=click.Path(exists=True),
    required=False,
    help="Input data file (Parquet/CSV). Auto-detected from run metadata if using --run-id.",
)
@click.option(
    "--split-dir",
    type=click.Path(exists=True),
    required=False,
    help="Directory containing split indices. Auto-detected from run metadata if using --run-id.",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="Model name (auto-detected from results directory if not provided, or filter when using --run-id)",
)
@click.option(
    "--stability-threshold",
    type=float,
    default=None,
    help="Minimum selection fraction for stable proteins (default: 0.75)",
)
@click.option(
    "--min-size",
    type=int,
    default=None,
    help="Minimum panel size to evaluate (default: 5)",
)
@click.option(
    "--min-auroc-frac",
    type=float,
    default=None,
    help="Early stop if AUROC drops below this fraction of max (default: 0.90)",
)
@click.option(
    "--cv-folds",
    type=int,
    default=None,
    help="CV folds for OOF AUROC estimation (default: 5)",
)
@click.option(
    "--step-strategy",
    type=click.Choice(["geometric", "fine", "linear"]),
    default=None,
    help="Feature elimination strategy (default: geometric)",
)
@click.option(
    "--outdir",
    type=click.Path(),
    default=None,
    help="Output directory (default: results_dir/aggregated/optimize_panel/)",
)
@click.option(
    "--verbose",
    type=int,
    default=None,
    help="Verbosity level (0=warnings, 1=info, 2=debug)",
)
@click.pass_context
def optimize_panel(ctx, config, **kwargs):
    """Find minimum viable panel from aggregated cross-split results.

    This command runs RFE on consensus stable proteins derived from ALL splits,
    providing a single authoritative panel size recommendation. Benefits:

    1. Uses consensus stable proteins from all splits (eliminates variability)
    2. Pools train/val data for maximum robustness
    3. Generates a single authoritative feature ranking
    4. Matches the aggregated analysis philosophy

    Requires prior aggregation:
        ced aggregate-splits --results-dir results/LR_EN/run_X

    Examples:

        # Use config file
        ced optimize-panel --config configs/optimize_panel.yaml

        # Optimize ALL models with aggregated results under a run-id (recommended)
        ced optimize-panel --run-id 20260127_115115

        # Optimize single model (explicit path)
        ced optimize-panel \\
          --results-dir results/LR_EN/run_20260127_115115 \\
          --infile data/Celiac_dataset_proteomics_w_demo.parquet \\
          --split-dir splits/

        # Optimize specific model(s) by run-id
        ced optimize-panel --run-id 20260127_115115 --model LR_EN

        # Override config values with CLI args
        ced optimize-panel --config configs/optimize_panel.yaml --cv-folds 10

    Outputs (in results_dir/aggregated/optimize_panel/)
        - panel_curve_aggregated.csv: AUROC vs panel size
        - feature_ranking_aggregated.csv: Protein elimination order
        - recommended_panels_aggregated.json: Minimum sizes at thresholds
        - panel_curve_aggregated.png: Pareto curve visualization
    """
    import json
    from pathlib import Path

    from ced_ml.cli.optimize_panel import discover_models_by_run_id, run_optimize_panel_aggregated

    # Load config file: use provided path, or auto-detect default if it exists
    config_params = {}
    default_config = Path(__file__).parent.parent.parent.parent / "configs" / "optimize_panel.yaml"
    config_path = config or (default_config if default_config.exists() else None)

    if config_path:
        import yaml

        with open(config_path) as f:
            config_params = yaml.safe_load(f) or {}
        if config:
            click.echo(f"Loaded config from {config_path}")
        else:
            click.echo(f"Loaded default config from {config_path}")

    # Merge config with CLI args (CLI takes precedence)
    # Only use config values if CLI arg is None (not provided)
    for key in [
        "results_dir",
        "run_id",
        "infile",
        "split_dir",
        "model",
        "stability_threshold",
        "min_size",
        "min_auroc_frac",
        "cv_folds",
        "step_strategy",
        "outdir",
        "verbose",
    ]:
        if kwargs.get(key) is None and key in config_params:
            kwargs[key] = config_params[key]
            click.echo(f"Using config value for {key}: {config_params[key]}")

    # Validate mutually exclusive options
    if kwargs.get("results_dir") and kwargs.get("run_id"):
        raise click.UsageError(
            "--results-dir and --run-id are mutually exclusive. Use one or the other."
        )

    if not kwargs.get("results_dir") and not kwargs.get("run_id"):
        raise click.UsageError("Either --results-dir or --run-id is required.")

    # Auto-discover models if using --run-id
    if kwargs.get("run_id"):
        run_id = kwargs["run_id"]
        results_root = Path(__file__).parent.parent.parent.parent.parent / "results"

        click.echo(f"Auto-discovering models for run_id={run_id} in {results_root}")

        model_dirs = discover_models_by_run_id(
            run_id=run_id,
            results_root=results_root,
            model_filter=kwargs.get("model"),
        )

        if not model_dirs:
            if kwargs.get("model"):
                raise click.ClickException(
                    f"No models matching '{kwargs['model']}' found with run_id={run_id} "
                    f"and aggregated results in {results_root}"
                )
            else:
                raise click.ClickException(
                    f"No models found with run_id={run_id} and aggregated results in {results_root}"
                )

        click.echo(f"Found {len(model_dirs)} model(s) with aggregated results:")
        for model_name, results_dir in model_dirs.items():
            click.echo(f"  - {model_name}: {results_dir}")

        # Auto-detect infile and split_dir from first model's run metadata
        # (they should be the same across all models in the same run)
        first_model_dir = next(iter(model_dirs.values()))
        # model_dirs values are aggregated directories, go up one level for run_metadata.json
        metadata_file = first_model_dir.parent / "run_metadata.json"

        # If not found at run level, search in split directories
        if not metadata_file.exists():
            run_dir = first_model_dir.parent
            split_dirs = list(run_dir.glob("split_seed*"))
            if split_dirs:
                # Check for run_metadata.json in the first split directory
                metadata_file = split_dirs[0] / "run_metadata.json"

        if not metadata_file.exists():
            # Fallback: require manual specification
            if not kwargs.get("infile") or not kwargs.get("split_dir"):
                raise click.ClickException(
                    f"Could not find run_metadata.json in {first_model_dir} or split directories. "
                    f"Please specify --infile and --split-dir manually."
                )
            infile = kwargs["infile"]
            split_dir = kwargs["split_dir"]
        else:
            with open(metadata_file) as f:
                metadata = json.load(f)

            # Use metadata values, but allow CLI overrides
            infile = kwargs.get("infile") or metadata.get("infile")
            # Handle both "split_dir" (per-split metadata) and "splits_dir" (run-level metadata)
            split_dir = (
                kwargs.get("split_dir") or metadata.get("split_dir") or metadata.get("splits_dir")
            )

            if not infile or not split_dir:
                raise click.ClickException(
                    "Could not auto-detect infile and split_dir from run metadata. "
                    "Please specify --infile and --split-dir manually."
                )

            click.echo("\nAuto-detected from run metadata:")
            click.echo(f"  Input file: {infile}")
            click.echo(f"  Split dir:  {split_dir}")

        click.echo("")

        # Determine verbosity (config/CLI, with CLI taking precedence via ctx.obj)
        verbose_level = (
            kwargs.get("verbose")
            if kwargs.get("verbose") is not None
            else ctx.obj.get("verbose", 0)
        )

        # Run optimization for each discovered model
        for model_name, results_dir in model_dirs.items():
            click.echo(f"\n{'='*70}")
            click.echo(f"Optimizing panel for: {model_name}")
            click.echo(f"{'='*70}\n")

            run_optimize_panel_aggregated(
                results_dir=results_dir,
                infile=infile,
                split_dir=split_dir,
                model_name=model_name,
                stability_threshold=kwargs.get("stability_threshold") or 0.75,
                min_size=kwargs.get("min_size") or 5,
                min_auroc_frac=kwargs.get("min_auroc_frac") or 0.90,
                cv_folds=kwargs.get("cv_folds") or 5,
                step_strategy=kwargs.get("step_strategy") or "geometric",
                outdir=kwargs.get("outdir"),
                verbose=verbose_level,
            )

        click.echo(f"\n{'='*70}")
        click.echo(f"Panel optimization complete for all {len(model_dirs)} model(s)")
        click.echo(f"{'='*70}\n")

    else:
        # Single model path provided explicitly
        # Require infile and split_dir for explicit path mode
        if not kwargs.get("infile") or not kwargs.get("split_dir"):
            raise click.UsageError(
                "When using --results-dir, both --infile and --split-dir are required."
            )

        # Determine verbosity (config/CLI, with CLI taking precedence via ctx.obj)
        verbose_level = (
            kwargs.get("verbose")
            if kwargs.get("verbose") is not None
            else ctx.obj.get("verbose", 0)
        )

        run_optimize_panel_aggregated(
            results_dir=kwargs["results_dir"],
            infile=kwargs["infile"],
            split_dir=kwargs["split_dir"],
            model_name=kwargs.get("model"),
            stability_threshold=kwargs.get("stability_threshold") or 0.75,
            min_size=kwargs.get("min_size") or 5,
            min_auroc_frac=kwargs.get("min_auroc_frac") or 0.90,
            cv_folds=kwargs.get("cv_folds") or 5,
            step_strategy=kwargs.get("step_strategy") or "geometric",
            outdir=kwargs.get("outdir"),
            verbose=verbose_level,
        )


@cli.command("consensus-panel")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    default=None,
    help="Path to YAML configuration file (auto-detects consensus_panel.yaml if exists)",
)
@click.option(
    "--run-id",
    type=str,
    required=True,
    help="Run ID to process (e.g., 20260127_115115). Auto-discovers all models.",
)
@click.option(
    "--infile",
    type=click.Path(exists=True),
    default=None,
    help="Input data file (auto-detected from run metadata if not provided)",
)
@click.option(
    "--split-dir",
    type=click.Path(exists=True),
    default=None,
    help="Directory containing split indices (auto-detected if not provided)",
)
@click.option(
    "--stability-threshold",
    type=float,
    default=None,
    help="Minimum selection fraction for stable proteins (default: 0.75)",
)
@click.option(
    "--corr-threshold",
    type=float,
    default=None,
    help="Correlation threshold for clustering (default: 0.85)",
)
@click.option(
    "--target-size",
    type=int,
    default=None,
    help="Target panel size (default: 25)",
)
@click.option(
    "--rfe-weight",
    type=float,
    default=None,
    help="Weight for RFE rank vs stability (0-1, default: 0.5)",
)
@click.option(
    "--rra-method",
    type=click.Choice(["geometric_mean", "borda", "median"]),
    default=None,
    help="RRA aggregation method (default: geometric_mean)",
)
@click.option(
    "--outdir",
    type=click.Path(),
    default=None,
    help="Output directory (default: results/consensus_panel/run_<RUN_ID>)",
)
@click.option(
    "--verbose",
    type=int,
    default=None,
    help="Verbosity level (0=warnings, 1=info, 2=debug)",
)
@click.pass_context
def consensus_panel(ctx, config, **kwargs):
    """Generate consensus protein panel from multiple models via RRA.

    Aggregates protein rankings from all base models (LR_EN, RF, XGBoost, etc.)
    to create a single consensus panel for clinical deployment. The workflow:

    1. Loads stability rankings from each model's aggregated results
    2. (Optional) Incorporates RFE rankings if available
    3. Aggregates via Robust Rank Aggregation (geometric mean)
    4. Clusters correlated proteins and selects representatives
    5. Outputs final panel for fixed-panel validation

    Requires prior aggregation:
        ced aggregate-splits --results-dir results/*/run_<RUN_ID>

    Examples:

        # Basic usage (auto-discovers all models)
        ced consensus-panel --run-id 20260127_115115

        # Custom parameters
        ced consensus-panel --run-id 20260127_115115 \\
            --target-size 30 \\
            --corr-threshold 0.90 \\
            --rfe-weight 0.3

        # Validate the resulting panel
        ced train --model LR_EN \\
            --fixed-panel results/consensus_panel/run_20260127_115115/final_panel.txt \\
            --split-seed 10

    Outputs (in results/consensus_panel/run_<RUN_ID>/):
        - final_panel.txt: One protein per line (for --fixed-panel)
        - final_panel.csv: With consensus scores
        - consensus_ranking.csv: All proteins with RRA scores
        - per_model_rankings.csv: Per-model composite rankings
        - correlation_clusters.csv: Cluster assignments
        - consensus_metadata.json: Run parameters and statistics
    """
    from pathlib import Path

    from ced_ml.cli.consensus_panel import run_consensus_panel

    # Load config file if provided or auto-detect
    config_params = {}
    default_config = Path(__file__).parent.parent.parent.parent / "configs" / "consensus_panel.yaml"
    config_path = config or (default_config if default_config.exists() else None)

    if config_path:
        import yaml

        with open(config_path) as f:
            config_params = yaml.safe_load(f) or {}
        if config:
            click.echo(f"Loaded config from {config_path}")
        else:
            click.echo(f"Loaded default config from {config_path}")

    # Merge config with CLI args (CLI takes precedence)
    param_keys = [
        "run_id",
        "infile",
        "split_dir",
        "stability_threshold",
        "corr_threshold",
        "target_size",
        "rfe_weight",
        "rra_method",
        "outdir",
        "verbose",
    ]
    for key in param_keys:
        if kwargs.get(key) is None and key in config_params:
            kwargs[key] = config_params[key]

    # Determine verbosity
    verbose_level = kwargs.get("verbose")
    if verbose_level is None:
        verbose_level = ctx.obj.get("verbose", 0)

    # Run consensus panel generation
    run_consensus_panel(
        run_id=kwargs["run_id"],
        infile=kwargs.get("infile"),
        split_dir=kwargs.get("split_dir"),
        stability_threshold=kwargs.get("stability_threshold") or 0.75,
        corr_threshold=kwargs.get("corr_threshold") or 0.85,
        target_size=kwargs.get("target_size") or 25,
        rfe_weight=kwargs.get("rfe_weight") or 0.5,
        rra_method=kwargs.get("rra_method") or "geometric_mean",
        outdir=kwargs.get("outdir"),
        verbose=verbose_level,
    )


@cli.group("config")
@click.pass_context
def config_group(ctx):
    """Configuration management tools (validate, diff)."""
    pass


@config_group.command("validate")
@click.argument("config_file", type=click.Path(exists=True))
@click.option(
    "--command",
    type=click.Choice(["save-splits", "train"]),
    default="train",
    help="Command type (default: train)",
)
@click.option(
    "--strict",
    is_flag=True,
    help="Treat warnings as errors",
)
@click.pass_context
def config_validate(ctx, config_file, command, strict):
    """Validate configuration file and report issues."""
    from pathlib import Path

    from ced_ml.cli.config_tools import run_config_validate

    run_config_validate(
        config_file=Path(config_file),
        command=command,
        strict=strict,
        verbose=ctx.obj.get("verbose", 0),
    )


@config_group.command("diff")
@click.argument("config_file1", type=click.Path(exists=True))
@click.argument("config_file2", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file for diff report",
)
@click.pass_context
def config_diff(ctx, config_file1, config_file2, output):
    """Compare two configuration files."""
    from pathlib import Path

    from ced_ml.cli.config_tools import run_config_diff

    run_config_diff(
        config_file1=Path(config_file1),
        config_file2=Path(config_file2),
        output_file=Path(output) if output else None,
        verbose=ctx.obj.get("verbose", 0),
    )


@cli.command("convert-to-parquet")
@click.argument(
    "csv_path",
    type=click.Path(exists=True),
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output Parquet file path (default: same as input with .parquet extension)",
)
@click.option(
    "--compression",
    type=click.Choice(["snappy", "gzip", "brotli", "zstd", "none"]),
    default="snappy",
    help="Compression algorithm (default: snappy)",
)
def convert_to_parquet(csv_path, output, compression):
    """
    Convert proteomics CSV file to Parquet format.

    This command reads a CSV file and converts it to Parquet format with
    compression. Only columns needed for modeling are included (same as
    the training pipeline).

    CSV_PATH: Path to input CSV file

    Example:
        ced convert-to-parquet data/celiac_proteomics.csv
        ced convert-to-parquet data/celiac_proteomics.csv -o data/celiac.parquet --compression gzip
    """

    from ced_ml.data.io import convert_csv_to_parquet

    try:
        parquet_path = convert_csv_to_parquet(
            csv_path=csv_path,
            parquet_path=output,
            compression=compression,
        )
        click.echo(f"Successfully converted to: {parquet_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort() from e


def main():
    """Entry point for console script."""
    cli(obj={})


if __name__ == "__main__":
    main()
