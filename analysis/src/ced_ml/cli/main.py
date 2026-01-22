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
@click.pass_context
def cli(ctx, verbose):
    """
    CeD-ML: Machine Learning Pipeline for Celiac Disease Risk Prediction

    A modular, reproducible ML pipeline for predicting incident Celiac Disease
    risk from proteomics biomarkers.
    """
    from ced_ml.utils.random import apply_seed_global

    # Store global options in context
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

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
    )


@cli.command("aggregate-splits")
@click.option(
    "--results-dir",
    type=click.Path(exists=True),
    required=True,
    help="Directory containing split_seedX subdirectories",
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

    Example:
        ced aggregate-splits --results-dir results_local/
        ced aggregate-splits --results-dir results_local/ --stability-threshold 0.80
        ced aggregate-splits --results-dir results_local/ --plot-formats png --plot-formats pdf
    """
    from ced_ml.cli.aggregate_splits import run_aggregate_splits

    # Convert tuple to list for plot_formats
    kwargs["plot_formats"] = list(kwargs["plot_formats"]) if kwargs["plot_formats"] else ["png"]

    run_aggregate_splits(**kwargs, verbose=ctx.obj.get("verbose", 0))


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
    required=True,
    help="Directory containing base model results (e.g., results/)",
)
@click.option(
    "--base-models",
    type=str,
    default=None,
    help="Comma-separated list of base models (e.g., LR_EN,RF,XGBoost)",
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

    Requirements:
        - Base models must be trained first using 'ced train'
        - OOF predictions must exist in results_dir/{model}/split_{seed}/preds/train_oof/

    Example:
        ced train-ensemble --results-dir results/ --base-models LR_EN,RF,XGBoost
        ced train-ensemble --config configs/training_config.yaml --results-dir results/
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
@click.pass_context
def convert_to_parquet(ctx, csv_path, output, compression):
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
