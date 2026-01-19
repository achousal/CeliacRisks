"""
Main CLI entry point for CeD-ML pipeline.

Provides subcommands:
  - ced save-splits: Generate train/val/test splits
  - ced train: Train ML models
  - ced postprocess: Aggregate results across models
  - ced eval-holdout: Evaluate on holdout set
"""

import sys
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
    # Store global options in context
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


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
    default="development",
    help="Split mode: development (TRAIN/VAL/TEST) or holdout (TRAIN/VAL/TEST + HOLDOUT)",
)
@click.option(
    "--scenarios",
    multiple=True,
    default=["IncidentOnly"],
    help="Scenarios to generate (can be repeated)",
)
@click.option(
    "--n-splits",
    type=int,
    default=1,
    help="Number of repeated splits with different seeds",
)
@click.option(
    "--val-size",
    type=float,
    default=0.0,
    help="Validation set proportion (0-1)",
)
@click.option(
    "--test-size",
    type=float,
    default=0.30,
    help="Test set proportion (0-1)",
)
@click.option(
    "--holdout-size",
    type=float,
    default=0.30,
    help="Holdout set proportion (only if mode=holdout)",
)
@click.option(
    "--seed-start",
    type=int,
    default=0,
    help="Starting random seed",
)
@click.option(
    "--prevalent-train-only",
    is_flag=True,
    help="Restrict prevalent cases to TRAIN set only (prevents reverse causality)",
)
@click.option(
    "--prevalent-train-frac",
    type=float,
    default=1.0,
    help="Fraction of prevalent cases to include in TRAIN (0-1)",
)
@click.option(
    "--train-control-per-case",
    type=float,
    help="Downsample TRAIN controls to N per case (e.g., 5 for 1:5 ratio)",
)
@click.option(
    "--eval-control-per-case",
    type=float,
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
    default="IncidentOnly",
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


@cli.command("postprocess")
@click.option(
    "--results-dir",
    type=click.Path(exists=True),
    required=True,
    help="Directory containing model results",
)
@click.option(
    "--mode",
    type=click.Choice(["models", "sensitivity", "single"]),
    default="models",
    help="Postprocessing mode",
)
@click.option(
    "--n-boot",
    type=int,
    default=500,
    help="Number of bootstrap iterations for CIs",
)
@click.option(
    "--compute-dca",
    is_flag=True,
    help="Compute decision curve analysis",
)
@click.pass_context
def postprocess(ctx, **kwargs):
    """Aggregate and compare results across models."""
    from ced_ml.cli.postprocess import run_postprocess
    
    run_postprocess(**kwargs, verbose=ctx.obj.get("verbose", 0))


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


@cli.group("config")
@click.pass_context
def config_group(ctx):
    """Configuration management tools (migrate, validate, diff)."""
    pass


@config_group.command("migrate")
@click.option(
    "--input-file",
    type=click.Path(exists=True),
    help="File containing legacy CLI args (one per line)",
)
@click.option(
    "--args",
    multiple=True,
    help="Legacy CLI arguments (can be repeated)",
)
@click.option(
    "--command",
    type=click.Choice(["save-splits", "train"]),
    default="save-splits",
    help="Command type (default: save-splits)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output YAML file (default: <command>_config.yaml)",
)
@click.pass_context
def config_migrate(ctx, input_file, args, command, output):
    """Convert legacy CLI arguments to YAML config."""
    from pathlib import Path
    from ced_ml.cli.config_tools import run_config_migrate

    run_config_migrate(
        input_file=Path(input_file) if input_file else None,
        args=list(args) if args else None,
        command=command,
        output_file=Path(output) if output else None,
        verbose=ctx.obj.get("verbose", 0),
    )


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


def main():
    """Entry point for console script."""
    cli(obj={})


if __name__ == "__main__":
    main()
