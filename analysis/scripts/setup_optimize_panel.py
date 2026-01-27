#!/usr/bin/env python3
"""Auto-detect latest run ID and set up optimize_panel.yaml for ced optimize-panel.

This helper script:
1. Finds the latest run across all models in results/
2. Lists available models and splits
3. Offers to configure optimize_panel.yaml with the detected run

Usage:
    python scripts/setup_optimize_panel.py
    python scripts/setup_optimize_panel.py --model LR_EN --split-seed 0
    python scripts/setup_optimize_panel.py --run-id 20260127_104409
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import re

def find_latest_run(results_dir: Path) -> str | None:
    """Find the latest run_* directory across all models."""
    run_ids = []
    for model_dir in results_dir.glob("*/"):
        if model_dir.name.startswith(".") or model_dir.name == "investigations":
            continue
        for run_dir in model_dir.glob("run_*"):
            if run_dir.is_dir():
                run_id = run_dir.name.replace("run_", "")
                run_ids.append(run_id)

    if not run_ids:
        return None

    # Sort by timestamp (format: YYYYMMDD_HHMMSS)
    run_ids.sort(reverse=True)
    return run_ids[0]

def list_models_and_splits(results_dir: Path, run_id: str | None = None) -> dict:
    """List all models and their available splits."""
    models = {}

    for model_dir in sorted(results_dir.glob("*/")):
        if model_dir.name.startswith(".") or model_dir.name == "investigations":
            continue

        model_name = model_dir.name
        splits = []

        if run_id:
            # Look for specific run
            run_dir = model_dir / f"run_{run_id}"
            if run_dir.exists():
                for split_dir in sorted(run_dir.glob("split_seed*")):
                    seed = split_dir.name.replace("split_seed", "")
                    model_file = split_dir / "core" / f"{model_name}__final_model.joblib"
                    if model_file.exists():
                        splits.append(int(seed))
        else:
            # Look for any split directories
            for split_dir in sorted(model_dir.glob("split_seed*")):
                seed = split_dir.name.replace("split_seed", "")
                model_file = split_dir / "core" / f"{model_name}__final_model.joblib"
                if model_file.exists():
                    splits.append(int(seed))

            # Also check run_* directories
            for run_dir in sorted(model_dir.glob("run_*")):
                for split_dir in sorted(run_dir.glob("split_seed*")):
                    seed = split_dir.name.replace("split_seed", "")
                    model_file = split_dir / "core" / f"{model_name}__final_model.joblib"
                    if model_file.exists():
                        splits.append(int(seed))

        if splits:
            models[model_name] = sorted(set(splits))

    return models

def find_model_path(results_dir: Path, model: str, split_seed: int, run_id: str | None = None) -> Path | None:
    """Find the model file for a given model and split seed."""
    model_dir = results_dir / model

    if run_id:
        # Try with run_id
        model_file = model_dir / f"run_{run_id}" / f"split_seed{split_seed}" / "core" / f"{model}__final_model.joblib"
        if model_file.exists():
            return model_file
    else:
        # Try without run_id (legacy paths)
        model_file = model_dir / f"split_seed{split_seed}" / "core" / f"{model}__final_model.joblib"
        if model_file.exists():
            return model_file

        # Try with run_* directory (find latest)
        run_dirs = sorted(model_dir.glob("run_*"), reverse=True)
        for run_dir in run_dirs:
            model_file = run_dir / f"split_seed{split_seed}" / "core" / f"{model}__final_model.joblib"
            if model_file.exists():
                return model_file

    return None

def update_config(config_path: Path, model_path: Path, split_seed: int, project_root: Path) -> bool:
    """Update optimize_panel.yaml with the model path."""
    try:
        with open(config_path, "r") as f:
            content = f.read()

        # Replace model_path line (relative to project root, not config dir)
        model_path_rel = model_path.relative_to(project_root)
        new_content = re.sub(
            r"model_path:.*",
            f"model_path: {model_path_rel}",
            content
        )

        # Replace split_seed line
        new_content = re.sub(
            r"split_seed:.*",
            f"split_seed: {split_seed}",
            new_content
        )

        with open(config_path, "w") as f:
            f.write(new_content)

        return True
    except Exception as e:
        print(f"Error updating config: {e}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Auto-detect latest run ID and configure optimize_panel.yaml"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name (if not provided, lists available models)"
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Split seed to use (default: 0)"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Specific run ID (if not provided, auto-detects latest)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/optimize_panel.yaml",
        help="Path to optimize_panel.yaml (default: configs/optimize_panel.yaml)"
    )
    parser.add_argument(
        "--no-update",
        action="store_true",
        help="Don't update config file, just print paths"
    )

    args = parser.parse_args()

    # Determine base directory
    base_dir = Path(__file__).parent.parent
    project_root = base_dir.parent  # Go up from analysis/
    results_dir = project_root / "results"
    config_path = base_dir / args.config

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}", file=sys.stderr)
        return 1

    # Auto-detect run ID if not provided
    run_id = args.run_id
    if not run_id:
        run_id = find_latest_run(results_dir)
        if run_id:
            print(f"Auto-detected latest run ID: {run_id}")
        else:
            print("Error: No runs found in results/", file=sys.stderr)
            return 1

    # List available models and splits
    print(f"\nAvailable models and splits for run {run_id}:")
    models_dict = list_models_and_splits(results_dir, run_id)

    if not models_dict:
        print("Error: No models found for this run", file=sys.stderr)
        return 1

    for model_name in sorted(models_dict.keys()):
        splits = models_dict[model_name]
        print(f"  {model_name}: seeds {splits}")

    # Determine which model to use
    if args.model:
        model = args.model
        if model not in models_dict:
            print(f"\nError: Model '{model}' not found", file=sys.stderr)
            return 1
    else:
        # Use first available model by default
        model = sorted(models_dict.keys())[0]
        print(f"\nUsing first available model: {model}")

    # Check if split_seed exists for this model
    if args.split_seed not in models_dict[model]:
        print(f"Error: Split seed {args.split_seed} not found for model {model}", file=sys.stderr)
        print(f"Available seeds: {models_dict[model]}", file=sys.stderr)
        return 1

    # Find model path
    model_path = find_model_path(results_dir, model, args.split_seed, run_id)
    if not model_path or not model_path.exists():
        print(f"Error: Model file not found for {model}, seed {args.split_seed}", file=sys.stderr)
        return 1

    print(f"\nSelected configuration:")
    print(f"  Model: {model}")
    print(f"  Split seed: {args.split_seed}")
    print(f"  Run ID: {run_id}")
    print(f"  Model path: {model_path.relative_to(project_root)}")

    # Update config file if requested
    if not args.no_update:
        if update_config(config_path, model_path, args.split_seed, project_root):
            print(f"\nUpdated config: {config_path}")
            print(f"\nYou can now run:")
            print(f"  cd {base_dir}")
            print(f"  ced optimize-panel")
        else:
            return 1
    else:
        print(f"\nTo use this model, you can run:")
        print(f"  cd {base_dir}")
        print(f"  ced optimize-panel \\")
        print(f"    --model-path {model_path.relative_to(project_root)} \\")
        print(f"    --split-seed {args.split_seed}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
