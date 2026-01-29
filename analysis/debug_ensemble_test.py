import sys
import tempfile
from pathlib import Path
from click.testing import CliRunner
from ced_ml.cli.main import cli

# Create temp dirs
with tempfile.TemporaryDirectory() as tmpdir:
    tmp_path = Path(tmpdir)
    splits_dir = tmp_path / "splits"
    results_dir = tmp_path / "results"
    splits_dir.mkdir()
    results_dir.mkdir()

    # Create minimal test data (same as in fixtures)
    import numpy as np
    import pandas as pd

    n = 180
    np.random.seed(42)

    data = {
        "celiac": np.concatenate([
            np.zeros(150),  # 150 controls
            np.ones(15),    # 15 incident
            np.ones(15)     # 15 prevalent
        ]).astype(int),
        "CaseControl": np.concatenate([
            ["Control"] * 150,
            ["Incident"] * 15,
            ["Prevalent"] * 15
        ]),
        "age": np.random.uniform(20, 80, n),
        "bmi_final": np.random.uniform(18, 35, n),
        "sex": np.random.choice(["M", "F"], n),
    }

    for i in range(10):
        data[f"protein_{i}_resid"] = np.random.randn(n)

    df = pd.DataFrame(data)
    data_path = tmp_path / "test_data.parquet"
    df.to_parquet(data_path, index=False)

    # Create minimal config
    config_content = """
cv:
  n_outer: 2
  n_repeats: 1
  n_inner: 2

features:
  feature_selection_strategy: none
  screen_method: mannwhitney
  screen_top_n: 10

hyperparams:
  n_iter: 2
  cv_folds: 2

calibration:
  enabled: false
  strategy: none
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content)

    runner = CliRunner()

    # Save splits
    print("Saving splits...")
    result = runner.invoke(cli, [
        "save-splits",
        "--infile", str(data_path),
        "--outdir", str(splits_dir),
        "--n-splits", "1",
        "--seed-start", "42",
    ])
    print(f"Save splits exit code: {result.exit_code}")
    if result.exit_code != 0:
        print(f"Save splits output: {result.output}")
        sys.exit(1)

    # Train LR_EN
    print("\nTraining LR_EN...")
    result = runner.invoke(cli, [
        "train",
        "--infile", str(data_path),
        "--split-dir", str(splits_dir),
        "--outdir", str(results_dir / "LR_EN"),
        "--config", str(config_path),
        "--model", "LR_EN",
        "--split-seed", "42",
        "--run-id", "test_run",
    ], catch_exceptions=False)
    print(f"LR_EN exit code: {result.exit_code}")
    if result.exit_code != 0:
        print(f"LR_EN output:\n{result.output}")
        sys.exit(1)

    # Check for OOF files
    print("\nChecking LR_EN outputs...")
    lr_split_dir = results_dir / "LR_EN" / "run_test_run" / "splits" / "split_seed42"
    print(f"LR split dir exists: {lr_split_dir.exists()}")
    if lr_split_dir.exists():
        print(f"LR split dir contents: {list(lr_split_dir.iterdir())}")
        oof_dir = lr_split_dir / "preds"
        if oof_dir.exists():
            print(f"OOF dir contents: {list(oof_dir.iterdir())}")

    # Train RF
    print("\nTraining RF...")
    result = runner.invoke(cli, [
        "train",
        "--infile", str(data_path),
        "--split-dir", str(splits_dir),
        "--outdir", str(results_dir / "RF"),
        "--config", str(config_path),
        "--model", "RF",
        "--split-seed", "42",
        "--run-id", "test_run",
    ], catch_exceptions=False)
    print(f"RF exit code: {result.exit_code}")
    if result.exit_code != 0:
        print(f"RF output:\n{result.output}")
        sys.exit(1)

    # Check for OOF files
    print("\nChecking RF outputs...")
    rf_split_dir = results_dir / "RF" / "run_test_run" / "splits" / "split_seed42"
    print(f"RF split dir exists: {rf_split_dir.exists()}")
    if rf_split_dir.exists():
        print(f"RF split dir contents: {list(rf_split_dir.iterdir())}")
        oof_dir = rf_split_dir / "preds"
        if oof_dir.exists():
            print(f"OOF dir contents: {list(oof_dir.iterdir())}")

    # Train ensemble
    print("\nTraining ensemble...")
    result = runner.invoke(cli, [
        "train-ensemble",
        "--run-id", "test_run",
        "--split-seed", "42",
    ], catch_exceptions=False)
    print(f"Ensemble exit code: {result.exit_code}")
    print(f"Ensemble output:\n{result.output}")
