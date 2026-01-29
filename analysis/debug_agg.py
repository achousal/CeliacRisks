"""Debug aggregation test failure."""

import tempfile
from pathlib import Path
from click.testing import CliRunner
from ced_ml.cli.main import cli

# Create temp directories
tmp_dir = Path(tempfile.mkdtemp())
splits_dir = tmp_dir / "splits"
results_dir = tmp_dir / "results"
splits_dir.mkdir()
results_dir.mkdir()

# Find test data
test_data = Path("tests/fixtures/tiny_proteomics.parquet")
if not test_data.exists():
    print(f"Test data not found at {test_data}")
    exit(1)

# Find test config
test_config = Path("tests/fixtures/minimal_training_config.yaml")
if not test_config.exists():
    print(f"Test config not found at {test_config}")
    exit(1)

runner = CliRunner()

# Generate splits
print("\n=== Generating splits ===")
result_splits = runner.invoke(
    cli,
    [
        "save-splits",
        "--infile", str(test_data),
        "--outdir", str(splits_dir),
        "--n-splits", "2",
        "--seed-start", "42",
    ],
)
print(f"Exit code: {result_splits.exit_code}")
if result_splits.output:
    print(result_splits.output)

# Train on both splits
for seed in [42, 43]:
    print(f"\n=== Training split {seed} ===")
    result = runner.invoke(
        cli,
        [
            "train",
            "--infile", str(test_data),
            "--split-dir", str(splits_dir),
            "--outdir", str(results_dir),
            "--config", str(test_config),
            "--model", "LR_EN",
            "--split-seed", str(seed),
        ],
        catch_exceptions=False,
    )
    print(f"Exit code: {result.exit_code}")
    if result.exit_code != 0:
        print(f"Training failed on seed {seed}")
        print(result.output)
        exit(1)

# Aggregate
print("\n=== Aggregating ===")
result_agg = runner.invoke(
    cli,
    ["aggregate-splits", "--run-id", "test_e2e_run", "--model", "LR_EN"],
    catch_exceptions=False,
)

print(f"Exit code: {result_agg.exit_code}")
print(f"Output:\n{result_agg.output}")

if result_agg.exit_code != 0:
    print("\n=== Aggregation failed ===")
    if result_agg.exception:
        print(f"Exception: {result_agg.exception}")
        import traceback
        traceback.print_exception(type(result_agg.exception), result_agg.exception, result_agg.exception.__traceback__)

# Check structure
agg_dir = results_dir / "run_test_e2e_run" / "aggregated"
print(f"\n=== Checking structure ===")
print(f"Aggregated dir exists: {agg_dir.exists()}")
if agg_dir.exists():
    print(f"Contents: {list(agg_dir.rglob('*'))[:20]}")

print(f"\nTemp dir: {tmp_dir}")
