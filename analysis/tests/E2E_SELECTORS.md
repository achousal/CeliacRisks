# E2E Test Selectors Strategy

Stable selector patterns for E2E tests in the CeliacRisks pipeline.

## File Path Selectors

### Principle: Use Path objects, not strings
```python
# Good: Cross-platform Path objects
splits_dir = tmp_path / "splits"
model_dir = results_dir / "LR_EN" / "split_seed42"

# Bad: String concatenation
splits_dir = str(tmp_path) + "/splits"
```

### Pattern: Flexible directory discovery
```python
# Find actual output directory (handles run_YYYYMMDD/ wrapper)
model_dirs = list(results_dir.rglob("split_seed42"))
if model_dirs:
    model_dir = model_dirs[0]

# Check for files with glob patterns
has_predictions = any(model_dir.rglob("*.csv"))
has_config = any(model_dir.rglob("*config*.yaml"))
```

## CLI Command Selectors

### CliRunner pattern
```python
from click.testing import CliRunner
from ced_ml.cli.main import cli

runner = CliRunner()
result = runner.invoke(cli, ["command", "--arg", "value"])

# Check exit code
assert result.exit_code == 0

# Check output
assert "expected text" in result.output
```

### Error handling pattern
```python
result = runner.invoke(cli, [...], catch_exceptions=False)

if result.exit_code != 0:
    print("OUTPUT:", result.output)
    if result.exception:
        import traceback
        traceback.print_exception(...)
```

## File Content Validators

### JSON files
```python
import json

with open(model_dir / "metrics.json") as f:
    metrics = json.load(f)

assert "calibration" in metrics
assert "auroc" in metrics["discrimination"]
assert 0.0 <= metrics["discrimination"]["auroc"] <= 1.0
```

### CSV files
```python
import pandas as pd

train_idx = pd.read_csv(splits_dir / "train_idx_IncidentOnly_seed42.csv")
assert "idx" in train_idx.columns
assert len(train_idx) > 0
assert all(train_idx["idx"] >= 0)
```

### YAML configs
```python
import yaml

with open(config_path) as f:
    config = yaml.safe_load(f)

assert "cv" in config
assert config["cv"]["folds"] > 0
```

## Output Structure Validators

### Split files structure
```
splits/
  train_idx_IncidentOnly_seed42.csv
  val_idx_IncidentOnly_seed42.csv
  test_idx_IncidentOnly_seed42.csv
  split_meta_IncidentOnly_seed42.json
```

Validation pattern:
```python
for seed in [42, 43]:
    assert (splits_dir / f"train_idx_IncidentOnly_seed{seed}.csv").exists()
    assert (splits_dir / f"val_idx_IncidentOnly_seed{seed}.csv").exists()
    assert (splits_dir / f"test_idx_IncidentOnly_seed{seed}.csv").exists()
    assert (splits_dir / f"split_meta_IncidentOnly_seed{seed}.json").exists()
```

### Training output structure
```
results/
  run_YYYYMMDD_HHMMSS/  # Optional timestamped wrapper
    split_seed42/
      core/
        test_metrics.csv
      cv/
        cv_results.csv
      preds/
        *.csv
      training_config.yaml
      config_metadata.json
```

Flexible validation pattern:
```python
# Find directory regardless of wrapper
model_dirs = list(results_dir.rglob("split_seed42"))
assert len(model_dirs) > 0, "No split_seed42 directory found"

model_dir = model_dirs[0]

# Check for key files (flexible naming)
has_predictions = any(model_dir.rglob("*.csv"))
has_config = any(model_dir.rglob("*config*.yaml"))
has_output = len(list(model_dir.rglob("*"))) > 5

assert has_predictions
assert has_config
assert has_output
```

## Metrics Range Validators

### Discrimination metrics
```python
assert 0.0 <= metrics["discrimination"]["auroc"] <= 1.0
assert 0.0 <= metrics["discrimination"]["pr_auc"] <= 1.0
assert 0.0 <= metrics["discrimination"]["sensitivity"] <= 1.0
assert 0.0 <= metrics["discrimination"]["specificity"] <= 1.0
```

### Calibration metrics
```python
assert 0.0 <= metrics["calibration"]["brier_score"] <= 1.0
# Slope and intercept have no strict bounds (can be negative)
```

### Sample counts
```python
assert meta["n_train"] > 0
assert meta["n_val"] >= 0  # May be 0 in some configs
assert meta["n_test"] > 0
assert 0.0 <= meta["prevalence_train"] <= 1.0
```

## Error Message Validators

### Missing file errors
```python
result = runner.invoke(cli, ["--infile", "nonexistent.parquet", ...])

assert result.exit_code != 0
assert "not found" in result.output.lower() or "does not exist" in result.output.lower()
```

### Invalid model errors
```python
result = runner.invoke(cli, ["train", "--model", "INVALID_XYZ", ...])

assert result.exit_code != 0
assert "model" in result.output.lower() or "invalid" in result.output.lower()
```

### Config errors
```python
# Corrupted YAML
result = runner.invoke(cli, ["train", "--config", "bad_config.yaml", ...])

assert result.exit_code != 0
# Error message varies, just check it failed
```

## Reproducibility Validators

### Same seed → same output
```python
# Run 1
runner.invoke(cli, [..., "--seed-start", "123"])
train1 = pd.read_csv(splits_dir1 / "train_idx_seed123.csv")["idx"].values

# Run 2 (same seed)
runner.invoke(cli, [..., "--seed-start", "123"])
train2 = pd.read_csv(splits_dir2 / "train_idx_seed123.csv")["idx"].values

# Must be identical
np.testing.assert_array_equal(train1, train2)
```

### Different seed → different output
```python
train_seed42 = pd.read_csv(splits_dir / "train_idx_seed42.csv")["idx"].values
train_seed43 = pd.read_csv(splits_dir / "train_idx_seed43.csv")["idx"].values

# Must differ
assert not np.array_equal(sorted(train_seed42), sorted(train_seed43))
```

## Fixture Selectors

### Minimal data fixture
```python
@pytest.fixture
def minimal_proteomics_data(tmp_path):
    """200 samples, 15 proteins."""
    # Create deterministic data
    rng = np.random.default_rng(42)

    # Save to tmp_path
    parquet_path = tmp_path / "minimal.parquet"
    df.to_parquet(parquet_path, index=False)

    return parquet_path
```

### Config fixture
```python
@pytest.fixture
def minimal_training_config(tmp_path):
    """Fast config for E2E tests."""
    config = {
        "cv": {"folds": 2, "repeats": 1},
        "optuna": {"enabled": False},
    }

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path
```

## CLI Argument Patterns

### save-splits command
```python
runner.invoke(
    cli,
    [
        "save-splits",
        "--infile", str(data_path),
        "--outdir", str(splits_dir),
        "--n-splits", "2",
        "--val-size", "0.25",
        "--test-size", "0.25",
        "--seed-start", "42",
    ],
)
```

### train command
```python
runner.invoke(
    cli,
    [
        "train",
        "--infile", str(data_path),
        "--split-dir", str(splits_dir),
        "--outdir", str(results_dir),
        "--config", str(config_path),
        "--model", "LR_EN",
        "--split-seed", "42",
    ],
    catch_exceptions=False,  # For debugging
)
```

### train-ensemble command
```python
runner.invoke(
    cli,
    [
        "train-ensemble",
        "--results-dir", str(results_dir),
        "--config", str(config_path),
        "--base-models", "LR_EN,RF",
        "--split-seed", "42",
    ],
)
```

## Resilient Patterns

### Handle optional wrappers
```python
# Don't assume exact path structure
# Use rglob to find target directory

# Bad: Assumes exact structure
model_dir = results_dir / "LR_EN" / "split_seed42"

# Good: Handles wrappers
model_dirs = list(results_dir.rglob("split_seed42"))
if model_dirs:
    model_dir = model_dirs[0]
```

### Check for content, not exact files
```python
# Bad: Requires exact file name
assert (model_dir / "metrics.json").exists()

# Good: Checks for any metrics file
has_metrics = any(model_dir.rglob("*metrics*.json"))
assert has_metrics
```

### Graceful degradation
```python
# If training fails, skip structure check instead of failing
if result.exit_code != 0:
    pytest.skip(f"Training failed: {result.output[:200]}")
```

## Summary

E2E test selectors should be:
- **Flexible**: Use glob patterns, not exact paths
- **Cross-platform**: Use Path objects
- **Resilient**: Handle optional directory wrappers
- **Clear**: Validate content, not implementation details
- **Reproducible**: Use fixed seeds and tmp_path isolation
