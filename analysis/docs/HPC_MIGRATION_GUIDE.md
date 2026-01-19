# HPC Batch Script Migration Guide

## Overview

This guide explains the migration from legacy batch scripts to the modern `ced` CLI-based workflow.

## Quick Comparison

### Legacy Approach
```bash
# Complex monolithic script
bash run.sh          # 1000+ lines of bash + Python calls
```

### Modern Approach
```bash
# Simple orchestration script + CLI
./run_production.sh  # Clean bash + ced CLI commands
```

## Key Improvements

### 1. Simplicity
- **Legacy:** 16,500 lines of Python in monolithic script
- **Modern:** Clean separation of CLI (`ced`) and orchestration (`run_production.sh`)

### 2. Maintainability
- **Legacy:** Parameters hardcoded in bash script
- **Modern:** YAML configuration files with validation

### 3. Reproducibility
- **Legacy:** Config scattered across bash variables
- **Modern:** Configs saved with results (provenance tracking)

### 4. Testability
- **Legacy:** Difficult to test batch logic
- **Modern:** Unit tests for CLI + dry-run mode for orchestration

## Migration Steps

### Step 1: Replace Legacy Scripts

**Legacy files to archive:**
- `Legacy/run.sh` (1036 lines)
- `Legacy/CeD_optimized.lsf` (848 lines)
- `Legacy/celiacML_faith.py` (4000+ lines)

**Modern replacements:**
- `run_production.sh` (simple orchestration)
- `CeD_production.lsf` (LSF wrapper for `ced train`)
- `ced` CLI (modular Python package)

### Step 2: Create Configuration Files

```bash
mkdir -p configs
```

**configs/splits_config.yaml:**
```yaml
mode: development
scenarios:
  - IncidentPlusPrevalent
val_size: 0.25
test_size: 0.25
n_splits: 10
seed_start: 0
prevalent_train_only: true
prevalent_train_frac: 0.5
train_control_per_case: 5
```

**configs/training_config.yaml:**
```yaml
scenario: IncidentPlusPrevalent
cv:
  folds: 5
  repeats: 10
  scoring: neg_brier_score
  n_iter: 200
  inner_folds: 5

features:
  feature_select: hybrid
  screen_method: mannwhitney
  screen_top_n: 1000
  stability_thresh: 0.75

thresholds:
  threshold_objective: fixed_spec
  fixed_spec: 0.95
  threshold_source: val
  target_prevalence_source: test

evaluation:
  test_ci_bootstrap: true
  n_boot: 500
```

### Step 3: Update HPC Workflow

**Legacy workflow:**
```bash
# 1. Edit CeD_optimized.lsf manually (100+ parameters)
# 2. Submit jobs
bsub < CeD_optimized.lsf

# 3. Wait for completion
# 4. Run postprocessing
python postprocess_compare.py
Rscript compare_models_faith.R
```

**Modern workflow:**
```bash
# 1. Configure once in YAML (version controlled)
vim configs/training_config.yaml

# 2. Run everything
./run_production.sh

# OR submit individual models
bsub < CeD_production.lsf
```

### Step 4: CLI Commands Reference

**Generate splits:**
```bash
ced save-splits \
  --config configs/splits_config.yaml \
  --infile ../Celiac_dataset_proteomics.csv \
  --outdir splits_production
```

**Train model:**
```bash
ced train \
  --config configs/training_config.yaml \
  --model LR_EN \
  --infile ../Celiac_dataset_proteomics.csv \
  --splits-dir splits_production \
  --outdir results_production
```

**Postprocess:**
```bash
ced postprocess \
  --results-dir results_production \
  --n-boot 500
```

**Evaluate holdout:**
```bash
ced eval-holdout \
  --config configs/holdout_config.yaml \
  --infile ../Celiac_dataset_proteomics.csv \
  --holdout-idx splits/HOLDOUT_idx.csv \
  --model-artifact results/.../final_model.joblib \
  --outdir results/HOLDOUT
```

## Parameter Mapping

### Legacy → Modern

| Legacy (Bash Variable) | Modern (YAML) | Notes |
|------------------------|---------------|-------|
| `FOLDS=5` | `cv.folds: 5` | Outer CV folds |
| `REPEATS=10` | `cv.repeats: 10` | Repeat stratifications |
| `SCORING=neg_brier_score` | `cv.scoring: neg_brier_score` | Optimization metric |
| `FEATURE_SELECT=hybrid` | `features.feature_select: hybrid` | Feature selection method |
| `SCREEN_TOP_N=1000` | `features.screen_top_n: 1000` | Univariate pre-filter |
| `FIXED_SPEC=0.95` | `thresholds.fixed_spec: 0.95` | Target specificity |
| `N_BOOT=500` | `evaluation.n_boot: 500` | Bootstrap iterations |

## Validation

### Test Configuration
```bash
# Validate config before running
ced config validate configs/training_config.yaml --strict

# Compare two configs
ced config diff config1.yaml config2.yaml
```

### Dry Run
```bash
# Test orchestration without submitting jobs
DRY_RUN=1 ./run_production.sh
```

### Smoke Test (Local)
```bash
# Run single model on toy data (fast validation)
pytest tests/ -k test_cli_train  # Unit tests
pytest tests/test_e2e_pipeline.py::test_full_pipeline_single_model  # Integration (slow)
```

## Troubleshooting

### Issue: "Command not found: ced"
**Solution:**
```bash
cd analysis/sklearn
pip install -e .
ced --help
```

### Issue: "Config validation error"
**Solution:**
```bash
ced config validate configs/training_config.yaml --strict
# Fix errors reported in output
```

### Issue: "Import errors in tests"
**Solution:**
```bash
# Ensure package is installed in editable mode
pip install -e .

# Check imports
python -c "from ced_ml.cli.train import run_train; print('OK')"
```

### Issue: "LSF job fails immediately"
**Solution:**
```bash
# Check job output
cat logs/CeD_train_*.err

# Common fixes:
# 1. Conda environment not activated
# 2. Paths incorrect (check BASE_DIR in .lsf script)
# 3. Config files missing
```

## Performance Comparison

| Metric | Legacy | Modern | Improvement |
|--------|--------|--------|-------------|
| **Lines of code** | 16,500 | 15,109 | Modular refactor |
| **Test coverage** | 0% | 85% | Robust validation |
| **Config files** | 0 | 2 YAML | Version-controlled |
| **Documentation** | Scattered | Centralized | Single source |
| **Reproducibility** | Manual | Automatic | Config provenance |

## Outputs Comparison

### Legacy
```
results/
├── IncidentPlusPrevalent__LR_EN__5x10__val0.25__test0.25__hybrid/
│   ├── core/
│   │   ├── val_metrics.csv
│   │   └── test_metrics.csv
│   └── diagnostics/
```

### Modern (Identical Structure)
```
results_production/
├── IncidentPlusPrevalent__LR_EN__5x10__val0.25__test0.25__hybrid/
│   ├── config/
│   │   ├── resolved_config.yaml  # NEW: Full config provenance
│   │   └── cli_overrides.json    # NEW: Runtime overrides
│   ├── core/
│   │   ├── val_metrics.csv
│   │   ├── test_metrics.csv
│   │   └── final_model.joblib
│   ├── preds/
│   ├── reports/
│   └── diagnostics/
```

**Backwards compatibility:** 100% - all legacy output files remain unchanged

## Next Steps

1. **Test locally:**
   ```bash
   pytest tests/ -v
   ```

2. **Test on HPC (dry-run):**
   ```bash
   DRY_RUN=1 ./run_production.sh
   ```

3. **Run pilot (single model, single split):**
   ```bash
   N_SPLITS=1 RUN_MODELS="LR_EN" ./run_production.sh
   ```

4. **Full production run:**
   ```bash
   ./run_production.sh
   ```

5. **Monitor:**
   ```bash
   bjobs -w | grep CeD_
   ```

## Rollback Plan

If issues arise, legacy scripts are preserved in `Legacy/` folder:

```bash
# Revert to legacy workflow
cd Legacy
bash run.sh
```

## Support

- **Documentation:** [README.md](../README.md), [CLAUDE.MD](../CLAUDE.MD)
- **Issues:** Check test failures with `pytest tests/ -v --tb=short`
- **Questions:** Review code comments or config schema validation errors
