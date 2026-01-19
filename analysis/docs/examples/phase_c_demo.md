# Phase C Demo: Configuration Management Tools

This document demonstrates the Phase C configuration management tools with real examples.

---

## 1. Config Migration: Legacy Args → YAML

### Example 1: Split Generation Config

**Legacy command:**
```bash
python save_splits.py \
  --infile ../Celiac_dataset_proteomics.csv \
  --outdir splits_production \
  --mode development \
  --scenarios IncidentPlusPrevalent \
  --n-splits 10 \
  --val-size 0.25 \
  --test-size 0.25 \
  --prevalent-train-only \
  --prevalent-train-frac 0.5 \
  --train-control-per-case 5
```

**New approach:**
```bash
# Migrate to YAML
ced config migrate --command save-splits \
  --args "--n-splits 10" \
  --args "--val-size 0.25" \
  --args "--test-size 0.25" \
  --args "--prevalent-train-only" \
  --args "--prevalent-train-frac 0.5" \
  --args "--train-control-per-case 5" \
  --args "--scenarios IncidentPlusPrevalent" \
  -o splits_config.yaml
```

**Generated YAML:**
```yaml
mode: development
scenarios:
  - IncidentPlusPrevalent
n_splits: 10
val_size: 0.25
test_size: 0.25
prevalent_train_only: true
prevalent_train_frac: 0.5
train_control_per_case: 5.0
outdir: splits
```

**Use it:**
```bash
ced save-splits --config splits_config.yaml \
  --infile ../Celiac_dataset_proteomics.csv
```

---

### Example 2: Training Config

**Legacy command:**
```bash
python celiacML_faith.py \
  --infile ../Celiac_dataset_proteomics.csv \
  --split-dir splits_production \
  --scenario IncidentPlusPrevalent \
  --model LR_EN \
  --folds 5 \
  --repeats 10 \
  --scoring neg_brier_score \
  --n-iter 200 \
  --feature-select hybrid \
  --screen-top-n 1000 \
  --stability-thresh 0.75 \
  --threshold-objective max_f1 \
  --threshold-source val \
  --target-prevalence-source test \
  --n-boot 500
```

**New approach:**
```bash
# Migrate to YAML
ced config migrate --command train \
  --args "--infile ../Celiac_dataset_proteomics.csv" \
  --args "--model LR_EN" \
  --args "--scenario IncidentPlusPrevalent" \
  --args "--folds 5" \
  --args "--repeats 10" \
  --args "--scoring neg_brier_score" \
  --args "--n-iter 200" \
  --args "--feature-select hybrid" \
  --args "--screen-top-n 1000" \
  --args "--stability-thresh 0.75" \
  --args "--threshold-objective max_f1" \
  --args "--threshold-source val" \
  --args "--target-prevalence-source test" \
  --args "--n-boot 500" \
  -o training_config.yaml
```

**Generated YAML:**
```yaml
infile: ../Celiac_dataset_proteomics.csv
model: LR_EN
scenario: IncidentPlusPrevalent

cv:
  folds: 5
  repeats: 10
  scoring: neg_brier_score
  n_iter: 200

features:
  feature_select: hybrid
  screen_top_n: 1000
  stability_thresh: 0.75

thresholds:
  objective: max_f1
  threshold_source: val
  target_prevalence_source: test

evaluation:
  n_boot: 500
```

**Use it:**
```bash
ced train --config training_config.yaml
```

---

## 2. Config Validation

### Validate a Config File

```bash
# Basic validation
ced config validate splits_config.yaml --command save-splits

# Strict mode (warnings = errors)
ced config validate training_config.yaml --command train --strict
```

**Example output (with warning):**
```
================================================================================
Validation Report: splits_config.yaml
================================================================================

WARNINGS (1):
  - prevalent_train_only=True but prevalent_train_frac=0.0. No prevalent cases will be included.

✓ Config is valid
================================================================================
```

**Example output (with error):**
```
================================================================================
Validation Report: bad_config.yaml
================================================================================

ERRORS (1):
  - val_size (0.6) + test_size (0.6) >= 1.0. No data left for training.

✗ Config is invalid
================================================================================
```

---

## 3. Config Diff

### Compare Two Configs

```bash
ced config diff configs/baseline.yaml configs/experiment_001.yaml
```

**Example output:**
```
================================================================================
Config Diff: baseline.yaml vs experiment_001.yaml
================================================================================

Different values:
  cv.folds:
    baseline.yaml: 5
    experiment_001.yaml: 10
  cv.repeats:
    baseline.yaml: 10
    experiment_001.yaml: 20
  features.screen_top_n:
    baseline.yaml: 1000
    experiment_001.yaml: 2000

Total differences: 3
================================================================================
```

**Save diff to file:**
```bash
ced config diff baseline.yaml experiment_001.yaml -o diff_report.txt
```

---

## 4. End-to-End Workflow

### Scenario: Migrating Existing LSF Script to YAML

**Step 1: Extract legacy args from LSF script**
```bash
cat CeD_optimized.lsf | grep "celiacML_faith.py" | \
  sed 's/.*celiacML_faith.py//' > legacy_args.txt
```

**Step 2: Migrate to YAML**
```bash
ced config migrate --input-file legacy_args.txt --command train -o prod_config.yaml
```

**Step 3: Validate**
```bash
ced config validate prod_config.yaml --command train --strict
```

**Step 4: Compare with baseline**
```bash
ced config diff docs/examples/training_config.yaml prod_config.yaml
```

**Step 5: Run training with new config**
```bash
ced train --config prod_config.yaml
```

---

## 5. Reproducibility Use Case

### Track Experiments with Version-Controlled Configs

```bash
# Create experiment configs
mkdir -p configs/experiments

# Baseline config
ced config migrate --command train \
  --args "--infile data.csv" \
  --args "--model LR_EN" \
  --args "--folds 5" \
  --args "--repeats 10" \
  --args "--scoring neg_brier_score" \
  -o configs/experiments/baseline.yaml

# Experiment 1: More folds
ced config migrate --command train \
  --args "--infile data.csv" \
  --args "--model LR_EN" \
  --args "--folds 10" \
  --args "--repeats 10" \
  --args "--scoring neg_brier_score" \
  -o configs/experiments/exp001_more_folds.yaml

# Experiment 2: Different scoring
ced config migrate --command train \
  --args "--infile data.csv" \
  --args "--model LR_EN" \
  --args "--folds 5" \
  --args "--repeats 10" \
  --args "--scoring roc_auc" \
  -o configs/experiments/exp002_roc_scoring.yaml

# Compare experiments
ced config diff configs/experiments/baseline.yaml configs/experiments/exp001_more_folds.yaml
ced config diff configs/experiments/baseline.yaml configs/experiments/exp002_roc_scoring.yaml

# Version control
git add configs/experiments/*.yaml
git commit -m "feat: add experiment configs for CV and scoring comparison"
```

---

## 6. CI/CD Integration

### GitHub Actions Example

```yaml
# .github/workflows/validate-configs.yml
name: Validate Configs

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install package
        run: pip install -e .

      - name: Validate all configs
        run: |
          for config in configs/**/*.yaml; do
            ced config validate "$config" --strict || exit 1
          done
```

---

## Summary

Phase C provides three essential tools for configuration management:

1. **`ced config migrate`** - One-time migration from legacy CLI args to YAML
2. **`ced config validate`** - Continuous validation for safety and correctness
3. **`ced config diff`** - Systematic comparison for experiment tracking

**Benefits:**
- **Reproducibility:** YAML configs can be version-controlled and shared
- **Safety:** Validation catches errors before expensive training runs
- **Transparency:** Diff tool enables systematic experiment comparison
- **Efficiency:** Migration automates conversion from legacy scripts

**Next Steps:**
- Proceed to Phase D: Extract library modules (split generation, feature selection, model training)
