# Production Run Guide

**Generated**: 2026-01-26
**Configs**: `training_config_production.yaml`, `splits_config_production.yaml`

---

## Executive Summary

Based on run_20260125_122411 (3 splits, 20 Optuna trials):

**Current Performance (Test Set)**:
- **ENSEMBLE**: AUROC 0.886, Sensitivity@95%Spec = 67.0%
- **LR_EN**: AUROC 0.877, Sensitivity@95%Spec = 64.2%
- **RF**: AUROC 0.867
- **LinSVM_cal**: AUROC 0.864
- **XGBoost**: AUROC 0.823 (BUT 0.894 in inner CV - severe overfitting)

**Key Finding**: XGBoost achieved best inner CV (0.894) but worst test (0.823). Gap of -7.1% indicates overfitting, NOT a bad model. Proper regularization can unlock its potential.

**Production Goals**:
- Fix XGBoost: target 0.87-0.89 AUROC
- Ensemble: target 0.90-0.92 AUROC
- Consensus panel: 40-80 proteins (0.70 threshold)
- Deployment panel: 20-30 proteins (RFE knee point)

---

## XGBoost Overfitting: Root Cause

**Best Trial Params (Split 0)**:
```yaml
max_depth: 10              # Too deep
learning_rate: 0.0146      # Fast learning
n_estimators: 1438         # Many rounds
reg_lambda: 0.18           # Weak L2
reg_alpha: 0.014           # Minimal L1
min_child_weight: 2.95     # Weak constraint
```

**Problem**: Deep trees + fast learning + weak regularization + hyperband pruner killed conservative trials + only 20 trials.

**Solution**: Constrain search space to force conservative hyperparams, 150 trials, median pruner.

---

## Production Workflow

### Step 1: Create Production Configs (10 min)

**`configs/training_config_production.yaml`**:
```yaml
cv:
  folds: 5              # Up from 3
  repeats: 3            # Up from 2
  inner_folds: 3

optuna:
  enabled: true
  n_trials: 150         # Up from 20
  sampler: tpe
  pruner: median        # More conservative than hyperband
  pruner_n_startup_trials: 20

features:
  feature_selection_strategy: hybrid_stability  # Fast, reproducible (30 min vs 22h for rfecv)
  screen_method: mannwhitney
  screen_top_n: 1200
  k_grid: [50, 100, 150, 200, 300, 400, 500]
  stability_thresh: 0.75      # Stricter than 0.70
  stable_corr_thresh: 0.85

calibration:
  enabled: true
  strategy: oof_posthoc       # Unbiased
  method: isotonic

ensemble:
  enabled: true
  method: stacking
  base_models: [LR_EN, RF, XGBoost, LinSVM_cal]
  meta_model:
    type: logistic_regression
    penalty: l2
    C: 1.0

evaluation:
  n_boot: 1000          # Tight 95% CIs

# XGBoost Anti-Overfitting
xgboost:
  optuna_max_depth: [3, 7]              # Shallow (was 3-10)
  optuna_learning_rate: [0.005, 0.05]   # Slow (was 0.005-0.3)
  optuna_min_child_weight: [5.0, 30.0]  # Conservative (was 0.1-20.0)
  optuna_n_estimators: [500, 2500]
  optuna_reg_lambda: [1.0, 100.0]       # Strong L2 (was 0.01-50.0)
  optuna_reg_alpha: [0.01, 10.0]        # L1 reg
  optuna_gamma: [0.5, 5.0]              # High split cost (was 0.0-2.0)
  optuna_subsample: [0.5, 0.8]
  optuna_colsample_bytree: [0.4, 0.7]
```

**`configs/splits_config_production.yaml`**:
```yaml
n_splits: 10            # Up from 3
val_size: 0.25
test_size: 0.25
train_control_per_case: 5.0
seed_start: 0
```

### Step 2: Generate Splits (5 min)
```bash
cd analysis/
ced save-splits \
  --config configs/splits_config_production.yaml \
  --infile ../data/Celiac_dataset_proteomics_w_demo.parquet
```

### Step 3: Update HPC Config (5 min)

Edit `configs/pipeline_hpc.yaml`:
```yaml
hpc:
  project: YOUR_ALLOCATION  # REQUIRED
  queue: medium
  cores: 4                  # 8 for XGBoost
  memory: 16G               # 32G for XGBoost
  walltime: "12:00"         # 14:00 for XGBoost

execution:
  models: [LR_EN, RF, XGBoost, LinSVM_cal]
  n_boot: 1000
```

### Step 4: Submit Jobs (10 min)
```bash
# Edit run_hpc.sh to use production configs
sed -i '' 's/training_config.yaml/training_config_production.yaml/g' run_hpc.sh

# Dry run
DRY_RUN=1 ./run_hpc.sh

# Submit
./run_hpc.sh
```

### Step 5: Monitor (ongoing)
```bash
bjobs -w | grep CeD_
tail -f logs/train_XGBoost_seed0.log

# Expected runtime per split:
# LR_EN:       6-8h
# RF:          8-10h
# XGBoost:     10-14h (more trials)
# LinSVM_cal:  6-8h
# Total:       ~60h wall time (4 models in parallel)
```

### Step 6: Post-Processing (2h after completion)
```bash
# Wait for jobs
while bjobs | grep -q CeD_; do sleep 60; done

# Extract run ID
RUN_ID=$(ls results/LR_EN/ | grep "run_" | tail -1 | cut -d'_' -f2-3)

# Run aggregation
bash scripts/post_training_pipeline.sh --run-id $RUN_ID
```

Generates:
- Pooled metrics (10 splits, bootstrap 95% CIs)
- Calibration plots and Brier scores
- Feature stability and consensus panels
- DCA curves

---

## Panel Optimization (Post-Training)

### Step 7: Panel Size Optimization (30 min)

**Goal**: Find minimum viable panel for deployment using post-hoc RFE.

```bash
ced optimize-panel \
  --model-path results/LR_EN/split_seed0/core/LR_EN__final_model.joblib \
  --infile ../data/Celiac_dataset_proteomics_w_demo.parquet \
  --split-dir ../splits/ \
  --split-seed 0 \
  --start-size 100 --min-size 5
```

**Output**: `results/LR_EN/split_seed0/optimize_panel/`
- `panel_curve.png` - Pareto curve
- `recommended_panels.json` - Knee points
- `feature_ranking.csv` - Elimination order

**Example recommendations**:
```json
{
  "max_auroc": 0.892,
  "recommended_panels": {
    "min_size_95pct": 28,  // Maintains 95% of max AUROC
    "min_size_90pct": 14,  // Maintains 90% of max AUROC
    "knee_point": 22       // Optimal cost-benefit
  }
}
```

**Deploy at knee point (22 proteins)**: ~$110/patient vs $500 for full panel.

### Step 8: Consensus Panel Validation (1h)

**Goal**: Extract multi-split consensus and validate with unbiased AUROC.

#### 8.1: Extract Consensus
```bash
# Proteins selected in ≥70% of splits
awk -F',' 'NR==1 || $2 >= 0.70 {print $1}' \
  results/LR_EN/aggregated/feature_stability.csv \
  > deployment_panel_70pct.csv

wc -l deployment_panel_70pct.csv  # Expected: 40-80 proteins
```

| Threshold | Expected Size | Use Case |
|-----------|---------------|----------|
| 0.80 | 20-40 | Publication (high-confidence) |
| 0.70 | 40-80 | Clinical deployment |
| 0.60 | 80-120 | Conservative |

#### 8.2: Validate with NEW Split Seed (CRITICAL)

**IMPORTANT**: Use NEW seed (never seen before) for unbiased estimate.

```bash
# Generate NEW validation split
ced save-splits \
  --config configs/splits_config_production.yaml \
  --infile ../data/Celiac_dataset_proteomics_w_demo.parquet \
  --seed-start 100 --n-splits 1

# Train on fixed panel
ced train \
  --model LR_EN \
  --fixed-panel deployment_panel_70pct.csv \
  --split-seed 100 \
  --config configs/training_config_production.yaml

# Check unbiased AUROC
cat results/LR_EN/split_seed100/evaluation/test_metrics.json | grep auroc
```

**Expected**:
- Post-hoc RFE: 0.880 (slightly optimistic)
- Fixed panel: 0.875 (unbiased ground truth)
- Difference: ≤1% (acceptable)

---

## Validation Checklist

### Model Performance
```bash
# XGBoost fixed?
cat results/XGBoost/run_*/aggregated/core/pooled_test_metrics.csv | grep AUROC
# Expected: ≥0.87 (was 0.823)

# Ensemble improved?
cat results/ENSEMBLE/run_*/aggregated/core/pooled_test_metrics.csv | grep AUROC
# Expected: ≥0.90 (was 0.886)
```

### Feature Stability
```bash
# Per-split stability
for seed in {0..9}; do
  wc -l results/LR_EN/split_seed${seed}/cv/feature_selection/stability/stable_panel_t0.75.csv
done
# Expected: 30-80 proteins per split

# Multi-split consensus
awk -F',' 'NR>1 && $2 >= 0.70' results/LR_EN/aggregated/feature_stability.csv | wc -l
# Expected: 40-80 proteins
```

### Panel Optimization
```bash
cat results/LR_EN/split_seed0/optimize_panel/recommended_panels.json
# Knee point: 20-30 proteins
# AUROC at knee: ≥0.85
```

### Fixed Panel Validation
```bash
grep '"auroc"' results/LR_EN/split_seed100/evaluation/test_metrics.json
# Expected: ≥0.87 (unbiased)
```

---

## Success Criteria

### Minimum Acceptable (Production Release)
**Models**:
- ✅ XGBoost AUROC ≥ 0.87
- ✅ Ensemble AUROC ≥ 0.88
- ✅ Sensitivity@95%Spec ≥ 65%
- ✅ Brier ≤ 0.08

**Panels**:
- ✅ Consensus (0.70) ≥ 40 proteins
- ✅ Consensus (0.80) ≥ 20 proteins
- ✅ RFE knee point: 15-40 proteins
- ✅ Fixed-panel AUROC ≥ 0.87
- ✅ Post-hoc optimism ≤ 1.5%

### Target Goals (Manuscript Quality)
- 🎯 Ensemble AUROC ≥ 0.90
- 🎯 Sensitivity@95%Spec ≥ 70%
- 🎯 Deployment panel: 20-50 proteins
- 🎯 Fixed-panel AUROC ≥ 0.88
- 🎯 Panel includes ≥5 known CeD biomarkers

---

## Troubleshooting

### XGBoost Jobs Timeout
```yaml
# configs/pipeline_hpc.yaml
walltime: "16:00"  # Up from 12:00
cores: 8           # More cores
```

### XGBoost Memory Issues
```yaml
memory: 48G  # Up from 32G
# tree_method='hist' already set (memory-efficient)
```

### Stability Panel Too Small (<20 proteins)
```yaml
# training_config_production.yaml
features:
  stability_thresh: 0.60  # Down from 0.75
  k_grid: [50, 100, 150, 200, 300, 400, 500, 700, 1000]  # Wider range
```

### Stability Panel Too Large (>200 proteins)
```yaml
features:
  stability_thresh: 0.80  # Up from 0.75
  stable_corr_thresh: 0.80  # More aggressive pruning
```

### RFECV Takes Too Long
**Symptom**: Job timeout with `feature_selection_strategy: rfecv`

**Solution 1** (recommended): Use `hybrid_stability` instead (~30 min vs ~4 hours):
```yaml
features:
  feature_selection_strategy: hybrid_stability
```

**Solution 2**: Enable k-best pre-filter (default, ~5× speedup):
```yaml
features:
  feature_selection_strategy: rfecv
  rfe_kbest_prefilter: true  # Enabled by default
  rfe_kbest_k: 100           # Reduces ~300 → ~100 proteins before RFECV
hpc:
  walltime: "06:00"          # 6 hours (vs 24 hours without pre-filter)
```

**Solution 3**: Reduce CV complexity if still too slow:
```yaml
cv:
  folds: 3       # Down from 5
  repeats: 2     # Down from 3
features:
  rfe_cv_folds: 2
  rfe_kbest_k: 50  # More aggressive pre-filter
hpc:
  walltime: "04:00"
```

**Expected runtimes** (per split):
- hybrid_stability: ~30 min
- rfecv (with k-best=100): ~4-5 hours
- rfecv (with k-best=50): ~2 hours
- rfecv (no pre-filter): ~22 hours

### Panel Validation Leakage
**CRITICAL**: Always use NEW split seed (100+) for fixed-panel validation. Never reuse discovery seeds.

---

## Quick Reference

### Complete Workflow
```bash
# 1-2. Create configs and generate splits
ced save-splits --config configs/splits_config_production.yaml --infile ../data/input.parquet

# 3-4. Update HPC config and submit
./run_hpc.sh

# 5. Monitor
bjobs -w | grep CeD_

# 6. Post-process
RUN_ID=$(ls results/LR_EN/ | grep "run_" | tail -1 | cut -d'_' -f2-3)
bash scripts/post_training_pipeline.sh --run-id $RUN_ID

# 7. Panel optimization
ced optimize-panel \
  --model-path results/LR_EN/split_seed0/core/LR_EN__final_model.joblib \
  --infile ../data/input.parquet --split-dir ../splits/ --split-seed 0

# 8. Consensus extraction & validation
awk -F',' 'NR==1 || $2 >= 0.70 {print $1}' \
  results/LR_EN/aggregated/feature_stability.csv > deployment_panel_70pct.csv

ced save-splits --config configs/splits_config_production.yaml \
  --infile ../data/input.parquet --seed-start 100 --n-splits 1

ced train --model LR_EN --fixed-panel deployment_panel_70pct.csv \
  --split-seed 100 --config configs/training_config_production.yaml
```

### Decision Checklist
- ✅ Ensemble AUROC ≥ 0.90
- ✅ XGBoost AUROC ≥ 0.87
- ✅ Consensus panel ≥ 40 proteins (0.70)
- ✅ RFE knee point 20-30 proteins
- ✅ Fixed-panel AUROC ≥ 0.87
- ✅ Post-hoc optimism ≤ 1.5%

**If all pass**: Proceed to clinical deployment and manuscript preparation.

---

## Resource Requirements

**Training (Steps 1-6)**:
- CPU-hours: ~400h (10 splits × 4 models × 10h avg)
- Wall time: ~60h with parallel execution
- Add ~37 days if using `rfecv` instead of `hybrid_stability` (NOT recommended)

**Panel work (Steps 7-8)**:
- Panel optimization: +30 min (negligible)
- Panel validation: +8h (single split)
- **Total: ~68h wall time**

**Storage**: ~3GB per run

---

## Summary

**Recommendation**: Proceed with production run.

**Critical Insights**:
1. **XGBoost potential**: Best inner CV (0.894) but overfitting to test (0.823). With constrained search space + 150 trials, can reach 0.87-0.89 and boost ensemble to 0.90+.

2. **Feature selection**: Use `hybrid_stability` (30 min per split) for production. RFECV is 45× slower (22h per split) and only needed for scientific validation.

3. **Panel optimization**: Post-hoc RFE (Step 7) finds deployment panels in ~5 min. Fixed-panel validation (Step 8) provides unbiased AUROC for regulatory submission.

**Expected Outcomes**:

**Models**:
- XGBoost: 0.87-0.89 AUROC (up from 0.823)
- Ensemble: 0.90-0.92 AUROC (up from 0.886)

**Panels**:
- Consensus (0.70): 40-80 proteins
- Consensus (0.75): 30-60 proteins
- RFE knee point: 20-30 proteins
- Fixed-panel AUROC: 0.87-0.88 (unbiased)
- Cost reduction: 70-80% vs full panel

**Timeline**: ~68h total (60h training + 8h panel validation)

---

**Generated**: 2026-01-26
**Based on**: run_20260125_122411 + consolidated FEATURE_SELECTION.md
