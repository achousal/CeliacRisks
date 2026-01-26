# Production Run Guide

**Generated**: 2026-01-26
**Recommended Config Files**:
- `configs/training_config_production.yaml`
- `configs/splits_config_production.yaml`

---

## Executive Summary

Based on analysis of run_20260125_122411 (3 splits, 20 Optuna trials, exploratory settings):

**Current Best Performance (Test Set)**:
- **ENSEMBLE**: AUROC 0.886, PR-AUC 0.747, Sensitivity@95%Spec = 67.0%
- **LR_EN**: AUROC 0.877, PR-AUC 0.732, Sensitivity@95%Spec = 64.2%
- **RF**: AUROC 0.867, PR-AUC 0.742, Sensitivity@95%Spec = 63.3%
- **LinSVM_cal**: AUROC 0.864, PR-AUC 0.732, Sensitivity@95%Spec = 63.3%
- **XGBoost**: AUROC 0.823, PR-AUC 0.686, Sensitivity@95%Spec = 54.1%

**Critical Finding**: XGBoost achieved **AUROC 0.894** in inner CV (best across all models) but only **0.823** on test set. This -7.1% gap indicates **overfitting, NOT a bad model**. Proper regularization and more trials can unlock its potential.

---

## Why XGBoost Is Overfitting (Root Cause Analysis)

### Evidence
1. **Inner CV**: AUROC 0.894 (best of all models)
2. **Test Set**: AUROC 0.823 (worst of all models)
3. **Gap**: -7.1% performance loss from CV to test

### Best Trial Hyperparameters (Split 0)
```python
learning_rate: 0.0146      # Too fast for deep trees
n_estimators: 1438         # Many boosting rounds
max_depth: 10              # Very deep trees
min_child_weight: 2.95     # Weak constraint
reg_lambda: 0.18           # Low L2 regularization
reg_alpha: 0.014           # Minimal L1 regularization
gamma: 0.81                # Low split penalty
```

### Why This Configuration Overfits
- **Deep trees (10) + many rounds (1438)** → memorizes training patterns
- **Fast learning (0.015) + weak regularization** → overshoots optimal
- **Low min_child_weight (2.95)** → fits noise in small subgroups
- **Hyperband pruner** may have killed conservative trials (low depth, high reg) early
- **Only 20 trials** insufficient to find optimal regularization balance

### Solution
Constrain XGBoost search space + 150 trials + median pruner to force conservative hyperparameters and avoid premature pruning of regularized trials.

---

## HPC Execution Plan

### Step 1: Create Production Config Files (10 min)

Create `configs/training_config_production.yaml`:
```yaml
cv:
  folds: 5              # Up from 3 (better CV estimates)
  repeats: 3            # Up from 2 (more stable selection)
  inner_folds: 3

optuna:
  enabled: true
  n_trials: 150         # Up from 20 (XGBoost needs more search!)
  sampler: tpe
  pruner: median        # More conservative than hyperband
  pruner_n_startup_trials: 20  # Up from 5 (protect early trials)

features:
  screen_method: mannwhitney
  screen_top_n: 1200    # Modest increase
  kbest_max: 1000       # Allow more features
  k_grid: [50, 100, 150, 200, 300, 400, 500]
  stability_thresh: 0.75  # Up from 0.70 (stricter stability)
  corr_thresh: 0.85

calibration:
  enabled: true
  strategy: oof_posthoc  # Unbiased calibration
  method: isotonic       # Or sigmoid for comparison

ensemble:
  enabled: true
  method: stacking
  base_models: [LR_EN, RF, XGBoost, LinSVM_cal]  # KEEP XGBoost - it has potential!
  meta_model:
    type: logistic_regression
    penalty: l2
    C: 1.0

evaluation:
  n_boot: 1000          # Up from 100 (tight 95% CIs)

# XGBoost Anti-Overfitting Settings
lr:
  optuna_C: [1.0e-4, 50.0]
  optuna_l1_ratio: [0.0, 1.0]
  max_iter: 15000

rf:
  optuna_n_estimators: [200, 1000]
  optuna_max_depth: [10, 70]
  optuna_min_samples_leaf: [1, 10]

xgboost:
  # Force conservative hyperparameters
  optuna_max_depth: [3, 7]              # Shallow trees (was 3-10)
  optuna_learning_rate: [0.005, 0.05]   # Slow learning (was 0.005-0.3)
  optuna_min_child_weight: [5.0, 30.0]  # Conservative (was 0.1-20.0)
  optuna_n_estimators: [500, 2500]      # More rounds to compensate

  # Strong regularization
  optuna_reg_lambda: [1.0, 100.0]       # L2 reg (was 0.01-50.0)
  optuna_reg_alpha: [0.01, 10.0]        # L1 reg (was 1e-8-10.0)
  optuna_gamma: [0.5, 5.0]              # High split cost (was 0.0-2.0)

  # Subsampling for regularization
  optuna_subsample: [0.5, 0.8]          # Row sampling (was 0.5-1.0)
  optuna_colsample_bytree: [0.4, 0.7]   # Column sampling (was 0.3-1.0)

  tree_method: hist                      # Memory-efficient

svm:
  optuna_C: [1.0e-5, 200.0]
  max_iter: 15000
```

Create `configs/splits_config_production.yaml`:
```yaml
n_splits: 10            # Up from 3 (statistical robustness)
val_size: 0.25
test_size: 0.25
train_control_per_case: 5.0
seed_start: 0
temporal_split: false   # Set true for chronological validation
```

### Step 2: Generate Production Splits (5 min)

```bash
cd analysis/

ced save-splits \
  --config configs/splits_config_production.yaml \
  --infile ../data/Celiac_dataset_proteomics_w_demo.parquet
```

Output: `../splits/splits_*.pkl` (10 splits for production)

### Step 3: Update HPC Configuration (5 min)

Edit `configs/pipeline_hpc.yaml`:
```yaml
hpc:
  project: YOUR_ALLOCATION  # REQUIRED - update with your HPC allocation
  queue: medium             # or "long" for >12h jobs
  cores: 4                  # 8 for XGBoost
  memory: 16G               # 32G for XGBoost
  walltime: "12:00"         # 14:00 for XGBoost, 16:00 if timeouts

execution:
  models: [LR_EN, RF, XGBoost, LinSVM_cal]
  n_boot: 1000
  overwrite_splits: false
```

### Step 4: Dry Run & Submit Jobs (10 min)

```bash
# Edit run_hpc.sh to use production configs
sed -i '' 's/training_config.yaml/training_config_production.yaml/g' run_hpc.sh

# Dry run to check job submissions
DRY_RUN=1 ./run_hpc.sh

# Submit for real
./run_hpc.sh
```

### Step 5: Monitor Progress (ongoing)

```bash
# Check job status
bjobs -w | grep CeD_

# Check specific model logs
tail -f logs/train_LR_EN_seed0.log
tail -f logs/train_RF_seed0.log
tail -f logs/train_XGBoost_seed0.log
tail -f logs/train_LinSVM_cal_seed0.log

# Expected runtime per model:
# LR_EN:       ~6-8h per split
# RF:          ~8-10h per split
# XGBoost:     ~10-14h per split (more trials, regularization)
# LinSVM_cal:  ~6-8h per split
# Total:       ~60h wall time (with parallel 4 models)
```

### Step 6: Post-Processing (2h after jobs complete)

```bash
# Wait for all jobs to finish
while bjobs | grep -q CeD_; do sleep 60; done

# Extract run ID from latest results
RUN_ID=$(ls results/LR_EN/ | grep "run_" | tail -1 | cut -d'_' -f2-3)

# Run automated aggregation pipeline
bash scripts/post_training_pipeline.sh --run-id $RUN_ID

# This generates:
# - Pooled metrics across all 10 splits
# - Bootstrap 95% CIs for AUROC, PR-AUC, sensitivity/specificity
# - Calibration plots and Brier scores
# - Feature stability and consensus panels
# - DCA curves and clinical utility analysis
```

---

## Expected Performance & Validation

### Expected Model Performance (Test Set)

| Model | Current | Expected | Improvement |
|-------|---------|----------|-------------|
| LR_EN | 0.877 | 0.88-0.89 | +0.3-1.3% |
| RF | 0.867 | 0.87-0.88 | +0.3-1.3% |
| XGBoost | 0.823 | 0.87-0.89 | **+4.7-6.7%** |
| LinSVM_cal | 0.864 | 0.87-0.88 | +0.6-1.6% |
| **ENSEMBLE** | **0.886** | **0.90-0.92** | **+1.4-3.4%** |

### Statistical Quality Targets
- **Bootstrap 95% CI width**: <0.05 for AUROC
- **Feature stability**: ≥0.75 (consensus panel)
- **Calibration**: Brier ≤0.08, slope 0.90-1.10, intercept -0.10 to +0.10
- **Clinical utility**: DCA net benefit > 0 in range [0.01, 0.10]

### Resource Requirements
- **Total CPU-hours**: ~400h (10 splits x 4 models x 10h avg)
- **Wall time**: ~60h (with parallel execution of 4 models)
- **Storage**: ~3GB per run

---

## Post-Run Validation Checklist

After aggregation completes, verify success:

### 1. XGBoost Overfitting Fixed?
```bash
cat results/XGBoost/run_*/aggregated/core/pooled_test_metrics.csv | grep AUROC
# Expected: ≥0.87 (was 0.823)
```

### 2. Ensemble Performance Improved?
```bash
cat results/ENSEMBLE/run_*/aggregated/core/pooled_test_metrics.csv | grep AUROC
# Expected: ≥0.90 (was 0.886)
```

### 3. Calibration Quality
```bash
cat results/ENSEMBLE/run_*/aggregated/diagnostics/calibration/calibration.csv
# Expected: Brier ≤0.08 (was 0.070), slope within [0.90, 1.10]
```

### 4. Feature Stability
```bash
find results/ENSEMBLE/run_*/aggregated/reports/stable_panel/ -name "*.csv" | wc -l
# Expected: ≥50 stable features at 0.75 threshold
```

---

## Success Criteria

### Minimum Acceptable (Production Release)
- ✅ XGBoost AUROC ≥ 0.87 (fixed overfitting)
- ✅ Ensemble AUROC ≥ 0.88 (current baseline)
- ✅ Sensitivity@95%Spec ≥ 65%
- ✅ Brier score ≤ 0.08
- ✅ DCA net benefit > 0 in [0.01, 0.10]

### Target Goals (Manuscript Quality)
- 🎯 Ensemble AUROC ≥ 0.90
- 🎯 Sensitivity@95%Spec ≥ 70%
- 🎯 PPV@95%Spec ≥ 80%
- 🎯 Brier score ≤ 0.07
- 🎯 Feature panel stability ≥ 0.80

---

## Troubleshooting

### XGBoost Jobs Timeout
**Symptom**: Job killed before completion (check `bjobs` output for EXIT status)

**Solution**:
```bash
# Edit configs/pipeline_hpc.yaml
walltime: "16:00"  # Increase from 12:00 to 16:00
# Or increase cores to reduce per-split time
cores: 8
```

### XGBoost Memory Issues
**Symptom**: "Out of memory" errors in logs

**Solution**:
```bash
# Update configs/pipeline_hpc.yaml for XGBoost
memory: 48G  # Increase from 32G
# Note: tree_method='hist' already set (memory-efficient)
```

### Jobs Fail with Convergence Warnings
**Symptom**: LR/SVM warnings about max iterations

**Solution**: Already set `max_iter: 15000` in production config. Check logs:
```bash
grep -i "convergence" logs/train_*.log
```

### Ensemble Training Fails
**Symptom**: "Missing OOF files" error during ensemble training

**Solution**: Validate base model outputs exist:
```bash
for model in LR_EN RF XGBoost LinSVM_cal; do
  echo "=== $model ==="
  find results/$model/run_*/split_seed*/preds/train_oof/ -name "*.csv" | wc -l
done
# Should show 10 files per model (one per split)
```

If missing, check base model logs:
```bash
tail -100 logs/train_XGBoost_seed0.log
```

---

## Risk Mitigation

| Risk | Mitigation | Monitoring |
|------|-----------|-----------|
| Jobs timeout | Start with n_trials=50, monitor first split time, adjust walltime | `bjobs -w` for EXIT status |
| XGBoost memory | Request 32-48GB, use tree_method='hist' | `sacct -j <JOBID> --format=MaxRSS` |
| Convergence failures | max_iter: 15000 for LR/SVM | `grep "convergence" logs/` |
| Ensemble fails | Validate base models first | `find results/*/run_*/split_seed*/preds/train_oof/` |

---

## Key Configuration Differences: Exploratory vs Production

| Setting | Exploratory | Production | Rationale |
|---------|------------|-----------|-----------|
| n_splits | 3 | 10 | Statistical robustness |
| CV folds | 3 | 5 | Standard biomarker study |
| CV repeats | 2 | 3 | Reduce split randomness |
| Optuna trials | 20 | 150 | Find optimal regularization |
| Optuna pruner | hyperband | median | Avoid killing conservative trials |
| n_boot | 100 | 1000 | Tight 95% confidence intervals |
| XGBoost max_depth | [3, 10] | [3, 7] | Shallow trees, less overfitting |
| XGBoost reg_lambda | [0.01, 50.0] | [1.0, 100.0] | Strong L2 regularization |
| Feature k_grid | [25, 50, ..., 400] | [50, 100, ..., 500] | Mid-range features optimal |

---

## Model-Specific Insights

### LR_EN (Logistic Regression - Elastic Net)
- **Current**: Top single model (AUROC 0.877)
- **Action**: Keep current settings, will see modest gains from more trials
- **Expected**: 0.88-0.89 AUROC

### RF (Random Forest)
- **Current**: Strong performer (AUROC 0.867), good calibration
- **Action**: Explore deeper trees and more estimators
- **Expected**: 0.87-0.88 AUROC

### XGBoost
- **Current**: PARADOX - Best inner CV (0.894) but worst test (0.823)
- **Diagnosis**: Severe overfitting despite good CV performance
- **Action**: Aggressive regularization + 150 trials + median pruner
- **Expected**: 0.87-0.89 AUROC (±6% closer to inner CV performance)

### LinSVM_cal
- **Current**: Solid performer (AUROC 0.864)
- **Action**: Minimal changes, widen C range slightly
- **Expected**: 0.87-0.88 AUROC

### Ensemble
- **Current**: AUROC 0.886 (best across all models)
- **Improvement**: +0.9% over best single model (LR_EN)
- **Expected**: 0.90-0.92 AUROC with regularized XGBoost

---

## Next Steps After Success

### Immediate (after production run completes)
1. **Validate** XGBoost AUROC ≥ 0.87 (overfitting fixed)
2. **Confirm** Ensemble AUROC ≥ 0.90 (success threshold)
3. **Check** calibration and feature stability

### Phase 2: Robustness Validation
1. **Temporal Validation**
   - Enable `temporal_split: true` in splits_config
   - Use chronological train/val/test splits
   - Check for dataset drift over time

2. **Subgroup Analysis**
   - Performance by ethnicity
   - Performance by age/BMI quartiles
   - Performance by sex

### Phase 3: Clinical Deployment
1. **Feature Interpretation**
   - Top 50 stable features
   - Literature comparison (known CeD biomarkers)
   - Biology validation

2. **Clinical Utility Optimization**
   - Cost-effectiveness analysis
   - Threshold optimization for clinical workflows
   - Decision curve analysis at different prevalence assumptions

### Phase 4: Manuscript Preparation
1. Generate publication-quality figures
2. Write methods section with reproducibility details
3. Prepare supplementary tables and panels

---

## Summary

**Recommendation**: Proceed with production run using these settings.

**Critical insight**: XGBoost is **not** a bad model. It achieved **0.894 AUROC in inner CV** (best of all models) but overfit to **0.823 on test**. With proper regularization (constrained search space) and more trials (150 vs 20), XGBoost could become the **top-performing single model** and significantly boost ensemble performance to **0.90+**.

**Timeline**: ~60h wall time on HPC with 4 parallel models.

**Expected outcomes**:
- XGBoost: 0.87-0.89 AUROC (up from 0.823)
- Ensemble: 0.90-0.92 AUROC (up from 0.886)
- XGBoost potential to match or exceed LR_EN as best single model

**Decision rule** after run completes:
- If XGBoost ≥ 0.87: Success - regularization worked, proceed to manuscript
- If Ensemble ≥ 0.90: Success threshold met
- If XGBoost still < 0.85: Consider dropping from ensemble, investigate further

---

**Generated**: 2026-01-26
**Based on**: run_20260125_122411 analysis
**Author**: Claude (analysis + consolidation)
