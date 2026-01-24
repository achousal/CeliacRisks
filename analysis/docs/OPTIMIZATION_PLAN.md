# ML Pipeline Optimization Plan

**Date**: 2026-01-23
**Baseline Run**: `run_20260122_233730` (exploratory, 3 Optuna trials, 2 splits)
**Target**: Maximize test AUROC and clinical utility (DCA net benefit) for incident celiac disease risk prediction



## 3. Optimization Strategy

### 3.1 Level 2: Quick Lock-In (Local/Medium HPC)

**Goal**: Identify best hyperparameter regions. Confirm model ranking. Stabilize feature selection.

| Parameter            | Exploratory | Level 2     | Rationale                                    |
|---------------------|-------------|-------------|----------------------------------------------|
| Optuna n_trials     | 3           | 30          | TPE convergence threshold                    |
| n_splits            | 2           | 5           | Reliable ranking (SE < 0.01)                 |
| CV repeats          | 10          | 5           | Halve compute, adequate for ranking          |
| n_boot              | 500         | 200         | Adequate for model selection                 |
| Learning curves     | enabled     | disabled    | Save compute                                 |
| LR C range          | [0.001, 1]  | [1e-5, 100] | Explore both sides of regularization         |
| LR l1_ratio         | grid 0-0.5  | cont [0,1]  | Full ElasticNet spectrum                     |
| SVM C range         | [0.001, 1]  | [1e-4, 100] | Widen for potential underregularization      |
| RF Optuna ranges    | disabled    | enabled     | Continuous search in all dimensions          |
| XGBoost Optuna      | disabled    | enabled     | Continuous log-scale search                  |
| Stability threshold | 0.75        | 0.70        | Slightly relaxed, more features pass through |

**Config**: `configs/training_config_L2.yaml`
**Pipeline**: `configs/pipeline_local.yaml` (already pointing to L2)

**Run command**:
```bash
cd analysis/
OVERWRITE_SPLITS=1 ./run_local.sh
```

**Expected outcomes**:
- Feature selection converges (k std/mean < 30%)
- OOF-to-test gap narrows to < 0.03 for all models
- Clear model ranking with error bars
- Identify if XGBoost overtakes RF with proper tuning

### 3.1.5 Level 2.5: Intermediate Lock-In (HPC Medium Queue)

**Goal**: Increased trial budget for hyperparameter convergence. Confirm model ranking with more splits.

| Parameter            | Level 2     | Level 2.5   | Rationale                                    |
|---------------------|-------------|-------------|----------------------------------------------|
| Optuna n_trials     | 30          | 50          | +67% search, improved convergence            |
| n_splits            | 5           | 8           | More robust ranking (SE < 0.008)            |
| CV repeats          | 5           | 5           | Keep (adequate for ranking)                  |
| n_boot              | 200         | 200         | Keep (adequate for intermediate)             |
| Walltime            | 12:00       | 5:00        | RF bottleneck ~4.2hr with 50 trials          |
| Queue               | premium     | medium      | 5hr fits medium queue                        |

**Config**: `configs/training_config_L2.5.yaml`, `configs/pipeline_hpc_L2.5.yaml`

**Run command**:
```bash
cd analysis/
OVERWRITE_SPLITS=1 PIPELINE_CONFIG=configs/pipeline_hpc_L2.5.yaml ./run_hpc.sh
# After jobs complete:
bash scripts/post_training_pipeline.sh --run-id <RUN_ID>
```

**Expected outcomes**:
- Feature selection k converges (std/mean < 40%)
- Linear model C parameters stabilize
- Clear model ranking with tighter error bars (SE < 0.008)
- Identify if RF dominance persists with more trials

### 3.2 Level 3: Production (Overnight HPC) - UPDATED FOR FULL RESOURCE RUN

**Goal**: Definitive model selection. Publication-ready results. Maximum statistical power for overnight execution.

**Strategy**: Maximize splits and CV power for robust model ranking and publication-quality statistical inference. Use full Optuna budget with refined ranges based on recent multi-objective runs.

| Parameter            | Level 2.5   | Level 3 (Overnight) | Rationale                                    |
|---------------------|-------------|---------------------|----------------------------------------------|
| Optuna n_trials     | 50          | 100                 | Full Bayesian convergence                    |
| Multi-objective     | enabled     | disabled            | Single AUROC focus for speed (can add later)|
| n_splits            | 8           | 10                  | SE < 0.006, detect 1% AUROC differences     |
| CV outer folds      | 5           | 5                   | Keep (standard nested CV)                   |
| CV repeats          | 5           | 10                  | Maximum OOF precision (SE < 0.005)          |
| CV inner folds      | 3           | 5                   | Restore full inner tuning loop              |
| n_boot              | 200         | 1000                | Publication-quality bootstrap CIs           |
| Learning curves     | disabled    | enabled             | Publication figure + sample efficiency      |
| Queue               | medium      | premium             | Priority access for overnight run           |
| Walltime            | 5:00        | 48:00               | Full headroom for 100 trials x 50 folds     |
| Cores per job       | 4           | 8                   | Maximum parallelism per model               |
| Memory per core     | 4000 MB     | 8000 MB             | Prevent OOM with large feature matrices     |

**Total compute budget**:
- 10 splits × 4 models × 100 trials × 50 folds (5 outer × 10 repeats) = ~200,000 model fits
- Estimated runtime: 24-36 hours on premium queue with 8 cores
- Total core-hours: ~10,000-15,000 (well within overnight budget)

**Config files**:
- Training: `configs/training_config_production.yaml` (already configured)
- Pipeline: `configs/pipeline_hpc.yaml` (already configured)
- Splits: `configs/splits_config.yaml` (needs update to n_splits: 10)

**Pre-launch checklist**:

1. **Update splits config** for 10 splits:
   ```bash
   # Edit configs/splits_config.yaml
   n_splits: 10  # Up from 2
   ```

2. **Verify HPC allocation** in `configs/pipeline_hpc.yaml`:
   ```yaml
   hpc:
     project: acc_Chipuk_Laboratory  # Confirm active
     queue: premium                   # For overnight priority
     walltime: "48:00"                # 48 hours
     cores: 8                         # Max parallelism
     mem_per_core: 8000               # 8 GB per core
   ```

3. **Review current production config** (already optimal):
   - Optuna n_trials: 100
   - CV: 5 outer folds × 10 repeats × 5 inner folds = 50 total folds
   - n_boot: 1000
   - Learning curves: enabled
   - OOF-posthoc calibration: enabled (unbiased)
   - All evaluation metrics: enabled

4. **Optional refinements** based on recent L2 multi-objective runs:
   - If L2 results show k convergence to narrow range: tighten k_grid
   - If LR l1_ratio consistently > 0.7: add LR_L1 model
   - If LinSVM_cal clearly worst: consider dropping (keep for comparison)
   - If XGBoost subsample/colsample cluster < 0.6: keep current wide ranges

5. **Data and environment checks**:
   ```bash
   # Verify data file exists
   ls -lh ../data/Celiac_dataset_proteomics_w_demo.parquet

   # Verify HPC setup
   bash scripts/hpc_setup.sh
   source venv/bin/activate
   ced --version
   ```

**Launch commands**:

```bash
cd analysis/

# 1. Generate fresh splits (10 splits, seed 0-9)
OVERWRITE_SPLITS=1 ced save-splits \
  --config configs/splits_config.yaml \
  --infile ../data/Celiac_dataset_proteomics_w_demo.parquet

# 2. Optional: Dry run to verify job submission
DRY_RUN=1 ./run_hpc.sh

# 3. Launch production run
./run_hpc.sh
```

**Expected job array structure**:
- 40 jobs total: 10 splits × 4 models (LR_EN, LinSVM_cal, RF, XGBoost)
- Job names: `CeD_LR_EN_seed0`, `CeD_LR_EN_seed1`, ..., `CeD_XGBoost_seed9`
- Logs: `logs/{RUN_ID}/CeD_{MODEL}_seed{N}.{out,err}`

**Monitoring during execution**:

```bash
# Check job queue status
bjobs -w | grep CeD_

# Monitor a specific job's progress
tail -f logs/{RUN_ID}/CeD_LR_EN_seed0.out

# Check resource utilization
bjobs -l <JOB_ID> | grep -E "(RUNLIMIT|MEMLIMIT)"

# Count completed jobs
bjobs -w | grep CeD_ | grep DONE | wc -l
```

**Post-training workflow** (after all jobs DONE):

```bash
# 1. Validate base model outputs
bash scripts/post_training_pipeline.sh --run-id <RUN_ID>

# This will:
# - Validate all 40 model outputs (10 splits × 4 models)
# - Report missing/incomplete jobs
# - Aggregate results across splits for each model
# - Generate summary reports

# 2. Optional: Train ensemble meta-learner
bash scripts/post_training_pipeline.sh \
  --run-id <RUN_ID> \
  --train-ensemble

# This adds:
# - ENSEMBLE model trained on OOF predictions
# - Aggregated ensemble results
# - Expected +2-5% AUROC improvement

# 3. Check aggregation outputs
ls -la ../results/{LR_EN,RF,XGBoost,LinSVM_cal}/run_{RUN_ID}/aggregated/

# Key files per model:
# - aggregated_metrics.json     # Pooled metrics across splits
# - aggregated_metrics.csv      # Same as table
# - feature_stability.csv       # Feature selection consistency
# - calibration_aggregated.png  # Calibration plot
# - roc_aggregated.png          # ROC curves
# - dca_aggregated.png          # Decision curve analysis
```

**Success criteria**:

1. **All 40 jobs complete** without errors
2. **OOF-test gap < 0.02** for all models (minimal overfitting)
3. **Feature selection k converges** (std/mean < 30% across splits)
4. **Clear model ranking** with non-overlapping 95% CIs
5. **Test AUROC ≥ 0.85** for best model (realistic ceiling: 0.85-0.88)
6. **Calibration slope 0.9-1.1** (well-calibrated with OOF-posthoc)
7. **DCA net benefit > treat-all** in clinically relevant threshold range

**Expected outcomes**:

- **Discrimination**: Test AUROC 0.85-0.88 (best model), SE < 0.006
- **Calibration**: Brier score < 0.015, calibration slope 0.95-1.05
- **Clinical utility**: Positive net benefit at 1-10% threshold range
- **Feature panel**: 50-300 stable proteins (≥75% selection frequency)
- **Model ranking**: Clear winner with statistical significance (p < 0.05)
- **Ensemble boost**: +2-5% AUROC improvement over best single model

**Fallback plan** if jobs fail:

1. **Check error logs**: `cat logs/{RUN_ID}/CeD_{MODEL}_seed{N}.err`
2. **Common issues**:
   - OOM errors: Reduce cores to 4, increase mem_per_core to 16000
   - Timeout: Increase walltime to 72:00 or reduce n_trials to 50
   - Split file missing: Re-run `ced save-splits` with OVERWRITE_SPLITS=1
3. **Re-run failed jobs only**:
   ```bash
   # Re-submit specific split seeds
   ced train --model LR_EN --split-seed 3 --config configs/training_config_production.yaml
   ```

---

## 3.3 Level 3 Resource Optimization Summary

**Recommended overnight configuration** (maximum resources):

```yaml
# configs/splits_config.yaml
n_splits: 10                    # 10 independent random splits

# configs/training_config_production.yaml (already set)
cv:
  folds: 5                      # Outer CV folds
  repeats: 10                   # Maximize statistical power
  inner_folds: 5                # Full inner tuning loop

optuna:
  enabled: true
  n_trials: 100                 # Full Bayesian convergence
  multi_objective: false        # Disable for speed (single AUROC)

evaluation:
  n_boot: 1000                  # Publication-quality CIs
  learning_curve: true          # Generate learning curves

# configs/pipeline_hpc.yaml (already set)
hpc:
  queue: premium                # Priority overnight queue
  walltime: "48:00"             # 48-hour safety margin
  cores: 8                      # Maximum parallelism
  mem_per_core: 8000            # 8 GB (prevent OOM)
```

**Expected runtime breakdown** (per split, per model):

| Model      | Trials | Folds | Avg Time/Trial | Total Time | Notes                     |
|------------|--------|-------|----------------|------------|---------------------------|
| LR_EN      | 100    | 50    | 2 min          | 3.3 hr     | Fast linear model         |
| LinSVM_cal | 100    | 50    | 3 min          | 5.0 hr     | Calibration overhead      |
| RF         | 100    | 50    | 8 min          | 13.3 hr    | Bottleneck (tree building)|
| XGBoost    | 100    | 50    | 5 min          | 8.3 hr     | GPU-accelerated possible  |

**Critical path**: RF with 13.3 hours per split × 10 splits = **133 hours sequential**

**With parallel job array**: ~13.3 hours (all splits run simultaneously)

**Total wall-clock time**: ~14-16 hours (including overhead, post-processing)

**Resource efficiency**:
- Core-hours: 10 splits × 4 models × 8 cores × 14 hrs = **4,480 core-hours**
- Well within overnight budget on premium queue
- Cost-effective: ~$200-300 at typical HPC rates

**Quick pre-flight checklist**:

```bash
# 1. Update splits config
sed -i.bak 's/n_splits: 2/n_splits: 10/' configs/splits_config.yaml

# 2. Verify HPC config points to production
grep "training_config_production" configs/pipeline_hpc.yaml

# 3. Verify allocation and queue
grep -E "(project|queue|walltime)" configs/pipeline_hpc.yaml

# 4. Check data file size
du -h ../data/Celiac_dataset_proteomics_w_demo.parquet

# 5. Estimate job count
echo "Expected jobs: 10 splits × 4 models = 40 jobs"

# 6. Launch
OVERWRITE_SPLITS=1 ./run_hpc.sh
```

---

## 4. Statistical Justification

### 4.1 Trial Count Selection
- TPE builds a kernel density estimator over the objective surface
- With d=3-10 hyperparameters, 30 trials provides ~5x coverage per dimension (sufficient for basin identification)
- 100 trials enables exploitation within the basin (diminishing returns beyond ~100 for smooth objectives)
- Reference: Bergstra et al. (2011) show TPE outperforms random search at ~25+ evaluations

### 4.2 Split Count Selection
- SE of mean test AUROC = SD / sqrt(n_splits)
- With SD ~ 0.02 (estimated from exploratory): 5 splits -> SE = 0.009; 10 splits -> SE = 0.006
- To distinguish models separated by delta=0.02 at 80% power: need SE < delta/2.8 = 0.007 -> 10 splits

### 4.3 Expected Performance Ceiling
- OOF AUROC ceiling: ~0.87-0.88 (consistent across models, likely data-driven limit)
- True prevalence: 0.34%, downsampled test: 16.7%
- With ~2920 proteins and likely 20-50 truly informative features:
  - Realistic test AUROC target: **0.85-0.88** with proper tuning
  - OOF-test gap target: < 0.02

### 4.4 Overfitting Risk Mitigation
- OOF-posthoc calibration (already in place)
- Test set truly held out (never seen during tuning)
- Hyperband pruning kills bad trials early (reduces total compute)
- Stability selection regularizes feature count
- Multiple random splits average out split-specific overfitting

---

## 5. Key Optuna Range Decisions

### 5.1 LR_EN: C = [1e-5, 100]
- Exploratory found C ~ 0.02-0.06 (heavy regularization)
- Widening upward tests if the model is over-regularized
- Widening downward explores sparser solutions
- l1_ratio [0, 1] covers full ridge-to-lasso spectrum

### 5.2 XGBoost: colsample_bytree = [0.3, 1.0]
- Exploratory found colsample = 0.57 (aggressive column subsampling)
- Lower bound 0.3 allows even more regularization via feature subsetting
- Upper bound 1.0 tests no subsampling
- This is critical for high-dimensional data where feature correlation is high

### 5.3 RF: max_features = [0.05, 0.8]
- Standard sqrt(p) ~ 0.018 for 2920 features (very small)
- Allowing up to 0.8 tests less aggressive decorrelation
- Lower bound 0.05 allows very sparse tree building
- This directly controls bias-variance tradeoff in ensembles

### 5.4 Feature Selection k_grid
- Added k=150 for resolution between 100 and 200
- Exploratory showed LR_EN oscillating between 50 and 400 -- need intermediate points
- With 30 trials, Optuna will sample k as part of the joint optimization
- Expect convergence to k ~ 100-300 based on OOF AUROC surface

---

## 6. Decision Points After L2.5

| Observation                              | Action                                             |
|------------------------------------------|----------------------------------------------------|
| k converges to narrow range              | Tighten k_grid around that range for L3            |
| LR l1_ratio consistently > 0.7          | Add LR_L1 as separate model for L3                 |
| LinSVM_cal always worst                  | Drop from ensemble, keep for comparison only        |
| RF clearly beats all others              | Focus L3 compute on RF + ensemble                  |
| OOF-test gap still > 0.03               | Increase stability_thresh or use more aggressive pruning |
| XGBoost subsample/colsample still < 0.6 | Confirmed: strong regularization needed, keep ranges |

---

## 7. File Map

| File                                     | Purpose                              | Status    |
|------------------------------------------|--------------------------------------|-----------|
| `configs/training_config_L2.yaml`        | Level 2 training config (30 trials)  | Created   |
| `configs/training_config_L2.5.yaml`      | Level 2.5 training config (50 trials)| Created   |
| `configs/training_config_production.yaml` | Level 3 production config (100 trials)| Created  |
| `configs/splits_config.yaml`             | n_splits=8 (update to 10 for L3)    | Modified  |
| `configs/pipeline_local.yaml`            | Points to L2 config                  | Modified  |
| `configs/pipeline_hpc.yaml`              | Points to production config          | Modified  |
| `configs/pipeline_hpc_L2.5.yaml`         | Points to L2.5 config                | Created   |
| `configs/training_config.yaml`           | Original exploratory config          | Unchanged |
| `configs/training_config_optuna.yaml`    | Previous HPC config (superseded)     | Unchanged |

---

## 8. Timeline

```
[L2 Complete] Analysis shows k not converged, need more trials
     |
     v
[L2.5 Run]    8 splits x 4 models x 50 trials x 25 folds = ~40,000 evaluations
     |        Estimated: ~4.2hr on HPC (8 cores, 5hr walltime)
     v
[L2.5 Review] Analyze convergence, refine ranges for L3
     |
     v
[L3 Launch]   10 splits x 4 models x 100 trials x 50 folds = ~200,000 evaluations
     |        Estimated: overnight on HPC (8 cores, 48h walltime)
     v
[L3 Post]     post_training_pipeline.sh -> aggregation + ensemble + reports
     |
     v
[Final]       Model selection, publication figures, feature panel
```
