# ML Pipeline Optimization Plan

**Date**: 2026-01-23
**Baseline Run**: `run_20260122_233730` (exploratory, 3 Optuna trials, 2 splits)
**Target**: Maximize test AUROC and clinical utility (DCA net benefit) for incident celiac disease risk prediction

---

## 1. Exploratory Run Diagnosis

### 1.1 Performance Summary (Pooled Test, n=438, prevalence=16.7%)

| Model      | Test AUROC | PR-AUC | Brier  | Sens@95Spec | DCA NB Improvement |
|------------|-----------|--------|--------|-------------|---------------------|
| RF         | 0.844     | 0.652  | 0.089  | 0.466       | +0.00092            |
| ENSEMBLE   | 0.841     | 0.690  | 0.101  | 0.575       | +0.00016            |
| XGBoost    | 0.836     | 0.689  | 0.096  | 0.493       | +0.00064            |
| LinSVM_cal | 0.821     | 0.664  | 0.090  | 0.534       | --                  |
| LR_EN      | 0.819     | 0.648  | 0.104  | 0.575       | -0.00120            |

### 1.2 Inner CV (OOF) Performance

| Model      | OOF AUROC | OOF PR-AUC | OOF-to-Test Gap |
|------------|-----------|------------|-----------------|
| XGBoost    | 0.874     | 0.694      | -0.038          |
| LR_EN      | 0.873     | 0.700      | -0.054          |
| LinSVM_cal | 0.872     | 0.695      | -0.051          |
| RF         | 0.863     | 0.677      | -0.019          |

### 1.3 Validation vs Test Consistency

| Model      | Val AUROC | Test AUROC | Gap    |
|------------|-----------|------------|--------|
| XGBoost    | 0.868     | 0.836      | -0.032 |
| RF         | 0.838     | 0.844      | +0.006 |
| ENSEMBLE   | 0.829     | 0.841      | +0.012 |
| LinSVM_cal | 0.819     | 0.821      | +0.002 |
| LR_EN      | 0.806     | 0.819      | +0.013 |

### 1.4 Hyperparameter Convergence

| Model      | Key Params (mean +/- std)                                        | Feature k (mean +/- std) |
|------------|------------------------------------------------------------------|--------------------------|
| LR_EN      | C=0.054+/-0.015, l1_ratio=0.088+/-0.116                         | 298 +/- 160              |
| LinSVM_cal | C=0.024+/-0.024                                                  | 489 +/- 360              |
| RF         | depth=28+/-4, leaf=8+/-3, n_est=392+/-33                         | 140 +/- 125              |
| XGBoost    | lr=0.027+/-0.007, depth=7.2+/-0.6, subsample=0.59, colsample=0.57| 792 +/- 75               |

---

## 2. Identified Bottlenecks (Priority Order)

### B1: Hyperparameter Search is Trivial (Critical)
- **3 Optuna trials per fold** is random sampling, not optimization
- TPE requires ~30 trials to build a useful surrogate model
- LR_EN explored only 3 distinct (C, l1_ratio) combinations across all 50 folds
- XGBoost regularization parameters may be artifacts of landing in a constrained region

### B2: Insufficient Split Seeds (High)
- Only 2 random seeds: model ranking may be due to split luck
- Cannot compute reliable inter-split variance
- Need 5+ seeds for ranking, 10 for publication-ready CIs

### B3: Feature Count Instability (High)
- LR_EN k: std/mean = 160/298 = 54% CV -- no convergence
- LinSVM_cal k: std/mean = 360/489 = 74% CV -- no convergence
- Each fold selects wildly different feature sets because hyperparameters never stabilized

### B4: Ensemble Underperformance (Medium)
- ENSEMBLE (0.841) does not beat RF (0.844) individually
- Stacking undertrained base models amplifies noise rather than combining signal
- Expected to resolve when base models are properly optimized

### B5: Large OOF-to-Test Gap for Linear Models (Medium)
- LR_EN gap: 0.054 (concerning)
- LinSVM_cal gap: 0.051 (concerning)
- RF gap: 0.019 (acceptable)
- Linear models may be overfit to inner CV or feature selection is unstable

### B6: LR_EN has Negative DCA Net Benefit (Low-Medium)
- LR_EN is WORSE than "treat all" strategy on test set
- Likely due to poor calibration from undertrained hyperparameters
- Should resolve with proper tuning and stable feature selection

---

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

### 3.2 Level 3: Production (Overnight HPC)

**Goal**: Definitive model selection. Publication-ready results. Maximum statistical power.

| Parameter            | Level 2     | Level 3     | Rationale                                    |
|---------------------|-------------|-------------|----------------------------------------------|
| Optuna n_trials     | 30          | 100         | Full Bayesian convergence                    |
| n_splits            | 5           | 10          | SE < 0.006, detect 1% differences           |
| CV repeats          | 5           | 10          | Tight OOF CIs for publication               |
| n_boot              | 200         | 1000        | Narrow bootstrap CIs                         |
| Learning curves     | disabled    | enabled     | Publication figure                           |
| Walltime            | --          | 48:00       | Safety margin for 100 trials x 50 folds      |
| Optuna ranges       | Same as L2  | Refined     | Tighten based on L2 convergence              |

**Config**: `configs/training_config_production.yaml`
**Pipeline**: `configs/pipeline_hpc.yaml` (already pointing to production)

**Pre-launch checklist** (after L2 completes):
1. Update `splits_config.yaml`: `n_splits: 10`
2. Review L2 hyperparameter distributions -- tighten Optuna ranges if clustering seen
3. Refine `k_grid` if L2 shows convergence to specific feature counts
4. Consider dropping dominated models from ensemble
5. Consider adding `LR_L1` if optimal l1_ratio from L2 is > 0.7

**Run command**:
```bash
cd analysis/
OVERWRITE_SPLITS=1 ./run_hpc.sh
# After jobs complete:
bash scripts/post_training_pipeline.sh --run-id <RUN_ID>
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
