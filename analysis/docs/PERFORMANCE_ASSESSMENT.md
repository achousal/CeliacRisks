# CeliacRisks ML Pipeline: Performance Assessment

**Date**: 2026-01-21
**Assessor**: Kaggle/Mathematical Perspective
**Codebase Version**: commit 62128e5

---

## Executive Summary

The pipeline is **well-engineered** with strong fundamentals: nested CV, proper threshold selection on validation, prevalence adjustment, and clinical utility metrics (DCA). However, several critical implementation gaps and opportunities exist to improve correctness and predictive performance.

**Current Status**: Production-grade infrastructure, research-grade modeling
**Key Opportunity**: +3-8% AUROC potential through ensemble and calibration improvements
**Blockers**: See [Critical/High Findings](#8-critical-findings-blocking-production) below

### Table of Contents

1. [Repo Map](#1-repo-map-entry-points-and-pipeline-stages)
2. [Evaluation Correctness Review](#2-evaluation-correctness-review)
3. [Validation Gap Analysis](#3-validation-gap-analysis)
4. [Prioritized Recommendations](#4-prioritized-recommendations)
5. [Implementation Priorities](#5-implementation-priorities-mathematicians-view)
6. [Quick Wins](#6-quick-wins-implement-today)
7. [Summary Metrics](#7-summary-metrics)
8. [Critical Findings (Blocking Production)](#8-critical-findings-blocking-production)
9. [Open Questions](#9-open-questions)
10. [Implementation Status Summary](#10-implementation-status-summary)

---

## 1. Repo Map: Entry Points and Pipeline Stages

### Entry Points

| Command | File | Purpose |
|---------|------|---------|
| `ced save-splits` | [cli/save_splits.py](../src/ced_ml/cli/save_splits.py) | Generate stratified train/val/test splits |
| `ced train` | [cli/train.py](../src/ced_ml/cli/train.py) | Train single model with nested CV |
| `ced aggregate-splits` | [cli/aggregate_splits.py](../src/ced_ml/cli/aggregate_splits.py) | Aggregate metrics across splits |
| `ced eval-holdout` | [cli/eval_holdout.py](../src/ced_ml/cli/eval_holdout.py) | One-time holdout evaluation |

### Pipeline Flow

```
Data (Parquet) --> Row Filters --> Scenario Filter --> Split Selection
                                                            |
                                                            v
                   +--------------------+-------------------+
                   |                    |                   |
                   v                    v                   v
              TRAIN SET            VAL SET             TEST SET
                   |                    |                   |
                   v                    |                   |
        Nested CV (5x10 folds)          |                   |
        +-------------------------+     |                   |
        | Outer fold:             |     |                   |
        |   Inner CV:             |     |                   |
        |     - K-best selection  |     |                   |
        |     - Hyperparam tuning |     |                   |
        |     - Optuna/Random     |     |                   |
        |   Fit on CV_train       |     |                   |
        |   Calibrate (isotonic)  |     |                   |
        |   Predict CV_test (OOF) |     |                   |
        +-------------------------+     |                   |
                   |                    |                   |
                   v                    |                   |
           Fit Final Model              |                   |
           on full TRAIN                |                   |
                   |                    |                   |
                   +--------------------+                   |
                   |                                        |
                   v                                        |
           Threshold Selection                              |
           (on VAL set)                                     |
                   |                                        |
                   +----------------------------------------+
                   |
                   v
           TEST Evaluation
           (using VAL threshold)
                   |
                   v
           Prevalence Adjustment
           + Metrics + Plots
```

### Critical Files for Performance

| Component | File | Impact |
|-----------|------|--------|
| Nested CV | [models/training.py:70-252](../src/ced_ml/models/training.py#L70-L252) | Core training loop with OOF predictions |
| Feature Selection | [features/kbest.py](../src/ced_ml/features/kbest.py) | ANOVA F-test k-best selection |
| Hyperparameter Tuning | [models/optuna_search.py](../src/ced_ml/models/optuna_search.py) | Bayesian TPE + median pruning |
| Calibration | [models/calibration.py](../src/ced_ml/models/calibration.py) | Isotonic/Platt scaling |
| Prevalence Adjustment | [models/prevalence.py:51-111](../src/ced_ml/models/prevalence.py#L51-L111) | Logit-space intercept shift |
| Threshold Selection | [metrics/thresholds.py](../src/ced_ml/metrics/thresholds.py) | Fixed-spec/Youden/F1 thresholds |

### Orchestration Scripts

| Script | Purpose | Key Commands |
|--------|---------|--------------|
| [run_production.sh](../run_production.sh) | Full pipeline execution | Splits, train all models, aggregate |
| [run_hpc.sh](../run_hpc.sh) | HPC batch submission | LSF/Slurm job arrays |
| [run_local.sh](../run_local.sh) | Local development | Single model, quick validation |

---

## 2. Evaluation Correctness Review

### CORRECT (No Issues Found)

| Component | Status | Evidence |
|-----------|--------|----------|
| **Nested CV Structure** | Correct | RepeatedStratifiedKFold with separate inner/outer loops ([training.py:129-133](../src/ced_ml/models/training.py#L129-L133)) |
| **Threshold Selection** | Correct | Computed on VAL, reused on TEST ([train.py:652-679](../src/ced_ml/cli/train.py#L652-L679)) |
| **Prevalence Adjustment Formula** | Correct | Uses logit-space intercept shift (Saerens et al. 2002) ([prevalence.py:51-111](../src/ced_ml/models/prevalence.py#L51-L111)) |
| **Bootstrap CI** | Correct | Stratified resampling maintains case/control ratio ([bootstrap.py:32-109](../src/ced_ml/metrics/bootstrap.py#L32-L109)) |
| **DCA Computation** | Correct | Standard Vickers & Elkin formula ([dca.py:29-61](../src/ced_ml/metrics/dca.py#L29-L61)) |
| **AUROC/PR-AUC** | Correct | Uses sklearn with single-class guards via `_validate_binary_labels` ([discrimination.py:32-55](../src/ced_ml/metrics/discrimination.py#L32-L55)) |
| **Feature Selection in CV** | Correct | K-best is inside pipeline, fitted per fold ([train.py:184-188](../src/ced_ml/cli/train.py#L184-L188)) |
| **Split Integrity** | Correct | Overlap/bounds validation in `validate_split_indices` ([persistence.py:33-95](../src/ced_ml/data/persistence.py#L33-L95)) |
| **Row Filter Alignment** | Correct | Centralized in `apply_row_filters` with stats tracking ([filters.py:15-80](../src/ced_ml/data/filters.py#L15-L80)) |

### POTENTIAL ISSUES (Minor)

#### 2.1 Calibration Within CV Loop
**Location**: [training.py:180-187](../src/ced_ml/models/training.py#L180-L187)

**Issue**: CalibratedClassifierCV is applied AFTER hyperparameter search on the SAME fold data. While CalibratedClassifierCV uses internal CV, the data informed both hyperparam selection and calibration.

**Impact**: Slight optimistic bias in OOF predictions (~0.5-1% AUROC inflation)

**Ideal**: Hold out a calibration subset within each outer fold, OR use a separate calibration set entirely.

#### 2.2 Bootstrap CI Method
**Location**: [bootstrap.py:108-109](../src/ced_ml/metrics/bootstrap.py#L108-L109)

**Issue**: Uses simple percentile method. For small samples (148 cases), bias-corrected and accelerated (BCa) provides better coverage.

**Impact**: CI may have undercoverage for extreme metrics

**Fix**: Replace with `scipy.stats.bootstrap` with BCa method (available in scipy 1.7+)

#### 2.3 DCA Threshold Range
**Location**: [dca.py:124-125](../src/ced_ml/metrics/dca.py#L124-L125)

**Issue**: Default range 0.001 to 0.10 may miss relevant thresholds for 0.34% prevalence scenarios.

**Impact**: Missing clinical utility information at very low thresholds

**Fix**: Make DCA range configurable based on target prevalence: `min_thr = target_prev / 10`

---

## 3. Validation Gap Analysis

### 3.1 Missing Ensemble Methods (HIGH IMPACT)

**Gap**: No stacking, blending, or weighted averaging across models.

**Current State**: Models (LR_EN, RF, XGBoost, LinSVM_cal) trained independently. OOF predictions exist ([training.py:70-120](../src/ced_ml/models/training.py#L70-L120)) but no meta-learner combines them.

**Kaggle Best Practice**:
- Level-1: Train diverse base models (done via [models/registry.py](../src/ced_ml/models/registry.py))
- Level-2: Stack predictions using regularized meta-learner
- Often yields +2-5% AUROC improvement

**Recommendation**:
```python
# Pseudo-code for stacking
# OOF predictions already generated in training.py:oof_predictions_with_nested_cv
oof_preds = {
    'LR_EN': lr_oof_predictions,
    'RF': rf_oof_predictions,
    'XGBoost': xgb_oof_predictions,
}
meta_features = pd.DataFrame(oof_preds)
meta_model = LogisticRegression(penalty='l2', C=1.0)
meta_model.fit(meta_features, y_train)
```

### 3.2 No Proper External Validation (HIGH IMPACT)

**Gap**: While `eval-holdout` exists ([cli/eval_holdout.py](../src/ced_ml/cli/eval_holdout.py)), temporal/geographic holdout is not enforced in the split logic.

**Current Support**: Temporal ordering is implemented ([save_splits.py:228-234](../src/ced_ml/cli/save_splits.py#L228-L234)) but not integrated into holdout evaluation by default.

**Risk**: For incident disease prediction, future samples may have different characteristics.

**Kaggle Best Practice**:
- Time-based splits for time-to-event outcomes
- Geographic/site-based splits for multi-center data
- "Adversarial validation" to detect train/test distribution shift

**Recommendation**: Enable `temporal_split=True` in config and verify holdout respects temporal ordering.

### 3.3 Limited Calibration Strategy (MEDIUM IMPACT)

**Gap**: Calibration is done inside CV via `CalibratedClassifierCV` ([training.py:180-187](../src/ced_ml/models/training.py#L180-L187)), not on a dedicated held-out calibration set.

**Current Approach**: [models/calibration.py](../src/ced_ml/models/calibration.py) provides isotonic/Platt scaling, applied within the same fold data used for hyperparameter selection.

**Risk**: Calibration quality may degrade on truly unseen data (~0.5-1% optimistic bias).

**Kaggle Best Practice**:
- Train/Val/Calibration/Test split (4-way)
- OR: Calibrate on validation set AFTER threshold selection
- Consider Venn-ABERS for uncertainty quantification

### 3.4 No Feature Interactions (MEDIUM IMPACT)

**Gap**: Linear models and trees, but no explicit polynomial/interaction features.

**Opportunity**: Protein-protein interactions may be biologically meaningful.

**Recommendation**:
```python
from sklearn.preprocessing import PolynomialFeatures
# Top-k proteins only to avoid curse of dimensionality
poly = PolynomialFeatures(degree=2, interaction_only=True)
```

### 3.5 Single-Model Hyperparameter Ranges (LOW-MEDIUM IMPACT)

**Gap**: Fixed hyperparameter grids may not be optimal for this dataset.

**Observation**: XGBoost grids include typical defaults but may miss dataset-specific optima.

**Recommendation**:
- Expand search space for best-performing model
- Use Optuna's `suggest_*` with wider ranges
- Consider learning rate scheduling for XGBoost

### 3.6 No Target Encoding for Categories (LOW IMPACT)

**Gap**: OneHotEncoder for categorical metadata (sex, ethnicity).

**Opportunity**: Target encoding can capture non-linear relationships.

**Caution**: Must be done carefully to avoid leakage (fit only on training fold).

### 3.7 Missing Model Selection Criterion (LOW IMPACT)

**Gap**: Models compared by AUROC, but selection criteria not formalized.

**Recommendation**: Define composite score:
```
selection_score = 0.5 * AUROC + 0.3 * (1 - Brier) + 0.2 * calibration_slope
```

---

## 4. Prioritized Recommendations

### Tier 0: Critical Fixes (Before Production)

| # | Recommendation | Severity | Effort | Reference |
|---|----------------|----------|--------|-----------|
| **0a** | **Fix holdout feature alignment** | HIGH | Low | [Section 8.2](#82-high-holdout-evaluation-feature-alignment) |
| **0b** | **Verify PPV threshold direction** | MEDIUM | Low | [Section 8.4](#84-medium-threshold-selection-direction-verify) |
| **0c** | **Document prevalent holdout intent** | MEDIUM | Trivial | [Section 8.3](#83-high-holdout-prevalent-case-leakage) |

### Tier 1: High Impact, Moderate Effort

| # | Recommendation | Expected Impact | Effort | Files to Modify |
|---|----------------|-----------------|--------|-----------------|
| **1** | **Implement Model Stacking** | +2-5% AUROC | Medium | New: `models/stacking.py`; extend [cli/train.py](../src/ced_ml/cli/train.py) with ensemble mode |
| **2** | **Dedicated Calibration Set** | Better probability estimates | Low | [cli/train.py:600-650](../src/ced_ml/cli/train.py#L600-L650), [data/splits.py](../src/ced_ml/data/splits.py) |
| **3** | **BCa Bootstrap CIs** | More accurate intervals | Low | [metrics/bootstrap.py:108](../src/ced_ml/metrics/bootstrap.py#L108) |

### Tier 2: Medium Impact, Low-Medium Effort

| # | Recommendation | Expected Impact | Effort | Files to Modify |
|---|----------------|-----------------|--------|-----------------|
| **4** | **Interaction Features (Top-K)** | +1-2% AUROC for linear models | Medium | New: `features/interactions.py` |
| **5** | **Expanded Optuna Search** | Better hyperparameters | Low | [models/optuna_search.py](../src/ced_ml/models/optuna_search.py), [config/defaults.py:150-200](../src/ced_ml/config/defaults.py#L150-L200) |
| **6** | **Temporal Validation** | Honest generalization estimate | Medium | [data/splits.py](../src/ced_ml/data/splits.py), [cli/save_splits.py:228-234](../src/ced_ml/cli/save_splits.py#L228-L234) (already supports temporal ordering) |

### Tier 3: Lower Impact, Worth Considering

| # | Recommendation | Expected Impact | Effort | Files to Modify |
|---|----------------|-----------------|--------|-----------------|
| **7** | Target Encoding | +0.5-1% for sparse categories | Low | New: `features/encoding.py` |
| **8** | DCA Range Auto-Config | Better clinical utility plots | Trivial | [metrics/dca.py:124-125](../src/ced_ml/metrics/dca.py#L124-L125) |
| **9** | Model Selection Score | Formal selection criteria | Low | [cli/aggregate_splits.py](../src/ced_ml/cli/aggregate_splits.py) |

---

## 5. Implementation Priorities (Mathematician's View)

### 5.1 Ensemble Implementation (Highest Priority)

**Rationale**: Averaging uncorrelated errors is the single most reliable way to improve predictions in competitions and real-world applications.

**Design**:
```
Level-0 (Base Models):
  - LR_EN: High bias, low variance, excellent calibration
  - RF: Medium bias, medium variance, captures interactions
  - XGBoost: Low bias, higher variance, strong with downsampling
  - LinSVM_cal: Different decision boundary geometry

Level-1 (Meta-Model):
  - Logistic Regression with L2 penalty
  - Input: OOF predictions from Level-0 models
  - Regularization prevents overfitting to training noise
```

**Mathematical Justification**:
If base model errors are uncorrelated with variance sigma^2 each:
- Single model MSE: sigma^2
- Average of k models MSE: sigma^2 / k

Even with correlation rho between errors:
- Average MSE: sigma^2 * (1 + (k-1)*rho) / k

For rho < 1, averaging always helps.

### 5.2 Calibration Improvement

**Current Formula** (correct):
```
P(Y=1|X, prev_target) = sigmoid(logit(p) + logit(prev_target) - logit(prev_sample))
```

**Improvement**: Platt scaling with held-out data
```
# After training, fit on calibration set (not training set)
logit(p_calibrated) = a * logit(p_raw) + b
# Where a, b are fitted on calibration data only
```

### 5.3 Bootstrap Improvement

**Current**: Percentile method
```python
ci = (np.percentile(vals, 2.5), np.percentile(vals, 97.5))
```

**Improved**: BCa method (bias-corrected and accelerated)
```python
from scipy.stats import bootstrap
result = bootstrap((y_true, y_pred), metric_fn, method='BCa', n_resamples=1000)
ci = result.confidence_interval
```

---

## 6. Quick Wins (Implement Today)

### 6.1 Fix BCa Bootstrap (~30 min)

```python
# In metrics/bootstrap.py, add option:
def stratified_bootstrap_ci(
    ...,
    method: str = "percentile",  # or "bca"
):
    if method == "bca" and HAS_SCIPY_BOOTSTRAP:
        from scipy.stats import bootstrap
        # Implementation
```

### 6.2 Auto-Configure DCA Range (~15 min)

```python
# In metrics/dca.py:
def decision_curve_analysis(..., prevalence_adjustment=None):
    if thresholds is None:
        if prevalence_adjustment:
            min_thr = max(0.0001, prevalence_adjustment / 10)
            max_thr = min(0.5, prevalence_adjustment * 10)
        else:
            min_thr, max_thr = 0.001, 0.10
        thresholds = generate_dca_thresholds(min_thr, max_thr)
```

### 6.3 Add Model Comparison Score (~20 min)

```python
# In cli/aggregate_splits.py:
def compute_selection_score(metrics: dict) -> float:
    """Composite score for model selection."""
    auroc = metrics.get('AUROC', 0.5)
    brier = metrics.get('Brier', 0.25)
    slope = metrics.get('calib_slope', 1.0)

    # Weights emphasize discrimination and calibration
    score = (
        0.50 * auroc +
        0.30 * (1.0 - brier) +
        0.20 * (1.0 - abs(slope - 1.0))
    )
    return score
```

---

## 7. Summary Metrics

| Aspect | Current Grade | Post-Fix Grade | Post-Enhancement Grade |
|--------|---------------|----------------|------------------------|
| **Core CV Logic** | A | A | A |
| **Leakage Prevention** | A | A | A |
| **Holdout Workflow** | C (bugs) | B+ (fixes applied) | A- (with temporal val) |
| **Calibration** | B+ | B+ | A- (with dedicated cal set) |
| **Ensemble Methods** | C (none) | C | A (with stacking) |
| **Validation Rigor** | B | B+ | A- (with temporal val) |
| **Bootstrap CIs** | B | B | A (with BCa) |
| **Feature Engineering** | B- | B- | B+ (with interactions) |
| **Production Readiness** | C (blockers) | B+ (fixes applied) | A- |
| **Overall** | B- | B+ | A- |

---

## Unresolved Questions (Performance)

These are open questions related to **model performance improvements** (see also [Section 9](#9-open-questions) for correctness-related questions):

1. **Sample collection dates**: Are temporal validation splits possible? Config supports it ([save_splits.py:228](../src/ced_ml/cli/save_splits.py#L228)).
2. **Multi-center data**: Is geographic holdout feasible?
3. **Target prevalence**: Should DCA range be auto-configured based on prevalence?
4. **Ensemble compute budget**: How many CPU-hours available for stacking?

---

## 8. Critical Findings (Blocking Production)

### 8.1 CRITICAL: Missing Aggregation Command

**Severity**: CRITICAL
**Impact**: Advertised `ced postprocess` command does not exist

**Evidence**: No `postprocess.py` file found in `cli/` directory. The `ced aggregate-splits` command exists but documentation references `ced postprocess`.

**Recommendation**: Clarify which CLI command handles post-training aggregation, or implement the missing entrypoint.

---

### 8.2 HIGH: Holdout Evaluation Feature Alignment

**Severity**: HIGH
**Location**: [evaluation/holdout.py:406-409](../src/ced_ml/evaluation/holdout.py#L406-L409)

**Issue**: Holdout evaluation builds feature matrix from **protein columns only**:
```python
# holdout.py:406-409
prot_cols = identify_protein_columns(df_filtered)
X_all = df_filtered[prot_cols]  # CODEX: Missing metadata columns
```

If training uses demographic metadata (age, BMI, sex, ethnicity), holdout evaluation will fail or produce misaligned predictions because those columns are excluded.

**Related**: [cli/train.py:180](../src/ced_ml/cli/train.py#L180) includes metadata in preprocessor:
```python
# train.py:142-155 (build_preprocessor)
numeric_cols = protein_cols + meta_num_cols
transformers = [("num", StandardScaler(), numeric_cols)]
if cat_cols:
    transformers.append(("cat", OneHotEncoder(...), cat_cols))
```

**Recommendation**: Extend `evaluate_holdout` to load model metadata and include the same column set used during training. The bundle already stores config ([holdout.py:369-371](../src/ced_ml/evaluation/holdout.py#L369-L371)).

---

### 8.3 HIGH: Holdout Prevalent Case Leakage

**Severity**: HIGH
**Location**: [cli/save_splits.py:246-258](../src/ced_ml/cli/save_splits.py#L246-L258)

**Issue**: When `mode="holdout"` and `prevalent_train_only=True`, the `_create_holdout` function samples from `df_scenario` which may contain prevalent cases. This means prevalent cases can appear in holdout even though the intent is incident-only evaluation.

**Code path**:
```python
# save_splits.py:237 - y_full includes all cases
y_full = df_scenario[TARGET_COL].isin(positives).astype(int).to_numpy()

# save_splits.py:246-258 - holdout sampled before prevalent filtering
if config.mode == "holdout":
    df_work, y_work, index_space, dev_to_global_map = _create_holdout(
        df_scenario,  # CODEX: Contains prevalent cases
        y_full,       # CODEX: Includes prevalent in positives
        ...
    )
```

**Recommendation**: Add prevalent filtering before holdout sampling when `prevalent_train_only=True`, OR explicitly document that holdout contains all case types.

---

### 8.4 MEDIUM: Threshold Selection Direction (Verify)

**Severity**: MEDIUM (potential issue, needs verification)
**Location**: [metrics/thresholds.py:269-275](../src/ced_ml/metrics/thresholds.py#L269-L275)

**Issue**: `threshold_for_precision` uses `idx = int(ok[-1])` to select threshold. The docstring says "lowest threshold" but in sklearn's `precision_recall_curve`, thresholds are sorted in **increasing** order (not decreasing).

```python
# thresholds.py:269-275
ok = np.where(prec_t >= target_ppv)[0]
if ok.size == 0:
    return threshold_max_f1(y_true, p)

# Want lowest threshold (most inclusive) among those achieving target
idx = int(ok[-1])  # CODEX: Takes LAST index; verify if this is lowest threshold
```

**Behavior**: Taking `ok[-1]` gets the **highest** threshold meeting PPV, not the lowest. This makes fixed-PPV operating points more conservative than intended.

**Recommendation**:
1. Add unit test with known data to verify threshold direction
2. If bug confirmed, change to `idx = int(ok[0])` for lowest qualifying threshold

---

### 8.5 MEDIUM: Row Filter Alignment in Holdout

**Severity**: MEDIUM
**Location**: [evaluation/holdout.py:384-399](../src/ced_ml/evaluation/holdout.py#L384-L399)

**Issue**: Holdout evaluation applies row filters AFTER loading holdout indices. If the original data changed or filters differ, indices created during split generation may not align with filtered data rows.

```python
# holdout.py:384-390
# Apply row filters (matching save_splits.py and train.py)
df_filtered, filter_stats = apply_row_filters(df_scenario_raw, meta_num_cols=meta_num_cols)

# holdout.py:398-399 - temporal reordering after filters
if temporal_split:
    df_filtered = df_filtered.iloc[order_idx].reset_index(drop=True)
```

The code attempts alignment via metadata ([holdout.py:386-388](../src/ced_ml/evaluation/holdout.py#L386-L388)), but if split metadata is missing or incomplete, indices may be misaligned.

**Recommendation**: Add validation that holdout indices map to expected rows (e.g., sample ID matching or hash verification).

---

## 9. Open Questions

1. **Legacy dependency**: The codebase references `celiacML_faith.py` ([persistence.py:16](../src/ced_ml/data/persistence.py#L16), [cli/train.py:4](../src/ced_ml/cli/train.py#L4)). Is this file outside the repo? Should we review it for correctness?

2. **Metadata in models**: Are demographic columns (age, BMI, sex, ethnicity) included in production models? This determines whether [Finding 8.2](#82-high-holdout-evaluation-feature-alignment) is a blocker.

3. **Prevalent case intent**: For holdout mode, should prevalent cases be excluded when `prevalent_train_only=True`? This determines whether [Finding 8.3](#83-high-holdout-prevalent-case-leakage) is a bug or intended behavior.

4. **Threshold test coverage**: Does the test suite cover `threshold_for_precision` with known expected values? ([test_metrics_thresholds.py:223](../tests/test_metrics_thresholds.py#L223) exists but needs verification of threshold direction assertions).

---

## 10. Implementation Status Summary

| Component | Status | Key Files | Notes |
|-----------|--------|-----------|-------|
| **Split Generation** | Complete | [cli/save_splits.py](../src/ced_ml/cli/save_splits.py), [data/persistence.py](../src/ced_ml/data/persistence.py) | Holdout mode has alignment concerns |
| **Training Pipeline** | Complete | [cli/train.py](../src/ced_ml/cli/train.py), [models/training.py](../src/ced_ml/models/training.py) | Nested CV, Optuna integration working |
| **Holdout Evaluation** | Partial | [evaluation/holdout.py](../src/ced_ml/evaluation/holdout.py), [cli/eval_holdout.py](../src/ced_ml/cli/eval_holdout.py) | Feature alignment issue, see 8.2 |
| **Aggregation** | Complete | [cli/aggregate_splits.py](../src/ced_ml/cli/aggregate_splits.py) | Works for development mode |
| **Threshold Selection** | Needs Verification | [metrics/thresholds.py](../src/ced_ml/metrics/thresholds.py) | PPV threshold direction unclear |
| **Stacking/Ensemble** | Not Implemented | - | High-impact opportunity |

---

## References

1. Saerens M et al. (2002). Adjusting the outputs of a classifier to new a priori probabilities. Neural Computation.
2. Van Calster B et al. (2016). Calibration of risk prediction models. Medical Decision Making.
3. Vickers AJ, Elkin EB (2006). Decision curve analysis. Med Decis Making.
4. Efron B, Tibshirani R (1993). An Introduction to the Bootstrap. Chapman & Hall.
5. Wolpert DH (1992). Stacked Generalization. Neural Networks.
