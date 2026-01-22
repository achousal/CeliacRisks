# CeliacRisks ML Pipeline: Performance Assessment

**Date**: 2026-01-21
**Assessor**: Kaggle/Mathematical Perspective
**Codebase Version**: commit 62128e5

---

## Executive Summary

The pipeline is **well-engineered** with strong fundamentals: nested CV, proper threshold selection on validation, prevalence adjustment, and clinical utility metrics (DCA). However, several opportunities exist to improve predictive performance, particularly around ensemble methods, calibration strategy, and validation rigor.

**Current Status**: Production-grade infrastructure, research-grade modeling
**Key Opportunity**: +3-8% AUROC potential through ensemble and calibration improvements

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
| Nested CV | [models/training.py:70-252](../src/ced_ml/models/training.py#L70-L252) | Core training loop |
| Feature Selection | [features/kbest.py](../src/ced_ml/features/kbest.py) | Dimensionality reduction |
| Hyperparameter Tuning | [models/optuna_search.py](../src/ced_ml/models/optuna_search.py) | Bayesian optimization |
| Calibration | [models/calibration.py](../src/ced_ml/models/calibration.py) | Probability calibration |
| Prevalence Adjustment | [models/prevalence.py](../src/ced_ml/models/prevalence.py) | Deploy-time adjustment |
| Threshold Selection | [metrics/thresholds.py](../src/ced_ml/metrics/thresholds.py) | Operating point |

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
| **AUROC/PR-AUC** | Correct | Uses sklearn with single-class guards ([discrimination.py:58-127](../src/ced_ml/metrics/discrimination.py#L58-L127)) |
| **Feature Selection in CV** | Correct | K-best is inside pipeline, fitted per fold ([train.py:184-188](../src/ced_ml/cli/train.py#L184-L188)) |

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

**Current State**: Models (LR_EN, RF, XGBoost, LinSVM_cal) trained independently, compared side-by-side.

**Kaggle Best Practice**:
- Level-1: Train diverse base models (done)
- Level-2: Stack predictions using regularized meta-learner
- Often yields +2-5% AUROC improvement

**Recommendation**:
```python
# Pseudo-code for stacking
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

**Gap**: While `eval-holdout` exists, no temporal or geographic holdout is enforced.

**Risk**: For incident disease prediction, future samples may have different characteristics.

**Kaggle Best Practice**:
- Time-based splits for time-to-event outcomes
- Geographic/site-based splits for multi-center data
- "Adversarial validation" to detect train/test distribution shift

**Recommendation**: Implement temporal validation if sample collection dates are available.

### 3.3 Limited Calibration Strategy (MEDIUM IMPACT)

**Gap**: Calibration is done inside CV, not on a dedicated held-out calibration set.

**Risk**: Calibration quality may degrade on truly unseen data.

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

### Tier 1: High Impact, Moderate Effort

| # | Recommendation | Expected Impact | Effort | Files to Modify |
|---|----------------|-----------------|--------|-----------------|
| **1** | **Implement Model Stacking** | +2-5% AUROC | Medium | New: `models/stacking.py`, `cli/train_ensemble.py` |
| **2** | **Dedicated Calibration Set** | Better probability estimates | Low | `cli/train.py:600-610`, `data/splits.py` |
| **3** | **BCa Bootstrap CIs** | More accurate intervals | Low | `metrics/bootstrap.py:108` |

### Tier 2: Medium Impact, Low-Medium Effort

| # | Recommendation | Expected Impact | Effort | Files to Modify |
|---|----------------|-----------------|--------|-----------------|
| **4** | **Interaction Features (Top-K)** | +1-2% AUROC for linear models | Medium | `features/interactions.py` (new) |
| **5** | **Expanded Optuna Search** | Better hyperparameters | Low | `models/hyperparams.py`, configs |
| **6** | **Temporal Validation** | Honest generalization estimate | Medium | `data/splits.py`, `cli/save_splits.py` |

### Tier 3: Lower Impact, Worth Considering

| # | Recommendation | Expected Impact | Effort | Files to Modify |
|---|----------------|-----------------|--------|-----------------|
| **7** | Target Encoding | +0.5-1% for sparse categories | Low | `features/encoding.py` (new) |
| **8** | DCA Range Auto-Config | Better clinical utility plots | Trivial | `metrics/dca.py:124-125` |
| **9** | Model Selection Score | Formal selection criteria | Low | `cli/aggregate_splits.py` |

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

| Aspect | Current Grade | Post-Improvement Grade |
|--------|---------------|------------------------|
| **Leakage Prevention** | A | A |
| **Calibration** | B+ | A- (with dedicated cal set) |
| **Ensemble Methods** | C (none) | A (with stacking) |
| **Validation Rigor** | B | A- (with temporal val) |
| **Bootstrap CIs** | B | A (with BCa) |
| **Feature Engineering** | B- | B+ (with interactions) |
| **Overall** | B+ | A- |

---

## Unresolved Questions

1. **Sample collection dates**: Are temporal validation splits possible?
2. **Multi-center data**: Is geographic holdout feasible?
3. **Target prevalence**: Should DCA range be auto-configured?
4. **Ensemble compute budget**: How many CPU-hours available for stacking?

---

## References

1. Saerens M et al. (2002). Adjusting the outputs of a classifier to new a priori probabilities. Neural Computation.
2. Van Calster B et al. (2016). Calibration of risk prediction models. Medical Decision Making.
3. Vickers AJ, Elkin EB (2006). Decision curve analysis. Med Decis Making.
4. Efron B, Tibshirani R (1993). An Introduction to the Bootstrap. Chapman & Hall.
5. Wolpert DH (1992). Stacked Generalization. Neural Networks.
