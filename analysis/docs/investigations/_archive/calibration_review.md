# Calibration Strategy Review and Recommendations

**Date**: 2026-01-27
**Context**: Investigation of prevalent vs incident case score differences
**Finding**: Calibration shows case-type bias

---

## Current Calibration Performance

### LR_EN Model (OOF Predictions)

| Metric | Incident Cases | Prevalent Cases | Difference | Interpretation |
|--------|---------------|-----------------|------------|----------------|
| **Intercept** | -0.88 | -0.63 | -0.25 | Incident MORE underpredicted |
| **Slope** | 0.94 | 0.98 | -0.05 | Both near-optimal |
| **Brier** | 0.048 | 0.052 | -0.005 | Incident slightly better |

**Optimal values**: Intercept ≈ 0, Slope ≈ 1.0, Brier → 0

---

## Identified Issues

### Issue 1: Systematic Underprediction
**Both case types** show negative intercepts (model predicts lower probabilities than true risk).

**Magnitude**:
- Incident: -0.88 (severe underprediction)
- Prevalent: -0.63 (moderate underprediction)

**Clinical Impact**: Model may miss high-risk individuals due to conservative predictions.

### Issue 2: Case-Type Bias
**Incident cases are underpredicted MORE** than prevalent cases (intercept difference = -0.25).

**Potential causes**:
1. **Training data composition**: 50% prevalent downsampling may bias calibration
2. **Feature selection**: Features may discriminate better for prevalent cases (32% prevalent-biased, only 4% incident-biased)
3. **Calibration method**: Per-fold calibration may not generalize to incident cases

### Issue 3: Calibration Strategy Not Validated
**Current approach**: Per-fold calibration fitted during nested CV.

**Concerns**:
- Calibration parameters optimized on training data
- No independent validation of calibration quality
- May introduce optimistic bias (~0.5-1% AUROC according to ADR-014)

---

## Available Calibration Strategies

### 1. Per-Fold Calibration (Current Default)
**Method**: Fit isotonic regression on each CV fold's training data.

**Pros**:
- Fast (calibrated during training)
- Integrated with hyperparameter tuning

**Cons**:
- Optimistic bias (~0.5-1% AUROC)
- Validated on same data used for feature selection
- May not generalize to OOF predictions

**Use case**: Rapid prototyping, when speed matters more than unbiased estimates.

### 2. OOF-Posthoc Calibration (Recommended)
**Method**: Collect all OOF predictions first, then fit single isotonic regression.

**Pros**:
- Unbiased calibration (ADR-014)
- No data leakage
- More reliable prevalence adjustment

**Cons**:
- Slightly slower (two-pass algorithm)
- Requires code changes (already implemented)

**Use case**: Production deployment, scientific publications, unbiased performance estimates.

**How to enable**:
```yaml
# configs/training_config.yaml
calibration:
  enabled: true
  strategy: oof_posthoc  # Change from 'per_fold'
  method: isotonic
```

### 3. No Calibration (Baseline)
**Method**: Use raw model probabilities.

**Pros**:
- Simplest
- No calibration bias

**Cons**:
- Poor calibration for most models
- Especially bad for tree-based models (RF, XGBoost)

**Use case**: Baseline comparison, discrimination-only tasks.

---

## Comparison Experiment

### Design
Train LR_EN on same data with three calibration strategies:

| Strategy | Config Setting | Expected Benefit |
|----------|---------------|------------------|
| Baseline | `enabled: false` | Measure raw model calibration |
| Per-fold | `strategy: per_fold` | Current production |
| OOF-posthoc | `strategy: oof_posthoc` | Unbiased estimate |

### Metrics to Compare
1. **Calibration**:
   - Intercept (should be closer to 0)
   - Slope (should be closer to 1.0)
   - Brier score (lower is better)

2. **Discrimination**:
   - AUROC (should be similar across strategies)
   - PR-AUC

3. **Case-type fairness**:
   - Intercept difference (incident - prevalent)
   - Slope difference
   - Median score difference

### Hypothesis
OOF-posthoc will show:
- Better calibration (intercept closer to 0)
- Reduced case-type bias (smaller intercept difference)
- Similar discrimination (AUROC unchanged)

---

## Case-Type-Specific Calibration

### Motivation
If model systematically underpredicts incident cases, calibrate them separately.

### Approach
1. **Train single model** on combined data (incident + prevalent + controls)
2. **Fit separate calibrators**:
   - Incident calibrator: fit on (incident vs controls) subset
   - Prevalent calibrator: fit on (prevalent vs controls) subset
3. **At inference**: route predictions through appropriate calibrator based on case type

### Implementation Strategy
```python
# Pseudocode
if case_type == "Incident":
    calibrated_prob = incident_calibrator.predict_proba(raw_prob)
elif case_type == "Prevalent":
    calibrated_prob = prevalent_calibrator.predict_proba(raw_prob)
else:
    calibrated_prob = combined_calibrator.predict_proba(raw_prob)
```

### Challenges
1. **Case type unknown at inference** (in real screening, we don't know if someone has prevalent disease)
2. **Smaller sample sizes** for each calibrator
3. **Added complexity** in deployment

### Recommendation
**NOT RECOMMENDED** for production unless:
- Case type can be reliably predicted from auxiliary features
- Separate calibration substantially improves outcomes (>10% better sensitivity at same specificity)

---

## Recommendations

### Immediate Actions (High Priority)

1. **Switch to OOF-posthoc calibration**
   ```yaml
   # configs/training_config.yaml
   calibration:
     enabled: true
     strategy: oof_posthoc
     method: isotonic
   ```
   **Expected outcome**: Reduced case-type bias, better calibration.

2. **Re-run experiment with OOF-posthoc**
   ```bash
   cd analysis/docs/investigations
   # Edit training_config.yaml first
   bash run_experiment.sh --models LR_EN,RF
   ```

3. **Validate calibration on test set**
   - Current analysis only uses OOF predictions (train set)
   - Need test set evaluation for true generalization

### Medium-Term Actions

4. **Compare calibration strategies empirically**
   - Train 3 models (no cal, per-fold, oof-posthoc)
   - Measure calibration metrics on held-out test set
   - Document in ADR

5. **Investigate feature bias**
   - 32% features are prevalent-biased vs 4% incident-biased
   - Consider feature re-weighting or separate selection per case type

### Long-Term Considerations

6. **Case-type-specific calibration** (only if bias persists)
   - Requires auxiliary features to predict case type
   - High implementation cost
   - Consider only if simpler methods fail

7. **External validation**
   - Test calibration on independent cohort
   - Check if case-type bias persists in new data

---

## Testing Checklist

Before deploying calibration changes:

- [ ] OOF-posthoc produces better intercepts (closer to 0)
- [ ] Case-type bias reduced (|intercept_diff| < 0.15)
- [ ] Discrimination unchanged (AUROC within 1%)
- [ ] Test set calibration validates OOF findings
- [ ] Clinical thresholds re-optimized after calibration change
- [ ] Documentation updated (ADR-014 amendment)

---

## Code Changes Required

### 1. Enable OOF-Posthoc (Already Implemented)
File: `configs/training_config.yaml`
```yaml
calibration:
  enabled: true
  strategy: oof_posthoc  # Just change this line
  method: isotonic
```

### 2. Add Test Set Calibration Evaluation
File: `src/ced_ml/evaluation/reports.py`
```python
# Add test set calibration analysis
test_cal_metrics = calculate_calibration_metrics(
    y_true=test_y,
    y_prob=test_preds,
    case_types=test_case_types  # New parameter
)
```

### 3. Case-Type Calibration Plots
File: `src/ced_ml/plotting/calibration.py`
```python
# Add case-type-stratified calibration curves
plot_calibration_by_case_type(
    y_true, y_prob, case_types,
    save_path="calibration_by_case_type.png"
)
```

---

## References

- **ADR-014**: OOF-posthoc calibration strategy
- **ADR-010**: Prevalence adjustment (downstream of calibration)
- **Investigation results**: `/analysis/docs/bugs/investigation_results_2026-01-27.md`
- **Calibration code**: `/analysis/src/ced_ml/models/calibration.py`

---

## Expected Outcomes

After implementing OOF-posthoc calibration:

| Metric | Current | Expected | Improvement |
|--------|---------|----------|-------------|
| Incident intercept | -0.88 | -0.20 to 0.00 | 70-100% reduction |
| Prevalent intercept | -0.63 | -0.15 to 0.00 | 75-100% reduction |
| Intercept difference | -0.25 | -0.10 to 0.00 | 60-100% reduction |
| Case-type bias | Significant | Minimal | Major improvement |

**Timeline**: 1-2 hours to re-run experiment + analysis.
