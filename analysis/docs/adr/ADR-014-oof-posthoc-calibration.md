# ADR-014: OOF Posthoc Calibration Strategy

**Status:** Accepted
**Date:** 2026-01-22

## Context

The pipeline supports probability calibration to improve alignment between predicted probabilities and true event frequencies. The original implementation used `per_fold` calibration, which applies `CalibratedClassifierCV` inside each nested CV fold.

**Problem with `per_fold` strategy:**
- Calibration is fitted on the same data used for hyperparameter selection
- This introduces subtle optimistic bias (~0.5-1% in Brier score)
- The calibrator "sees" the validation data indirectly through the model selection process
- Bias is small but compounds across CV folds

**Need:**
- A calibration strategy that eliminates this subtle leakage
- Maintain full data efficiency (no additional holdout set)
- Support per-model strategy selection

## Decision

Add `oof_posthoc` calibration strategy as an alternative to `per_fold`:

1. **`per_fold`** (default, existing): Apply CalibratedClassifierCV inside each CV fold
2. **`oof_posthoc`** (new): Collect raw OOF predictions from all CV folds, then fit a single calibrator post-hoc on the aggregated OOF predictions
3. **`none`**: No calibration applied

**Implementation:**
- New classes: `OOFCalibrator`, `OOFCalibratedModel`
- Config-driven: `CalibrationConfig.strategy` field
- Per-model overrides via `CalibrationConfig.per_model` dict
- Backward compatible: existing configs default to `per_fold`

**Strategy Comparison:**

| Approach | Data Efficiency | Leakage Risk | Optimism Bias | Stability |
|----------|-----------------|--------------|---------------|-----------|
| `per_fold` | Full | Subtle (~0.5-1%) | ~0.5-1% | Lower |
| `oof_posthoc` | Full | None | None | Higher |
| 4-way split | Reduced | None | None | Medium |

## Alternatives Considered

1. **4-way split (train/val/calibration/test):**
   - Dedicated calibration holdout set
   - Rejected: Reduces training data, problematic for small datasets

2. **Nested calibration within inner CV:**
   - Calibrate inside inner CV loop
   - Rejected: Excessive complexity, minimal benefit over oof_posthoc

3. **Always use oof_posthoc:**
   - Replace per_fold entirely
   - Rejected: Per_fold has lower variance for some model types; choice should be configurable

4. **Temperature scaling:**
   - Single-parameter calibration (Guo et al., 2017)
   - Rejected: Less flexible than isotonic; doesn't address fundamental leakage issue

## Consequences

### Positive
- Eliminates ~0.5-1% optimistic bias in Brier score
- Higher stability: single calibrator vs. multiple per-fold calibrators
- Full data efficiency maintained (no additional holdout)
- Per-model flexibility: can use different strategies for different models
- Backward compatible: existing configs unchanged

### Negative
- Slightly higher variance in calibration for small datasets
- Additional complexity in calibration pipeline
- OOF predictions must be stored during training (already done for other reasons)
- May show slightly worse calibration curves for well-calibrated base models

## Evidence

### Code Pointers
- [models/calibration.py](../../src/ced_ml/models/calibration.py) - `OOFCalibrator`, `OOFCalibratedModel`
- [config/schema.py:301-337](../../src/ced_ml/config/schema.py#L301-L337) - `CalibrationConfig` with strategy field
- [models/training.py](../../src/ced_ml/models/training.py) - Strategy-aware calibration in CV loop

### Test Coverage
- `tests/test_models_calibration.py` - 24 tests for calibration strategies
- Tests validate: strategy selection, OOF collection, posthoc fitting, prediction flow

### Configuration Example
```yaml
calibration:
  enabled: true
  strategy: oof_posthoc  # or per_fold or none
  method: isotonic       # or sigmoid
  per_model:             # Optional per-model overrides
    LR_EN: oof_posthoc
    RF: per_fold
    XGBoost: oof_posthoc
```

### References
- Guo, C., et al. (2017). On Calibration of Modern Neural Networks. ICML.
- Van Calster, B., et al. (2019). Calibration: the Achilles heel of predictive analytics. BMC Medicine.
- Steyerberg, E.W. (2019). Clinical Prediction Models (2nd ed.), Chapter 15.

## Related ADRs

- Depends on: [ADR-006: Nested CV Structure](ADR-006-nested-cv.md) - Provides OOF predictions
- Complements: [ADR-010: Prevalence Adjustment](ADR-010-prevalence-adjustment.md) - Applied after calibration
- Related: [ADR-013: Prevalence Wrapper](ADR-013-prevalence-wrapper.md) - Final model wrapping
