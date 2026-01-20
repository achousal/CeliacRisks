# ADR-005: Prevalence Adjustment via Logit Shift

**Status:** Accepted
**Date:** 2026-01-20
**Decision Makers:** Elahi Lab + Computational Team

## Context

Due to control downsampling (ADR-003), the model is trained on:
- **Training prevalence:** 1:5 case:control (≈ 16.7%)
- **Deployment prevalence:** 1:300 real-world (≈ 0.33%)

Without adjustment, predicted probabilities will be 50× too high, leading to:
- Excessive false positives in screening
- Poor calibration in deployment

Steyerberg (2019) describes a logit shift method for prevalence adjustment.

## Decision

Apply **logit shift formula** (Steyerberg, 2019, Chapter 13) to adjust predicted probabilities from training prevalence to deployment prevalence:

```
P(Y=1|X, prev_new) = sigmoid(logit(p) + logit(prev_new) - logit(prev_old))
```

Where:
- `p` = Model's predicted probability (at training prevalence)
- `prev_old` = Training prevalence (e.g., 1/6 ≈ 0.167)
- `prev_new` = Deployment prevalence (e.g., 1/300 ≈ 0.0033)

Wrap final model in `PrevalenceAdjustedModel` to apply adjustment automatically at prediction time.

## Alternatives Considered

### Alternative A: No Prevalence Adjustment
- Simpler deployment
- **Rejected:** Predicted probabilities 50× too high → clinically unusable

### Alternative B: Platt Scaling on Deployment Data
- Re-calibrate on deployment data with true prevalence
- **Rejected:** Requires labeled deployment data (not available at model training time)

### Alternative C: Sample Weights During Training
- Weight samples to reflect true prevalence during training
- **Rejected:** Discards control information; equivalent to downsampling further

### Alternative D: Threshold Adjustment Only
- Adjust decision threshold instead of probabilities
- **Rejected:** Does not fix calibration; probabilities remain incorrect for continuous risk assessment

## Consequences

### Positive
- Predicted probabilities calibrated to deployment prevalence
- Maintains discrimination (AUROC unchanged)
- Mathematically principled (derived from Bayes' theorem)
- Wrapper ensures adjustment is always applied (prevents deployment errors)

### Negative
- Adds complexity to model serialization
- Requires specifying target prevalence (from domain knowledge or deployment data)

## Evidence

### Code Pointers
- [models/calibration.py:117-149](../../src/ced_ml/models/calibration.py#L117-L149) - `adjust_probabilities_for_prevalence` function
- [models/calibration.py:152-188](../../src/ced_ml/models/calibration.py#L152-L188) - `PrevalenceAdjustedModel` wrapper class
- [evaluation/reports.py](../../src/ced_ml/evaluation/reports.py) - `ResultsWriter.save_model_artifact` (uses wrapper)

### Test Coverage
- `tests/test_models_calibration.py::test_adjust_probabilities_for_prevalence` - Validates adjustment formula
- `tests/test_models_calibration.py::test_prevalence_adjusted_model_wrapper` - Validates wrapper behavior
- `tests/test_prevalence.py` - End-to-end prevalence adjustment tests

### References
- Steyerberg, E. W. (2019). *Clinical Prediction Models: A Practical Approach to Development, Validation, and Updating* (2nd ed.), Chapter 13 (Updating for a New Setting).
- Docstring in `adjust_probabilities_for_prevalence` cites Steyerberg (2019).

## Related ADRs

- Depends on: [ADR-003: Control Downsampling](ADR-003-control-downsampling.md)
- Depends on: [ADR-004: Brier Optimization](ADR-004-brier-optimization.md) (calibration-focused)
- Supports: [ADR-011: PrevalenceAdjustedModel Wrapper](ADR-011-prevalence-wrapper.md)
