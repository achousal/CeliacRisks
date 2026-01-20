# ADR-003: Control Downsampling (1:5 Case:Control)

**Status:** Accepted
**Date:** 2026-01-20
**Decision Makers:** Elahi Lab + Computational Team

## Context

The original dataset has extreme class imbalance:
- 148 incident cases
- 43,662 controls
- Ratio: ~1:300 (0.33% prevalence)

Training with this imbalance requires:
- 300× more computational resources
- Handling class weights or resampling
- Prevalence adjustment for deployment

## Decision

Downsample controls to **1:5 case:control ratio** via random sampling without replacement, stratified by split. This reduces controls from 43,662 to ~740 (148 × 5) in the incident-only scenario.

## Alternatives Considered

### Alternative A: No Downsampling (Full 1:300 Ratio)
- Preserves natural prevalence
- **Rejected:** 300× longer training time, minimal accuracy gain, still requires prevalence adjustment

### Alternative B: 1:10 Case:Control Ratio
- More negative signal
- **Rejected:** 2× longer training than 1:5, marginal accuracy improvement in pilot tests

### Alternative C: 1:2 Case:Control Ratio
- Faster training
- **Rejected:** Insufficient negative signal in pilot tests; worse calibration

### Alternative D: SMOTE or Synthetic Oversampling
- Generate synthetic positive samples
- **Rejected:** Proteomics data has complex structure; synthetic samples may not generalize well

## Consequences

### Positive
- Reduces computational cost by 60× (300 → 5 controls per case)
- Preserves adequate negative signal for discrimination
- Faster hyperparameter tuning (50,000 fits per model becomes feasible)

### Negative
- Distribution shift from natural prevalence (1:300 → 1:5)
- Requires prevalence adjustment for deployment (see [ADR-005](ADR-005-prevalence-adjustment.md))
- Loss of some control variability (though sampled controls remain representative)

## Evidence

### Code Pointers
- [data/splits.py:193-250](../../src/ced_ml/data/splits.py#L193-L250) - `downsample_controls` function
- [config/schema.py](../../src/ced_ml/config/schema.py) - `SplitsConfig.train_control_per_case` parameter

### Test Coverage
- `tests/test_data_splits.py::test_downsample_controls` - Validates downsampling logic
- `tests/test_data_splits.py::test_case_control_ratio` - Verifies 1:5 ratio enforcement

### References
- Project memory: `project_overview` - "Control downsampling: 1:5 case:control ratio"

## Related ADRs

- Depends on: [ADR-001: Split Strategy](ADR-001-split-strategy.md)
- Requires: [ADR-005: Prevalence Adjustment](ADR-005-prevalence-adjustment.md) for deployment
