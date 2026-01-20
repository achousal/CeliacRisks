# ADR-009: Threshold Selection on VAL

**Status:** Accepted  
**Date:** 2026-01-20  
**Decision Makers:** Elahi Lab + Computational Team  

## Context

Binary classifiers output continuous probabilities; a decision threshold converts probabilities to binary predictions. The threshold must be chosen to optimize a specific criterion (e.g., Youden's J, max F1, fixed specificity).

Threshold selection requires data:
- **TRAIN:** Biased (model trained on this data)
- **VAL:** Unbiased for threshold selection (not used for hyperparameter tuning)
- **TEST:** Must remain completely held-out for final evaluation

Selecting threshold on TEST leads to optimistic bias in reported TEST metrics (leakage).

## Decision

**Select threshold on VAL set**, never on TEST.

TEST remains completely held-out until final evaluation. Threshold selected on VAL is then applied to TEST for unbiased performance estimation.

## Alternatives Considered

### Alternative A: Threshold on TEST
- Simpler (no VAL set needed)
- **Rejected:** Optimistic bias in TEST metrics (threshold tuned to TEST data)

### Alternative B: Threshold on TRAIN (OOF Predictions)
- No need for VAL set
- **Rejected:** TRAIN may be overfitted; threshold may not generalize to VAL/TEST

### Alternative C: Fixed Threshold (e.g., 0.5)
- No tuning needed
- **Rejected:** Arbitrary threshold may be suboptimal; ignores class imbalance and cost considerations

### Alternative D: Nested Threshold Selection (Inner CV)
- Threshold tuned during hyperparameter tuning
- **Rejected:** Couples threshold to hyperparameters; less flexible for post-hoc threshold adjustment

## Consequences

### Positive
- VAL provides unbiased threshold selection
- TEST remains completely held-out (no leakage)
- Threshold can be adjusted post-hoc without retraining

### Negative
- Requires 3-way split (reduces TRAIN size)
- VAL size (25%) must be large enough for stable threshold selection

## Evidence

### Code Pointers
- [config/schema.py:198-208](../../src/ced_ml/config/schema.py#L198-L208) - `ThresholdConfig.threshold_source`
- [cli/train.py](../../src/ced_ml/cli/train.py) - Threshold selection logic
- [metrics/thresholds.py:326-377](../../src/ced_ml/metrics/thresholds.py#L326-L377) - `choose_threshold_objective`

### Test Coverage
- `tests/test_metrics_thresholds.py::test_choose_threshold_on_val` - Validates threshold selection on VAL
- `tests/test_config.py::test_threshold_source_validation` - Enforces `threshold_source='val'`

### References
- Steyerberg, E. W. (2019). *Clinical Prediction Models* (2nd ed.), Chapter 11 (Choosing Between Alternative Strategies).

## Related ADRs

- Depends on: [ADR-001: Split Strategy](ADR-001-split-strategy.md) (provides VAL set)
- Supports: [ADR-010: Fixed Spec 95%](ADR-010-fixed-spec-95.md) (threshold objective)
