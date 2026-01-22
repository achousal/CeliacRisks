# ADR-010: Fixed Specificity Objective (95%)

**Status:** Accepted
**Date:** 2026-01-20

## Context

Threshold selection requires choosing an objective function. Common objectives:
- **Youden's J:** Sensitivity + Specificity - 1 (balances both)
- **Max F1:** Maximize F1 score (balances precision and recall)
- **Fixed Specificity:** Achieve target specificity (e.g., 95%)
- **Fixed PPV:** Achieve target positive predictive value

For **clinical screening**, high specificity minimizes false positives:
- False positives → unnecessary follow-up tests, anxiety, costs
- High specificity (95%) → only 5% of controls flagged

## Decision

Support **fixed specificity objective** with default 95% specificity target.

This objective selects the threshold that achieves the closest specificity to the target (e.g., 95%), prioritizing specificity over sensitivity.

**Note:** Youden and max_f1 are also supported for comparison.

## Alternatives Considered

### Alternative A: Youden's J Only
- Balances sensitivity and specificity equally
- **Rejected:** May yield lower specificity than desired for screening (e.g., 85% spec, 80% sens)

### Alternative B: Max F1 Only
- Optimizes F1 score (harmonic mean of precision and recall)
- **Rejected:** F1 emphasizes precision/recall balance, not specificity

### Alternative C: Fixed PPV
- Targets positive predictive value instead of specificity
- **Rejected:** PPV depends on prevalence; less stable across deployment scenarios

### Alternative D: Hard-Coded 95% Specificity
- No configurability
- **Rejected:** Inflexible; different use cases may require different specificity targets

## Consequences

### Positive
- 95% specificity minimizes false positives for screening
- Configurable target allows flexibility (e.g., 90%, 95%, 99%)
- Clinical interpretation: "Only 5% of controls flagged for follow-up"

### Negative
- Lower sensitivity than Youden or max_f1 (trade-off)
- Specificity target may not be achievable (model may plateau at lower spec)

## Evidence

### Code Pointers
- [config/schema.py:198-208](../../src/ced_ml/config/schema.py#L198-L208) - `ThresholdConfig.fixed_spec` parameter
- [metrics/thresholds.py:326-377](../../src/ced_ml/metrics/thresholds.py#L326-L377) - `choose_threshold_objective` (supports `fixed_spec`)
- Example config: `docs/examples/training_config.yaml` - `thresholds.objective: fixed_spec`, `fixed_spec: 0.95`

### Test Coverage
- `tests/test_metrics_thresholds.py::test_choose_threshold_fixed_spec` - Validates fixed_spec objective
- `tests/test_config.py::test_threshold_config_fixed_spec` - Validates config validation

### References
- Clinical screening guidelines typically target high specificity (90-99%) to minimize false positives.

## Related ADRs

- Depends on: [ADR-009: Threshold on VAL](ADR-009-threshold-on-val.md) (threshold source)
