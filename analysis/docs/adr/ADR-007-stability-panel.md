# ADR-007: Stability Panel Extraction (0.75 Threshold)

**Status:** Accepted  
**Date:** 2026-01-20  
**Decision Makers:** Elahi Lab + Computational Team  

## Context

Nested CV with 10 repeats generates 50 trained models (5 outer folds × 10 repeats). Each model may select different features via KBest hyperparameter tuning.

Stability selection aggregates feature selections across CV folds to identify robust biomarkers:
- Features selected frequently are more likely to generalize
- Features selected rarely may be spurious or fold-specific

Choosing a stability threshold (selection frequency) balances panel size and robustness.

## Decision

Use **stability threshold = 0.75** (75% selection rate):
- Protein must appear in ≥37.5 of 50 CV folds (75% of 50)
- Ensures stable panels that generalize across data subsets

**Fallback:** If no proteins meet threshold, keep top 20 by frequency.

## Alternatives Considered

### Alternative A: Stability Threshold = 0.5 (50%)
- Larger panels
- **Rejected:** Includes less stable features; higher overfitting risk

### Alternative B: Stability Threshold = 0.9 (90%)
- Very robust panels
- **Rejected:** Too strict; often yields 0-5 proteins (insufficient for modeling)

### Alternative C: Fixed Panel Size (Top K)
- Predictable panel size
- **Rejected:** Ignores stability; may include unstable features

### Alternative D: No Fallback (Hard Threshold)
- Simplest logic
- **Rejected:** Risk of 0-feature panels if threshold too strict

## Consequences

### Positive
- 0.75 threshold balances robustness and panel size
- Fallback (top 20) ensures non-empty panels
- Stable panels generalize better than single-fold selections

### Negative
- Threshold choice is somewhat arbitrary (0.75 vs. 0.7 vs. 0.8)
- Fallback adds complexity (but prevents catastrophic failure)

## Evidence

### Code Pointers
- [features/stability.py:124-216](../../src/ced_ml/features/stability.py#L124-L216) - `extract_stable_panel` function
- [config/schema.py](../../src/ced_ml/config/schema.py) - `FeatureConfig.stability_thresh` parameter

### Test Coverage
- `tests/test_features_stability.py::test_extract_stable_panel_threshold` - Validates threshold logic
- `tests/test_features_stability.py::test_extract_stable_panel_fallback` - Validates fallback (top 20)

### References
- Meinshausen, N., & Bühlmann, P. (2010). Stability selection. *Journal of the Royal Statistical Society: Series B*, 72(4), 417-473.

## Related ADRs

- Depends on: [ADR-006: Hybrid Feature Selection](ADR-006-hybrid-feature-selection.md)
- Depends on: [ADR-008: Nested CV Structure](ADR-008-nested-cv.md) (provides 50 folds)
