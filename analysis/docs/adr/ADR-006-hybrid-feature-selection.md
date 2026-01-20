# ADR-006: Hybrid Feature Selection

**Status:** Accepted  
**Date:** 2026-01-20  
**Decision Makers:** Elahi Lab + Computational Team  

## Context

With 2,920 proteins, feature selection is essential to:
1. Reduce overfitting
2. Improve computational efficiency
3. Identify stable biomarker panels

Multiple feature selection methods exist:
- **Screening:** Univariate filters (fast, no interactions)
- **KBest:** sklearn SelectKBest (fast, tunable k)
- **Stability Selection:** Track features selected across CV folds

Each method has strengths; combining them may yield better panels.

## Decision

Use **hybrid feature selection**: Screening → KBest → Stability.

**Pipeline:**
1. **Screening** (Mann-Whitney U or F-statistic) → top 1,000 proteins
2. **KBest** (SelectKBest with f_classif) → tune k via inner CV
3. **Stability** → extract proteins selected in ≥75% of CV folds

Order configurable via `hybrid_kbest_first` flag (default: True).

## Alternatives Considered

### Alternative A: KBest Only
- Simplest approach
- **Rejected:** Less stable panels, overfitting risk (k optimized per CV fold without stability constraint)

### Alternative B: Stability Only
- Most robust panels
- **Rejected:** Slower (requires full CV); no k tuning

### Alternative C: Screening Only
- Fastest
- **Rejected:** No multivariate optimization; ignores feature interactions

### Alternative D: L1 Regularization (Lasso)
- Embedded feature selection
- **Rejected:** Model-specific; hybrid approach is model-agnostic

## Consequences

### Positive
- Screening reduces search space (2,920 → 1,000)
- KBest optimizes k via CV
- Stability ensures robust panels (≥75% selection rate)
- Hybrid approach balances speed, tunability, and robustness

### Negative
- More complex than single-method selection
- Stability requires tracking selections across CV folds (memory overhead)

## Evidence

### Code Pointers
- [config/schema.py:83-105](../../src/ced_ml/config/schema.py#L83-L105) - `FeatureConfig` class
- [features/screening.py](../../src/ced_ml/features/screening.py) - Screening methods
- [features/kbest.py](../../src/ced_ml/features/kbest.py) - KBest wrapper
- [features/stability.py:124-216](../../src/ced_ml/features/stability.py#L124-L216) - `extract_stable_panel`

### Test Coverage
- `tests/test_features_screening.py` - Validates screening methods
- `tests/test_features_kbest.py` - Validates KBest wrapper
- `tests/test_features_stability.py` - Validates stability extraction

### References
- Meinshausen, N., & Bühlmann, P. (2010). Stability selection. *Journal of the Royal Statistical Society: Series B*, 72(4), 417-473.

## Related ADRs

- Supports: [ADR-007: Stability Panel](ADR-007-stability-panel.md)
- Depends on: [ADR-008: Nested CV Structure](ADR-008-nested-cv.md) (provides CV folds for stability)
