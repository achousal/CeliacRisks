# ADR-006: Nested CV Structure (5x10x5)

**Status:** Accepted
**Date:** 2026-01-20

## Context

Hyperparameter tuning requires cross-validation to avoid overfitting. However, evaluating hyperparameter selection on the same data used for tuning leads to optimistic bias.

Nested CV addresses this by:
- **Outer CV:** Generates unbiased OOF predictions for model evaluation
- **Inner CV:** Tunes hyperparameters independently per outer fold

Choosing CV structure involves trade-offs:
- More folds → more stable estimates, longer runtime
- Fewer folds → faster, less stable estimates

## Decision

Use **nested CV structure: 5 outer folds × 10 repeats × 5 inner folds**.

- **Outer CV:** 5-fold repeated 10 times (50 total outer folds)
  - Generates OOF predictions for robust evaluation
- **Inner CV:** 5-fold RandomizedSearchCV with 200 iterations
  - Tunes hyperparameters independently per outer fold

**Total fits:** 5 (outer) × 10 (repeats) × 5 (inner) × 200 (iterations) = **50,000 model fits** per model type.

## Alternatives Considered

### Alternative A: Single 5-Fold CV (No Repeats)
- Faster (5 outer folds instead of 50)
- **Rejected:** Less stable OOF predictions; single fold assignment may be unlucky

### Alternative B: 10-Fold CV × 5 Repeats
- Same 50 total folds, different structure
- **Rejected:** 5×10 provides better balance (more repeats, fewer folds per repeat)

### Alternative C: 3-Fold Inner CV
- Faster hyperparameter tuning
- **Rejected:** 3 folds too few for stable hyperparameter selection

### Alternative D: Grid Search Instead of Randomized Search
- Exhaustive search over hyperparameter grid
- **Rejected:** Computationally infeasible (exponential grid size)

## Consequences

### Positive
- 50 outer folds provide robust OOF predictions
- 5 inner folds × 200 iterations provide thorough hyperparameter search
- Nested structure prevents optimistic bias in hyperparameter selection

### Negative
- 50,000 fits per model type (computationally expensive)
- 12-hour HPC job runtime per model
- High memory requirements (16 cores × 8 GB/core = 128 GB per job)

## Evidence

### Code Pointers
- [models/training.py:29-192](../../src/ced_ml/models/training.py#L29-L192) - `oof_predictions_with_nested_cv` function
- [config/schema.py](../../src/ced_ml/config/schema.py) - `CVConfig` (folds=5, repeats=10, inner_folds=5)

### Test Coverage
- `tests/test_training.py::test_oof_predictions_with_nested_cv` - Validates nested CV logic
- `tests/test_training.py::test_nested_cv_structure` - Validates fold counts

### References
- Varma, S., & Simon, R. (2006). Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*, 7(1), 91.

## Related ADRs

- Supports: [ADR-006: Hybrid Feature Selection](ADR-006-hybrid-feature-selection.md) (provides CV folds for stability)
- Supports: [ADR-007: Stability Panel](ADR-007-stability-panel.md) (50 folds for stability extraction)
- Depends on: [ADR-004: AUROC Optimization](ADR-004-auroc-optimization.md) (optimization metric)
