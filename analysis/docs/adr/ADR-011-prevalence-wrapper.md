# ADR-011: PrevalenceAdjustedModel Wrapper Serialization

**Status:** Accepted  
**Date:** 2026-01-20  
**Decision Makers:** Elahi Lab + Computational Team  

## Context

Prevalence adjustment (ADR-005) is required for deployment due to control downsampling (ADR-003). The adjustment must be applied at prediction time:

```python
adjusted_proba = adjust_probabilities_for_prevalence(
    raw_proba, prev_old=0.167, prev_new=0.0033
)
```

However, if prevalence adjustment is a separate post-processing step:
- **Risk:** User forgets to apply adjustment → incorrect probabilities
- **Complexity:** Deployment code must remember to call adjustment function
- **Non-reproducibility:** Adjustment parameters may be lost or misconfigured

## Decision

Wrap final trained model in **`PrevalenceAdjustedModel`** wrapper class before serialization.

The wrapper:
- Inherits from sklearn `BaseEstimator` and `ClassifierMixin`
- Stores `base_model`, `prev_old`, and `prev_new`
- Applies prevalence adjustment automatically in `predict_proba()`

Serialized model artifact (`final_model.pkl`) contains the wrapper, ensuring adjustment is always applied.

## Alternatives Considered

### Alternative A: Separate Adjustment Step
- User calls `adjust_probabilities_for_prevalence()` after loading model
- **Rejected:** Error-prone (user may forget); not reproducible

### Alternative B: Document Adjustment in README
- Rely on documentation to remind users
- **Rejected:** Documentation not enforced; high risk of deployment error

### Alternative C: Model Retraining at True Prevalence
- Retrain model with true prevalence (1:300)
- **Rejected:** Computationally infeasible (300× longer training)

### Alternative D: Calibration at True Prevalence
- Use CalibratedClassifierCV with true prevalence
- **Rejected:** Requires labeled data at true prevalence (not available)

## Consequences

### Positive
- Serialized model automatically applies prevalence adjustment
- Prevents deployment errors (no separate adjustment step)
- Sklearn-compatible (`BaseEstimator`, `ClassifierMixin`) → works with standard pipelines
- Adjustment parameters stored in model artifact (reproducible)

### Negative
- Adds wrapper layer (slightly more complex serialization)
- `base_model` must be extracted if raw probabilities needed

## Evidence

### Code Pointers
- [models/calibration.py:152-188](../../src/ced_ml/models/calibration.py#L152-L188) - `PrevalenceAdjustedModel` class
- [evaluation/reports.py](../../src/ced_ml/evaluation/reports.py) - `ResultsWriter.save_model_artifact` (uses wrapper)
- [models/calibration.py:117-149](../../src/ced_ml/models/calibration.py#L117-L149) - `adjust_probabilities_for_prevalence` (used by wrapper)

### Test Coverage
- `tests/test_models_calibration.py::test_prevalence_adjusted_model_wrapper` - Validates wrapper behavior
- `tests/test_models_calibration.py::test_prevalence_adjusted_model_serialization` - Validates pickle roundtrip
- `tests/test_models_calibration.py::test_prevalence_adjusted_model_sklearn_compat` - Validates sklearn compatibility

### References
- sklearn API documentation: `BaseEstimator` and `ClassifierMixin` contracts.

## Related ADRs

- Depends on: [ADR-005: Prevalence Adjustment](ADR-005-prevalence-adjustment.md) (adjustment formula)
- Depends on: [ADR-003: Control Downsampling](ADR-003-control-downsampling.md) (creates need for adjustment)
