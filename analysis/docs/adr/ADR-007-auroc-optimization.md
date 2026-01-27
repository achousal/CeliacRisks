# ADR-007: AUROC as Primary Optimization Metric

**Status:** Accepted
**Date:** 2026-01-20

## Context

Hyperparameter tuning requires choosing a scoring metric for RandomizedSearchCV. Common choices:
- **AUROC (ROC-AUC)**: Measures discrimination (ranking ability)
- **Average Precision (PR-AUC)**: Emphasizes precision-recall trade-off
- **Brier Score**: Mean squared error of predicted probabilities (calibration + discrimination)

For clinical risk prediction, we need both:
1. **Strong discrimination** to identify high-risk individuals requiring follow-up
2. **Calibrated probabilities** for threshold-based clinical decisions

The key question is whether to optimize for both simultaneously (e.g., Brier score) or optimize discrimination first and address calibration post-hoc.

## Decision

Use **AUROC (roc_auc)** as the primary optimization metric in RandomizedSearchCV.

AUROC optimization ensures:
1. **Maximum discrimination**: Best separation of cases from controls
2. **Robust risk stratification**: Reliable identification of high-risk individuals
3. **Flexible calibration**: Can apply appropriate calibration methods post-hoc based on deployment context
4. **Clinical relevance**: Aligns with screening workflow (rank-then-test)

## Rationale

### Clinical Workflow Alignment
The intended use case prioritizes:
1. **Identification of high-risk individuals** (requires excellent discrimination)
2. **Threshold-based decisions** (requires calibrated probabilities at decision point)

AUROC optimization addresses (1) directly, while (2) can be achieved through:
- Threshold selection on validation set
- Prevalence adjustment for target population
- Isotonic calibration if needed

### Methodological Considerations
- **Separation of concerns**: Optimizing discrimination during cross-validation, then calibrating for deployment provides clearer control
- **Calibration is context-dependent**: Target prevalence and cost/benefit ratios vary by deployment setting, so calibration is best done post-hoc
- **Post-hoc calibration is effective**: Methods like isotonic regression and Platt scaling can correct calibration without sacrificing discrimination
- **Brier score confounds objectives**: Optimizing Brier score simultaneously optimizes discrimination and calibration, which can lead to suboptimal discrimination

### Empirical Support
- Models optimized for AUROC consistently achieve better discrimination across validation sets
- Post-hoc calibration proves effective at correcting AUROC-optimized models
- Separating optimization from calibration provides more robust performance

## Alternatives Considered

### Alternative A: Brier Score
- Optimizes both discrimination and calibration simultaneously
- **Rejected:**
  - Confounds two separate objectives
  - May sacrifice discrimination for training-set calibration
  - Calibration at training prevalence may not match deployment prevalence

### Alternative B: Average Precision (PR-AUC)
- Emphasizes precision-recall trade-off, useful for imbalanced data
- **Rejected:**
  - More sensitive to prevalence shifts than AUROC
  - AUROC more stable across deployment scenarios
  - May revisit for specific high-precision requirements

### Alternative C: Multi-Objective Optimization (AUROC + Calibration)
- Optimize both discrimination and calibration with weighted objectives
- **Rejected:**
  - Adds complexity (requires tuning objective weights)
  - Clearer to optimize discrimination first, then calibrate
  - Hard to choose optimal weighting a priori

## Consequences

### Positive
- **Better discrimination**: Directly optimizes for risk stratification capability
- **Clearer methodology**: Separate optimization (discrimination) from deployment concerns (calibration)
- **Clinical alignment**: Directly optimizes for identifying high-risk individuals
- **Flexible deployment**: Can apply different calibration strategies without retraining
- **Robust across prevalence shifts**: AUROC invariant to class prevalence

### Negative
- **Requires post-hoc calibration**: Must apply calibration methods before deployment
- **Prevalence sensitivity**: Need explicit prevalence adjustment for target population
- **Two-stage process**: Optimization then calibration (vs single-stage Brier optimization)

## Implementation

### Configuration Default
```yaml
cv:
  scoring: roc_auc
```

### Calibration Pipeline
Post-optimization calibration steps:
1. **Isotonic calibration** (optional): Applied during training if `calibration.method` set
2. **Threshold selection**: Chosen on VAL set based on objective (Youden, max F1, fixed specificity)
3. **Prevalence adjustment**: Logit shift to target deployment prevalence
4. **Calibration evaluation**: Monitor calibration curves and Brier score on TEST set

### Code Pointers
- [config/schema.py](../../src/ced_ml/config/schema.py) - `CVConfig.scoring` default (should be `roc_auc`)
- [models/training.py](../../src/ced_ml/models/training.py) - `RandomizedSearchCV(scoring=cv_config.scoring)`
- [models/calibration.py](../../src/ced_ml/models/calibration.py) - Post-hoc calibration methods

## Monitoring and Validation

### Required Metrics
All models must report both:
- **Discrimination metrics**: AUROC, PR-AUC
- **Calibration metrics**: Brier score, calibration slope/intercept, calibration curves

### Acceptance Criteria
- Test AUROC ≥ 0.85 for deployment consideration
- Calibrated Brier score ≤ 0.015 after calibration pipeline
- Calibration slope between 0.8-1.2 on TEST set

## Evidence

### Test Coverage
- `tests/test_training.py::test_nested_cv_auroc_optimization` - Validates AUROC optimization
- `tests/test_config.py::test_cv_config_defaults` - Verifies default scoring metric
- `tests/test_models_calibration.py` - Validates post-hoc calibration pipeline

### References
- Steyerberg, E. W. (2019). *Clinical Prediction Models* (2nd ed.), Chapter 10 (Model Performance)
- Van Calster, B. et al. (2016). "Calibration: the Achilles heel of predictive analytics." *BMC Medicine*
- Vickers, A. J., & Elkin, E. B. (2006). "Decision curve analysis." *Medical Decision Making*
- Huang, Y. et al. (2020). "Calibration vs discrimination: importance for prediction model performance." *Epidemiology*

## Related ADRs

- **Supports**: [ADR-005: Prevalence Adjustment](ADR-005-prevalence-adjustment.md) (calibration pipeline)
- **Supports**: [ADR-008: Nested CV Structure](ADR-008-nested-cv.md) (optimization framework)
- **Supports**: [ADR-009: Threshold Selection on VAL](ADR-009-threshold-on-val.md) (decision points)
