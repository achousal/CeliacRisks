# ADR-009: OOF Stacking Ensemble

**Status:** Accepted
**Date:** 2026-01-22
**Decision Makers:** Elahi Lab + Computational Team

## Context

Single models (LR_EN, RF, XGBoost) achieve AUROC ~0.85-0.89 on the Celiac dataset. Model ensembling can improve predictive performance by combining diverse base models, but naive ensembling risks information leakage if training data predictions are used to train the meta-learner.

The challenge: how to combine multiple trained models without leaking information from the training process into the ensemble weights.

## Decision

Implement a **stacking ensemble** using **out-of-fold (OOF) predictions** to train the meta-learner:

1. **OOF predictions**: Each base model's nested CV produces OOF predictions where each training sample's prediction comes from a fold where it was held out
2. **Meta-learner**: Logistic Regression (L2 penalty) trained on stacked OOF predictions
3. **Calibration**: Optional isotonic calibration of meta-learner output
4. **No leakage**: Meta-learner never sees predictions made on samples that were in the training fold

Architecture:
```
Base Models (trained independently)
    |
    v
OOF Predictions (n_samples x n_models)
    |
    v
Meta-Learner (LogisticRegression L2)
    |
    v
Calibrated Ensemble Probability
```

## Alternatives Considered

1. **Simple averaging**: Average base model probabilities
   - Rejected: No learned weighting, suboptimal combination

2. **Train meta-learner on training predictions**:
   - Rejected: Information leakage - meta-learner overfits to training predictions

3. **Separate holdout for meta-learner**:
   - Rejected: Reduces training data for base models

4. **Blending with single holdout**:
   - Rejected: Less efficient use of data than OOF approach

## Consequences

### Positive
- Expected +2-5% AUROC improvement over best single model
- No information leakage from training to meta-learner
- Interpretable meta-learner weights show model contributions
- Calibrated final predictions
- Reuses existing OOF predictions from nested CV

### Negative
- Requires all base models trained on same splits
- Additional complexity in training pipeline
- Meta-learner hyperparameters add tuning burden
- Ensemble predictions harder to explain than single model

## Evidence

### Code Pointers
- [models/stacking.py:61-160](../../src/ced_ml/models/stacking.py#L61-L160) - StackingEnsemble class
- [models/stacking.py:200-280](../../src/ced_ml/models/stacking.py#L200-L280) - fit_from_oof method
- [cli/train_ensemble.py](../../src/ced_ml/cli/train_ensemble.py) - CLI for ensemble training

### Test Coverage
- `tests/test_models_stacking.py` - Unit tests for stacking logic (23 tests)
- Tests validate OOF prediction handling, meta-learner fitting, and prediction aggregation

### References
- Wolpert (1992). Stacked Generalization. Neural Networks.
- Breiman (1996). Stacked Regressions. Machine Learning.
- Van der Laan et al. (2007). Super Learner. Statistical Applications in Genetics.

## Related ADRs

- Depends on: [ADR-006: Nested CV Structure](ADR-006-nested-cv.md) - Nested CV provides OOF predictions
- Depends on: [ADR-010: Prevalence Adjustment](ADR-010-prevalence-adjustment.md) - Applied for final calibration
- Related: [ADR-008: Optuna Hyperparameter Optimization](ADR-008-optuna-hyperparameter-optimization.md) - Optimizes base model hyperparameters
