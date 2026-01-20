# ADR-001: Split Strategy (50/25/25 Three-Way)

**Status:** Accepted
**Date:** 2026-01-20
**Decision Makers:** Elahi Lab + Computational Team

## Context

In supervised machine learning for clinical risk prediction, data must be split to enable:
1. Model training and hyperparameter tuning
2. Threshold selection for decision-making
3. Unbiased final evaluation

A common approach is a 2-way split (train/test), but this forces threshold selection on the test set, leading to optimistic bias in reported test metrics (leakage).

## Decision

Use a **3-way stratified split: 50% TRAIN / 25% VAL / 25% TEST**.

- **TRAIN (50%)**: For nested CV hyperparameter tuning and feature selection
- **VAL (25%)**: For threshold selection (avoid TEST leakage)
- **TEST (25%)**: For final unbiased evaluation

Stratification is performed by target (`incident_CeD`) to preserve class balance.

## Alternatives Considered

### Alternative A: 2-Way Split (Train 75% / Test 25%)
- Simpler structure
- More training data
- **Rejected:** Forces threshold selection on TEST → optimistic bias in reported metrics

### Alternative B: 2-Way with Nested Threshold Selection
- Use inner CV for threshold selection within TRAIN
- **Rejected:** Less stable threshold estimates, still no held-out VAL set for calibration verification

### Alternative C: 4-Way Split (Train / Val / Test / Holdout)
- Additional holdout set for final verification
- **Rejected:** Insufficient sample size for 4-way split (only 148 incident cases)

### Alternative D: Larger TRAIN Fraction (70/15/15)
- More training data
- **Rejected:** Reduces VAL and TEST sizes, leading to less stable threshold and metric estimates

## Consequences

### Positive
- Threshold selected on VAL prevents TEST leakage
- TEST remains completely held-out until final evaluation
- VAL and TEST sizes (25% each) provide stable estimates despite small incident count

### Negative
- Smaller TRAIN set (50%) compared to 2-way split (75%)
- Reduced statistical power for hyperparameter tuning

## Evidence

### Code Pointers
- [data/splits.py:374-438](../../src/ced_ml/data/splits.py#L374-L438) - `stratified_train_val_test_split` function
- [config/schema.py](../../src/ced_ml/config/schema.py) - `SplitsConfig.validate_split_sizes` validator

### Test Coverage
- `tests/test_data_splits.py` - Validates stratified split logic, fraction enforcement

### References
- Steyerberg, E. W. (2019). *Clinical Prediction Models: A Practical Approach to Development, Validation, and Updating* (2nd ed.). Springer.

## Related ADRs

- Depends on: None
- Supports: [ADR-009: Threshold Selection on VAL](ADR-009-threshold-on-val.md)
- Supports: [ADR-002: Prevalent→TRAIN](ADR-002-prevalent-train-only.md)
