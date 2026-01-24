# ADR-010: Prevalence Adjustment Strategy (Future Deployment Concern)

**Status:** Accepted (deployment-only; not integrated into training)
**Date:** 2026-01-20
**Updated:** 2026-01-24 (clarified as deployment-only concern)

---

## Context

Our pipeline uses control downsampling uniformly across all data splits:

- **TRAIN prevalence:** 5:1 case:control (16.7%) via `train_control_per_case: 5`
- **VAL prevalence:** 5:1 case:control (16.7%) via `eval_control_per_case: 5`
- **TEST prevalence:** 5:1 case:control (16.7%) via `eval_control_per_case: 5`
- **Real-world deployment prevalence:** ~1:300 (0.34%)

**Critical observation:** All three pipeline sets (TRAIN/VAL/TEST) are at the same prevalence (16.7%).

### Why This Matters for Deployment

Without prevalence adjustment, predicted probabilities will be ~50× too high in real-world use (0.34% prevalence), leading to:
- Excessive false positives in screening
- Clinically invalid risk scores
- Unusable thresholds

### Why Not During Training

Since TRAIN, VAL, and TEST are all at 16.7% prevalence, **no adjustment is needed during training**:
- Model trains, validates, and tests in the same prevalence environment
- Threshold selection on VAL is unbiased (no prevalence mismatch)
- Calibration is valid (OOF calibrator fit on 16.7% data, applied to 16.7% test data)
- Adding adjustment during testing would be a no-op and add unnecessary complexity

---

## Decision

**DO NOT apply prevalence adjustment during training, validation, or testing.** All three sets are at the same prevalence (16.7%), so adjustment would provide no benefit.

**DO plan for logit-scale adjustment at clinical deployment time** if the model is used for real-world screening at 0.34% prevalence.

### Logit-Scale Adjustment Formula (Steyerberg, 2019)

For future deployment, apply the following transformation:

```
P_adjusted(Y=1|X) = sigmoid(logit(p) + logit(π_new) - logit(π_old))

where:
  p           = Model's predicted probability (at training prevalence = 16.7%)
  π_old       = Training/test prevalence (0.167)
  π_new       = Real-world deployment prevalence (0.0034)
  sigmoid(x)  = 1 / (1 + exp(-x))
  logit(π)    = log(π / (1 - π))
```

### Example Deployment Code (Speculative)

This is provided for future reference; **not currently integrated**:

```python
import numpy as np
from scipy.special import expit, logit

def adjust_for_deployment(p, training_prev=0.167, deployment_prev=0.0034):
    """
    Adjust predicted probabilities from training prevalence to deployment prevalence.

    Use case: Load a trained model and wrap predictions for clinical screening.

    Args:
        p: Model's predicted probability (at training prevalence = 16.7%)
        training_prev: Prevalence during model training (default: 0.167)
        deployment_prev: Real-world prevalence (default: 0.0034)

    Returns:
        Adjusted probability at deployment prevalence
    """
    # Avoid numerical issues
    p = np.clip(p, 1e-10, 1 - 1e-10)

    # Logit-scale adjustment
    logit_p = logit(p)
    adjustment = logit(deployment_prev) - logit(training_prev)
    adjusted_logit = logit_p + adjustment

    return expit(adjusted_logit)  # sigmoid
```

---

## Alternatives Considered

### Alternative A: No Prevalence Adjustment
- Simpler deployment
- **Rejected:** Predicted probabilities 50× too high → clinically unusable

### Alternative B: Platt Scaling on Deployment Data
- Re-calibrate on deployment data with true prevalence
- **Rejected:** Requires labeled deployment data at model training time (unavailable)

### Alternative C: Sample Weights During Training
- Weight samples to reflect true prevalence (0.34%) during training
- **Rejected:** Discards control information; performance would suffer

### Alternative D: Threshold Adjustment Only
- Adjust decision threshold instead of probabilities
- **Rejected:** Does not fix calibration; probabilities remain incorrect for risk assessment

---

## Consequences

### Positive
- **Clean separation of concerns:** Training remains simple; deployment handles prevalence adaptation
- **Valid training logic:** No no-op wrappers during testing; threshold selection unbiased
- **Mathematically principled:** Logit-space adjustment derived from Bayes' theorem
- **Preserves discrimination:** AUROC unchanged by adjustment
- **Future-proof:** When deployment occurs, adjustment is straightforward

### Negative
- **Not implemented yet:** Requires future work for actual clinical deployment
- **Requires domain knowledge:** Target prevalence (0.0034) must be known at deployment time
- **External dependency:** Deployment code is outside pipeline (not version-controlled in this repo)

---

## Deployment Checklist (When Needed)

Before deploying the model for real-world screening:

1. [ ] Implement `adjust_for_deployment()` function (or use external library)
2. [ ] Load trained model from `results/{MODEL}/split_seed*/core/final_model.pkl`
3. [ ] Wrap predictions with prevalence adjustment (training_prev=0.167, deployment_prev=0.0034)
4. [ ] Obtain labeled deployment cohort (real-world 0.34% prevalence)
5. [ ] Validate discrimination (AUROC should match test set ~0.89)
6. [ ] Validate calibration (probability calibration at 0.34% prevalence)
7. [ ] Re-optimize or validate clinical thresholds on deployment cohort
8. [ ] Document model card, deployment protocol, and monitoring procedures
9. [ ] Set up drift detection for incoming patient data
10. [ ] Monitor performance metrics over time

---

## Evidence and References

- **Steyerberg, E. W.** (2019). *Clinical Prediction Models: A Practical Approach to Development, Validation, and Updating* (2nd ed.), Chapter 13 (Updating for a New Setting).
  - Describes logit-shift method and theoretical justification
  - Provides guidance on prevalence adjustment and recalibration

- **Best practices guide:** See [DEPLOYMENT.md](../development/DEPLOYMENT.md) for speculative deployment workflow, threshold re-calibration, and validation on deployment data.

---

## Related ADRs

- Depends on: [ADR-001: Split Strategy](ADR-001-split-strategy.md) (ensures uniform 16.7% prevalence across splits)
- Depends on: [ADR-003: Control Downsampling](ADR-003-control-downsampling.md) (establishes 5:1 ratio)
- Supports: [ADR-011: Threshold Optimization on Validation Set](ADR-011-threshold-on-val.md) (threshold selection unbiased due to no prevalence mismatch)

---

## Implementation Notes

**Current Status:**
- ✅ Logit-scale formula documented
- ✅ Deployment guide written ([DEPLOYMENT.md](../development/DEPLOYMENT.md))
- ❌ Not integrated into training pipeline
- ❌ Not integrated into inference

**If Deploying:**
1. Implement adjustment function (code provided above)
2. Create `DeploymentModel` wrapper class for sklearn compatibility
3. Validate thoroughly on real-world cohort
4. Version control deployment code separately (not in this repo)
5. Monitor for data drift and model decay

---

**Last Updated:** 2026-01-24
**Status:** Speculative design for future deployment (not production-ready)
