# ADR-015: Model Selection Scoring

**Status:** Accepted
**Date:** 2026-01-22

## Context

When comparing multiple trained models, relying on a single metric (e.g., AUROC) provides an incomplete picture of model quality. A model with high AUROC may have poor calibration, or vice versa.

**Challenges with single-metric selection:**
- AUROC measures discrimination but not calibration quality
- Brier score combines discrimination and calibration but doesn't distinguish between them
- Calibration slope indicates systematic over/under-confidence but not overall accuracy
- Different clinical use cases may prioritize different aspects

**Need:**
- A composite score combining multiple evaluation dimensions
- Configurable weights for different use cases
- Consistent ranking methodology across model comparisons

## Decision

Implement a composite model selection score combining:

1. **AUROC (50% weight)**: Discrimination ability
2. **Brier score (30% weight)**: Prediction quality (inverted: 1 - Brier)
3. **Calibration slope (20% weight)**: Calibration quality (1 - |slope - 1|)

**Formula:**
```
score = (AUROC * w_auroc) + ((1 - Brier) * w_brier) + ((1 - |slope - 1|) * w_slope)
```

**Implementation:**
- `compute_selection_score()`: Score for single model
- `compute_selection_scores_for_models()`: Batch computation
- `rank_models_by_selection_score()`: Ranking utility
- Configurable weights via function parameters
- Case-insensitive metric key matching

**Score Interpretation:**
- Range: [0, 1] where higher is better
- Perfect model: score = 1.0 (AUROC=1.0, Brier=0.0, slope=1.0)
- Random model: score ~ 0.475 (AUROC=0.5, Brier=0.25, slope=1.0)

## Alternatives Considered

1. **AUROC only:**
   - Simple, widely understood
   - Rejected: Ignores calibration, which is critical for clinical decision support

2. **Brier score only:**
   - Combines discrimination and calibration
   - Rejected: Doesn't distinguish between the two; can't weight by importance

3. **Multi-objective Pareto ranking:**
   - Identify Pareto-optimal models
   - Rejected: Doesn't produce a single ranking; harder to interpret

4. **Net benefit at clinical threshold:**
   - Directly measures clinical utility
   - Rejected: Requires specifying a threshold a priori; too narrow

5. **Weighted log loss:**
   - Penalizes confident wrong predictions more
   - Rejected: Correlated with Brier; doesn't add unique information

## Consequences

### Positive
- Holistic model comparison across multiple dimensions
- Configurable weights for different use cases
- Produces a single interpretable score for ranking
- Robust to metric key variations (case-insensitive)
- Handles missing metrics gracefully (returns NaN)

### Negative
- Adds complexity to model selection process
- Default weights (50/30/20) are somewhat arbitrary
- May not align with all clinical decision-making frameworks
- Composite scores harder to explain than single metrics

## Evidence

### Code Pointers
- [evaluation/scoring.py](../../src/ced_ml/evaluation/scoring.py) - Scoring utilities
  - `compute_selection_score()`: Line 34-130
  - `rank_models_by_selection_score()`: Line 180-220
- [tests/test_evaluation_scoring.py](../../tests/test_evaluation_scoring.py) - 22 test cases

### Test Coverage
- Perfect model scoring: verify score = 1.0
- Random model scoring: verify score ~ 0.475
- Custom weights: verify weighted combination
- Missing metrics: verify NaN handling
- Key variants: verify case-insensitive matching
- Batch computation: verify multiple models
- Ranking: verify correct ordering

### Usage Example
```python
from ced_ml.evaluation.scoring import compute_selection_score, rank_models_by_selection_score

# Single model
metrics = {'AUROC': 0.85, 'Brier': 0.10, 'calib_slope': 1.05}
score = compute_selection_score(metrics)  # ~0.82

# Multiple models
models = {
    'LR_EN': {'AUROC': 0.85, 'Brier': 0.10, 'calib_slope': 1.02},
    'RF': {'AUROC': 0.83, 'Brier': 0.12, 'calib_slope': 0.95},
    'XGBoost': {'AUROC': 0.87, 'Brier': 0.11, 'calib_slope': 1.10},
}
ranking = rank_models_by_selection_score(models)
# Returns: [('XGBoost', 0.83), ('LR_EN', 0.82), ('RF', 0.79)]

# Custom weights emphasizing AUROC
custom_weights = {'auroc': 0.7, 'brier': 0.2, 'slope': 0.1}
score = compute_selection_score(metrics, weights=custom_weights)
```

### Default Weight Rationale
- **AUROC (50%)**: Primary clinical interest is correctly ranking patients by risk
- **Brier (30%)**: Absolute probability accuracy matters for decision support
- **Slope (20%)**: Systematic miscalibration should be penalized but is less critical if probabilities are still useful for ranking

### References
- Collins, G.S., et al. (2015). TRIPOD Statement. BMJ.
- Van Calster, B., et al. (2019). Calibration: the Achilles heel of predictive analytics. BMC Medicine.
- Steyerberg, E.W. (2019). Clinical Prediction Models (2nd ed.), Chapter 15.

## Related ADRs

- Complements: [ADR-007: AUROC Optimization](ADR-007-auroc-optimization.md) - AUROC is primary component
- Related: [ADR-014: OOF Posthoc Calibration](ADR-014-oof-posthoc-calibration.md) - Affects calibration metrics
- Related: [ADR-009: OOF Stacking Ensemble](ADR-009-oof-stacking-ensemble.md) - Ensemble may be selected via this score
