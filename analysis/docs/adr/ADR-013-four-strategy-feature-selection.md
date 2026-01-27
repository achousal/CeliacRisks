# ADR-013: Four-Strategy Feature Selection Framework

**Status:** Accepted
**Date:** 2026-01-26

## Context

Feature selection is critical for biomarker discovery and clinical deployment, but no single approach satisfies all use cases:

1. **Production models** need fast, reproducible feature selection with explicit hyperparameter tuning
2. **Scientific discovery** requires feature stability analysis across CV folds to identify robust biomarkers
3. **Clinical deployment** demands panel size optimization balancing cost vs. performance
4. **Regulatory validation** requires unbiased performance estimates on predetermined panels

Initial implementation provided only hybrid feature selection (screen → k-best → stability). This proved insufficient for:
- Understanding which proteins are consistently selected across folds (scientific rigor)
- Exploring panel size trade-offs after model training (deployment decisions)
- Validating specific panels without selection bias (regulatory compliance)

## Decision

Implement **four distinct feature selection strategies**, each optimized for different phases and objectives:

### Strategy 1: Hybrid Stability (Default, During Training)
**Pipeline:** screen → k-best (tuned) → stability → correlation pruning → model

**When to use:**
- Production models requiring fast training (~30 minutes)
- Reproducible feature selection with explicit k tuning
- Default choice for model development

**Config:** `feature_selection_strategy: hybrid_stability`

**Outputs:** `feature_selection/stability/stable_panel_t{thresh}.csv`

### Strategy 2: Nested RFECV (During Training)
**Pipeline:** screen → [k-best pre-filter] → RFECV (per fold) → consensus panel → model

**When to use:**
- Scientific papers requiring feature stability metrics
- Understanding which proteins are robust vs. unstable
- Automatic panel size discovery (no manual k_grid)
- Willing to accept 5-10× longer training time

**Config:** `feature_selection_strategy: rfecv`

**Outputs:** `cv/rfecv/consensus_panel.csv`, `feature_stability.csv`

### Strategy 3: Post-hoc RFE (After Training)
**Pipeline:** Trained model → RFE on stability ranking → Pareto curve

**When to use:**
- Exploring panel size trade-offs after training (~5 minutes)
- Stakeholder decisions: "What's the smallest panel maintaining 0.90 AUROC?"
- Rapid iteration on deployment sizing

**CLI:** `ced optimize-panel --model-path ... --start-size 100 --min-size 5`

**Outputs:** `optimize_panel/panel_curve.png`, `recommendations.json`

### Strategy 4: Fixed Panel Validation (During Training)
**Pipeline:** Bypass feature selection → train on predetermined panel → unbiased AUROC

**When to use:**
- Validating consensus panels from discovery phase
- Comparing to published panels
- Regulatory submission (FDA, clinical deployment)
- Literature benchmarking

**CLI:** `ced train --fixed-panel panel.csv --split-seed 10`  (new seed critical)

**Outputs:** Standard training outputs with unbiased test metrics

## Mutually Exclusive vs. Complementary Methods

**Mutually exclusive (choose during training):**
- Hybrid Stability vs. Nested RFECV
- Both operate during training, controlled by `feature_selection_strategy`

**Complementary (post-training tools):**
- Post-hoc RFE: Runs after any training method, explores panel sizes
- Fixed Panel: Validates specific panels on new splits

**Typical workflow:**
```bash
# Phase 1: Discovery (choose one)
ced train --model LR_EN --split-seed 0  # hybrid_stability OR rfecv

# Phase 2: Optimization (optional, complements both)
ced optimize-panel --model-path results/LR_EN/split_seed0/core/LR_EN__final_model.joblib

# Phase 3: Validation (complements all methods)
ced train --fixed-panel deployment_panel.csv --split-seed 10  # NEW SEED
```

## Alternatives Considered

### Alternative A: Single Unified Method
- Simplest implementation
- **Rejected:** No single method satisfies speed, stability, and deployment trade-off analysis simultaneously

### Alternative B: Two Methods (Hybrid + RFECV Only)
- Covers training-time needs
- **Rejected:** Missing post-training panel optimization (stakeholder decisions) and validation (regulatory needs)

### Alternative C: Nested RFECV Only
- Most scientifically rigorous
- **Rejected:** Too slow for routine use (45× slower than hybrid); not suitable as default

### Alternative D: Post-hoc RFE Only
- Fastest for panel size exploration
- **Rejected:** Cannot be used during training; requires pre-trained model; slight optimism bias (~0.5%)

## Consequences

### Positive
- Clear use case separation: production vs. discovery vs. deployment vs. validation
- Users can choose appropriate speed-rigor trade-off
- Post-hoc RFE enables rapid iteration without retraining
- Fixed panel validation provides regulatory-grade unbiased estimates
- Scientific papers can report feature stability metrics (nested RFECV)
- Clinical deployment gets stakeholder-friendly Pareto curves (post-hoc RFE)

### Negative
- More complex than single-method approach
- Users must understand which method suits their goal
- Four different output formats/locations to track
- Documentation overhead (this ADR, consolidated FEATURE_SELECTION.md)

### Performance Characteristics

**Runtime comparison** (LR_EN, 43k samples, 2920 proteins, 5 folds × 3 repeats):
```
Hybrid Stability:  ~30 minutes  (baseline, 1.0×)
Nested RFECV:      ~5 hours     (10× slower, with k-best pre-filter)
                   ~22 hours    (45× slower, without pre-filter)
Post-hoc RFE:      ~5 minutes   (6× faster than training)
Fixed Panel:       ~30 minutes  (same as hybrid, no selection overhead)
```

**AUROC comparison** (typical results):
```
Hybrid Stability (k=100):  0.945 ± 0.015  (tuned panel size)
Nested RFECV (optimal):    0.943 ± 0.018  (automatic sizing, slightly lower due to conservatism)
Post-hoc RFE (50 proteins): 0.940 ± 0.012  (optimistic by ~0.5% due to post-hoc selection)
Fixed Panel (50 proteins):  0.938 ± 0.014  (unbiased ground truth)
```

### Data Leakage Prevention

All strategies prevent data leakage through different mechanisms:

**Hybrid Stability:** Per-fold feature selection (screening, k-best), post-hoc stability aggregation

**Nested RFECV:** RFECV uses internal CV within each outer fold only

**Post-hoc RFE:** Uses validation set for AUROC; slight optimism expected and documented

**Fixed Panel:** No feature selection; new split seed prevents peeking at discovery splits

## Evidence

### Code Pointers

**Hybrid Stability:**
- [features/screening.py](../../src/ced_ml/features/screening.py)
- [features/kbest.py](../../src/ced_ml/features/kbest.py)
- [features/stability.py](../../src/ced_ml/features/stability.py)

**Nested RFECV:**
- [features/nested_rfe.py](../../src/ced_ml/features/nested_rfe.py)

**Post-hoc RFE:**
- [features/rfe.py](../../src/ced_ml/features/rfe.py)
- [cli/optimize_panel.py](../../src/ced_ml/cli/optimize_panel.py)

**Fixed Panel:**
- [cli/train.py](../../src/ced_ml/cli/train.py) - `--fixed-panel` flag

**Config Schema:**
- [config/schema.py:FeatureConfig](../../src/ced_ml/config/schema.py) - `feature_selection_strategy` enum

### Test Coverage
- `tests/test_features_screening.py` - Screening methods
- `tests/test_features_kbest.py` - K-best tuning
- `tests/test_features_stability.py` - Stability extraction
- `tests/test_features_nested_rfe.py` - Nested RFECV
- `tests/test_features_rfe.py` - Post-hoc RFE
- `tests/test_cli_train.py` - Fixed panel integration

### Documentation
- [FEATURE_SELECTION.md](../reference/FEATURE_SELECTION.md) - Complete consolidated guide (all 4 strategies, decision tree, workflows, troubleshooting)
- [CLI_REFERENCE.md](../reference/CLI_REFERENCE.md) - Command syntax
- [CLAUDE.md](../../../CLAUDE.md) - Quick start and workflows

### References
- Guyon, I., et al. (2002). Gene selection for cancer classification using support vector machines. *Machine Learning*, 46(1-3), 389-422. (RFE)
- Meinshausen, N., & Bühlmann, P. (2010). Stability selection. *Journal of the Royal Statistical Society: Series B*, 72(4), 417-473. (Stability selection)

## Related ADRs

### Supersedes/Refines
- [ADR-004: Hybrid Feature Selection](ADR-004-hybrid-feature-selection.md) - Now Strategy 1 in unified framework
- [ADR-005: Stability Panel Extraction](ADR-005-stability-panel.md) - Used by Strategies 1, 2, 3

### Depends on
- [ADR-006: Nested CV Structure](ADR-006-nested-cv.md) - Provides folds for stability and RFECV

### Complements
- [ADR-008: Optuna Hyperparameter Optimization](ADR-008-optuna-hyperparameter-optimization.md) - Hyperparameter tuning applies to all strategies
- [ADR-009: OOF Stacking Ensemble](ADR-009-oof-stacking-ensemble.md) - Feature selection happens per base model before stacking

## Implementation Notes

### Config Schema Changes
Added `feature_selection_strategy` enum to `FeatureConfig`:
```python
feature_selection_strategy: Literal["hybrid_stability", "rfecv", "none"] = "hybrid_stability"
```

**Note:** `"none"` is automatically set when `--fixed-panel` is used.

### CLI Changes
Added `--fixed-panel` flag to `ced train`:
```bash
ced train --fixed-panel panel.csv --split-seed 10
```

Added `ced optimize-panel` command:
```bash
ced optimize-panel --model-path ... --start-size 100 --min-size 5
```

### Backward Compatibility
All existing configs default to `hybrid_stability`, maintaining backward compatibility.

## Future Considerations

### Potential Extension: Nested RFECV + Post-hoc RFE Combo
Use nested RFECV for discovery, then post-hoc RFE on consensus panel for fine-grained size exploration.

### Potential Extension: Multi-split Fixed Panel Validation
Validate fixed panel across multiple new split seeds for more robust unbiased estimates.

### Potential Extension: Ensemble Feature Selection
Combine features selected by multiple strategies (intersection or union) for ultra-robust panels.
