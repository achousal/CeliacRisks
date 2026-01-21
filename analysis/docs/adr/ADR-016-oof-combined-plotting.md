# ADR-016: Out-of-Fold Combined Plotting with Confidence Bands

**Status:** Accepted
**Date:** 2026-01-20
**Decision Makers:** Elahi Lab + Computational Team

## Context

During nested cross-validation, we generate out-of-fold (OOF) predictions across multiple CV repeats (typically 10 repeats x 5 folds). Each repeat produces slightly different predictions for the same samples due to different fold assignments and hyperparameter selections. Understanding the variability across these repeats is critical for assessing model stability and reliability.

Previously, we only visualized final model performance on TRAIN/VAL/TEST sets, missing the opportunity to visualize the uncertainty inherent in the CV process itself. Without aggregated OOF visualization, it was difficult to:
- Assess prediction stability across CV repeats
- Identify samples with high prediction variance
- Compare OOF performance distribution to final model performance
- Communicate model uncertainty to stakeholders

## Decision

Implement a dedicated OOF combined plotting system that aggregates predictions across all CV repeats and visualizes them with confidence bands.

**Key Design Elements:**
1. **Single function interface:** `plot_oof_combined()` in `plotting/oof.py`
2. **Three plot types:** ROC curves, PR curves, and calibration plots
3. **Confidence bands:** Display mean curve with shaded 95% confidence interval across repeats
4. **Per-repeat visualization:** Overlay individual repeat curves (semi-transparent) to show variation
5. **Consistent metadata:** Pin metadata to bottom using shared `apply_plot_metadata` function
6. **Output location:** `plots/oof_*.png` (e.g., `oof_roc.png`, `oof_pr.png`, `oof_calibration.png`)

**Data Requirements:**
- OOF predictions CSV must include `repeat` column
- Each repeat must have consistent sample coverage (all TRAIN samples)
- Predictions aggregated by calculating mean and std across repeats per sample

## Alternatives Considered

1. **Separate plot per repeat:**
   - Would generate 10+ separate files, difficult to compare
   - No aggregate view of uncertainty
   - Rejected: Not scalable for 10+ repeats

2. **Box plots at discrete probability bins:**
   - Could show distribution at specific probability levels
   - Rejected: Less intuitive than confidence bands, loses curve continuity

3. **Violin plots overlaid on curves:**
   - Could show full distribution shape
   - Rejected: Visually cluttered, difficult to interpret

4. **Histogram-style uncertainty visualization:**
   - Could use 2D histograms to show density
   - Rejected: Doesn't work well for ROC/PR curves (discrete steps)

## Consequences

### Positive
- Provides quantitative assessment of model stability across CV repeats
- Visually communicates prediction uncertainty to stakeholders
- Identifies samples with high variance (potential data quality issues)
- Enables comparison of OOF variability vs. final model performance
- Reuses shared metadata formatting infrastructure (`apply_plot_metadata`)
- Lightweight implementation (single module, ~120 lines)

### Negative
- Requires OOF predictions to include `repeat` column (already implemented)
- Adds 3 additional plot files to output directory (minor storage overhead)
- Confidence bands can be misleading if CV repeats are not truly independent (mitigated by stratified CV)

## Evidence

### Code Pointers
- [plotting/oof.py](../../src/ced_ml/plotting/oof.py) - `plot_oof_combined` implementation
- [evaluation/reports.py:ResultsWriter](../../src/ced_ml/evaluation/reports.py) - Integration into reporting pipeline
- [plotting/dca.py:apply_plot_metadata](../../src/ced_ml/plotting/dca.py) - Shared metadata formatting function

### Test Coverage
- `tests/test_plotting_oof.py` - Validates OOF plotting logic (to be added)
- Manual testing: Smoke tests confirm correct rendering with 10 CV repeats

### References
- Efron, B., & Tibshirani, R. J. (1994). *An Introduction to the Bootstrap*. CRC Press.
- Bouckaert, R. R., & Frank, E. (2004). Evaluating the Replicability of Significance Tests for Comparing Learning Algorithms. *PAKDD 2004*.

## Related ADRs

- Depends on: [ADR-008: Nested CV Structure](ADR-008-nested-cv.md) - CV repeat structure
- Related to: [ADR-012: Pydantic Config Schema](ADR-012-pydantic-config.md) - Config for CV repeats
- Complements: Standard plotting modules (ROC, PR, calibration)
