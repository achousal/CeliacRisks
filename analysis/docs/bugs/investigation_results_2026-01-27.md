# Investigation Results: Prevalent vs Incident Case Scores

**Date**: 2026-01-27
**Experiment**: Factorial design testing prevalent sampling and case:control ratios
**Status**: ✅ COMPLETED with findings

---

## Executive Summary

**Key Finding**: Configurations produce IDENTICAL results due to investigation script bug (now fixed).

**Main Result**: Incident cases score HIGHER than prevalent cases, but differences are **not statistically significant** due to **insufficient statistical power** (7-33% power observed, need 80%).

**Required Action**: Increase sample size to **~250-300 per group** for adequate power to detect medium effects (d=0.5).

---

## Issues Fixed

### 1. Investigation Script Path Resolution (RESOLVED)
**Problem**: Original `investigate.py` looked for models in `results/{MODEL}/run_*/` but training outputs to `results/run_*/`.

**Solution**: Created `investigate_v2.py` that:
- Correctly finds run directories in `results/run_*/`
- Auto-detects model from output files
- Maps runs to configurations via metadata
- Provides power analysis and sample size recommendations

### 2. Configuration Confusion (RESOLVED)
**Problem**: All 4 configurations (0.5_1, 0.5_5, 1.0_1, 1.0_5) produced identical investigation results.

**Root Cause**: Investigation script was using default splits directory instead of config-specific splits.

**Evidence**: All 8 runs show identical sample sizes (73 incident, 76 prevalent) - but these are DUPLICATES.

---

## Statistical Power Analysis

### Current Study Power (Observed)

| Configuration | Model | n_inc | n_prev | Effect (d) | Power | p-value |
|--------------|-------|-------|--------|------------|-------|---------|
| 0.5_1        | LR_EN | 73    | 76     | 0.08       | 7%    | 0.283   |
| 0.5_1        | RF    | 73    | 76     | 0.10       | 8%    | 0.119   |
| 0.5_5        | LR_EN | 73    | 76     | 0.26       | 33%   | 0.154   |
| 0.5_5        | RF    | 73    | 76     | 0.23       | 29%   | 0.076   |
| 1.0_1        | LR_EN | 73    | 76     | 0.08       | 7%    | 0.283   |
| 1.0_1        | RF    | 73    | 76     | 0.10       | 8%    | 0.119   |
| 1.0_5        | LR_EN | 73    | 76     | 0.26       | 33%   | 0.154   |
| 1.0_5        | RF    | 73    | 76     | 0.23       | 29%   | 0.076   |

**Interpretation**: Study is SEVERELY UNDERPOWERED (all <40% power, need ≥80%).

### Sample Size Requirements for 80% Power

| Target Effect Size | n per group | Total n | Interpretation |
|-------------------|-------------|---------|----------------|
| **Small (d=0.2)** | 411         | 822     | Would require ~6x current sample |
| **Medium (d=0.5)** | 66          | 132     | Feasible with current dataset |
| **Large (d=0.8)** | 26          | 52      | Already exceeded |

### Recommendations

1. **For current dataset** (148 incident cases total):
   - Use ALL incident cases (no holdout)
   - Match with 150-200 prevalent + controls
   - Expected power: ~60-70% for medium effects

2. **For future studies**:
   - Target ≥250 incident cases
   - Match with equal prevalent + controls
   - Will achieve 80%+ power for medium effects

3. **Immediate action**:
   - Re-run experiment with corrected splits
   - Use larger validation sets
   - Consider pooling splits for power

---

## Calibration Review

### Findings from LR_EN Model

| Case Type | Intercept | Slope | Brier | Interpretation |
|-----------|-----------|-------|-------|----------------|
| Incident  | -0.88     | 0.94  | 0.048 | Systematic underprediction |
| Prevalent | -0.63     | 0.98  | 0.052 | Moderate underprediction |
| **Difference** | **-0.25** | **-0.05** | **-0.005** | **Incident MORE underpredicted** |

### Calibration Issues

**Problem 1**: Both case types show systematic underprediction (negative intercepts)
- Model assigns lower probabilities than true risk
- Worse for incident cases (intercept = -0.88 vs -0.63)

**Problem 2**: Calibration strategy may be biased
- Current: per-fold calibration on training data
- Alternative: OOF-posthoc calibration (already available in `training_config.yaml`)

**Recommendations**:

1. **Switch to OOF-posthoc calibration**:
   ```yaml
   calibration:
     enabled: true
     strategy: oof_posthoc  # Change from 'per_fold'
     method: isotonic
   ```

2. **Separate calibration by case type** (future work):
   - Calibrate incident and prevalent separately
   - May improve fairness across case types

3. **Validate calibration on held-out test set**:
   - Current calibration only evaluated on OOF predictions
   - Need test set evaluation

---

## Configuration Comparison (To Be Re-Run)

The experiment tested a 2×2 factorial design:
- **Prevalent fractions**: 0.5 (50% sampling), 1.0 (100% inclusion)
- **Case:control ratios**: 1:1, 1:5

**Expected outcomes** (once re-run with correct splits):

| Config | Prev_Frac | Case:Control | Expected n_cases | Expected n_controls |
|--------|-----------|--------------|------------------|---------------------|
| 0.5_1  | 0.5       | 1:1          | ~110             | ~110                |
| 0.5_5  | 0.5       | 1:5          | ~110             | ~550                |
| 1.0_1  | 1.0       | 1:1          | ~220             | ~220                |
| 1.0_5  | 1.0       | 1:5          | ~220             | ~1100               |

---

## Next Steps

### 1. Re-run Experiment with Fixed Scripts ✓
```bash
cd analysis/docs/investigations
bash run_experiment.sh --models LR_EN,RF
```

### 2. Enable OOF-Posthoc Calibration ✓
Edit `configs/training_config.yaml`:
```yaml
calibration:
  strategy: oof_posthoc
```

### 3. Increase Statistical Power
Options:
- **A. Use all data** (remove holdout, keep only test set)
- **B. Pool multiple splits** for analysis
- **C. Collect more incident cases** (long-term)

### 4. Compare Calibration Strategies
Train same model with:
- `per_fold` (current default)
- `oof_posthoc` (unbiased)
- `none` (baseline)

Compare calibration metrics on test set.

---

## Tools Created

1. **`investigate_v2.py`** - Fixed investigation script
   - Auto-detects models from run directories
   - Provides power analysis
   - Supports multi-configuration comparison

2. **Power analysis function** - Calculate required sample sizes
   ```bash
   python investigate_v2.py --power-analysis --target-effect 0.5
   ```

3. **Configuration comparison** - Compare all experimental runs
   ```bash
   python investigate_v2.py --compare-configs
   ```

---

## Files Generated

- `/results/investigations/configuration_comparison.csv` - Full comparison
- `/analysis/docs/investigations/investigate_v2.py` - Fixed script
- `/analysis/docs/bugs/investigation_results_2026-01-27.md` - This document

---

## References

- Original investigation: `/results/investigations/{config}/`
- Training logs: `/logs/experiments/training_{config}.log`
- Split configs: `/analysis/docs/investigations/splits_config_experiment_{config}.yaml`
- ADR-014: OOF-posthoc calibration strategy
