# Uncertainty Quantification in Panel Optimization

**Version**: 1.0.0
**Last Updated**: 2026-01-28

---

## Overview

The CeliacRisks pipeline provides comprehensive uncertainty quantification for panel optimization and consensus panel generation. This document describes the metrics, their interpretation, and how to use them for deployment decisions.

## Table of Contents

1. [RFE Uncertainty Metrics](#rfe-uncertainty-metrics)
2. [Consensus Panel Uncertainty](#consensus-panel-uncertainty)
3. [Statistical Comparisons](#statistical-comparisons)
4. [Interpretation Guidelines](#interpretation-guidelines)
5. [Examples](#examples)

---

## RFE Uncertainty Metrics

### Bootstrap Confidence Intervals

**What it measures**: Variability of AUROC estimates across cross-validation folds.

**How it's computed**:
- 50 bootstrap resamples of the OOF predictions
- 95% confidence interval: mean ± 1.96 × std
- Displayed as shaded region in panel size vs AUROC plots

**Files**:
- `panel_curve.csv`: Contains `auroc_cv_std` column
- `metrics_summary.csv`: Full metrics at each panel size
- `panel_curve_aggregated.png`: Visualizes CIs

**Interpretation**:
- **Narrow CI** (std < 0.02): Stable performance estimate
- **Wide CI** (std > 0.05): High variability, less reliable estimate
- **Non-overlapping CIs**: Statistically significant difference between panel sizes

### Statistical Comparisons

**What it measures**: Whether differences between recommended panel sizes are statistically significant.

**How it's computed**:
- Z-score for difference: |AUROC₁ - AUROC₂| / √(std₁² + std₂²)
- Significant if Z > 1.96 (p < 0.05)
- Also checks for CI overlap

**Displayed as**:
- Green annotation ("NS"): Not significantly different
- Red annotation ("p<0.05, Δ=..."): Significantly different

**Interpretation**:
- **NS (green)**: Can choose smaller panel without significant performance loss
- **p<0.05 (red)**: Larger panel provides statistically significant improvement

---

## Consensus Panel Uncertainty

### Cross-Model Agreement Metrics

#### 1. n_models_present
**What it measures**: Number of models that selected this protein.

**Range**: 1 to N (number of base models)

**Interpretation**:
- **n = N**: Protein selected by all models (high confidence)
- **n > N/2**: Majority agreement (moderate confidence)
- **n ≤ N/2**: Minority agreement (lower confidence, may be model-specific)

#### 2. agreement_strength
**What it measures**: Fraction of models agreeing (n_models_present / n_models).

**Range**: 0.0 to 1.0

**Interpretation**:
- **≥ 0.75**: Strong cross-model agreement
- **0.50 - 0.75**: Moderate agreement
- **< 0.50**: Weak agreement, consider investigating why

#### 3. rank_std
**What it measures**: Standard deviation of ranks across models (only for models where protein is present).

**Range**: 0 to ∞

**Interpretation**:
- **Low (< 10)**: Consistent ranking across models
- **Medium (10-30)**: Some disagreement on importance
- **High (> 30)**: Substantial disagreement, may be context-dependent

#### 4. rank_cv
**What it measures**: Coefficient of variation (rank_std / rank_mean) - normalized uncertainty.

**Range**: 0.0 to ∞

**Interpretation**:
- **< 0.2**: Very stable ranking
- **0.2 - 0.5**: Moderate stability
- **> 0.5**: Unstable ranking, investigate cause

### Uncertainty Summary Statistics

The `uncertainty_summary.csv` file provides:
- Mean agreement strength across final panel
- Minimum agreement strength (identifies least certain proteins)
- Mean rank CV (overall ranking stability)
- Maximum rank CV (identifies most unstable protein)
- Count of proteins in all models
- Count of proteins in majority of models

---

## Statistical Comparisons

### Adjacent Panel Size Comparisons

**Displayed in**: `panel_curve_aggregated.png`

**What it shows**: Statistical significance of differences between adjacent recommended panel sizes (e.g., min_size_95pct vs min_size_90pct).

**Annotations**:
- **Green line + "NS"**: Not significantly different
  - CIs overlap OR Z-score < 1.96
  - **Decision**: Can safely use smaller panel
- **Red line + "p<0.05"**: Significantly different
  - CIs do not overlap AND Z-score > 1.96
  - **Decision**: Larger panel provides meaningful improvement

---

## Interpretation Guidelines

### For Clinical Deployment

1. **High-confidence panel** (recommended):
   - All proteins have `n_models_present ≥ 3` (if 4 models)
   - Mean `agreement_strength ≥ 0.75`
   - Mean `rank_cv < 0.3`
   - Panel size recommended by 95% threshold (conservative)

2. **Moderate-confidence panel**:
   - Most proteins have `n_models_present ≥ 2`
   - Mean `agreement_strength ≥ 0.50`
   - Mean `rank_cv < 0.5`
   - Panel size recommended by 90% threshold

3. **Exploratory panel** (not for deployment):
   - Low agreement strength or high rank CV
   - Use for scientific discovery, not clinical decisions

### Red Flags

- **Low agreement strength** (< 0.50): Protein may be model-specific, not robust
- **High rank CV** (> 0.5): Ranking unstable, may indicate overfitting or noise
- **Wide AUROC CI** (std > 0.05): Performance estimate unreliable
- **Significant drop between sizes**: Avoid cutting to smaller panel

---

## Examples

### Example 1: High-Confidence Consensus Panel

```bash
ced consensus-panel --run-id 20260127_115115

# Output:
Uncertainty Summary:
  Mean agreement strength: 0.92
  Min agreement strength: 0.75
  Mean rank CV: 0.18
  Max rank CV: 0.42
  Proteins in all models: 22/25
  Proteins in majority: 25/25
```

**Interpretation**: Excellent consensus. All proteins supported by majority, 88% in all models. Low rank CV indicates stable rankings. **Ready for clinical deployment.**

### Example 2: Moderate-Confidence Panel

```bash
ced consensus-panel --run-id 20260127_115115

# Output:
Uncertainty Summary:
  Mean agreement strength: 0.65
  Min agreement strength: 0.50
  Mean rank CV: 0.35
  Max rank CV: 0.68
  Proteins in all models: 12/25
  Proteins in majority: 20/25
```

**Interpretation**: Moderate consensus. 80% supported by majority, but only 48% in all models. Higher rank CV suggests some ranking instability. **Consider validation before deployment or increase stability threshold.**

### Example 3: Statistical Comparison

From `panel_curve_aggregated.png`:

**Scenario A**:
- 50-protein panel: AUROC 0.92 ± 0.03
- 25-protein panel: AUROC 0.91 ± 0.03
- Annotation: "NS" (green)

**Decision**: No significant difference. Choose 25-protein panel (more cost-effective).

**Scenario B**:
- 50-protein panel: AUROC 0.92 ± 0.02
- 25-protein panel: AUROC 0.87 ± 0.03
- Annotation: "p<0.05, Δ=0.05" (red)

**Decision**: Significant drop. Recommend 50-protein panel to maintain performance.

---

## Files Reference

| File | Location | Content |
|------|----------|---------|
| **RFE Outputs** |
| `panel_curve_aggregated.csv` | `results/{MODEL}/run_{ID}/aggregated/optimize_panel/` | Full RFE curve with CI metrics |
| `metrics_summary_aggregated.csv` | `results/{MODEL}/run_{ID}/aggregated/optimize_panel/` | All metrics at each panel size |
| `panel_curve_aggregated.png` | `results/{MODEL}/run_{ID}/aggregated/optimize_panel/` | Visualization with CIs and comparisons |
| **Consensus Outputs** |
| `consensus_ranking.csv` | `results/consensus_panel/run_{ID}/` | All proteins with uncertainty metrics |
| `uncertainty_summary.csv` | `results/consensus_panel/run_{ID}/` | Focused report for final panel |
| `final_panel.csv` | `results/consensus_panel/run_{ID}/` | Panel with n_models_present, agreement_strength, rank_cv |
| `consensus_metadata.json` | `results/consensus_panel/run_{ID}/` | Summary statistics under "uncertainty" key |

---

## Command Reference

### Generate RFE with Uncertainty

```bash
# Single model
ced optimize-panel --run-id 20260127_115115 --model LR_EN

# All models (auto-detects)
ced optimize-panel --run-id 20260127_115115
```

### Generate Consensus with Uncertainty

```bash
# Default settings
ced consensus-panel --run-id 20260127_115115

# Stricter stability threshold (more conservative)
ced consensus-panel --run-id 20260127_115115 --stability-threshold 0.80

# More RFE weight (prioritize importance over stability)
ced consensus-panel --run-id 20260127_115115 --rfe-weight 0.7
```

### Interpret Outputs

```bash
# View uncertainty summary
cat results/consensus_panel/run_20260127_115115/uncertainty_summary.csv

# View full consensus ranking
cat results/consensus_panel/run_20260127_115115/consensus_ranking.csv | head -30

# View metadata with uncertainty stats
cat results/consensus_panel/run_20260127_115115/consensus_metadata.json
```

---

## Best Practices

1. **Always check uncertainty metrics** before deployment decisions
2. **Require majority agreement** (`agreement_strength ≥ 0.50`) for clinical panels
3. **Investigate high-CV proteins**: May indicate noise or model-specific features
4. **Use statistical comparisons**: Avoid arbitrary size cuts without checking significance
5. **Document uncertainty** in regulatory submissions (FDA, clinical validation)

---

## Troubleshooting

### High Rank CV
**Causes**:
- Models trained on different feature selection strategies
- Feature instability across CV folds
- Correlated features selected differently by models

**Solutions**:
- Increase `--stability-threshold` to 0.80+
- Use `--rfe-weight 0.3` (prioritize stability over RFE)
- Investigate feature correlation structure

### Low Agreement Strength
**Causes**:
- Models selecting different but correlated features
- Overfitting to split-specific noise
- Model-specific feature interactions

**Solutions**:
- Correlation clustering (already done by consensus)
- Increase buffer size in consensus selection
- Retrain with more stringent stability requirements

### Wide Confidence Intervals
**Causes**:
- Small sample size (low case count)
- High prediction variance across folds
- Model instability

**Solutions**:
- Use larger CV folds (`--cv-folds 10`)
- Increase bootstrap iterations (modify `n_bootstrap` in code)
- Check for data leakage or split issues

---

**Last Updated**: 2026-01-28
**Version**: 1.0.0
**Maintainer**: Andres Chousal (Chowell Lab)
