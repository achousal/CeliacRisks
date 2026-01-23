# Ensemble Plotting Quick Reference

**Module**: `ced_ml.plotting.ensemble`
**Version**: 1.1.0
**Added**: 2026-01-22

---

## Function Summary

| Function | Purpose | When Generated | Output Location |
|----------|---------|----------------|-----------------|
| `plot_meta_learner_weights()` | Show base model coefficients in meta-learner | Per-split training | `{ENSEMBLE}/split_{seed}/diagnostics/ensemble/` |
| `plot_model_comparison()` | Compare ENSEMBLE vs base models on test metrics | Per-split training + aggregation | Per-split: `diagnostics/ensemble/`<br>Aggregated: `aggregated/diagnostics/plots/` |
| `plot_aggregated_weights()` | Show mean ± SD of coefficients across splits | Aggregation | `aggregated/diagnostics/plots/` |

---

## Quick Examples

### 1. Meta-Learner Weights (Per-Split)

```python
from ced_ml.plotting.ensemble import plot_meta_learner_weights

coef = {"LR_EN": 0.45, "RF": 0.35, "XGBoost": 0.20}
plot_meta_learner_weights(
    coef=coef,
    out_path="results/ENSEMBLE/split_0/diagnostics/ensemble/weights.png",
    title="Meta-Learner Coefficients",
    subtitle="split_seed=0",
    meta_penalty="l2",
    meta_c=1.0
)
```

**Output**: Horizontal bar chart, teal (positive) / coral (negative), sorted by magnitude

**Use Case**: Understand which base models contribute most to ensemble predictions

---

### 2. Model Comparison (Test Metrics)

```python
from ced_ml.plotting.ensemble import plot_model_comparison

metrics = {
    "LR_EN": {"AUROC": 0.89, "PR_AUC": 0.30, "Brier": 0.012},
    "RF": {"AUROC": 0.87, "PR_AUC": 0.28, "Brier": 0.013},
    "ENSEMBLE": {"AUROC": 0.91, "PR_AUC": 0.33, "Brier": 0.011}
}
plot_model_comparison(
    metrics=metrics,
    out_path="results/ENSEMBLE/split_0/diagnostics/ensemble/comparison.png",
    title="Model Comparison (Test Set)",
    subtitle="split_seed=0",
    highlight_model="ENSEMBLE"
)
```

**Output**: Grouped bar chart (3 subplots: AUROC, PR-AUC, Brier), ENSEMBLE highlighted with hatch pattern

**Use Case**: Visualize ensemble improvement over base models

---

### 3. Aggregated Weights (Across Splits)

```python
from ced_ml.plotting.ensemble import plot_aggregated_weights

coefs_per_split = {
    0: {"LR_EN": 0.45, "RF": 0.35, "XGBoost": 0.20},
    1: {"LR_EN": 0.50, "RF": 0.30, "XGBoost": 0.22},
    2: {"LR_EN": 0.42, "RF": 0.38, "XGBoost": 0.18}
}
plot_aggregated_weights(
    coefs_per_split=coefs_per_split,
    out_path="results/aggregated/diagnostics/plots/ensemble_weights_agg.png",
    title="Aggregated Meta-Learner Coefficients"
)
```

**Output**: Horizontal bar chart with error bars (±1 SD), auto-subtitle includes `n_splits`

**Use Case**: Assess stability of meta-learner coefficients across random splits

---

## Command-Line Workflow

### Step 1: Train Base Models
```bash
ced train --model LR_EN --split-seed 0
ced train --model RF --split-seed 0
ced train --model XGBoost --split-seed 0
```

### Step 2: Train Ensemble
```bash
ced train-ensemble --base-models LR_EN,RF,XGBoost --split-seed 0
```

**Auto-Generated Plots** (split_seed=0):
- ✅ `diagnostics/plots/ENSEMBLE__test_roc.png` (standard)
- ✅ `diagnostics/plots/ENSEMBLE__test_pr.png` (standard)
- ✅ `diagnostics/plots/ENSEMBLE__test_calibration.png` (standard)
- ✅ `diagnostics/plots/ENSEMBLE__test_dca.png` (standard)
- ✅ `diagnostics/plots/ENSEMBLE__test_risk_dist.png` (standard)
- ✅ **`diagnostics/ensemble/ENSEMBLE__meta_weights.png`** (NEW)
- ✅ **`diagnostics/ensemble/ENSEMBLE__model_comparison.png`** (NEW)

### Step 3: Aggregate Results
```bash
ced aggregate-splits --results-dir results/
```

**Auto-Generated Plots** (aggregated):
- ✅ **`aggregated/diagnostics/plots/ensemble_weights_aggregated.png`** (NEW)
- ✅ **`aggregated/diagnostics/plots/model_comparison.png`** (NEW)

---

## Visual Design

### Color Scheme (No Purple)
- **Teal** (`#2a9d8f`): Positive coefficients
- **Coral** (`#e76f51`): Negative coefficients
- **Dark Teal** (`#264653`): ENSEMBLE highlight
- **Sand/Orange** (`#e9c46a`, `#f4a261`): Other model bars

### Typography
- Title: 12-13pt bold
- Annotations: 8-9pt (bold for highlighted model)
- Metadata footer: 8pt grey

### Export
- Format: PNG
- DPI: 300 (publication quality)
- Backend: matplotlib `Agg`

---

## Data Sources

| Plot | Data Source | Format |
|------|-------------|--------|
| Meta-learner weights | `StackingEnsemble.get_meta_model_coef()` | `Dict[str, float]` |
| Model comparison (per-split) | `{model}/split_{seed}/core/metrics.json` | `Dict[str, Dict[str, float]]` |
| Aggregated weights | `{ENSEMBLE}/split_{seed}/core/run_settings.json` | `Dict[int, Dict[str, float]]` |
| Model comparison (aggregated) | Computed from pooled test metrics | `Dict[str, Dict[str, float]]` |

---

## Interpretation Guide

### Meta-Learner Weights

**What it shows**: How much each base model contributes to the ensemble decision

**Example**:
```
LR_EN:    0.451 █████████████████████▌
RF:       0.348 ████████████████▌
XGBoost:  0.201 █████████▌
```

**Interpretation**:
- All positive → meta-learner uses weighted averaging
- Negative weights → meta-learner uses "correction" strategy (rare)
- Magnitude differences → some models more trusted than others
- Zero weight → base model ignored (very rare if included in training)

**Clinical Insight**: LR_EN (interpretable linear model) contributes most → ensemble decision partly explainable

---

### Model Comparison

**What it shows**: AUROC, PR-AUC, and Brier score for ENSEMBLE vs base models

**Example**:
```
        AUROC    PR-AUC   Brier
LR_EN   0.89     0.30     0.012
RF      0.87     0.28     0.013
ENSEMBLE 0.91    0.33     0.011  ← highlighted
```

**Interpretation**:
- ENSEMBLE > all base models → stacking provides lift
- ENSEMBLE ≈ best base model → stacking adds minimal value (consider single model)
- ENSEMBLE < best base model → meta-learner overfitting (check regularization)

**Expected Lift**: +2-5% AUROC over best single model (per ADR-009)

---

### Aggregated Weights

**What it shows**: Mean coefficient ± SD across multiple random splits

**Example**:
```
LR_EN:    0.450 ± 0.025  (stable)
RF:       0.340 ± 0.045  (moderate variability)
XGBoost:  0.210 ± 0.018  (stable but lower)
```

**Interpretation**:
- Low SD (< 0.05) → robust coefficient assignment
- High SD (> 0.10) → meta-learner sensitive to training data
- Negative mean with positive SD → coefficient flips sign across splits (concerning)

**Clinical Insight**: Stable coefficients → ensemble decision reproducible across cohorts

---

## Customization Examples

### Custom Metrics
```python
plot_model_comparison(
    metrics=metrics,
    out_path="custom.png",
    metric_names=["AUROC", "Sensitivity", "Specificity"],  # Custom subset
    highlight_model="ENSEMBLE"
)
```

### Custom Metadata
```python
meta_lines = [
    "Model: ENSEMBLE (base: LR_EN, RF, XGBoost)",
    "Cohort: UKBB, n=43,960",
    "Prevalence: 0.34%",
    "Meta-learner: LR(C=1.0, penalty=l2)"
]
plot_meta_learner_weights(coef, out_path, meta_lines=meta_lines)
```

### No Subtitle
```python
plot_aggregated_weights(
    coefs_per_split,
    out_path,
    subtitle=""  # Suppress auto-generated "(n_splits=X)"
)
```

---

## Troubleshooting

### Plot Not Generated
**Symptom**: No PNG file, no error
**Cause**: Empty input dict (< 2 models for comparison, 0 models for weights)
**Solution**: Check logs for "skipping" warnings

### Matplotlib Warning: "Tight layout not applied"
**Symptom**: Warning during multi-subplot plots
**Impact**: Non-critical, plots still correct
**Solution**: Ignore (automatic spacing limitation with suptitle + footer)

### Missing Base Model Metrics
**Symptom**: Model comparison chart incomplete
**Cause**: `metrics.json` or `test_metrics.csv` not found
**Solution**:
1. Check `{model}/split_{seed}/core/` directory exists
2. Verify base model training completed successfully
3. Check logs for "Could not load metrics" debug messages

### Inconsistent Models Across Splits
**Symptom**: Aggregated weights plot shows different models per split
**Impact**: Reduced sample size for mean/SD calculation
**Solution**: Normal if some splits failed or used different base models; verify intentional

---

## API Reference

### `plot_meta_learner_weights()`

```python
def plot_meta_learner_weights(
    coef: dict[str, float],                  # Required
    out_path: Path | str,                    # Required
    title: str = "Meta-Learner Coefficients",
    subtitle: str = "",
    meta_penalty: str = "l2",
    meta_c: float = 1.0,
    meta_lines: Sequence[str] | None = None,
) -> None
```

**Raises**: None (logs warnings, skips gracefully)

---

### `plot_model_comparison()`

```python
def plot_model_comparison(
    metrics: dict[str, dict[str, float]],    # Required: {model: {metric: value}}
    out_path: Path | str,                    # Required
    title: str = "Model Comparison",
    subtitle: str = "",
    highlight_model: str = "ENSEMBLE",
    metric_names: list[str] | None = None,   # Default: ["AUROC", "PR_AUC", "Brier"]
    meta_lines: Sequence[str] | None = None,
) -> None
```

**Raises**: None (requires ≥2 models, skips if not met)

---

### `plot_aggregated_weights()`

```python
def plot_aggregated_weights(
    coefs_per_split: dict[int, dict[str, float]],  # Required: {split_seed: {model: coef}}
    out_path: Path | str,                           # Required
    title: str = "Aggregated Meta-Learner Coefficients",
    subtitle: str = "",                             # Default: "(n_splits={n})"
    meta_lines: Sequence[str] | None = None,
) -> None
```

**Raises**: None (skips if empty, logs warnings)

---

## Related Documentation

- [ADR-009: OOF Stacking Ensemble](../docs/adr/ADR-009-oof-stacking-ensemble.md) - Ensemble design rationale
- [ENSEMBLE_PLOTTING_VALIDATION.md](../ENSEMBLE_PLOTTING_VALIDATION.md) - Comprehensive validation report
- [analysis/src/ced_ml/plotting/ensemble.py](../src/ced_ml/plotting/ensemble.py) - Source code
- [analysis/tests/test_plotting_ensemble.py](../tests/test_plotting_ensemble.py) - Test suite

---

**Version**: 1.1.0
**Last Updated**: 2026-01-22
**Maintainer**: CeD-ML Team
