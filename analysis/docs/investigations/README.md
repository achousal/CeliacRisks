# Investigation: Prevalent vs Incident Case Scores

Analyzes whether prevalent and incident celiac cases receive different risk scores from trained models.

**Status**: Ready to run (consolidated single script)

---

## Quick Start

### OOF Analysis (Recommended - fastest, most statistical power)
```bash
cd analysis/docs/investigations/

# Full analysis (distributions + calibration + features)
python investigate.py --mode oof --model LR_EN

# All models
python investigate.py --mode oof --all-models

# Distributions only (fastest)
python investigate.py --mode oof --model LR_EN --analyses distributions
```

### Test Set Analysis (requires investigation splits)
```bash
# First: generate splits with prevalent in test set
cd analysis/
mv ../splits ../splits_backup_$(date +%Y%m%d)
ced save-splits --config docs/investigations/splits_config_investigation.yaml

# Train models (if needed)
ced train --model LR_EN --split-seed 0

# Run investigation
cd docs/investigations/
python investigate.py --mode test --model LR_EN
```

---

## Command Reference

```bash
# Full analysis (all three: distributions + calibration + features)
python investigate.py --mode oof --model LR_EN

# Specific analyses only
python investigate.py --mode oof --model LR_EN --analyses distributions
python investigate.py --mode oof --model LR_EN --analyses distributions,calibration
python investigate.py --mode oof --model LR_EN --analyses calibration,features

# All models
python investigate.py --mode oof --all-models

# Specific split seed
python investigate.py --mode oof --model LR_EN --split-seed 1

# Specific run ID
python investigate.py --mode oof --model LR_EN --run-id 20260125_220157

# Help
python investigate.py --help
```

---

## Modes

| Mode | Data Source | Sample Size | Use Case |
|------|-------------|-------------|----------|
| `oof` | Out-of-fold training predictions | 74-75 per group | **Recommended** - fastest, most power |
| `test` | Test set predictions | 37 per group (investigation config) | Requires investigation splits |

**OOF advantages:**
- No retraining needed
- 2x sample size (better statistical power)
- Uses existing model outputs

**Test set advantages:**
- True holdout evaluation
- Matches deployment scenario

---

## What It Analyzes

### 1. Score Distributions (✓ implemented)
**Purpose**: Determine if incident and prevalent cases receive different risk scores

**Metrics**:
- Descriptive statistics (mean, median, quartiles)
- Statistical tests (Mann-Whitney U, t-test, Kolmogorov-Smirnov)
- Effect size (Cohen's d)
- Power analysis

**Output**: Histograms, box plots, violin plots, ECDF

**Interpretation**:
- p < 0.05: Significant difference exists
- Cohen's d > 0.5: Medium-to-large effect size
- Direction: positive = incidents score higher (unexpected)

---

### 2. Calibration Analysis (✓ implemented)
**Purpose**: Check if model predictions are accurate (well-calibrated) for each case type

**Metrics**:
- Calibration intercept (optimal: ~0)
- Calibration slope (optimal: ~1)
- Brier score (lower = better)

**Output**: Calibration curves, slope comparison, Brier score comparison

**Interpretation**:
- **Intercept ≠ 0**: Systematic over/underprediction
- **Slope < 0.85**: Underconfident predictions (predicts too low)
- **Slope > 1.15**: Overconfident predictions (predicts too high)
- **Different calibration by type**: Methodological artifact (training imbalance)
- **Similar calibration**: Biological difference (true risk variation)

---

### 3. Feature Bias Analysis (✓ implemented)
**Purpose**: Identify if selected features discriminate better for one case type

**Metrics**:
- Per-protein AUROC for incident vs control
- Per-protein AUROC for prevalent vs control
- Bias score: incident AUROC - prevalent AUROC

**Output**: Scatter plot, bias distribution, category counts

**Interpretation**:
- **>60% incident-biased**: Feature selection favors incident discrimination
- **>60% prevalent-biased**: Feature selection favors prevalent discrimination
- **Balanced (40-60%)**: Features discriminate equally
- **Mean bias > 0.10**: Strong systematic bias

---

## Outputs

All outputs saved to `../results/investigations/`:

| File | Description |
|------|-------------|
| **Distributions** | |
| `distributions_{MODEL}_{MODE}_seed{N}.png` | Score distribution plots (histogram, box, violin, ECDF) |
| `scores_{MODEL}_{MODE}_seed{N}.csv` | Raw scores by case type |
| **Calibration** | |
| `calibration_{MODEL}_{MODE}_seed{N}.png` | Calibration curves, slope comparison, Brier scores |
| **Feature Bias** | |
| `feature_bias_{MODEL}_{MODE}_seed{N}.png` | Scatter plot, bias distribution, category counts |
| `feature_bias_details_{MODEL}_{MODE}_seed{N}.csv` | Per-protein AUROC and bias scores |
| **Summary** | |
| `summary_{MODE}_seed{N}.csv` | Statistical summary across models and analyses |

---

## Statistical Power

**With 74-75 cases per group** (OOF mode):
- Small effect (d=0.3): 65% power
- Medium effect (d=0.5): **95% power**
- Large effect (d=0.8): **99% power**

**With 37 cases per group** (test mode, investigation config):
- Small effect (d=0.3): 45% power
- Medium effect (d=0.5): **81% power**
- Large effect (d=0.8): **96% power**

---

## Interpreting Results

The three analyses work together to diagnose the cause:

### Scenario 1: Methodological Artifact (training imbalance)
**Evidence**:
- **Distributions**: Incidents score 25% higher (p<0.001)
- **Calibration**: Prevalent slope 0.70, intercept -0.18 (poor)
- **Features**: 68% incident-biased

**Conclusion**: Training imbalance caused systematic bias

**Action**: Retrain with `prevalent_train_frac: 1.0`

---

### Scenario 2: Biological Difference (true risk variation)
**Evidence**:
- **Distributions**: Incidents score 20% higher (p=0.002)
- **Calibration**: Both well-calibrated (slopes ~0.95, intercepts ~0)
- **Features**: 52% incident-biased (balanced)

**Conclusion**: Pre-diagnostic biomarkers genuinely differ from prevalent disease

**Action**: Document as expected, report metrics separately

---

### Scenario 3: Mixed Causes
**Evidence**:
- **Distributions**: Incidents score 28% higher (p<0.001)
- **Calibration**: Prevalent slope 0.82 (moderate miscalibration)
- **Features**: 60% incident-biased

**Conclusion**: Combination of methodological bias and biological difference

**Action**: Balanced sampling + case-aware calibration

---

### Quick Interpretation Guide

| Pattern | Likely Cause |
|---------|--------------|
| High score difference + poor calibration + strong feature bias | **Methodological artifact** |
| High score difference + good calibration + balanced features | **Biological difference** |
| Moderate difference + moderate calibration + moderate bias | **Mixed causes** |
| No difference | **Model treats both equally** |

---

## Configuration

### Investigation Splits Config
Edit `splits_config_investigation.yaml` to control test set composition:

```yaml
# Include prevalent in all splits (not just training)
prevalent_train_only: false

# No downsampling of prevalent cases
prevalent_train_frac: 1.0

# Better balance (10 controls per case)
train_control_per_case: 10
```

**Comparison to production config:**

| Setting | Production | Investigation |
|---------|-----------|---------------|
| `prevalent_train_only` | `true` | `false` |
| `prevalent_train_frac` | `0.5` | `1.0` |
| Test incident | 37 | 37 |
| Test prevalent | 0-1 | 37 |

---

## Dependencies

All dependencies included in project environment:
```bash
cd analysis/
pip install -e ".[dev]"
```

Requires: pandas, numpy, scipy, matplotlib, seaborn, scikit-learn

---

## Troubleshooting

### "No run directory found for {MODEL}"
Train the model first:
```bash
cd analysis/
ced train --model LR_EN --split-seed 0
```

### "Predictions not found"
Check that training completed successfully:
```bash
ls ../results/LR_EN/run_*/split_seed0/preds/
```

### "Missing case types in OOF predictions"
OOF mode requires both incident and prevalent in training set. This is the default.

### "0-1 prevalent cases in test set"
Use OOF mode, or generate investigation splits first:
```bash
cd analysis/
ced save-splits --config docs/investigations/splits_config_investigation.yaml
```

---

## Background

**The Problem**: Test predictions may show prevalent cases scoring 20-30% lower than incidents, which is unexpected since both are confirmed celiac cases.

**Root Cause Hypotheses**:
1. **Training imbalance** (40-50%): Prevalent downsampled 50%, incidents 100%
2. **Feature selection bias** (20-30%): Features optimized for incident discrimination
3. **Biological differences** (15-25%): Treatment effects, disease stage
4. **Calibration issues** (10-15%): Calibrator trained on mixed cases
5. **CV fold variability** (5-10%): Small sample sizes

**Why It Matters**: Affects clinical validity, threshold calibration, and feature selection strategies.

---

## Next Steps After Running Investigation

1. **Review outputs** in `../results/investigations/`
2. **Check consistency** across models and splits
3. **Interpret using the three-analysis framework**:
   - Distributions: Is there a problem?
   - Calibration: Is it methodological or biological?
   - Features: What's the mechanistic cause?
4. **If methodological artifact detected:**
   - Retrain with balanced sampling (`prevalent_train_frac: 1.0`)
   - Implement case-aware calibration
   - Validate on independent test set
5. **If biological difference confirmed:**
   - Document in project ADRs
   - Report metrics stratified by case type
   - Consider case-specific thresholds
6. **Document findings** in project documentation

---

**Last Updated**: 2026-01-26
**Script Version**: 3.0 (full implementation: distributions + calibration + features)
