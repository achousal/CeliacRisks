# Investigation: Prevalent vs Incident Case Scores

Analyzes whether prevalent and incident celiac cases receive different risk scores from trained models.

**Purpose**: Diagnose if score differences reflect biological reality or methodological artifacts (training imbalance, feature selection bias).

---

## Quick Start

### Local Analysis (Recommended)
```bash
cd analysis/docs/investigations/

# Single split, fast diagnostic (~30 sec)
bash run_investigation.sh --mode single

# Full coverage across all splits (~10-15 min)
bash run_investigation.sh --mode full

# HPC distributed execution (~2-3 min wall-clock)
bash run_investigation.sh --mode hpc --queue medium
```

### Manual Execution
```bash
# Single model, single split
python investigate.py --mode oof --model LR_EN --split-seed 0

# Specific analyses only
python investigate.py --mode oof --model LR_EN --analyses distributions,calibration

# All models
python investigate.py --mode oof --all-models

# Test set instead of OOF
python investigate.py --mode test --model LR_EN --split-seed 0
```

---

## Investigation Modes

| Mode | Coverage | Runtime | Use Case |
|------|----------|---------|----------|
| **Single (OOF)** | ~75 incident cases (1 split) | 30 sec | Quick diagnostic |
| **Single (Test)** | ~15 incident cases (1 split) | 30 sec | Validation check |
| **Full** | ALL ~148 incident cases (10 splits × [OOF+test]) | 10-15 min | Comprehensive statistics |
| **HPC** | ALL cases, parallelized | 2-3 min | Production analysis |

### Why Multiple Splits?

**Dataset**: ~148 total incident cases, but each split uses only a subset:
- Training (OOF): ~75 incident cases (50% of data)
- Test: ~15 incident cases (25% of data)

**Single split limitation**: Only sees ~50-75 cases out of 148 total.

**Full coverage solution**: Aggregate across all 10 splits (OOF + test) to analyze ALL cases.

**Example**:
- Split 0: cases A-K in OOF, L-M in test
- Split 1: cases B-L in OOF, A,M-N in test
- Combined: ALL cases A-Z covered

---

## What It Analyzes

### 1. Score Distributions
**Purpose**: Determine if incident and prevalent cases receive different risk scores.

**Metrics**:
- Descriptive statistics (mean, median, quartiles)
- Statistical tests (Mann-Whitney U, t-test, Kolmogorov-Smirnov)
- Effect size (Cohen's d)
- Power analysis

**Interpretation**:
- p < 0.05: Significant difference exists
- Cohen's d > 0.5: Medium-to-large effect
- Direction: positive = incidents score higher (unexpected)

### 2. Calibration Analysis
**Purpose**: Check if predictions are accurate (well-calibrated) for each case type.

**Metrics**:
- Calibration intercept (optimal: ~0)
- Calibration slope (optimal: ~1)
- Brier score (lower = better)

**Interpretation**:
- **Intercept ≠ 0**: Systematic over/underprediction
- **Slope < 0.85**: Underconfident predictions
- **Slope > 1.15**: Overconfident predictions
- **Different calibration by type**: Methodological artifact
- **Similar calibration**: Biological difference

### 3. Feature Bias Analysis
**Purpose**: Identify if selected features discriminate better for one case type.

**Metrics**:
- Per-protein AUROC for incident vs control
- Per-protein AUROC for prevalent vs control
- Bias score: incident AUROC - prevalent AUROC

**Interpretation**:
- >60% incident-biased: Feature selection favors incident discrimination
- >60% prevalent-biased: Feature selection favors prevalent discrimination
- Balanced (40-60%): Features discriminate equally
- Mean bias > 0.10: Strong systematic bias

---

## Interpreting Results

The three analyses work together to diagnose the cause:

### Scenario 1: Methodological Artifact
**Evidence**:
- Distributions: Incidents score 25% higher (p<0.001)
- Calibration: Prevalent slope 0.70, intercept -0.18 (poor)
- Features: 68% incident-biased

**Conclusion**: Training imbalance caused systematic bias

**Action**: Retrain with `prevalent_train_frac: 1.0`

### Scenario 2: Biological Difference
**Evidence**:
- Distributions: Incidents score 20% higher (p=0.002)
- Calibration: Both well-calibrated (slopes ~0.95, intercepts ~0)
- Features: 52% incident-biased (balanced)

**Conclusion**: Pre-diagnostic biomarkers genuinely differ from prevalent disease

**Action**: Document as expected, report metrics separately

### Scenario 3: Mixed Causes
**Evidence**:
- Distributions: Incidents score 28% higher (p<0.001)
- Calibration: Prevalent slope 0.82 (moderate miscalibration)
- Features: 60% incident-biased

**Conclusion**: Combination of methodological bias and biological difference

**Action**: Balanced sampling + case-aware calibration

### Quick Reference

| Pattern | Likely Cause |
|---------|--------------|
| High score difference + poor calibration + strong feature bias | **Methodological artifact** |
| High score difference + good calibration + balanced features | **Biological difference** |
| Moderate difference + moderate calibration + moderate bias | **Mixed causes** |
| No difference | **Model treats both equally** |

---

## Outputs

All outputs saved to `../results/investigations/`:

| File | Description |
|------|-------------|
| **Single Split Outputs** | |
| `distributions_{MODEL}_{MODE}_seed{N}.png` | Score distribution plots (histogram, box, violin, ECDF) |
| `scores_{MODEL}_{MODE}_seed{N}.csv` | Raw scores by case type |
| `calibration_{MODEL}_{MODE}_seed{N}.png` | Calibration curves, slope comparison, Brier scores |
| `feature_bias_{MODEL}_{MODE}_seed{N}.png` | Scatter plot, bias distribution, category counts |
| `feature_bias_details_{MODEL}_{MODE}_seed{N}.csv` | Per-protein AUROC and bias scores |
| `summary_{MODE}_seed{N}.csv` | Statistical summary across models |
| **Full Coverage Outputs** | |
| `FULL_COVERAGE_SUMMARY.csv` | All runs combined (all models, splits, modes) |
| `FULL_COVERAGE_MODEL_SUMMARY.csv` | Per-model aggregates with total case coverage |

---

## HPC Execution

### Setup (First Time Only)
```bash
cd analysis/
bash scripts/hpc_setup.sh
source venv/bin/activate
```

### Submit Jobs
```bash
cd analysis/docs/investigations/

# Full investigation (all models, all splits)
bash run_investigation.sh --mode hpc

# Specific models only
bash run_investigation.sh --mode hpc --models LR_EN,RF

# Custom resources
bash run_investigation.sh --mode hpc --cores 4 --memory 16G --walltime 02:00 --queue long

# Dry run (preview without submitting)
bash run_investigation.sh --mode hpc --dry-run
```

### Monitor Progress
```bash
# Check job status
bjobs -w JOB_ID

# View logs
tail -f ../../../results/investigations/logs/investigate_*.log

# Count completed jobs
ls ../../../results/investigations/logs/investigate_*.log | wc -l
```

### Timing
- **Setup**: 1-2 minutes (environment loading)
- **Per-job**: 30-60 seconds
- **Parallelization**: ~80 jobs simultaneously (4 models × 10 splits × 2 modes)
- **Total wall-clock**: ~2-3 minutes

### Resource Recommendations

**Conservative** (recommended for first run):
```bash
bash run_investigation.sh --mode hpc --cores 2 --memory 8G --walltime 01:00 --queue medium
```

**Balanced**:
```bash
bash run_investigation.sh --mode hpc --cores 4 --memory 16G --walltime 02:00 --queue medium
```

**Aggressive** (if queue allows):
```bash
bash run_investigation.sh --mode hpc --cores 8 --memory 32G --walltime 04:00 --queue long
```

---

## Configuration

### Investigation-Specific Splits

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
| Test incident | ~15 | ~15 |
| Test prevalent | 0 | ~15 |

**Generate investigation splits**:
```bash
cd analysis/
ced save-splits --config docs/investigations/splits_config_investigation.yaml
```

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
OOF mode requires both incident and prevalent in training set (default behavior).

### "0-1 prevalent cases in test set"
Expected with production config (ADR-002: prevalent excluded from test).
Use OOF mode, or generate investigation splits to include prevalent in test.

### HPC Jobs Fail
Check job logs:
```bash
cat ../../../results/investigations/logs/investigate_1.log
```

Verify environment:
```bash
source venv/bin/activate
python -c "import ced_ml; print('OK')"
```

### No Results After HPC Completion
Results are saved during each job run. Check:
```bash
ls ../../../results/investigations/summary_*.csv | wc -l  # Should match job count
```

If files are missing, check job logs for errors.

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

## Statistical Power

**With 74-75 cases per group** (OOF mode):
- Small effect (d=0.3): 65% power
- Medium effect (d=0.5): **95% power**
- Large effect (d=0.8): **99% power**

**With 15 cases per group** (test mode, production config):
- Small effect (d=0.3): 25% power
- Medium effect (d=0.5): **50% power**
- Large effect (d=0.8): **81% power**

**Full coverage** (aggregating across all splits):
- Analyzes all ~148 incident cases
- Maximum statistical power
- Robust effect size estimates

---

## Attack Plan: Understanding Prevalent vs Incident Score Differences

This section provides a systematic approach to answering three key questions:
1. How much of the score difference is class imbalance vs. biology?
2. Which features bias toward incident discrimination?
3. How does this affect clinical deployment?

### Overview

The core strategy separates **methodological artifacts** (class imbalance, feature selection bias) from **biological differences** by training and analyzing models under different conditions:

```
PHASE 1: Baseline (production conditions)
├─ Train on downsampled ratios (5:1 controls)
└─ Investigate on balanced ratios (1:1)
   └─ Reveals: Score differences with class imbalance confounding

PHASE 2: Isolated Biology (remove class imbalance)
├─ Train on balanced ratios (1:1 controls)
└─ Investigate on balanced ratios (1:1)
   └─ Reveals: Pure biological differences

PHASE 3: Root Cause Analysis (feature perspective)
├─ Compare feature selection across phases
├─ Measure per-protein discrimination bias
└─ Identify incident-specific vs. prevalent-specific features

DECISION: Deployment Strategy
├─ If artifact-driven: Retrain with balanced sampling
├─ If biology-driven: Use case-aware thresholds
└─ If mixed: Implement both strategies
```

---

### PHASE 1: Baseline Investigation (Current Setup)

**Goal**: Establish the observed score difference under realistic conditions.

**Setup**:
- Models trained on production config: `train_control_per_case: 5`
- Investigated on balanced splits: `train_control_per_case: 1`

**Run Investigation**:
```bash
cd analysis/

# BASELINE: Single split for quick diagnostic (~30 sec)
bash docs/investigations/run_investigation.sh --mode single

# OR FULL: All splits for comprehensive statistics (~10-15 min)
bash docs/investigations/run_investigation.sh --mode full
```

**Expected Outputs**:
- `distributions_*.png` – Score distributions by case type
- `summary_oof_seed*.csv` – Statistical metrics
- `feature_bias_details_*.csv` – Per-protein discrimination

**Key Metrics to Record**:
```
Phase 1 Results (Downsampled Training, Balanced Investigation)
┌─────────────────────────────────────────────┐
│ Model: LR_EN, RF, XGBoost, LinSVM_cal      │
├─────────────────────────────────────────────┤
│ Incident median score:  [record value]      │
│ Prevalent median score: [record value]      │
│ Median difference:      [record ± %]        │
│ Mann-Whitney p-value:   [record]            │
│ Cohen's d (effect):     [record]            │
│ % incident-biased features: [record]        │
│ Prevalent calibration slope: [record]       │
└─────────────────────────────────────────────┘
```

**Interpretation at This Stage**:
- High difference (>20%) + poor calibration (slope <0.85) + biased features (>65%)
  → **Suggests strong class imbalance artifact**
- Moderate difference (10-20%) + moderate calibration (slope 0.85-1.15) + balanced features (45-55%)
  → **Suggests biological difference with some artifact**

---

### PHASE 2: Isolated Biology (Remove Class Imbalance)

**Goal**: Determine what the score difference looks like without training imbalance.

**Step 1: Generate Balanced Training Splits**
```bash
cd analysis/

# Copy investigation config to production config
cp docs/investigations/splits_config_investigation.yaml configs/splits_config.yaml

# OR manually edit configs/splits_config.yaml:
# Change:
#   train_control_per_case: 5  →  train_control_per_case: 1
#   prevalent_train_frac: 0.5  →  prevalent_train_frac: 1.0

# Regenerate splits (will overwrite)
ced save-splits --config configs/splits_config.yaml --infile ../data/Celiac_dataset_proteomics_w_demo.parquet --overwrite
```

**Step 2: Retrain Models**
```bash
# Train on balanced splits (slower, but critical for separation)
ced train --model LR_EN --split-seed 0 --config configs/training_config.yaml
ced train --model RF --split-seed 0
ced train --model XGBoost --split-seed 0
ced train --model LinSVM_cal --split-seed 0
```

**Step 3: Investigate (Same Balanced Setup)**
```bash
# Now both training and investigation use 1:1 ratios
bash docs/investigations/run_investigation.sh --mode full
```

**Key Metrics to Record**:
```
Phase 2 Results (Balanced Training, Balanced Investigation)
┌─────────────────────────────────────────────┐
│ Model: LR_EN, RF, XGBoost, LinSVM_cal      │
├─────────────────────────────────────────────┤
│ Incident median score:  [record value]      │
│ Prevalent median score: [record value]      │
│ Median difference:      [record ± %]        │
│ Mann-Whitney p-value:   [record]            │
│ Cohen's d (effect):     [record]            │
│ % incident-biased features: [record]        │
│ Prevalent calibration slope: [record]       │
└─────────────────────────────────────────────┘
```

**Comparison Analysis**:
```
ARTIFACT ATTRIBUTION TABLE
┌──────────────────────┬─────────────┬──────────────────┐
│ Metric               │ Phase 1 (5:1) │ Phase 2 (1:1)    │
├──────────────────────┼─────────────┼──────────────────┤
│ Score difference     │ [from Ph1]  │ [from Ph2]       │
│ (Difference - Ph2)   │             │ = ARTIFACT SIZE  │
├──────────────────────┼─────────────┼──────────────────┤
│ Effect size (d)      │ [from Ph1]  │ [from Ph2]       │
│ (Difference - Ph2)   │             │ = IMBALANCE-DRIVEN │
├──────────────────────┼─────────────┼──────────────────┤
│ Calibration slope    │ [from Ph1]  │ [from Ph2]       │
│ (Change in slope)    │             │ = CALIBRATION ARTIFACT │
├──────────────────────┼─────────────┼──────────────────┤
│ Feature bias %       │ [from Ph1]  │ [from Ph2]       │
│ (% reduction)        │             │ = SELECTION BIAS SIZE │
└──────────────────────┴─────────────┴──────────────────┘
```

**Interpretation**:
- Phase 2 score difference **much smaller** than Phase 1? → **Artifact-driven (retrain)**
- Phase 2 score difference **similar** to Phase 1? → **Biology-driven (accept)**
- Phase 2 calibration slope **much better**? → **Class imbalance is calibration problem**

---

### PHASE 3: Root Cause Analysis (Feature Perspective)

**Goal**: Identify mechanistically which features drive the score difference.

**Step 1: Compare Feature Selection Across Phases**
```bash
cd analysis/

# Generate feature stability reports
ced aggregate-splits --config configs/aggregate_config.yaml

# Compare Phase 1 vs Phase 2 feature stability
diff <(sort results/LR_EN/*/aggregated/feature_stability.csv | cut -d, -f1) \
     <(sort results/LR_EN/*/aggregated/feature_stability.csv | cut -d, -f1)
```

**Step 2: Per-Protein Discrimination Analysis**

Run feature bias analysis (included in investigation):
```bash
python docs/investigations/investigate.py \
    --mode oof \
    --model LR_EN \
    --split-seed 0 \
    --analyses features
```

This produces:
- `feature_bias_details_*.csv` – Per-protein incident vs prevalent AUROC
- `feature_bias_*.png` – Visualization of bias distribution

**Step 3: Manual Protein Prioritization**

```python
# Create a summary of biased proteins (run in analysis/ dir)
import pandas as pd

# Load feature bias from Phase 1 (downsampled training)
bias_phase1 = pd.read_csv('results/investigations/feature_bias_details_LR_EN_oof_seed0.csv')

# Identify top incident-biased proteins
incident_biased = bias_phase1[bias_phase1['bias_score'] > 0.10].sort_values('bias_score', ascending=False)
print("\nTOP INCIDENT-BIASED PROTEINS (Phase 1):")
print(incident_biased[['feature', 'incident_auc', 'prevalent_auc', 'bias_score']].head(20))

# Identify top prevalent-biased proteins
prevalent_biased = bias_phase1[bias_phase1['bias_score'] < -0.10].sort_values('bias_score')
print("\nTOP PREVALENT-BIASED PROTEINS (Phase 1):")
print(prevalent_biased[['feature', 'incident_auc', 'prevalent_auc', 'bias_score']].head(20))

# Load Phase 2 and compare
bias_phase2 = pd.read_csv('results/LR_EN/run_*/aggregated/feature_stability.csv')  # or rerun investigate
print("\nBIAS REDUCTION FROM PHASE 1 → PHASE 2:")
print(f"  Incident-biased reduction: {(incident_biased.shape[0] - ...) / incident_biased.shape[0] * 100:.1f}%")
```

**Key Metrics to Record**:
```
FEATURE BIAS SUMMARY
┌────────────────────────────────────────────┐
│ Incident-Biased Proteins (bias > 0.10):    │
│  Count: [n]                                │
│  Examples: [list top 5]                    │
│  Reduction Ph1→Ph2: [%]                    │
├────────────────────────────────────────────┤
│ Prevalent-Biased Proteins (bias < -0.10):  │
│  Count: [n]                                │
│  Examples: [list top 5]                    │
│  Reduction Ph1→Ph2: [%]                    │
├────────────────────────────────────────────┤
│ Feature Selection Consistency:              │
│  Proteins selected in both phases: [n]     │
│  New selections in Phase 2: [n]            │
│  Lost selections in Phase 2: [n]           │
└────────────────────────────────────────────┘
```

---

### PHASE 4: Interpretation & Decision Making

**Create a Summary Report**:

```markdown
# Prevalent vs Incident Score Difference: Root Cause Analysis

## Finding #1: Magnitude of Score Difference

**Phase 1 (Downsampled Training)**: Prevalent score is [X]% lower
**Phase 2 (Balanced Training)**: Prevalent score is [Y]% lower
**Artifact Size**: [X-Y]% difference attributable to class imbalance

### Interpretation:
- If [X-Y] > 50% of [X]: **Primarily methodological**
- If [X-Y] < 20% of [X]: **Primarily biological**
- If 20-50%: **Mixed causes**

## Finding #2: Calibration Issues

**Phase 1 Prevalent Slope**: [slope value]
**Phase 2 Prevalent Slope**: [slope value]
**Improvement**: [% improvement]

### Interpretation:
- Slope improved >0.10 in Phase 2 → Class imbalance affects calibration
- Slope unchanged → Biological difference requires case-specific calibration

## Finding #3: Feature Selection Bias

**Incident-Biased Features (Phase 1)**: [n] proteins
**Incident-Biased Features (Phase 2)**: [n] proteins
**Reduction**: [%]

**Top 5 Problematic Proteins**:
1. [Protein] – bias was [Phase 1], now [Phase 2]
2. ...

### Interpretation:
- >50% reduction in bias → Feature selection heavily influenced by class imbalance
- <20% reduction → Features genuinely discriminate better for incidents

## Recommendation

### If Artifact-Driven (majority of difference is methodological):
✓ **Action**: Retrain on balanced sampling
- Use `train_control_per_case: 1` (or higher)
- Use `prevalent_train_frac: 1.0`
- Re-validate model performance
- This ensures fair treatment of both case types

### If Biology-Driven (most difference is real):
✓ **Action**: Accept and document
- Create ADR explaining the biological basis
- Report metrics stratified by case type
- Use case-specific thresholds if needed (high-sensitivity for prevalent)

### If Mixed Causes:
✓ **Action**: Implement both
- Retrain with balanced sampling
- Add case-aware calibration (fit calibrator separately)
- Validate on independent test set
- Document trade-offs in ADR
```

---

### Practical Workflow (Quick Checklist)

**Day 1: Baseline**
```bash
# 15 min total time
cd analysis/docs/investigations/
bash run_investigation.sh --mode full  # 10-15 min
# → Record Phase 1 metrics above
```

**Day 2: Retrain & Investigate**
```bash
# 1-2 hours total time (depends on model count)
cd analysis/
cp docs/investigations/splits_config_investigation.yaml configs/splits_config.yaml
ced save-splits --config configs/splits_config.yaml --infile ../data/Celiac_dataset_proteomics_w_demo.parquet --overwrite
ced train --model LR_EN --split-seed 0  # ~30 min per model
ced train --model RF --split-seed 0
ced train --model XGBoost --split-seed 0
ced train --model LinSVM_cal --split-seed 0
bash docs/investigations/run_investigation.sh --mode full  # 10-15 min
# → Record Phase 2 metrics above
```

**Day 3: Feature Analysis & Decision**
```bash
# 30 min total time
cd analysis/
python docs/investigations/investigate.py --mode oof --all-models --analyses features --split-seed 0
# → Create summary report using template above
```

---

### Expected Outcomes

| Outcome | Expected Finding | Action |
|---------|-----------------|--------|
| **Artifact-Dominated** | Phase 2 score diff < 5% of Phase 1 | Retrain with balanced config |
| **Biology-Dominated** | Phase 2 score diff ≥ 80% of Phase 1 | Accept, document, stratify reporting |
| **Mixed** | Phase 2 score diff 30-70% of Phase 1 | Retrain + case-aware calibration |

---

## Next Steps After Running Investigation

1. **Review outputs** in `../results/investigations/`
2. **Execute Attack Plan** phases above to separate artifact from biology
3. **Document findings** in project ADRs (create ADR-015 or update ADR-002)
4. **Decision gate**: Retrain or accept?
5. **If retraining**: Compare Phase 1 vs Phase 2 metrics
6. **If accepting**: Implement case-specific reporting and thresholds

---

**Last Updated**: 2026-01-27
**Version**: 2.1 (added comprehensive attack plan)
