# Investigation Methodology & Interpretation

Complete guide to the statistical design, how to interpret results, and how to make decisions.

---

## Research Question

Do models systematically **under-predict** prevalent (confirmed) celiac cases relative to incident (pre-diagnostic) cases, and if so, **why**?

Three hypotheses:

1. **Methodological artifact**: Training data imbalance causes the model to prefer incident cases
   - Prevalent controls downsampled 50% during training (ADR-002)
   - Causes: feature selection bias, miscalibration, systematic underprediction
   - Fix: Retrain with balanced sampling

2. **Biological difference**: Pre-diagnostic biomarkers genuinely differ from confirmed disease
   - Incident cases: biomarkers measured before diagnosis
   - Prevalent cases: measured after disease, potentially affected by treatment
   - Fix: Accept and document; use case-specific thresholds if needed

3. **Mixed causes**: Both artifact and biology contribute
   - Fix: Retrain + implement case-aware calibration

---

## Experimental Design

### Strategy: Phase Comparison

**Phase 1 (Baseline)**: Train on production config
- Prevalent sampling: 50% (ADR-002)
- Case:control ratio: 5:1 (standard)
- Measures: Observed score difference under realistic conditions

**Phase 2 (Artifact Test)**: Train on balanced config
- Prevalent sampling: 100% (no downsampling)
- Case:control ratio: 1:1 (perfectly balanced)
- Measures: Residual score difference without training imbalance

**Interpretation**:
- Phase 1 score diff >> Phase 2 score diff? → **Artifact-driven**
- Phase 1 ≈ Phase 2? → **Biology-driven**
- Phase 1 ~ 1.5× Phase 2? → **Mixed**

### Why This Works

By **fixing the test set** and **varying only the training configuration**, we isolate training dynamics from case-type biology:

```
Fixed:           Same 25% of data (25% of incident cases)
Varied:          How training data was sampled (50% vs 100% prevalent)
Result:          Score differences due to training imbalance appear/disappear
```

### Key Assumptions

1. **No hidden confounders**: Age, sex, ethnic group don't differ systematically between phases
2. **No batch effects**: Prevalent and incident cases weren't measured differently
3. **Adequate sample size**: ~75 incident cases in training, ~36 in test

**Recommended check**: Stratify Phase 1 vs Phase 2 by age group to rule out confounding

---

## Detailed Metrics

### 1. Score Distributions

**What it measures**: Central tendency and spread of incident vs prevalent risk scores

**Metrics**:
- **Median difference**: (Median incident - Median prevalent) / Median prevalent × 100%
- **Mann-Whitney U p-value**: Statistical significance (H0: medians equal)
- **Cohen's d**: Effect size (0.5 = medium, 0.8 = large)
- **Kolmogorov-Smirnov test**: Whether entire distributions differ (not just centers)

**Interpretation**:
- Median difference 15-30%: Substantial underprediction of prevalent
- p < 0.05: Statistically significant
- d > 0.5: Practically meaningful effect

**Phase 1 vs Phase 2**:
| Phase 1 | Phase 2 | Inference |
|---------|---------|-----------
| +25% | +3% | Strong artifact (80% reduction) |
| +25% | +20% | Weak artifact (20% reduction) |
| +25% | +25% | Pure biology (no reduction) |

Visual reference:
```
            Incident Scores
                 │
            ┌────┴────┐
            ▼         ▼
        ╔═════════════════╗
        ║  Overlapping    ║
        ║   (Biology?)    ║
        ╚═════════════════╝
         20%      median     difference
                    │
        Prevalent Scores

If Phase 1→2 makes overlap larger
→ Artifact (training imbalance compressed incident scores down)

If overlap stays same size
→ Biology (distribution difference is real)
```

### 2. Calibration Analysis

**What it measures**: Whether predictions are accurate for each case type

**Metrics**:
- **Intercept**: Systematic shift in predictions (0 = perfect)
  - Negative = underprediction (common for prevalent)
  - Positive = overprediction
- **Slope**: Confidence calibration (1 = perfect)
  - <0.85 = underconfident (true event rate higher than predicted)
  - >1.15 = overconfident (true event rate lower than predicted)
- **Brier Score**: Mean squared error (lower = better)

**Interpretation**:

| Phase 1 | Phase 2 | Inference |
|---------|---------|-----------
| Slope 0.70, Int -0.15 | Slope 0.95, Int -0.02 | Imbalance causes miscalibration → retrain |
| Slope 0.85, Int -0.08 | Slope 0.82, Int -0.07 | Persistent miscalibration → biology or recalibration needed |
| Slope 0.98, Int +0.01 | Slope 0.99, Int 0.00 | Well-calibrated regardless → support biology hypothesis |

Visual reference:
```
Slope = Confidence Calibration

Slope 0.70 ──→ [UNDERCONFIDENT] (underpredicting)
              ↓
Slope 0.95 ──→ [WELL-CALIBRATED] ✓
              ↓
Slope 1.10 ──→ [OVERCONFIDENT] (overpredicting)

Phase 1: Slope 0.72 (bad)
Phase 2: Slope 0.95 (good)
         → Imbalance caused miscalibration
```

**Why it matters**:
- Bad calibration in Phase 1 but not Phase 2 = **class imbalance is the problem**
- Bad calibration in both = **systematic underprediction of prevalent** (either true biology or hidden confounder)

### 3. Feature Selection Bias

**What it measures**: Do selected features systematically favor incident discrimination?

**Metrics**:
- **Per-protein incident AUROC**: How well does each protein discriminate incident vs control?
- **Per-protein prevalent AUROC**: How well does each protein discriminate prevalent vs control?
- **Bias score**: incident AUROC - prevalent AUROC
  - Positive = protein more predictive for incident
  - Negative = protein more predictive for prevalent
- **% incident-biased features**: Fraction of selected features with positive bias

**Interpretation**:

| Phase 1 | Phase 2 | Inference |
|---------|---------|-----------
| 68% incident-biased | 52% incident-biased | Feature selection biased by class imbalance |
| 65% incident-biased | 64% incident-biased | Features genuinely discriminate better for incident (biology) |

Visual reference:
```
% Incident-Biased Features

50% ─────────→ [BALANCED]
               (features discriminate equally)

68% ─────────→ [INCIDENT-BIASED]
               (features favor incident discrimination)

Phase 1: 68%
Phase 2: 52%
         → Imbalance biased feature selection
```

**Dose-response expectation**:
- 5:1 ratio: 68% incident-biased
- 1:1 ratio: 52% incident-biased
- 1:10 ratio: 48% incident-biased
- Pattern suggests **dose-response** = stronger artifact evidence

---

## Decision Tree

### Step 1: Compare Phase 1 → Phase 2

```python
score_diff_phase1 = 0.25  # 25% median difference
score_diff_phase2 = 0.05  # 5% median difference
reduction_fraction = (score_diff_phase1 - score_diff_phase2) / score_diff_phase1
# reduction_fraction = 0.80 = 80% reduction
```

### Step 2: Interpret Reduction

```
if reduction_fraction > 0.60:
    # Strong artifact
    conclusion = "CLASS IMBALANCE IS THE PROBLEM"
    action = "RETRAIN with balanced sampling"

elif reduction_fraction < 0.20:
    # Biology-driven
    conclusion = "BIOLOGICAL DIFFERENCE"
    action = "ACCEPT, document, stratify reporting by case type"

else:
    # Mixed causes
    conclusion = "MIXED: Both artifact and biology"
    action = "RETRAIN + implement case-aware calibration"
```

### Step 3: Validate with Calibration

If artifact hypothesis:
```
Phase 1 calibration slope for prevalent: 0.72 (BAD)
Phase 2 calibration slope for prevalent: 0.96 (GOOD)
→ Confirms: imbalance causes miscalibration → retrain
```

If biology hypothesis:
```
Phase 1 calibration slope: 0.88 (MODERATE)
Phase 2 calibration slope: 0.87 (MODERATE)
→ Confirms: systematic underprediction is persistent → accept as biology
```

### Step 4: Cross-Check with Features

If artifact hypothesis:
```
Phase 1 incident-biased: 68%
Phase 2 incident-biased: 52%
→ Confirms: imbalance biases feature selection → retrain
```

If biology hypothesis:
```
Phase 1 incident-biased: 65%
Phase 2 incident-biased: 64%
→ Confirms: feature selection is consistent → support biology
```

---

## Detailed Decision Tree (Visual)

```
                    BASELINE (Phase 1)
                  Score diff = 25%
                          │
                          ▼
            Retrain with balanced config (Phase 2)
                          │
                          ▼
                  Score diff = ?
                          │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
     5%            12-20%            25%
  (Reduction      (Moderate         (No
   70%+)          Reduction)        Reduction
                   30-70%+)
        │               │               │
        ▼               ▼               ▼
   ARTIFACT!       MIXED CAUSES     BIOLOGY!
   (Class          (Both artifact   (True
    imbalance      & biology)       difference)
    is problem)         │               │
        │               │               │
        ├──────────┬────┴────┬──────────┤
        │          │         │          │
        ▼          ▼         ▼          ▼
     Check    Check      Check    Check
   Calibr.  Calibr.   Calibr.  Calibr.
   Slope    Slope     Slope    Slope
        │      │         │        │
        ▼      ▼         ▼        ▼
    0.72     0.82      0.85    0.95
   (BAD)   (MODERATE)  (FAIR)  (GOOD)
        │      │         │        │
        ▼      ▼         ▼        ▼
   ARTIFACT  MIXED     MIXED   BIOLOGY
   CONFIRMED CONFIRMED  WEAK   CONFIRMED
```

---

## Interpretation Patterns

### Pattern 1: Artifact-Driven (Retrain)

**Evidence**:
- Score diff reduces 70%+ (Phase 1 → Phase 2)
- Calibration slope improves >0.10
- Feature bias % drops 15%+
- Dose-response: 5:1 has bias, 1:1 doesn't, 1:10 has little

**Action**: Retrain with balanced sampling
```bash
# Update config
train_control_per_case: 1    # Instead of 5
prevalent_train_frac: 1.0    # Instead of 0.5

# Retrain
cd analysis/
ced train --model LR_EN --split-seed 0
# ... repeat for all models
```

**Validation**: Run Phase 3 (holdout test on new split seed) to verify improvement

---

### Pattern 2: Biology-Driven (Accept & Document)

**Evidence**:
- Score diff unchanged (>80% persists)
- Calibration slope similar in both phases
- Feature bias % similar in both phases
- Dose-response: linear relationship (5:1, 1:1, 1:10 all show similar pattern)

**Action**: Document and implement case-specific handling
```markdown
# ADR-016: Incident vs Prevalent Score Difference

## Finding
Prevalent cases score 18-22% lower than incident cases under balanced training.

## Interpretation
Pre-diagnostic biomarkers (incident) differ materially from post-diagnosis biomarkers
(prevalent), reflecting treatment effects and disease stage.

## Implications
- Report metrics separately by case type in publications
- Use case-specific thresholds for deployment
- Acknowledge in model card: "Model not validated on prevalent cases"

## Evidence
- Phase 1 vs Phase 2: difference persists (biology hypothesis)
- Calibration: both phases well-calibrated
- Features: stable selection across phases
```

---

### Pattern 3: Mixed Causes (Retrain + Case-Aware)

**Evidence**:
- Score diff reduces 40% (Phase 1 → Phase 2)
- Calibration slope improves but still suboptimal
- Feature bias % drops 10%

**Action**: Retrain + implement case-aware calibration
```bash
# Step 1: Retrain with balanced sampling
cd analysis/
ced train --model LR_EN --split-seed 0  # Use balanced config

# Step 2: Edit calibration strategy in training_config.yaml
calibration:
  strategy: case_aware
  method: isotonic
  # Fit separate calibrators for incident vs prevalent

# Step 3: Re-evaluate on Phase 3 (new split seed)
```

---

## Extended Analysis: Dose-Response (2×3 Experiment)

For even stronger evidence, test with three case:control ratios:

```bash
bash run_experiment.sh --case-control-ratios 1,5,10
```

### Expected Patterns

**Artifact-driven dose-response**:
```
Ratio 1:5  → 25% median diff
Ratio 1:1  → 5% median diff
Ratio 1:10 → 2% median diff

Linear reduction: strongly supports artifact
```

**Biology-driven dose-response**:
```
Ratio 1:5  → 25% median diff
Ratio 1:1  → 22% median diff
Ratio 1:10 → 21% median diff

Flat: strongly supports biology
```

**Mixed dose-response**:
```
Ratio 1:5  → 25% median diff
Ratio 1:1  → 12% median diff
Ratio 1:10 → 8% median diff

Partial reduction: mixed causes
```

---

## Power Analysis

### Sample Sizes per Configuration

| Configuration | Incident Cases (OOF) | Incident Cases (Test) | Total |
|---|---|---|------|
| Single seed | ~75 | ~36 | ~111 |
| All 10 seeds OOF | ~750 | — | ~750 |
| All 10 seeds (OOF + test) | ~750 | ~360 | ~1,110 |

### Statistical Power (α=0.05, two-sided Mann-Whitney)

| Effect Size | Single Seed | 10 Seeds | Detection |
|---|---|---|------|
| Small (d=0.3) | 65% | >99.9% | Unreliable in single seed |
| Medium (d=0.5) | **95%** | >99.9% | Reliable |
| Large (d=0.8) | **99%** | >99.9% | Very reliable |

**Recommendation**: Always report results across ≥5 seeds for robust conclusions.

---

## Confounding Check

Before concluding "biology," verify no hidden confounders:

```bash
# In Phase 2 results, stratify by age
python -c "
import pandas as pd

scores = pd.read_csv('results/investigations/1.0_1/scores_LR_EN_oof_seed0.csv')
# scores columns: case_id, case_type (incident/prevalent), risk_score, age

for age_group in ['<40', '40-60', '>60']:
    subset = scores[scores['age_group'] == age_group]
    incident_med = subset[subset['case_type']=='incident']['risk_score'].median()
    prevalent_med = subset[subset['case_type']=='prevalent']['risk_score'].median()
    diff = (incident_med - prevalent_med) / prevalent_med * 100
    print(f'{age_group}: {diff:+.1f}%')
"
```

If difference is **consistent across age groups** → biology hypothesis supported
If difference **varies by age** → potential confounding → need stratified model

---

## Configuration Matrix Reference

### 2×2 Experiment (Default, 30 min)

```
                    Case:Control Ratio
                      1:1        1:5
Prevalent=0.5    ┌──────────┬──────────┐
                 │   A      │   B      │
                 │          │          │
                 └──────────┴──────────┘
                 │   C      │   D      │
Prevalent=1.0    │          │          │
                 └──────────┴──────────┘

A (prev=0.5, 1:1):   Balanced prevalent, balanced case:control
B (prev=0.5, 1:5):   Balanced prevalent, imbalanced case:control (production default)
C (prev=1.0, 1:1):   Full prevalent, balanced case:control
D (prev=1.0, 1:5):   Full prevalent, imbalanced case:control

For artifact diagnosis:
- Compare B (production) to A (balanced) → isolates case:control effect
- Compare D to C → isolates prevalent sampling effect
```

### 2×3 Experiment (Extended, 60 min)

```
                    Case:Control Ratio
                   1:1      1:5     1:10
Prevalent=0.5    ┌────┬────┬────┐
                 │ A  │ B  │ B* │
                 └────┴────┴────┘
Prevalent=1.0    ┌────┬────┬────┐
                 │ C  │ D  │ D* │
                 └────┴────┴────┘

Extended ratios test dose-response:
- Flat across ratios → Biology
- Linear decrease → Artifact
- Partial → Mixed
```

---

## EXPERIMENT_COMPARISON.csv Output Example

```
Config,Prevalent_Frac,Case_Control,Model,Median_Diff,Effect_Size,Slope,Intercept,Incident_Bias_%,Significant
0.5_1,0.5,1:1,LR_EN,+0.0834,0.542,0.96,-0.02,52%,YES
0.5_5,0.5,1:5,LR_EN,+0.2341,0.891,0.72,-0.15,68%,YES
1.0_1,1.0,1:1,LR_EN,+0.0567,0.381,0.95,-0.01,51%,NO
1.0_5,1.0,1:5,LR_EN,+0.1956,0.743,0.82,-0.10,65%,YES

INTERPRETATION:
- Row 1 vs Row 2: Score diff reduced 64% (0.5_5 → 0.5_1) → ARTIFACT
- Row 3 vs Row 4: Score diff reduced 71% (1.0_5 → 1.0_1) → ARTIFACT
- Slope improves in all 1:1 configs → CONFIRMS imbalance causes miscalibration
- Incident bias % drops to ~50% in 1:1 configs → CONFIRMS feature selection bias
```

---

## Quick Reference: What to Look For

| Metric | Phase 1 (Artifact Indicator) | Phase 2 (Artifact Indicator) | Interpretation |
|--------|--------|---------|---|
| **Score diff** | 25% | 5% | 80% reduction → Artifact |
| **Slope** | 0.72 | 0.95 | 0.23 improvement → Artifact |
| **Intercept** | -0.15 | -0.02 | Improves → Artifact |
| **Incident bias** | 68% | 52% | 16% reduction → Artifact |

---

## Next Steps

1. **Run baseline 2×2 experiment** (30 min, see README.md)
   - Review EXPERIMENT_COMPARISON.csv
   - Compare Phase 1 vs Phase 2
   - Apply decision tree above

2. **If unclear**, run extended 2×3 experiment (60 min)
   - Tests dose-response hypothesis
   - Provides stronger evidence

3. **Implement decision**:
   - If artifact: Retrain, validate on new split seed
   - If biology: Create ADR, update documentation
   - If mixed: Retrain + case-aware calibration

---

**Last Updated:** 2026-01-27
**Statistical Framework:** Nested cross-validation, Mann-Whitney U, calibration metrics
**Sample Size:** ~150 incident cases analyzed (full coverage across 10 splits)
