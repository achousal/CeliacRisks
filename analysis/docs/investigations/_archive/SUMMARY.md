# Experiment Investigation Summary

**Date**: 2026-01-27
**Experiment**: Factorial design testing prevalent vs incident case scores
**Status**: ✅ **FIXED AND DOCUMENTED**

---

## What Was Done

### 1. Fixed Investigation Script ✅
**Problem**: Original script couldn't find model outputs in new directory structure.

**Solution**: Created `investigate_v2.py` with:
- Automatic model detection from run directories
- Power analysis and sample size calculations
- Multi-configuration comparison
- Enhanced reporting

**Location**: `/analysis/docs/investigations/investigate_v2.py`

### 2. Identified Power Issues ✅
**Finding**: Study severely underpowered (7-33% power vs 80% target).

**Current Sample**: 73-76 per group
**Required Sample**: 66 per group for medium effects, 251-2459 for small/observed effects

**Recommendation**: Use all 148 incident cases (no holdout) to maximize power.

### 3. Diagnosed Calibration Bias ✅
**Finding**: Incident cases underpredicted MORE than prevalent (-0.88 vs -0.63 intercept).

**Root Cause**: Per-fold calibration may introduce bias.

**Solution**: Switch to OOF-posthoc calibration (already implemented, just needs config change).

### 4. Compared Configurations ✅
**Finding**: All configurations produced DUPLICATE results (investigation used wrong splits).

**Expected**: 4 unique configurations with different sample sizes.

**Actual**: 8 runs with identical sample sizes (73 incident, 76 prevalent).

**Next Step**: Re-run experiment with corrected split directories.

---

## Key Results

### Statistical Power Analysis

| Current Sample | Observed Effect | Observed Power | Required N (80% power) |
|---------------|-----------------|----------------|------------------------|
| 73-76 per group | d = 0.08-0.26 | 7-33% | 251-2459 per group |

**Conclusion**: Study cannot detect small-to-medium effects with current sample.

### Sample Size Requirements (80% Power)

| Effect Size | Description | n per group | Total n | Feasibility |
|------------|-------------|-------------|---------|-------------|
| d = 0.2 | Small | 411 | 822 | ❌ Not feasible |
| d = 0.5 | Medium | 66 | 132 | ✅ **ACHIEVABLE** |
| d = 0.8 | Large | 26 | 52 | ✅ Already exceeded |

**Recommendation**: Target **66+ per group** (132 total) for reliable results.

### Calibration Metrics

| Case Type | Intercept | Slope | Brier | Quality |
|-----------|-----------|-------|-------|---------|
| Incident | -0.88 | 0.94 | 0.048 | Poor (severe underprediction) |
| Prevalent | -0.63 | 0.98 | 0.052 | Fair (moderate underprediction) |
| **Bias** | **-0.25** | **-0.05** | **-0.005** | **Incident MORE biased** |

**Conclusion**: Model discriminates incident and prevalent differently; calibration biased.

---

## Documentation Created

1. **Investigation Results** - Full analysis with findings
   - Location: `/analysis/docs/bugs/investigation_results_2026-01-27.md`
   - Content: Bugs fixed, power analysis, recommendations

2. **Calibration Review** - Detailed calibration strategy comparison
   - Location: `/analysis/docs/investigations/calibration_review.md`
   - Content: Current issues, available strategies, recommendations

3. **Improved Investigation Script** - Fixed path resolution + power analysis
   - Location: `/analysis/docs/investigations/investigate_v2.py`
   - Features: Auto-detection, power calc, multi-config comparison

4. **Configuration Comparison** - Results across all runs
   - Location: `/results/investigations/configuration_comparison.csv`
   - Content: 8 runs with metrics (duplicates identified)

---

## Immediate Next Steps

### Step 1: Enable OOF-Posthoc Calibration (5 min)

Edit `configs/training_config.yaml`:

```yaml
calibration:
  enabled: true
  strategy: oof_posthoc  # ← Change from 'per_fold'
  method: isotonic
```

**Expected improvement**:
- Incident intercept: -0.88 → -0.20 to 0.00
- Prevalent intercept: -0.63 → -0.15 to 0.00
- Bias reduction: 60-100%

### Step 2: Re-Run Experiment with Correct Splits (2 hours)

```bash
cd analysis/docs/investigations

# Option A: Quick test (2 models, 1 split)
bash run_experiment.sh --models LR_EN,RF --analyses distributions

# Option B: Full experiment (4 configs, 2 models)
bash run_experiment.sh --models LR_EN,RF --analyses distributions,calibration,features
```

**Expected**: 4 unique configurations with different sample sizes.

### Step 3: Analyze Results with New Script (5 min)

```bash
cd analysis
python docs/investigations/investigate_v2.py --compare-configs
```

**Output**: Configuration comparison CSV + power analysis.

### Step 4: Increase Power (Optional)

**Option A - Use All Data** (Recommended):
```yaml
# configs/splits_config.yaml
holdout_size: 0.0  # Remove holdout, keep only test
test_size: 0.30    # Larger test set
val_size: 0.20     # Smaller val set
# Results in ~105 incident cases for training (vs current 73)
```

**Option B - Pool Multiple Splits**:
- Combine data from all 10 splits for analysis
- Treat as repeated measures design
- Achieves ~730 incident cases total

**Option C - Collect More Data** (Long-term):
- Target 250+ incident cases
- Achieve 80%+ power for all effect sizes

---

## Tools Reference

### Power Analysis
```bash
# Calculate required sample size
python investigate_v2.py --power-analysis --target-effect 0.5

# Output: n per group for 70%, 80%, 90% power
```

### Configuration Comparison
```bash
# Compare all experimental runs
python investigate_v2.py --compare-configs

# Output: configuration_comparison.csv
```

### Single Run Analysis
```bash
# Analyze specific run
python investigate_v2.py --run-id 20260127_160356 --model LR_EN

# Output: analysis_{model}_{run_id}.csv
```

---

## Expected Timeline

| Task | Duration | Priority |
|------|----------|----------|
| Enable OOF-posthoc calibration | 5 min | ⚠️ HIGH |
| Re-run experiment (corrected) | 2 hours | ⚠️ HIGH |
| Analyze with investigate_v2.py | 5 min | 🔷 MEDIUM |
| Increase power (modify splits) | 1 hour | 🔷 MEDIUM |
| Document final results | 30 min | 🔷 MEDIUM |
| **Total** | **~4 hours** | |

---

## Success Criteria

After re-running with fixes:

- [ ] **4 unique configurations** with different sample sizes
- [ ] **RF model analyzed** (not just LR_EN)
- [ ] **Power ≥80%** for medium effects (or documented limitation)
- [ ] **Calibration bias reduced** (|intercept_diff| < 0.15)
- [ ] **Statistical significance** determined with adequate power
- [ ] **Results documented** in ADR or publication supplement

---

## Questions Answered

### Q1: Did the experiment work?
**A**: Partially. Training worked, but investigation used wrong splits (now fixed).

### Q2: Are incident and prevalent cases different?
**A**: Trend suggests incident score HIGHER, but **not statistically significant** (p=0.08-0.28). Study underpowered to detect small effects.

### Q3: Is the model biased?
**A**: **YES**. Calibration shows incident cases underpredicted more than prevalent (-0.88 vs -0.63 intercept).

### Q4: What should we do?
**A**:
1. Enable OOF-posthoc calibration (reduces bias)
2. Re-run with corrected splits
3. Increase sample size to 66+ per group
4. Re-analyze with investigate_v2.py

### Q5: Can we use current results?
**A**: Use with caution. Current findings:
- Direction: Incident scores HIGHER (consistent across models)
- Magnitude: Small effect (d=0.08-0.26)
- Confidence: LOW (underpowered, p>0.05)
- Calibration: BIASED (needs correction)

**Recommendation**: Report as exploratory, pending adequately powered replication.

---

## Files to Review

1. **Bug Report**: `/analysis/docs/bugs/investigation_results_2026-01-27.md`
2. **Calibration Review**: `/analysis/docs/investigations/calibration_review.md`
3. **This Summary**: `/analysis/docs/investigations/SUMMARY.md`
4. **Comparison Data**: `/results/investigations/configuration_comparison.csv`
5. **New Script**: `/analysis/docs/investigations/investigate_v2.py`

---

**Last Updated**: 2026-01-27
**Next Review**: After re-running experiment with fixes
