# Factorial Experiment: Ready to Run

**Status**: All setup complete
**Date**: 2026-01-27
**Estimated Time**: 1 hour (quick test) or 5 hours (full experiment)

---

## What's Ready

### Core Files Created

| File | Purpose | Status |
|------|---------|--------|
| `run_experiment_v2.sh` | Main experiment runner | Ready |
| `generate_fixed_panel.py` | Panel generation | Ready |
| `investigate_factorial.py` | Statistical analysis | Ready |
| `training_config_frozen.yaml` | Frozen experimental controls | Ready |
| `CORRECTED_FACTORIAL_PLAN.md` | Experimental design doc | Ready |
| `QUICKSTART.md` | Quick reference guide | Ready |

### Key Improvements from v1

1. **5 random seeds** (was 3) for robust variance estimation
2. **Fixed 100-protein panel** eliminates feature selection variability
3. **Frozen config** ensures only sampling varies between configs
4. **Statistical testing**: Paired t-tests with Bonferroni correction
5. **Effect sizes**: Cohen's d for practical significance
6. **Power analysis**: Post-hoc calculation for sample size justification

---

## Run Commands

### Option 1: Quick Test (Recommended First)

```bash
cd /Users/andreschousal/Projects/Elahi_Lab/CeliacRisks/analysis/docs/investigations

# Run 1 seed to verify everything works
bash run_experiment_v2.sh --skip-splits --split-seeds 0
```

**Expected**:
- 8 training runs (4 configs × 1 seed × 2 models)
- ~45-60 minutes runtime
- Output in `results/investigations/experiment_*/`

### Option 2: Full Experiment

```bash
# After quick test succeeds
bash run_experiment_v2.sh --skip-splits --split-seeds 0,1,2,3,4
```

**Expected**:
- 40 training runs (4 configs × 5 seeds × 2 models)
- ~4-5 hours runtime
- Full statistical power for comparisons

---

## Verification Checklist

Before running, verify:

- [x] `training_config_frozen.yaml` exists
- [x] `generate_fixed_panel.py` exists
- [x] `investigate_factorial.py` exists
- [x] `run_experiment_v2.sh` is executable
- [ ] Splits exist in `../../../splits_experiments/` (should have 0.5_1, 0.5_5, 1.0_1, 1.0_5)
- [ ] Data file exists at `../../../data/Celiac_dataset_proteomics_w_demo.parquet`

**To check splits**:
```bash
ls -la ../../../splits_experiments/
```

**If splits don't exist**, remove `--skip-splits` from the run command.

---

## Expected Output

### During Run

```
################################################################################################
  FACTORIAL EXPERIMENT v2 (Statistical Rigor)
################################################################################################

Experiment Design:
  Experiment ID:        20260127_HHMMSS
  Prevalent fractions:  0.5 1.0
  Case:control ratios:  1 5
  Models:               LR_EN RF
  Random seeds:         0 1 2 3 4
  Fixed panel:          top100_panel.csv
  Frozen config:        training_config_frozen.yaml

  Total configurations: 4
  Total runs:           40 (configs × seeds × models)

################################################################################################
  PHASE 0: Generating Fixed 100-Protein Panel
################################################################################################

[2026-01-27 HH:MM:SS] Generating panel (Mann-Whitney screening + k-best selection)...
[✓] Panel generated: 101 proteins (including header)

################################################################################################
  PHASE 2: Training Models (Frozen Config + Fixed Panel)
################################################################################################

[Config 1/4] Configuration: prevalent_frac=0.5, case_control=1
  [Seed 0]
    [Run 1/40] Training LR_EN...
    [✓] [Config 1/4]  [Seed 0]    [Run 1/40] LR_EN complete
    [Run 2/40] Training RF...
    [✓] [Config 1/4]  [Seed 0]    [Run 2/40] RF complete
  [Seed 1]
    ...

################################################################################################
  PHASE 3: Statistical Analysis
################################################################################################

[2026-01-27 HH:MM:SS] Running factorial analysis with statistical testing...
[✓] Analysis complete

  Output files:
  - metrics_all.csv (raw metrics, 40 rows)
  - comparison_table.csv (config summary, 8 rows)
  - statistical_tests.csv (paired comparisons, 8 tests)
  - power_analysis.csv (post-hoc power)
  - summary.md (human-readable findings)

  Location: /path/to/results/investigations/experiment_20260127_HHMMSS
```

### After Completion

```bash
# View summary
cat results/investigations/experiment_*/summary.md

# Check comparison table
head results/investigations/experiment_*/comparison_table.csv

# Check statistical tests
head results/investigations/experiment_*/statistical_tests.csv
```

---

## Experimental Controls (Frozen)

| Parameter | Value | Why Fixed |
|-----------|-------|-----------|
| Feature panel | 100 top proteins | Eliminates FS variability across configs |
| Feature selection | None (uses fixed panel) | Only sampling varies |
| Hyperparameters | Optuna 100 trials | Each config finds optimal HP |
| Calibration | OOF-posthoc isotonic | Matches production (ADR-014) |
| CV structure | 5×3×3 nested CV | Standard pipeline |
| Val/test sets | Same across configs | Only training set varies |

---

## What Varies Between Configs

| Config | Prevalent Frac | Case:Control | n_train | Train Prevalence |
|--------|----------------|--------------|---------|------------------|
| 0.5_1  | 50% | 1:1 | 298 | 50% |
| 0.5_5  | 50% | 1:5 | 894 | 17% |
| 1.0_1  | 100% | 1:1 | 298 | 50% |
| 1.0_5  | 100% | 1:5 | 894 | 17% |

---

## Questions Answered

After running, you'll have statistical evidence for:

1. **Case:control ratio effect** (1:1 vs 1:5)
   - Is LR_EN sensitive to class imbalance?
   - Does more controls improve AUROC?

2. **Prevalent sampling effect** (50% vs 100%)
   - Does including all prevalent cases help?
   - Biology vs sampling artifact?

3. **Model robustness** (LR_EN vs RF)
   - Which model is more stable across configs?
   - Which is production-ready?

4. **Optimal configuration**
   - Statistically justified recommendation
   - Effect sizes and power analysis

---

## Troubleshooting

See [QUICKSTART.md](QUICKSTART.md) for detailed troubleshooting steps.

Quick checks:
```bash
# Check environment
cd ../../
python -c "import ced_ml; print('OK')"

# Check data
ls -la ../data/Celiac_dataset_proteomics_w_demo.parquet

# Check splits
ls -la ../splits_experiments/

# Dry run
cd docs/investigations
bash run_experiment_v2.sh --dry-run --split-seeds 0
```

---

## Next Steps

1. **Run quick test** (1 seed, ~1 hour)
2. **Review test results** (verify non-duplicate metrics)
3. **Run full experiment** (5 seeds, ~5 hours)
4. **Analyze findings** (statistical_tests.csv + summary.md)
5. **Update production config** based on optimal configuration
6. **Document in main summary** (link to this experiment)

---

**Ready to proceed!** Start with the quick test command above.
