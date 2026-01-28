# Factorial Experiment Quick Start

## Files Ready

### Scripts
- `run_experiment_v2.sh` - Main experiment runner (v2 with statistical rigor)
- `generate_fixed_panel.py` - Panel generation utility
- `investigate_factorial.py` - Statistical analysis script

### Configs
- `training_config_frozen.yaml` - Frozen experimental controls
- `CORRECTED_FACTORIAL_PLAN.md` - Full experimental plan

## Run the Experiment

### Step 1: Quick Test (1 seed, ~1 hour)

```bash
cd analysis/docs/investigations

# If splits don't exist yet
bash run_experiment_v2.sh --split-seeds 0

# If splits already exist (recommended)
bash run_experiment_v2.sh --skip-splits --split-seeds 0
```

**Expected output**: 8 training runs (4 configs × 1 seed × 2 models)

### Step 2: Review Quick Test Results

```bash
# Check the experiment output directory
ls -la ../../../results/investigations/experiment_*/

# Read summary
cat ../../../results/investigations/experiment_*/summary.md

# Check metrics
head ../../../results/investigations/experiment_*/comparison_table.csv
head ../../../results/investigations/experiment_*/statistical_tests.csv
```

### Step 3: Full Experiment (5 seeds, ~5 hours)

```bash
# After confirming quick test works
bash run_experiment_v2.sh --skip-splits --split-seeds 0,1,2,3,4
```

**Expected output**: 40 training runs (4 configs × 5 seeds × 2 models)

## Verify Setup

### Check Prerequisites

```bash
# 1. Splits exist for all configs
ls -la ../../../splits_experiments/
# Should show: 0.5_1/, 0.5_5/, 1.0_1/, 1.0_5/

# 2. Frozen config exists
cat training_config_frozen.yaml

# 3. Panel generation script works
python generate_fixed_panel.py \
    --infile ../../../data/Celiac_dataset_proteomics_w_demo.parquet \
    --outfile test_panel.csv \
    --final-k 10
# Should create test_panel.csv with 11 lines (header + 10 proteins)
rm test_panel.csv
```

## Troubleshooting

### Panel Generation Fails

```bash
# Check data file exists
ls -la ../../../data/Celiac_dataset_proteomics_w_demo.parquet

# Test imports
cd ../../
python -c "from ced_ml.data.io import load_data; print('OK')"
python -c "from ced_ml.features.screening import screen_features; print('OK')"
python -c "from ced_ml.features.kbest import select_k_best; print('OK')"
```

### Training Fails

```bash
# Check logs
tail -n 50 ../../../logs/experiments/training_0.5_1_seed0.log

# Test frozen config
cd ../../
ced train --help
ced train --config docs/investigations/training_config_frozen.yaml --help
```

### Analysis Fails

```bash
# Check if runs completed
find ../../../results -name "config_metadata.json" | wc -l
# Should match expected run count

# Check log
tail -n 100 ../../../logs/experiments/analysis_*.log

# Test analysis script manually
cd ../../
python docs/investigations/investigate_factorial.py \
    --results-dir ../results \
    --output-dir ../results/investigations/test_output
```

## Expected Timeline

| Step | Duration | Output |
|------|----------|--------|
| Panel generation | 2-5 min | top100_panel.csv |
| Quick test (1 seed) | 45-60 min | 8 runs |
| Full experiment (5 seeds) | 4-5 hours | 40 runs |
| Statistical analysis | 5 min | 5 CSV + summary.md |

## Key Files Generated

```
results/investigations/experiment_{TIMESTAMP}/
├── metrics_all.csv              # All 40 runs (or 8 for quick test)
├── comparison_table.csv         # 8 config-model combinations
├── statistical_tests.csv        # 8 paired comparisons
├── power_analysis.csv           # Power calculations
└── summary.md                   # Human-readable findings
```

## Decision Points

After reviewing results, you'll be able to answer:

1. **Does case:control ratio (1:1 vs 1:5) affect AUROC?**
   - Compare 0.5_1 vs 0.5_5 and 1.0_1 vs 1.0_5 in statistical_tests.csv
   - Look for p_adj < 0.00625 (Bonferroni-corrected alpha)

2. **Does prevalent sampling (50% vs 100%) affect AUROC?**
   - Compare 0.5_1 vs 1.0_1 and 0.5_5 vs 1.0_5 in statistical_tests.csv

3. **Which effects are large vs small?**
   - Check Cohen's d in statistical_tests.csv
   - |d| >= 0.8: large, 0.5-0.8: medium, 0.2-0.5: small

4. **Production recommendation?**
   - Choose config with highest AUROC_mean in comparison_table.csv
   - Verify it's statistically superior (significant in tests)

## Next Steps After Completion

1. Review `summary.md` for automated interpretation
2. Check `statistical_tests.csv` for significant differences
3. Verify adequate power in `power_analysis.csv` (target: >0.80)
4. Update production config based on optimal configuration
5. Document findings in main investigation summary

## References

- Full plan: [CORRECTED_FACTORIAL_PLAN.md](CORRECTED_FACTORIAL_PLAN.md)
- Statistical methods: Paired t-tests, Bonferroni correction, Cohen's d
- Alpha threshold: 0.00625 (0.05/8 comparisons)
