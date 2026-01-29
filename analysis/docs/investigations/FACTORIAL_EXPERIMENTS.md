# Factorial Experiments: Prevalent Cases and Control Ratios

**Last Updated**: 2026-01-28
**Status**: Production-ready consolidated runner

## Purpose

Investigate the impact of two key training set design choices on model performance:

1. **Prevalent case sampling fraction** (`prevalent_frac`): How many prevalent cases to include in training
2. **Case:control ratio** (`case_control_ratio`): Ratio of controls to incident cases in training

These experiments use a **fixed 100-protein panel** to eliminate feature selection variability and isolate the effects of training set composition.

---

## Quick Start

### Quick Test (30 minutes)
```bash
cd analysis/docs/investigations/
bash run_factorial_experiment.sh --quick
```

### Overnight Run (6-8 hours)
```bash
cd analysis/docs/investigations/
bash run_factorial_experiment.sh --overnight
```

### Full Experiment (24-30 hours)
```bash
cd analysis/docs/investigations/
bash run_factorial_experiment.sh --full
```

---

## Experimental Design

### Factors

| Factor | Levels | Values |
|--------|--------|--------|
| **Prevalent fraction** | 2 | 0.5 (50%), 1.0 (100%) |
| **Case:control ratio** | 3 | 1:1, 1:5, 1:10 |
| **Models** | 2-4 | LR_EN, RF, XGBoost, LinSVM_cal |
| **Random seeds** | 1-10 | 0-9 |

### Configurations

| Config | Prevalent Fraction | Case:Control | Train Size (approx) |
|--------|-------------------|--------------|---------------------|
| 0.5_1  | 50%               | 1:1         | ~220 samples        |
| 0.5_5  | 50%               | 1:5         | ~812 samples        |
| 0.5_10 | 50%               | 1:10        | ~1,624 samples      |
| 1.0_1  | 100%              | 1:1         | ~298 samples        |
| 1.0_5  | 100%              | 1:5         | ~890 samples        |
| 1.0_10 | 100%              | 1:10        | ~1,702 samples      |

### Fixed Panel

**File**: `top100_panel.csv`
**Size**: 100 proteins (no header)
**Selection**: Mann-Whitney screening (top 1000) → k-best (top 100) → stability filtering (0.75 threshold, yielded 100)

Proteins are residualized biomarker values (`*_resid` columns).

### Frozen Configuration

**File**: `training_config_frozen.yaml`

Key settings:
- **Feature selection**: DISABLED (uses fixed panel)
- **CV structure**: 5 outer folds × 3 repeats × 3 inner folds
- **Optuna**: 100 trials, TPE sampler, median pruner
- **Calibration**: Isotonic regression (per-fold strategy)
- **Thresholds**: Fixed specificity 0.95

---

## Usage

### Basic Usage

```bash
# Quick test (12 runs: 6 configs × 1 seed × 2 models)
bash run_factorial_experiment.sh --quick

# Overnight run (60 runs: 6 configs × 5 seeds × 2 models)
bash run_factorial_experiment.sh --overnight

# Full experiment (240 runs: 6 configs × 10 seeds × 4 models)
bash run_factorial_experiment.sh --full
```

### Custom Configuration

```bash
# Custom prevalent fractions
bash run_factorial_experiment.sh \
  --prevalent-fracs 0.25,0.5,0.75,1.0 \
  --case-control-ratios 1,5,10 \
  --models LR_EN,RF \
  --split-seeds 0,1,2

# Skip split generation (use existing)
bash run_factorial_experiment.sh --overnight --skip-splits

# Force panel regeneration
bash run_factorial_experiment.sh --overnight --force-panel

# Dry run (preview without execution)
bash run_factorial_experiment.sh --overnight --dry-run
```

### All Options

| Option | Description |
|--------|-------------|
| `--quick` | Preset: 1 seed, 2 models (12 runs, ~30 min) |
| `--overnight` | Preset: 5 seeds, 2 models (60 runs, ~6-8 hours) |
| `--full` | Preset: 10 seeds, 4 models (240 runs, ~24-30 hours) |
| `--prevalent-fracs FRAC1,FRAC2,...` | Comma-separated fractions (default: 0.5,1.0) |
| `--case-control-ratios R1,R2,...` | Comma-separated ratios (default: 1,5,10) |
| `--models MODEL1,MODEL2,...` | Comma-separated models (default: LR_EN,RF) |
| `--split-seeds SEED1,SEED2,...` | Comma-separated seeds (default: 0,1,2,3,4) |
| `--skip-training` | Skip training (analyze existing results) |
| `--skip-splits` | Skip split generation (use existing) |
| `--skip-panel` | Skip panel generation (use existing) |
| `--force-panel` | Force panel regeneration |
| `--dry-run` | Preview without execution |
| `--help` | Show help message |

---

## Output Structure

### Results Directory

```
results/investigations/
├── 0.5_1/                                # Config: 50% prevalent, 1:1 ratio
│   ├── LR_EN/
│   │   └── split_seed0/                  # Single seed results
│   │       ├── metrics/
│   │       │   ├── val_metrics.json
│   │       │   └── test_metrics.json
│   │       ├── preds/
│   │       │   ├── val_predictions.csv
│   │       │   └── test_predictions.csv
│   │       └── plots/
│   │           ├── roc_pr_val.png
│   │           └── calibration_val.png
│   └── RF/
│       └── split_seed0/
├── 0.5_5/                                # Config: 50% prevalent, 1:5 ratio
├── 0.5_10/                               # Config: 50% prevalent, 1:10 ratio
├── 1.0_1/                                # Config: 100% prevalent, 1:1 ratio
├── 1.0_5/                                # Config: 100% prevalent, 1:5 ratio
├── 1.0_10/                               # Config: 100% prevalent, 1:10 ratio
└── experiment_20260128_123456/           # Statistical analysis
    ├── summary.md                        # Executive summary
    ├── results_by_config.csv             # Mean ± SD per config
    ├── results_by_seed.csv               # Individual seed results
    ├── statistical_tests.csv             # Paired t-tests
    ├── effect_sizes.csv                  # Cohen's d
    └── metadata.json                     # Experiment metadata
```

### Statistical Analysis Outputs

#### summary.md
Executive summary with:
- Best configuration per model
- Statistical significance of factor effects
- Effect sizes (Cohen's d)
- Recommendations

#### results_by_config.csv
Mean ± SD across seeds for each configuration:
```csv
model,prevalent_frac,case_control_ratio,auroc_mean,auroc_sd,prauc_mean,prauc_sd,...
LR_EN,0.5,1,0.923,0.012,0.456,0.034,...
LR_EN,0.5,5,0.945,0.008,0.512,0.028,...
```

#### results_by_seed.csv
Individual results for each run:
```csv
model,prevalent_frac,case_control_ratio,split_seed,auroc,prauc,brier,sensitivity,specificity,...
LR_EN,0.5,1,0,0.931,0.478,0.042,0.45,0.95,...
LR_EN,0.5,1,1,0.918,0.441,0.045,0.42,0.95,...
```

#### statistical_tests.csv
Paired t-tests with Bonferroni correction:
```csv
model,comparison,mean_diff,t_statistic,p_value,p_value_bonf,significant
LR_EN,1.0_5 vs 0.5_5,0.012,3.45,0.003,0.018,True
LR_EN,1.0_10 vs 0.5_10,0.008,2.31,0.025,0.150,False
```

#### effect_sizes.csv
Cohen's d effect sizes:
```csv
model,comparison,cohens_d,interpretation
LR_EN,1.0_5 vs 0.5_5,0.82,Large
LR_EN,1.0_10 vs 0.5_10,0.54,Medium
```

---

## Logs

All logs are saved to `analysis/../logs/experiments/`:

| Log File | Contents |
|----------|----------|
| `experiment_<TIMESTAMP>.log` | Full experiment run log |
| `panel_generation_<TIMESTAMP>.log` | Panel generation details |
| `splits_<pf>_<ccr>_<TIMESTAMP>.log` | Split generation per config |
| `train_<model>_<pf>_<ccr>_seed<N>_<TIMESTAMP>.log` | Individual training runs |
| `analysis_<TIMESTAMP>.log` | Statistical analysis details |

---

## Workflow Details

### Phase 0: Panel Generation

1. **Check existing panel**: If `top100_panel.csv` exists, prompt to overwrite (unless `--skip-panel`)
2. **Generate panel** (if needed):
   - Mann-Whitney screening (top 1000 proteins)
   - K-best selection (top 100)
   - Stability filtering (0.75 threshold)
   - Write to `top100_panel.csv` (100 proteins, no header)

### Phase 1: Split Generation

For each configuration (prevalent_frac × case_control_ratio):

1. **Create config directory**: `splits_experiments/<pf>_<ccr>/`
2. **Generate temporary splits config**:
   - Set `train_control_per_case: <ccr>`
   - Set `prevalent_sampling_frac: <pf>`
   - Set `n_splits: <num_seeds>`
3. **Run split generation**: `ced save-splits`
4. **Save splits**: `splits_<seed>.pkl` for each seed

### Phase 2: Model Training

For each configuration × seed × model:

1. **Create results directory**: `results/investigations/<pf>_<ccr>/<model>/split_seed<N>/`
2. **Run training**: `ced train` with:
   - `--fixed-panel top100_panel.csv` (no feature selection)
   - `--config training_config_frozen.yaml` (frozen hyperparameters)
   - `--split-file splits_<seed>.pkl`
   - Metadata: experiment_id, prevalent_frac, case_control_ratio, split_seed
3. **Save outputs**: Metrics, predictions, plots, model artifacts

### Phase 3: Statistical Analysis

1. **Collect results**: Scan `results/investigations/` for all runs
2. **Aggregate by config**: Mean ± SD across seeds
3. **Statistical tests**: Paired t-tests with Bonferroni correction
4. **Effect sizes**: Cohen's d for significant comparisons
5. **Generate summary**: Markdown report with recommendations

---

## Expected Results

### Hypotheses

**H1: Prevalent fraction effect**
- More prevalent cases → better discrimination (more positive examples)
- Expected effect: +1-3% AUROC for 1.0 vs 0.5

**H2: Case:control ratio effect**
- Higher ratios (more controls) → better calibration (balanced class distribution)
- Expected effect: +2-5% AUROC for 1:10 vs 1:1

**H3: Interaction effect**
- Optimal configuration may depend on model type
- Expected: LR_EN prefers balanced data, RF handles imbalance better

### Validation Criteria

1. **Statistical significance**: p < 0.05 (Bonferroni corrected)
2. **Effect size**: Cohen's d > 0.3 (medium effect)
3. **Consistency**: Significant across multiple models
4. **Robustness**: Low variance across random seeds (SD < 0.02 for AUROC)

---

## Troubleshooting

### Panel Generation Issues

**Problem**: Panel generation fails
**Solution**: Check data file path and ensure `ced_ml` is installed
```bash
cd analysis/
python -c "import ced_ml; print('OK')"
```

**Problem**: Panel has wrong number of proteins
**Solution**: Regenerate with `--force-panel`:
```bash
bash run_factorial_experiment.sh --overnight --force-panel
```

### Split Generation Issues

**Problem**: Split generation fails for some configs
**Solution**: Check case:control ratio is compatible with sample sizes:
- Incident cases: ~148
- Prevalent cases: ~150
- Controls: ~43,662

**Problem**: Splits already exist
**Solution**: Use `--skip-splits` to reuse existing splits

### Training Issues

**Problem**: Training fails for some runs
**Solution**: Check individual training logs in `logs/experiments/train_*.log`

**Problem**: Out of memory
**Solution**: Reduce number of Optuna trials in `training_config_frozen.yaml`

### Analysis Issues

**Problem**: Statistical analysis fails
**Solution**: Ensure all training runs completed successfully:
```bash
find results/investigations/ -name "val_metrics.json" | wc -l
# Should equal: num_configs × num_seeds × num_models
```

**Problem**: Missing results for some configurations
**Solution**: Rerun training for missing configs:
```bash
bash run_factorial_experiment.sh \
  --prevalent-fracs 0.5 \
  --case-control-ratios 5 \
  --models LR_EN \
  --split-seeds 3
```

---

## Technical Notes

### Reproducibility

- **Fixed panel**: Eliminates feature selection variability
- **Frozen config**: Identical hyperparameters across all runs
- **Seeded splits**: Deterministic train/val/test assignment
- **Metadata tracking**: All runs tagged with experiment_id, config, seed

### Statistical Rigor

- **Multiple seeds**: 5-10 seeds per configuration (estimate variance)
- **Paired tests**: Same splits across configs (paired t-tests valid)
- **Multiple comparisons**: Bonferroni correction (15 comparisons per model)
- **Effect sizes**: Cohen's d (distinguish statistical vs practical significance)

### Computational Cost

| Preset | Runs | Models | Seeds | Duration | Total CPU-hours |
|--------|------|--------|-------|----------|-----------------|
| Quick | 12 | 2 | 1 | 30 min | 0.5 |
| Overnight | 60 | 2 | 5 | 6-8 hrs | 6-8 |
| Full | 240 | 4 | 10 | 24-30 hrs | 40-50 |

**Per-run cost**: ~5 minutes (nested CV with 45 folds, 100 Optuna trials)

---

## Key Files

| File | Purpose |
|------|---------|
| **run_factorial_experiment.sh** | Consolidated runner (replaces run_overnight.sh and run_experiment_v2.sh) |
| **top100_panel.csv** | Fixed 100-protein panel (no header) |
| **training_config_frozen.yaml** | Frozen hyperparameter config |
| **generate_fixed_panel.py** | Panel generation script |
| **analyze_factorial_results.py** | Statistical analysis script |

---

## Next Steps

### After Running Experiments

1. **Review summary**: Check `experiment_<TIMESTAMP>/summary.md`
2. **Identify best config**: Highest mean AUROC with statistical significance
3. **Validate robustness**: Check variance across seeds (SD < 0.02)
4. **Update pipeline**: Use optimal config for production training
5. **Document findings**: Update CLAUDE.md and relevant ADRs

### Follow-up Investigations

1. **Temporal validation**: Do effects hold with temporal splits?
2. **External validation**: Test optimal config on holdout set
3. **Feature ablation**: Are all 100 proteins necessary?
4. **Calibration impact**: Does config affect Brier score?
5. **Clinical utility**: DCA net benefit at optimal thresholds

---

## References

### Documentation
- [ADR-002: Prevalent cases in training only](../adr/ADR-002-prevalent-train-only.md)
- [ADR-003: Control downsampling ratio](../adr/ADR-003-control-downsampling.md)
- [FEATURE_SELECTION.md](../reference/FEATURE_SELECTION.md) - Panel generation methods

### Related Tools
- `ced save-splits` - Split generation CLI
- `ced train` - Model training CLI
- `ced aggregate-splits` - Results aggregation CLI

---

**Last Updated**: 2026-01-28
**Maintainer**: Andres Chousal
**Status**: Production-ready
