# Level 3 Overnight Production Run - Quick Reference

**Date**: 2026-01-23
**Goal**: Maximum statistical power for publication-ready model selection
**Expected runtime**: 14-16 hours on HPC premium queue

---

## Quick Launch (TL;DR)

```bash
cd analysis/
./run_overnight_L3.sh
# Answer prompts, confirm launch
# Monitor: bjobs -w | grep CeD_
# After completion: bash scripts/post_training_pipeline.sh --run-id <RUN_ID>
```

---

## Configuration Summary

### Splits (10 independent random splits)
```yaml
# configs/splits_config.yaml
n_splits: 10
seed_start: 0
```

### Cross-Validation (Maximum statistical power)
```yaml
# configs/training_config_production.yaml
cv:
  folds: 5              # Outer CV
  repeats: 10           # Maximum repeats for tight CIs
  inner_folds: 5        # Full inner tuning
```

### Optuna (Full Bayesian optimization)
```yaml
optuna:
  enabled: true
  n_trials: 100         # Full convergence budget
  multi_objective: false  # Single AUROC focus
  sampler: tpe
  pruner: hyperband
```

### Evaluation (Publication quality)
```yaml
evaluation:
  n_boot: 1000          # Narrow bootstrap CIs
  learning_curve: true  # Sample efficiency analysis
```

### HPC Resources (Maximum allocation)
```yaml
# configs/pipeline_hpc.yaml
hpc:
  project: acc_Chipuk_Laboratory
  queue: premium        # Overnight priority queue
  walltime: "48:00"     # 48-hour safety margin
  cores: 8              # Maximum parallelism per job
  mem_per_core: 8000    # 8 GB (prevent OOM)
```

---

## Compute Budget

**Job array**: 40 jobs (10 splits × 4 models)

| Model      | Time/Split | Parallel Time | Bottleneck |
|------------|------------|---------------|------------|
| LR_EN      | 3.3 hr     | 3.3 hr        | -          |
| LinSVM_cal | 5.0 hr     | 5.0 hr        | -          |
| RF         | 13.3 hr    | 13.3 hr       | **YES**    |
| XGBoost    | 8.3 hr     | 8.3 hr        | -          |

**Expected wall-clock time**: 14-16 hours (all splits run in parallel)
**Total core-hours**: ~4,500 (efficient use of premium queue)

---

## Pre-Flight Checklist

- [ ] Verify HPC allocation active: `acc_Chipuk_Laboratory`
- [ ] Data file exists: `ls ../data/Celiac_dataset_proteomics_w_demo.parquet`
- [ ] Splits config set to 10: `grep n_splits configs/splits_config.yaml`
- [ ] Pipeline points to production config: `grep training_config_production configs/pipeline_hpc.yaml`
- [ ] Disk space available (≥50 GB): `df -h ../results`
- [ ] HPC environment active: `source venv/bin/activate && ced --version`

---

## Launch Commands

### Option 1: Automated launch (recommended)
```bash
cd analysis/
./run_overnight_L3.sh
```

### Option 2: Manual launch
```bash
cd analysis/

# 1. Update splits config (if needed)
sed -i.bak 's/n_splits: 2/n_splits: 10/' configs/splits_config.yaml

# 2. Generate splits
OVERWRITE_SPLITS=1 ced save-splits \
  --config configs/splits_config.yaml \
  --infile ../data/Celiac_dataset_proteomics_w_demo.parquet

# 3. Dry run (optional)
DRY_RUN=1 ./run_hpc.sh

# 4. Launch
./run_hpc.sh
```

---

## Monitoring

### Check job status
```bash
# All jobs
bjobs -w | grep CeD_

# Count by status
bjobs -w | grep CeD_ | grep DONE | wc -l
bjobs -w | grep CeD_ | grep RUN | wc -l
bjobs -w | grep CeD_ | grep PEND | wc -l

# Specific model
bjobs -w | grep CeD_LR_EN
```

### Monitor progress
```bash
# Watch latest job output
tail -f logs/{RUN_ID}/CeD_LR_EN_seed0.out

# Check for errors
grep -i error logs/{RUN_ID}/CeD_*.err

# Resource usage
bjobs -l <JOB_ID> | grep -E "(RUNLIMIT|MEMLIMIT)"
```

---

## Post-Processing (After All Jobs Complete)

### Step 1: Validate and aggregate base models
```bash
# Wait for all jobs to show DONE status
bjobs -w | grep CeD_

# Run post-processing pipeline
bash scripts/post_training_pipeline.sh --run-id <RUN_ID>
```

This validates 40 model outputs and aggregates results across 10 splits per model.

### Step 2: Train ensemble (optional)
```bash
bash scripts/post_training_pipeline.sh \
  --run-id <RUN_ID> \
  --train-ensemble
```

Expected +2-5% AUROC improvement over best single model.

### Step 3: Review aggregated results
```bash
# Check aggregation outputs
ls ../results/{LR_EN,RF,XGBoost,LinSVM_cal}/run_{RUN_ID}/aggregated/

# Key files per model:
# - aggregated_metrics.json     # Pooled metrics across splits
# - feature_stability.csv       # Feature selection consistency
# - calibration_aggregated.png  # Calibration plot
# - roc_aggregated.png          # ROC curves
# - dca_aggregated.png          # Decision curve analysis
```

---

## Expected Outcomes

### Performance Metrics
- **Test AUROC**: 0.85-0.88 (best model), SE < 0.006
- **Calibration**: Brier score < 0.015, slope 0.95-1.05
- **Clinical utility**: Positive net benefit at 1-10% threshold range
- **OOF-test gap**: < 0.02 (minimal overfitting)

### Model Ranking
- Clear winner with non-overlapping 95% CIs
- Expected order: RF > XGBoost > LR_EN > LinSVM_cal
- Ensemble: +2-5% AUROC boost over best single model

### Feature Panel
- 50-300 stable proteins (≥75% selection frequency across splits)
- Feature count convergence (std/mean < 30%)
- Consensus panel for clinical translation

---

## Troubleshooting

### Job fails with OOM error
```bash
# Edit configs/pipeline_hpc.yaml
mem_per_core: 16000  # Increase from 8000 to 16 GB
cores: 4             # Reduce from 8 to 4

# Re-submit failed split
ced train --model RF --split-seed 3 \
  --config configs/training_config_production.yaml
```

### Job times out (>48 hours)
```bash
# Option 1: Reduce trial count
# Edit training_config_production.yaml
optuna:
  n_trials: 50  # Down from 100

# Option 2: Increase walltime
# Edit pipeline_hpc.yaml
walltime: "72:00"  # Up from 48:00
```

### Missing split files
```bash
# Re-generate splits
OVERWRITE_SPLITS=1 ced save-splits \
  --config configs/splits_config.yaml \
  --infile ../data/Celiac_dataset_proteomics_w_demo.parquet
```

### Aggregation fails
```bash
# Check which models completed
ls ../results/*/run_{RUN_ID}/split_seed*/core/test_metrics.csv

# Aggregate only completed models
ced aggregate-splits \
  --results-dir ../results/LR_EN/run_{RUN_ID}
```

---

## File Outputs

### Per-split outputs (40 directories)
```
results/{MODEL}/run_{RUN_ID}/split_seed{0-9}/
├── core/
│   ├── test_metrics.csv          # Test set performance
│   ├── val_metrics.csv           # Validation set performance
│   └── oof_metrics.csv           # OOF performance
├── preds/
│   ├── train_oof/                # OOF predictions (for ensemble)
│   ├── val_preds/                # Validation predictions
│   └── test_preds/               # Test predictions
├── cv/
│   ├── optuna/                   # Optuna study database
│   │   └── best_params.json     # Best hyperparameters
│   └── fold_*/                   # Per-fold metrics
├── plots/
│   ├── roc.png                   # ROC curve
│   ├── calibration.png           # Calibration plot
│   ├── dca.png                   # Decision curve
│   └── learning_curve.png        # Learning curve (if enabled)
└── config.yaml                   # Full config used
```

### Aggregated outputs (4 directories, one per model)
```
results/{MODEL}/run_{RUN_ID}/aggregated/
├── aggregated_metrics.json       # Pooled metrics (mean ± SE)
├── aggregated_metrics.csv        # Same as table
├── feature_stability.csv         # Feature selection across splits
├── calibration_aggregated.png    # Aggregated calibration
├── roc_aggregated.png            # Aggregated ROC curves
└── dca_aggregated.png            # Aggregated DCA
```

### Logs
```
logs/{RUN_ID}/
├── CeD_LR_EN_seed0.out           # stdout for split 0
├── CeD_LR_EN_seed0.err           # stderr for split 0
├── ...                           # (40 pairs total)
└── run_summary.txt               # Job submission summary

logs/post/post_{RUN_ID}/
├── post_training.log             # Post-processing log
└── pipeline_summary.json         # Validation report
```

---

## References

- [OPTIMIZATION_PLAN.md](docs/OPTIMIZATION_PLAN.md) - Full optimization strategy
- [POST_TRAINING_GUIDE.md](POST_TRAINING_GUIDE.md) - Detailed post-processing guide
- [CLAUDE.md](.claude/CLAUDE.md) - Complete project documentation
- [ADR-009](docs/adr/ADR-009-oof-stacking-ensemble.md) - Ensemble design
- [ADR-014](docs/adr/ADR-014-oof-posthoc-calibration.md) - Calibration strategy

---

**Last Updated**: 2026-01-23
