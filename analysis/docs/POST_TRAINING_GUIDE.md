# Post-Training Pipeline Guide

## Overview

After running `run_hpc.sh` and your HPC jobs complete, you need to run **post-processing** to:
1. Validate base model outputs
2. Train ensemble meta-learner (optional, via flag)
3. Aggregate results across splits
4. Generate validation reports

## Quick Start

```bash
cd analysis/

# Check job status
bjobs -w | grep CeD_

# When all jobs are DONE, run post-processing
bash scripts/post_training_pipeline.sh --run-id <YOUR_RUN_ID>

# Check results
cat logs/post/post_20260122_120000/pipeline_summary_20260122_120000.json
ls results/*/run_20260122_120000/aggregated/
```

Replace `<YOUR_RUN_ID>` with the timestamp from `run_hpc.sh` output (e.g., `20260122_120000`).

## post_training_pipeline.sh configurable options avalable

#==============================================================
### ARGUMENT PARSING
#==============================================================
TRAIN_ENSEMBLE=1
SKIP_ENSEMBLE=0
RUN_ID=""
RESULTS_DIR=""
CONFIG_FILE=""
BASE_MODELS=""
MIN_SPLITS=1

## What the Pipeline Does

### Step 1: Validate Base Model Outputs
- Checks each model's output for required files:
  - `core/test_metrics.csv`
  - `preds/train_oof/train_oof__{MODEL}.csv`
  - `preds/test_preds/test_preds__{MODEL}.csv`
  - `preds/val_preds/val_preds__{MODEL}.csv`
- Reports which models completed successfully
- Only aggregates validated models

### Step 2: Train Ensemble Meta-Learner
- Collects OOF predictions from base models
- Trains L2 logistic regression meta-learner
- Generates ensemble predictions on val/test sets
- Saves to `results/ENSEMBLE/run_{run_id}/split_seed{X}/`
- **Enabled by default** (use `--skip-ensemble` flag to disable)

### Step 3: Aggregate Results
- Runs `ced aggregate-splits` for each validated model
- Includes ENSEMBLE if trained
- Computes:
  - Pooled metrics across splits
  - Per-split summary statistics
  - Feature stability analysis
  - Consensus panels
  - Aggregated plots
- Saves to `results/{MODEL}/run_{run_id}/aggregated/`

### Step 4: Validation Report
- Generates summary JSON with:
  - Validated vs. missing models
  - Ensemble training status
  - Aggregation success/failures
  - Total aggregated files
- Saves to `logs/post/run_{run_id}/pipeline_summary.json`

## Logging

All post-processing steps are logged to:
```
logs/post/run_{run_id}/post_training.log
```

Log entries include:
- `[SUCCESS]` - Successful operations
- `[WARNING]` - Non-critical issues (e.g., missing splits)
- `[ERROR]` - Critical failures
- Timestamped entries for tracing

## Expected Output

### Successful Run
```
[2026-01-22 14:30:00] ============================================
[2026-01-22 14:30:00] Post-Training Pipeline
[2026-01-22 14:30:00] ============================================
[2026-01-22 14:30:00] Run ID: 20260122_120000
...
[2026-01-22 14:30:10] [SUCCESS] LR_EN: 10/10 splits completed
[2026-01-22 14:30:11] [SUCCESS] RF: 10/10 splits completed
[2026-01-22 14:30:12] [SUCCESS] XGBoost: 10/10 splits completed
...
[2026-01-22 14:31:00] [SUCCESS] Ensemble trained for seed 0
...
[2026-01-22 14:35:00] [SUCCESS] Aggregated LR_EN
[2026-01-22 14:35:30] [SUCCESS] Aggregated RF
[2026-01-22 14:36:00] [SUCCESS] Aggregated XGBoost
[2026-01-22 14:36:30] [SUCCESS] Aggregated ENSEMBLE
...
[2026-01-22 14:37:00] ============================================
[2026-01-22 14:37:00] Post-Training Pipeline Complete
[2026-01-22 14:37:00] ============================================
```

### Summary JSON
```json
{
  "timestamp": "2026-01-22T14:37:00-05:00",
  "run_id": "20260122_120000",
  "results_dir": "/path/to/results",
  "base_models": {
    "validated": ["LR_EN", "RF", "XGBoost"],
    "missing": []
  },
  "ensemble": {
    "requested": 1,
    "trained": 10,
    "failed": 0
  },
  "aggregation": {
    "successful": ["LR_EN", "RF", "XGBoost", "ENSEMBLE"],
    "failed": []
  },
  "files": {
    "total_aggregated": 542,
    "log_file": "logs/post/post_20260122_120000/training_20260122_120000.log"
  }
}
```

## Best Practices

1. **Always check job status** before post-processing
2. **Review validation step output** to catch failed models early
3. **Save log files** for reproducibility

## See Also

- [CLAUDE.md](../.claude/CLAUDE.md) - Full project documentation
- [ARCHITECTURE.md](../analysis/docs/ARCHITECTURE.md) - Technical architecture
- [ADR-009](../analysis/docs/adr/ADR-009-oof-stacking-ensemble.md) - Ensemble design
