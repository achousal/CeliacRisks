# Quick Start: HPC Pipeline

## Problem Summary

After running `run_hpc.sh`, you noticed:
1. No aggregated results when jobs finished
2. Ensemble training issues

**Root cause**: HPC jobs run asynchronously, so the aggregation step in `run_hpc.sh` runs before jobs finish.

## Solution: Post-Training Pipeline

A automated workflow handles everything after base model training:

```bash
cd analysis/

# 1. Submit training jobs
./run_hpc.sh
# Note the RUN_ID from output (e.g., 20260122_143022)

# 2. When all jobs done, run post-processing
bash scripts/post_training_pipeline.sh --run-id 20260122_143022
```

## What It Does

### 1. Validate Base Model Outputs
Checks each model's split directories for required files:
- `core/test_metrics.csv`
- `preds/train_oof/train_oof__{MODEL}.csv`
- `preds/test_preds/test_preds__{MODEL}.csv`
- `preds/val_preds/val_preds__{MODEL}.csv`

Only proceeds with models that have complete outputs.

### 2. Train Ensemble Meta-Learner
For each split seed:
- Collects OOF predictions from validated base models
- Trains L2 logistic regression meta-learner
- Generates ensemble predictions on val/test sets
- Saves to `results/ENSEMBLE/run_{run_id}/split_seed{X}/`

Skipped if `ensemble.enabled: false` or `--skip-ensemble` flag.

### 3. Aggregate Results
For each model (base + ENSEMBLE):
- Pools predictions across splits
- Computes metrics with bootstrap CIs
- Generates aggregated plots
- Builds feature stability reports
- Saves to `results/{MODEL}/run_{run_id}/aggregated/`

### 4. Generate Reports
Creates validation summary JSON:
```json
{
  "base_models": {"validated": [...], "missing": [...]},
  "ensemble": {"enabled": 1, "trained": 10, "failed": 0},
  "aggregation": {"successful": [...], "failed": [...]},
  "files": {"total_aggregated": 542, "log_file": "..."}
}
```

## Logging

All steps logged to `logs/post/post_{run_id}/training_{run_id}.log`:

```
[2026-01-22 14:30:00] ============================================
[2026-01-22 14:30:00] Post-Training Pipeline
[2026-01-22 14:30:00] ============================================
[2026-01-22 14:30:10] [SUCCESS] LR_EN: 10/10 splits completed
[2026-01-22 14:30:11] [SUCCESS] RF: 10/10 splits completed
...
[2026-01-22 14:31:00] [SUCCESS] Ensemble trained for seed 0
...
[2026-01-22 14:35:00] [SUCCESS] Aggregated LR_EN
[2026-01-22 14:36:30] [SUCCESS] Aggregated ENSEMBLE
...
[2026-01-22 14:37:00] [SUCCESS] All steps completed successfully
```

View in real-time:
```bash
tail -f logs/post/post_{run_id}/training_{run_id}.log
```

## Monitoring Jobs

Monitor job status with LSF commands:

```bash
# Check running jobs
bjobs -w | grep CeD_

# View live logs
tail -f logs/base/run_20260122_143022/*.live.log

# Check error logs
cat logs/base/run_20260122_143022/*.err
```

## Options

### Skip ensemble training
```bash
bash scripts/post_training_pipeline.sh --run-id 20260122_143022 --skip-ensemble
```

### Manual base models
```bash
bash scripts/post_training_pipeline.sh --run-id 20260122_143022 --base-models LR_EN,RF
```

### Custom config
```bash
bash scripts/post_training_pipeline.sh --run-id 20260122_143022 --config configs/custom.yaml
```

### Minimum splits required
```bash
bash scripts/post_training_pipeline.sh --run-id 20260122_143022 --min-splits 5
```

## Troubleshooting

### "No validated base models found"
**Check**: Did training jobs fail?
```bash
cat logs/base/run_{run_id}/*.err
bjobs -w | grep CeD_
```

### "Ensemble training failed for seed X"
**Check**: Are OOF predictions present?
```bash
ls results/{MODEL}/run_{run_id}/split_seed{X}/preds/train_oof/
```

### "Aggregation failed for {MODEL}"
**Check**: Are split directories complete?
```bash
ls results/{MODEL}/run_{run_id}/split_seed*/core/test_metrics.csv
```

View full logs:
```bash
cat logs/post/post_{run_id}/training_{run_id}.log
```

## Full Workflow Example

```bash
cd analysis/

# Submit jobs
./run_hpc.sh
# Output: Run ID: 20260122_143022

# Monitor (wait for completion)
bjobs -w | grep CeD_
tail -f logs/base/run_20260122_143022/*.live.log

# Run post-processing (when all done)
bash scripts/post_training_pipeline.sh --run-id 20260122_143022

# View results
cat logs/post/post_20260122_143022/pipeline_summary_20260122_143022.json
ls results/*/run_20260122_143022/aggregated/
```

## Documentation

- **Full guide**: [POST_TRAINING_GUIDE.md](POST_TRAINING_GUIDE.md)
- **Project overview**: [.claude/CLAUDE.md](../.claude/CLAUDE.md)
- **Architecture**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Changes to run_hpc.sh

The HPC runner now:
1. Submits base model training jobs
2. Submits ensemble jobs (with dependencies)
3. Saves run metadata
4. **Prints post-processing instructions** (instead of trying to aggregate immediately)

Old behavior (broken):
```bash
# Step 4: Aggregate (runs immediately, finds nothing)
ced aggregate-splits --results-dir results/{model}/run_{id}
```

New behavior (correct):
```bash
# Step 4: Instructions
echo "Run after jobs complete:"
echo "  bash scripts/post_training_pipeline.sh --run-id {id}"
```

## Benefits

1. **Comprehensive logging** - Every step tracked with timestamps
2. **Validation** - Catches missing outputs before aggregation
3. **Idempotent** - Safe to re-run (skips already-completed steps)
4. **Flexible** - Skip ensemble, customize models, adjust thresholds
5. **Status tracking** - Summary JSON for reproducibility

## Next Steps

After successful post-processing:

```bash
# View model comparison
cat results/LR_EN/run_20260122_143022/aggregated/reports/model_comparison.csv

# View ensemble improvement
cat results/ENSEMBLE/run_20260122_143022/aggregated/aggregation_metadata.json

# Generate R plots (optional)
Rscript scripts/compare_models.R --results_root results/
```
