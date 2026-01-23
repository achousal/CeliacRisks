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
```

Replace `<YOUR_RUN_ID>` with the timestamp from `run_hpc.sh` output (e.g., `20260122_120000`).

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
- **Skipped by default** (use `--train-ensemble` flag to enable)

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
- Saves to `logs/post/post_{run_id}/pipeline_summary_{run_id}.json`

## Logging

All post-processing steps are logged to:
```
logs/post/post_{run_id}/training_{run_id}.log
```

View in real-time:
```bash
tail -f logs/post/post_{run_id}/training_{run_id}.log
```

Log entries include:
- `[SUCCESS]` - Successful operations
- `[WARNING]` - Non-critical issues (e.g., missing splits)
- `[ERROR]` - Critical failures
- Timestamped entries for tracing

## Options

### Train Ensemble (Opt-In)
```bash
bash scripts/post_training_pipeline.sh --run-id 20260122_120000 --train-ensemble
```
By default, ensemble training is skipped to save time. Use this flag to train the stacking ensemble.

### Custom Results Directory
```bash
bash scripts/post_training_pipeline.sh --run-id 20260122_120000 --results-dir /path/to/results
```

### Custom Config File
```bash
bash scripts/post_training_pipeline.sh --run-id 20260122_120000 --config configs/custom.yaml
```

### Manual Base Models
```bash
bash scripts/post_training_pipeline.sh --run-id 20260122_120000 --base-models LR_EN,RF,XGBoost
```

### Minimum Splits Required
```bash
bash scripts/post_training_pipeline.sh --run-id 20260122_120000 --min-splits 5
```

## Troubleshooting

### No validated base models
**Symptom**: Pipeline exits with "No validated base models found"

**Cause**: Model training jobs failed or are still running

**Fix**:
1. Check job status: `bjobs -w | grep CeD_`
2. Check error logs: `cat logs/{run_id}/*.err`
3. Wait for jobs to complete
4. Re-run post-processing

### Ensemble training fails
**Symptom**: "Ensemble training failed for seed X"

**Cause**: Missing OOF predictions or calibration files

**Fix**:
1. Check base model validation output
2. Verify OOF files exist: `ls results/{MODEL}/run_{id}/split_seed{X}/preds/train_oof/`
3. Re-train failed base model splits
4. Re-run post-processing

### Aggregation fails for a model
**Symptom**: "Aggregation failed for {MODEL}"

**Cause**: Insufficient completed splits or corrupt output files

**Fix**:
1. Check aggregation log section for details
2. Verify split directories: `ls results/{MODEL}/run_{id}/`
3. Manually run: `ced aggregate-splits --results-dir results/{MODEL}/run_{id}`

### Pipeline log not found
**Symptom**: Cannot find log file

**Fix**:
```bash
ls -ltr logs/post/
# Use most recent log file
cat logs/post/post_*/training_*.log
```

## Manual Alternatives

### Train Ensemble Manually
```bash
# For each split
for seed in 0 1 2; do
  ced train-ensemble \
    --results-dir results/ \
    --base-models LR_EN,RF,XGBoost \
    --split-seed $seed \
    --outdir results/ENSEMBLE/run_20260122_120000/split_seed${seed}
done
```

### Aggregate Manually (Per Model)
```bash
# Base models
ced aggregate-splits --results-dir results/LR_EN/run_20260122_120000
ced aggregate-splits --results-dir results/RF/run_20260122_120000
ced aggregate-splits --results-dir results/XGBoost/run_20260122_120000

# Ensemble (if trained)
ced aggregate-splits --results-dir results/ENSEMBLE/run_20260122_120000
```

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

## Integration with HPC Workflow

### Full Workflow
```bash
# 1. Submit training jobs
cd analysis/
./run_hpc.sh
# Note the RUN_ID from output (e.g., 20260122_120000)

# 2. Monitor jobs
bjobs -w | grep CeD_
tail -f logs/20260122_120000/*.live.log

# 3. When all jobs complete, run post-processing
bash scripts/post_training_pipeline.sh --run-id 20260122_120000 --train-ensemble

# 4. Check results
cat logs/post/post_20260122_120000/pipeline_summary_20260122_120000.json
ls results/*/run_20260122_120000/aggregated/
```

### Automated Post-Processing (Advanced)
You can submit the post-processing script as a dependent HPC job:

```bash
# After run_hpc.sh, get job IDs
JOB_IDS=$(bjobs -w | grep "CeD_" | awk '{print $1}' | tr '\n' ',' | sed 's/,$//')

# Submit post-processing job with dependency
bsub -P YOUR_PROJECT \
  -q medium \
  -J "CeD_PostProcess" \
  -n 1 \
  -W "02:00" \
  -w "done($JOB_IDS)" \
  bash scripts/post_training_pipeline.sh --run-id 20260122_120000
```

## Best Practices

1. **Always check job status** before post-processing
2. **Review validation step output** to catch failed models early
3. **Save log files** for reproducibility
4. **Run post-processing once per run_id** (idempotent but time-consuming)
5. **Use --train-ensemble** only when you need the stacking ensemble (adds ~10-30 min per split)

## See Also

- [CLAUDE.md](../.claude/CLAUDE.md) - Full project documentation
- [ARCHITECTURE.md](../analysis/docs/ARCHITECTURE.md) - Technical architecture
- [ADR-009](../analysis/docs/adr/ADR-009-oof-stacking-ensemble.md) - Ensemble design
