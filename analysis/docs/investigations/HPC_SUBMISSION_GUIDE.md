# HPC Investigation Submission Guide

**TL;DR**: Use `submit_hpc_investigation.sh` to distribute jobs across HPC nodes instead of running locally.

---

## Why HPC?

Running 40 investigations (4 models × 10 seeds) locally is computationally intensive:
- **Per-job cost**: 30-60 seconds (feature AUROC calculations on 2,920 proteins)
- **Total time**: ~20-40 minutes sequential
- **Memory**: Can spike during feature analysis
- **CPU**: Fully utilized during processing

**HPC solution**: Parallelize across 40 separate jobs, finish in 1-2 minutes of wall-clock time.

---

## Setup

### 1. HPC Environment
First time only:

```bash
cd analysis/
bash scripts/hpc_setup.sh
source venv/bin/activate
```

This creates a Python virtual environment on HPC with all dependencies.

---

## Submission Workflow

### Step 1: Preview Job Configuration (Optional)
```bash
cd analysis/docs/investigations/

# Show what will be submitted (no jobs sent)
bash submit_hpc_investigation.sh --dry-run
```

This displays:
- Total job count
- Model/seed mapping
- Resource allocation
- How to actually submit

### Step 2: Submit Full Investigation
**All models, all 10 seeds:**
```bash
bash submit_hpc_investigation.sh
```

**Specific models only:**
```bash
bash submit_hpc_investigation.sh --models LR_EN,RF
```

**Specific seed range:**
```bash
bash submit_hpc_investigation.sh --seeds 0-4
```

**Custom resources:**
```bash
bash submit_hpc_investigation.sh --cores 4 --memory 16G --walltime 02:00 --queue long
```

### Output
```
Job array submitted with ID: 123456789

Monitor progress:
  bjobs -w 123456789                    # Job status
  tail -f logs/investigate_*.log        # Live logs
  bjobs -a 123456789 | wc -l            # Count completed jobs

After all jobs complete:
  python consolidate_hpc_results.py
```

---

## Monitoring

### Check Job Status
```bash
# Active jobs
bjobs -w JOB_ID

# All jobs (active + completed)
bjobs -a JOB_ID

# Jobs per array index
bjobs -l JOB_ID
```

### View Logs
```bash
# See first 50 lines of completed job
head -50 logs/investigate_1.log

# Tail live logs as they write
tail -f logs/investigate_*.log

# Count how many jobs completed
ls logs/investigate_*.log | wc -l
```

### Estimated Timing
- **Setup**: 1-2 minutes (environment loading)
- **Per-job**: 30-60 seconds (depending on model complexity)
- **Parallelization**: ~40 jobs simultaneously
- **Total wall-clock**: ~2-3 minutes from submission to completion
- **Overhead**: Job queue wait time (typically < 5 minutes)

---

## Consolidation

### After All Jobs Complete
```bash
# Check that all 40 jobs finished
bjobs JOB_ID        # Should show "DONE" status

# Run consolidation script (from analysis directory)
cd analysis/
python docs/investigations/consolidate_hpc_results.py
```

This:
1. Verifies all output files exist
2. Combines per-seed summaries into aggregate tables
3. Generates per-model statistics
4. Produces interpretation guide

### Outputs
All files saved to `results/investigations/`:

| File | Description |
|------|-------------|
| `CONSOLIDATED_SUMMARY.csv` | All 40 runs combined (1 row per job) |
| `MODEL_SUMMARY.csv` | Aggregates by model (4 rows) |
| `summary_*.csv` | Individual seed results |
| `distributions_*.png` | Score distribution plots |
| `calibration_*.png` | Calibration quality analysis |
| `feature_bias_*.png` | Feature selection bias |
| `feature_bias_details_*.csv` | Per-protein AUROC scores |

---

## Troubleshooting

### Job Submission Failed
```bash
# Check that you're in analysis directory
pwd        # Should end with /analysis

# Verify HPC setup
source venv/bin/activate
python -c "import ced_ml; print('OK')"

# Try manual submission
bsub < logs/submit_investigation_*.sh
```

### Jobs Completed But No Results
```bash
# Check job output
cat logs/investigate_1.log

# Verify results directory exists
ls -la results/investigations/

# Manually run one investigation to test
python docs/investigations/investigate.py --mode oof --model LR_EN --split-seed 0 --analyses distributions
```

### Consolidation Script Fails
```bash
# Check number of summary files
ls results/investigations/summary_*.csv | wc -l    # Should be ~40

# Check for file corruption
head results/investigations/summary_*.csv

# Run with verbose error handling
python -u consolidate_hpc_results.py
```

### Slow Job Performance
Check available resources:
```bash
# Queue status
bqueues

# Utilization
bhosts

# If queue is full, try:
bash submit_hpc_investigation.sh --queue short --seeds 0-3
```

---

## Resource Recommendations

### Conservative (guaranteed fast)
```bash
--cores 2 --memory 8G --walltime 01:00 --queue medium
```
- Safe for most nodes
- Fastest queue time
- Recommended for first run

### Balanced
```bash
--cores 4 --memory 16G --walltime 02:00 --queue medium
```
- Good for large feature sets
- Still reasonable queue time

### Aggressive (if queue allows)
```bash
--cores 8 --memory 32G --walltime 04:00 --queue long
```
- For parallel internal operations
- Longer queue wait but faster per-job execution

**Recommendation**: Start with conservative, scale up only if jobs timeout.

---

## Advanced: Partial Re-runs

### Re-run Only Failed Jobs
If some jobs fail:

1. Check which models/seeds failed:
   ```bash
   # Count successful outputs
   ls results/investigations/summary_*.csv | wc -l

   # Compare to expected (e.g., 40 for all models/seeds)
   ```

2. Re-run specific subset:
   ```bash
   # Example: re-run only XGBoost and LinSVM_cal
   bash submit_hpc_investigation.sh --models XGBoost,LinSVM_cal --seeds 0-9
   ```

3. Consolidate including both batches:
   ```bash
   python consolidate_hpc_results.py    # Automatically combines all summary_*.csv
   ```

---

## Environment Variables

If you need to override defaults:

```bash
# Example: submit with custom resource class
export HPC_QUEUE="long"
export HPC_CORES=8
bash submit_hpc_investigation.sh
```

---

## Next Steps After Consolidation

1. **Review results**: Open `CONSOLIDATED_SUMMARY.csv`
2. **Check per-model**: Review `MODEL_SUMMARY.csv`
3. **Visual inspection**: Look at `distributions_*.png`, `calibration_*.png`, `feature_bias_*.png`
4. **Interpret**: Use consolidation script's interpretation guide
5. **Document**: If findings are significant, update project CLAUDE.md

---

**Last Updated**: 2026-01-26
