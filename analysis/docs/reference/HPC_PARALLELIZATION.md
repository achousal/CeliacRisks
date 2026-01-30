# HPC Parallelization Architecture

**Last Updated**: 2026-01-30
**Status**: Production-ready

---

## Overview

The CeD-ML pipeline supports HPC job submission with intelligent parallelization to minimize wall-clock time. When using `ced run-pipeline --hpc`, the system submits multiple LSF jobs with dependency chains to execute training, aggregation, panel optimization, and consensus generation efficiently.

---

## Job Dependency Architecture

### Four-Tier Execution

```
Tier 1: Training Jobs (parallel)
  ├── CeD_{RUN_ID}_seed0
  ├── CeD_{RUN_ID}_seed1
  └── CeD_{RUN_ID}_seed2
         ↓ (all complete)
Tier 2: Post-Processing Job
  └── CeD_{RUN_ID}_post (aggregation + ensemble)
         ↓ (complete)
Tier 3: Panel Optimization Jobs (parallel)
  ├── CeD_{RUN_ID}_panel_LR_EN
  ├── CeD_{RUN_ID}_panel_RF
  └── CeD_{RUN_ID}_panel_XGBoost
         ↓ (all complete)
Tier 4: Consensus Panel Job
  └── CeD_{RUN_ID}_consensus
```

### Job Details

**Tier 1: Training Jobs**
- **Count**: One job per split seed (default: 3 jobs for seeds 0, 1, 2)
- **Purpose**: Train all models for a single split seed
- **Command**: `ced run-pipeline --models ... --split-seeds N --no-ensemble --no-consensus --no-optimize-panel`
- **Parallelization**: All training jobs run simultaneously
- **Dependency**: None (first tier)
- **Typical duration**: 30-60 minutes per job (depends on model complexity)

**Tier 2: Post-Processing Job**
- **Count**: 1 job
- **Purpose**: Aggregate results across splits and train ensemble meta-learner
- **Command**: Multi-line bash script running:
  - `ced aggregate-splits --run-id {RUN_ID} --model {MODEL}` (for each base model)
  - `ced train-ensemble --run-id {RUN_ID} --split-seed {SEED}` (if ensemble enabled)
  - `ced aggregate-splits --run-id {RUN_ID} --model ENSEMBLE` (if ensemble enabled)
- **Parallelization**: Sequential within job (but only 1 job needed)
- **Dependency**: `done(CeD_{RUN_ID}_seed*)`
- **Typical duration**: 5-10 minutes (mostly I/O and metrics computation)

**Tier 3: Panel Optimization Jobs** (NEW)
- **Count**: One job per base model (e.g., 3 jobs for LR_EN, RF, XGBoost)
- **Purpose**: Run computationally expensive RFE to find minimum viable protein panels
- **Command**: `ced optimize-panel --run-id {RUN_ID} --model {MODEL}`
- **Parallelization**: All panel optimization jobs run simultaneously
- **Dependency**: `done(CeD_{RUN_ID}_post)`
- **Typical duration**: 10-30 minutes per job (depends on RFE configuration)
- **Why parallel?**: Panel optimization is the most expensive post-processing step. Running one job per model provides ~3x speedup for 3 models.

**Tier 4: Consensus Panel Job**
- **Count**: 1 job
- **Purpose**: Generate cross-model consensus panel via Robust Rank Aggregation
- **Command**: `ced consensus-panel --run-id {RUN_ID}`
- **Parallelization**: N/A (single job, relatively fast)
- **Dependency**: `done(CeD_{RUN_ID}_post) && done(CeD_{RUN_ID}_panel_*)`
- **Typical duration**: 5-10 minutes

---

## Speedup Analysis

### Before (Sequential Post-Processing)

```
Training jobs (parallel)           : 60 min
Post-processing (sequential)       : 50 min
  ├── Aggregation (3 models)       : 5 min
  ├── Ensemble training            : 5 min
  ├── Panel optimization (3 models): 30 min  ← bottleneck
  └── Consensus panel              : 10 min
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total wall-clock time              : 110 min
```

### After (Parallel Panel Optimization)

```
Training jobs (parallel)           : 60 min
Post-processing (aggregation only) : 10 min
Panel optimization (parallel)      : 10 min  ← 3x faster
Consensus panel                    : 10 min
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total wall-clock time              : 90 min
```

**Speedup**: 20 minutes (18% reduction) for 3 models
**Scaling**: Speedup increases linearly with number of models

---

## Usage Examples

### Basic HPC Submission

```bash
# Submit with default config (configs/pipeline_hpc.yaml)
ced run-pipeline --hpc

# Preview without submitting
ced run-pipeline --hpc --dry-run
```

### Custom Configuration

```bash
# Custom models and seeds
ced run-pipeline --hpc \
  --models LR_EN,RF,XGBoost,LinSVM_cal \
  --split-seeds 0,1,2,3,4

# Custom HPC config
ced run-pipeline --hpc \
  --hpc-config configs/pipeline_custom_resources.yaml
```

### Monitoring Jobs

```bash
# Check job status
bjobs -w | grep CeD_

# Live logs (all jobs)
tail -f logs/training/run_*/CeD_*.live.log

# Error logs
cat logs/training/run_*/CeD_*.err

# Check specific job
bjobs -l <JOB_ID>
```

### Example Output

```
====================================================================
Pipeline Submission Complete
====================================================================
Run ID: 20260130_120000
Training jobs: 3
Post-processing job: 12345
Panel optimization jobs: 3 (parallel)
Consensus panel job: 12349

Monitor jobs:
  bjobs -w | grep CeD_

Live logs:
  tail -f logs/training/run_20260130_120000/*.live.log

Error logs:
  cat logs/training/run_20260130_120000/*.err

Results: results/run_20260130_120000/
====================================================================
```

---

## Configuration

### HPC Resource Settings

Edit `analysis/configs/pipeline_hpc.yaml`:

```yaml
hpc:
  project: YOUR_ALLOCATION        # LSF project code
  queue: medium                   # LSF queue (short, medium, long)
  cores: 4                        # CPU cores per job
  mem_per_core: 4096              # Memory in MB per core
  walltime: "12:00"               # Wall time limit (HH:MM)

pipeline:
  models:
    - LR_EN
    - RF
    - XGBoost
  ensemble: true                  # Enable ensemble training
  consensus: true                 # Enable consensus panel
  optimize_panel: true            # Enable parallel panel optimization
```

### Disabling Features

```bash
# Skip ensemble training
ced run-pipeline --hpc --no-ensemble

# Skip panel optimization (saves ~20 min)
ced run-pipeline --hpc --no-optimize-panel

# Skip consensus panel
ced run-pipeline --hpc --no-consensus

# Minimal pipeline (training + aggregation only)
ced run-pipeline --hpc --no-ensemble --no-optimize-panel --no-consensus
```

---

## Implementation Details

### LSF Job Script Structure

Each job uses the following template:

```bash
#!/bin/bash
#BSUB -P PROJECT_ALLOCATION
#BSUB -q QUEUE_NAME
#BSUB -J JOB_NAME
#BSUB -n NUM_CORES
#BSUB -W WALLTIME
#BSUB -R "span[hosts=1] rusage[mem=MEM_PER_CORE]"
#BSUB -oo /dev/null
#BSUB -eo ERROR_LOG.err
#BSUB -w "DEPENDENCY_EXPRESSION"  # Optional

set -euo pipefail
export PYTHONUNBUFFERED=1
export FORCE_COLOR=1

# Activate Python environment (venv or conda)
source venv/bin/activate  # or: conda activate ENV_NAME

# Run command with live logging
stdbuf -oL -eL COMMAND 2>&1 | tee -a "LIVE_LOG.live.log"

exit ${PIPESTATUS[0]}
```

### Dependency Expressions

LSF dependency syntax:

- `done(CeD_{RUN_ID}_seed*)`: Wait for all training jobs
- `done(CeD_{RUN_ID}_post)`: Wait for post-processing job
- `done(CeD_{RUN_ID}_panel_*)`: Wait for all panel optimization jobs
- `A && B`: Wait for both A and B

---

## Troubleshooting

### Job Fails Immediately

**Check error log:**
```bash
cat logs/training/run_*/CeD_*.err
```

**Common causes:**
- Python environment not activated (check `analysis/venv/bin/activate`)
- Missing data file (check `--infile` path)
- Invalid project allocation (check `hpc.project` in config)

### Panel Optimization Timeout

**Symptoms:** Job killed by scheduler after walltime limit

**Solutions:**
1. Increase walltime in `pipeline_hpc.yaml`:
   ```yaml
   hpc:
     walltime: "24:00"  # Increase to 24 hours
   ```

2. Use faster RFE settings in `training_config.yaml`:
   ```yaml
   features:
     rfe_target_size: 100        # Larger target (stops earlier)
     rfe_step_strategy: geometric  # Faster than fine
   ```

3. Reduce RFE cross-validation folds (sacrifice robustness for speed):
   ```yaml
   cv:
     n_inner: 3  # Reduce from 5
   ```

### Consensus Job Fails

**Check dependencies:**
```bash
# Ensure all panel optimization jobs completed successfully
bjobs -w | grep panel

# Check if aggregation completed
ls -l results/run_*/*/aggregated/panels/feature_stability_summary.csv
```

### Environment Detection Issues

**Symptoms:** "No Python environment detected"

**Solutions:**
1. Run HPC setup script:
   ```bash
   bash analysis/scripts/hpc_setup.sh
   ```

2. Manually activate before submitting:
   ```bash
   source analysis/venv/bin/activate  # or conda activate
   ced run-pipeline --hpc
   ```

---

## Design Rationale

### Why Not Parallelize Everything?

**Training jobs**: Already parallelized (one per split seed)

**Aggregation**: Sequential by design (reads all splits for one model). Cannot parallelize across splits without violating data integrity. Could parallelize across models, but aggregation is fast (~1-2 min per model) so overhead of separate jobs outweighs benefits.

**Ensemble training**: Sequential dependency (must wait for base model aggregation). Also fast (~5 min total for all seeds).

**Panel optimization**: PARALLELIZED (this feature). Computationally expensive (~10-30 min per model) with no cross-model dependencies. Ideal candidate for parallelization.

**Consensus panel**: Single job by design (aggregates across all models). Fast (~5-10 min) and must wait for all panel optimizations anyway.

### Why Not Submit Even More Jobs?

**Panel optimization per split seed?** No. The aggregated RFE approach is preferred because it:
- Uses consensus stable proteins from ALL splits (more robust)
- Pools train+val data for maximum robustness
- Generates single authoritative panel size recommendation

Per-split panel optimization would create 3x more jobs (one per seed) but provide less robust recommendations. The current design balances parallelism with statistical soundness.

---

## Future Enhancements

Potential areas for further optimization:

1. **Parallel aggregation** (per model)
   Current: Sequential aggregation in post-processing job
   Proposed: One aggregation job per model (3 parallel jobs for 3 models)
   Speedup: ~3 min (minor, may not justify job overhead)

2. **GPU support** for XGBoost/Neural networks
   Current: CPU-only jobs
   Proposed: Request GPU nodes for specific models
   Speedup: 2-5x for GPU-compatible models

3. **Dynamic resource allocation**
   Current: Same resources for all jobs
   Proposed: Variable cores/memory based on job type (e.g., more cores for ensemble training)
   Benefit: More efficient cluster utilization

4. **Smart job retry**
   Current: Manual re-submission if job fails
   Proposed: Automatic retry with increased resources
   Benefit: Improved fault tolerance

---

## References

- [CLI Reference](CLI_REFERENCE.md) - Complete command documentation
- [Feature Selection Guide](FEATURE_SELECTION.md) - Panel optimization details
- [Architecture](../ARCHITECTURE.md) - System design overview
- [LSF Documentation](https://www.ibm.com/docs/en/spectrum-lsf/10.1.0) - IBM Spectrum LSF reference

---

**Last Updated**: 2026-01-30
**Maintainer**: Andres Chousal
