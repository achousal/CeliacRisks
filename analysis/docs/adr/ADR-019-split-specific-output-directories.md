# ADR-019: Split-Specific Output Subdirectories

**Status:** Accepted
**Date:** 2026-01-20
**Decision Makers:** Elahi Lab + Computational Team

## Context

The original pipeline design assumed a single split seed per training run, with outputs written directly to the specified output directory (e.g., `results_hpc/LR_EN/`). This worked well for single-split workflows but created problems for multi-split experiments:

**Problems with Flat Output Structure:**
1. **Overwriting:** Running multiple split seeds sequentially overwrites previous results
2. **No parallelization:** Cannot run multiple splits in parallel (file conflicts)
3. **Manual organization:** Users must manually rename/move directories between runs
4. **Lost provenance:** Difficult to track which results came from which split seed
5. **HPC job arrays:** Job array parallelism requires isolated output directories per task

**Use Cases Requiring Multi-Split Support:**
- **Split stability analysis:** Train same model on 5+ different splits to assess variance
- **HPC job arrays:** Parallelize across split seeds (e.g., `#BSUB -J "train[1-5]"`)
- **Ensemble models:** Combine predictions from models trained on different splits
- **Reproducibility audits:** Re-run experiments with different split seeds without losing prior results

Without split-specific subdirectories, these workflows require complex bash scripting and manual directory management, increasing error risk.

## Decision

Automatically nest all outputs under split-specific subdirectories when `split_seed` is provided.

**Directory Structure:**
```
results_hpc/
  LR_EN/
    split_seed42/           # Auto-created based on config.split_seed
      core/                 # Model artifacts, predictions, metrics
      cv/                   # CV results
      plots/                # Visualizations
      diag_splits/          # Split diagnostics
    split_seed43/
      core/
      cv/
      plots/
      diag_splits/
    split_seed44/
      ...
```

**Behavior:**
- If `split_seed` is `None`: Output directly to `{outdir}/` (backward compatible)
- If `split_seed` is set (e.g., `42`): Output to `{outdir}/split_seed42/`
- Subdirectory creation is automatic and transparent to user
- All existing output paths (`core/`, `cv/`, `plots/`) remain unchanged within subdirectory

**Implementation:**
```python
# In OutputDirectories.__init__()
if split_seed is not None:
    root_path = root_path / f"split_seed{split_seed}"
```

## Alternatives Considered

1. **User-specified output directory per split:**
   - Require users to pass `--outdir results_hpc/LR_EN_seed42` manually
   - Rejected: Error-prone, verbose CLI, no standardization

2. **Timestamp-based directories:**
   - Use `{outdir}/{timestamp}/` for isolation
   - Rejected: Harder to identify split seed from directory name, breaks reproducibility

3. **Flat structure with filename prefixes:**
   - Save files as `final_model_seed42.pkl`, `oof_predictions_seed42.csv`, etc.
   - Rejected: Doesn't prevent file conflicts in parallel runs, cluttered directory

4. **Separate top-level directories per split:**
   - `results_hpc_seed42/LR_EN/`, `results_hpc_seed43/LR_EN/`, ...
   - Rejected: Breaks existing directory conventions, harder to aggregate results

5. **Database storage for results:**
   - Store all outputs in SQLite/DuckDB with split_seed column
   - Rejected: Overkill, reduces accessibility, harder to inspect results

## Consequences

### Positive
- **Parallel safety:** Multiple splits can run simultaneously without conflicts
- **Provenance:** Split seed encoded in directory path (clear, self-documenting)
- **Backward compatible:** Existing single-split workflows unchanged (`split_seed=None`)
- **HPC-friendly:** Enables job array parallelism (e.g., `bsub -J "train[1-5]"`)
- **Aggregation-ready:** Easy to glob all splits for downstream analysis (`split_seed*/core/test_metrics.json`)

### Negative
- **Directory depth:** One additional nesting level (minor)
- **Migration:** Existing results in flat structure must be manually moved if re-run with split_seed (one-time cost)
- **Wildcard paths:** Some scripts may need updating to handle `split_seed*/` glob patterns

## Evidence

### Code Pointers
- [evaluation/reports.py:OutputDirectories.__init__](../../src/ced_ml/evaluation/reports.py#L105-L106) - Split-specific nesting logic
- [cli/train.py](../../src/ced_ml/cli/train.py) - Passes `split_seed` to `OutputDirectories`
- [config/schema.py:TrainingConfig.split_seed](../../src/ced_ml/config/schema.py) - Configuration parameter

### Test Coverage
- `tests/test_evaluation_reports.py` - Validates output directory creation
- Manual testing: Confirmed parallel runs with different split seeds work correctly

### HPC Integration
**Job Array Example:**
```bash
#BSUB -J "train[1-5]"      # 5 parallel jobs
#BSUB -o logs/train_%I.out

# Map LSB_JOBINDEX to split seed
SPLIT_SEED=$((LSB_JOBINDEX * 10 + 42))  # 42, 52, 62, 72, 82

ced train --config config.yaml --override split_seed=${SPLIT_SEED}
# Outputs to: results_hpc/LR_EN/split_seed42/, split_seed52/, etc.
```

### References
- LSF Job Arrays: https://www.ibm.com/docs/en/spectrum-lsf/10.1.0?topic=jobs-job-arrays

## Related ADRs

- Related to: [ADR-013: Split Persistence Format](ADR-013-split-persistence.md) - Split index file naming
- Related to: [ADR-014: HPC Job Array](ADR-014-hpc-job-array.md) - HPC parallelization strategy
- Complements: Output artifact organization (core/, cv/, plots/ structure)
