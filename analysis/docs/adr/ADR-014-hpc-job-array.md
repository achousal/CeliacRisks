# ADR-014: HPC Job Array Pattern (4-Model Array)

**Status:** Accepted
**Date:** 2026-01-20
**Decision Makers:** Elahi Lab + Computational Team

## Context

Training 4 models (RF, XGBoost, LinSVM_cal, LR_EN) with nested CV (50,000 fits each) is computationally expensive:
- **Sequential execution:** 4 × 12 hours = 48 hours total wall time
- **Parallel execution:** 4 jobs × 12 hours = 12 hours wall time (75% reduction)

HPC systems (e.g., LSF, SLURM) support job arrays for parallel execution of similar tasks.

## Decision

Use **LSF job array** with 4 array indices (1-4), one per model.

**LSB_JOBINDEX mapping:**
- 1 → RF
- 2 → XGBoost
- 3 → LinSVM_cal
- 4 → LR_EN

**Resource Allocation per Job:**
- 16 cores
- 8 GB/core = 128 GB total
- 12-hour wall time

## Alternatives Considered

### Alternative A: Sequential Jobs
- Simpler submission (single job)
- **Rejected:** 4× longer wall time (48h vs. 12h)

### Alternative B: Manual Parallel Submission
- Submit 4 separate jobs manually
- **Rejected:** More error-prone; job array cleaner

### Alternative C: GNU Parallel or Similar
- Parallel execution within single job
- **Rejected:** Less HPC-native; job array better resource management

### Alternative D: Larger Job Array (10 Models)
- Plan for future model expansion
- **Rejected:** Only 4 models currently; overprovisioning

## Consequences

### Positive
- 75% reduction in wall time (48h → 12h)
- Parallel execution leverages HPC resources efficiently
- Job array pattern is standard HPC practice
- Easy to extend (add more models → increment array size)

### Negative
- Requires 4× resources simultaneously (16 cores × 4 = 64 cores total)
- Resource contention if HPC queue is full

## Evidence

### Code Pointers
- `CeD_production.lsf.template` - LSF batch script template
- `scripts/hpc_setup.sh` - Environment setup script
- `WORKFLOW.md` - Pipeline execution documentation

### Test Coverage
- No automated tests for HPC submission (requires HPC environment)
- Manual verification on HPC cluster

### References
- LSF documentation: Job arrays

## Related ADRs

- Depends on: [ADR-008: Nested CV Structure](ADR-008-nested-cv.md) (computational cost justifies parallel execution)
