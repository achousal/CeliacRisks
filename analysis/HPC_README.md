# HPC Deployment - Quick Reference

**CeliacRisks v1.0.0 - HPC Setup Guide**

---

## One-Time Setup

```bash
# 1. Clone repository
git clone git@github.com:achousal/CeliacRisks.git
cd CeliacRisks/analysis

# 2. Load Python module (adjust for your HPC)
module load python/3.9.0

# 3. Run automated setup
bash scripts/hpc_setup.sh

# 4. Verify installation
source venv/bin/activate
ced --help
```

---

## Before First Run

### 1. Copy Data File
```bash
# Place dataset in data directory
cp /path/to/shared/Celiac_dataset_proteomics.csv ../data/
```

### 2. Customize Batch Script
```bash
# Copy template
cp CeD_hpc.lsf.template CeD_hpc.lsf

# Edit these lines in CeD_hpc.lsf:
#   #BSUB -P YOUR_PROJECT_ALLOCATION
#   #BSUB -q normal
#   BASE_DIR="/path/to/your/CeliacRisks/analysis"
```

### 3. Verify Configuration
```bash
# Check config files
ls configs/*.yaml

# Edit if needed (optional)
vim configs/training_config.yaml
```

---

## Running the Pipeline

### Option A: Automated (Recommended)
```bash
bash run_hpc.sh
```

### Option B: LSF Array Job
```bash
# Submit all 4 models
bsub < CeD_hpc.lsf

# Monitor
bjobs -w
```

### Option C: Individual Models
```bash
# Activate environment
source venv/bin/activate

# Run single model
ced train \
  --config configs/training_config.yaml \
  --model LR_EN \
  --infile ../data/Celiac_dataset_proteomics.csv \
  --splits-dir splits_hpc
```

---

## Monitoring

```bash
# Check job status
bjobs -w | grep CeD

# Check for running jobs (via .live logs)
bash scripts/check_jobs.sh
ls logs/*.live

# View real-time logs
tail -f logs/CeD_train_*.out.live   # While running
tail -f logs/CeD_train_*.out         # After completion

# Check for errors
grep -i error logs/*.err

# Clean up stale .live logs (if jobs crashed)
bash scripts/check_jobs.sh --cleanup
```

**Note:** Jobs write to `.live` log files while running. On successful completion,
logs are automatically renamed to remove `.live` extension. This makes it easy to
identify active jobs by looking for `.live` files.

---

## After Completion

```bash
# Verify outputs
ls results_hpc/IncidentPlusPrevalent__*/core/final_model.joblib

# Post-process results
ced postprocess --results-dir results_hpc --n-boot 500

# View aggregated metrics
column -t -s, results_hpc/COMBINED/aggregated_metrics.csv
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ced: command not found` | `source venv/bin/activate` |
| `Data file not found` | Verify `../data/Celiac_dataset_proteomics.csv` exists |
| `Out of memory` | Increase `-R "rusage[mem=16000]"` in batch script |
| `Job timeout` | Increase `-W 24:00` in batch script |

---

## File Structure

```
CeliacRisks/analysis/
├── scripts/
│   ├── hpc_setup.sh           # Automated setup
│   └── load_modules.sh        # Module loading
├── configs/
│   ├── splits_config.yaml     # Split generation config
│   └── training_config.yaml   # Model training config
├── CeD_hpc.lsf         # LSF batch script (customize)
├── CeD_hpc.lsf.template # Generic template
├── run_hpc.sh          # Orchestration script
└── docs/
    └── ARCHITECTURE.md        # Technical architecture
```

---

## Resources

- **Project Overview:** [CLAUDE.MD](CLAUDE.MD)
- **Architecture:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Workflow Guide:** [WORKFLOW.md](WORKFLOW.md)

---

## Quick Commands

```bash
# Setup
bash scripts/hpc_setup.sh

# Activate
source venv/bin/activate

# Submit
bsub < CeD_hpc.lsf

# Monitor
bjobs -w

# Postprocess
ced postprocess --results-dir results_hpc
```

---

**Ready to run? Start with:** `bash scripts/hpc_setup.sh`
